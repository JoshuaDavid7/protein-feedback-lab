"""One finite GPU job; no deployment, background service, retries, or persistent volume.

Run from the project root after authenticating:
    modal run cloud/modal_run.py --output cloud-output/gpu-study.zip
"""

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import modal

app = modal.App("protein-feedback-lab")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1",
        "numpy==1.26.4",
        "scipy==1.15.3",
        "scikit-learn==1.6.1",
        "matplotlib==3.10.3",
    )
    .add_local_dir("src/protein_feedback", "/root/protein_feedback", copy=True)
)

# Pricing checked 2026-10-31: A100-40GB $0.000583/sec; CPU
# $0.0000131/core/sec; memory $0.00000222/GiB/sec.
# At requested resources, 4h execution is about $9.30, before build overhead.
# This is a conservative planning guard, not a provider billing guarantee.
EXECUTION_SECONDS = 4 * 60 * 60
EXECUTION_RATE_PER_SECOND = 0.000583 + 2 * 0.0000131 + 16 * 0.00000222
PLANNING_CEILING_USD = 20.0
USER_BUDGET_USD = 50.0
if EXECUTION_SECONDS * EXECUTION_RATE_PER_SECOND > PLANNING_CEILING_USD:
    raise ValueError("The job exceeds the declared execution planning ceiling")


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=2,
    memory=16384,
    timeout=EXECUTION_SECONDS,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def train(config_json: str) -> bytes:
    import sys

    sys.path.insert(0, "/root")
    from protein_feedback.experiment import run_experiment
    from protein_feedback.report import make_report, verify_results

    config = json.loads(config_json)
    # Cap externally supplied configuration as well as the provider timeout.
    if config["device"] != "cuda" or len(config["seeds"]) > 5:
        raise ValueError("Unexpected cloud configuration")
    if config["protein_corpus_steps"] > 1500 or config["budget"] > 256:
        raise ValueError("Requested workload exceeds the approved study")
    output = Path("/tmp/protein-feedback-results")
    data = Path("/tmp/protein-feedback-data")
    run_experiment(config, output, data)
    make_report(output)
    verification = verify_results(output, data / "raw" / "gb1.csv.zip")
    (output / "verification.json").write_text(json.dumps(verification, indent=2))
    (output / "compute_plan.json").write_text(
        json.dumps(
            {
                "gpu": "A100-40GB",
                "cpu": 2,
                "memory_gib": 16,
                "timeout_seconds": EXECUTION_SECONDS,
                "planned_execution_upper_usd": EXECUTION_SECONDS * EXECUTION_RATE_PER_SECOND,
                "planning_ceiling_including_overhead_usd": PLANNING_CEILING_USD,
                "user_budget_usd": USER_BUDGET_USD,
                "pricing_source": "https://modal.com/pricing",
                "actual_bill": "Consult the provider dashboard; not inferred from model wall time.",
            },
            indent=2,
        )
    )
    stream = io.BytesIO()
    first_seed = str(config["seeds"][0])
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if not path.is_file():
                continue
            # Keep all numerical evidence and the first seed's model artifacts.
            if path.suffix == ".pt" and not (
                path.name.startswith(first_seed + "_")
                or path.name == f"reference_{first_seed}.pt"
                or path.name == "protein_initialization.pt"
            ):
                continue
            archive.write(path, path.relative_to(output))
    return stream.getvalue()


@app.local_entrypoint()
def main(output: str = "cloud-output/gpu-study.zip", smoke: bool = False):
    config = json.loads(Path("configs/smoke.json" if smoke else "configs/gpu.json").read_text())
    config["device"] = "cuda"
    config["source_code_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    config["source_code_dirty"] = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    files = sorted(Path("src").rglob("*.py")) + [Path("cloud/modal_run.py")]
    config["source_files_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    config_text = json.dumps(config)
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError("Use a new output path; do not overwrite completed evidence")
    print(
        f"Finite job: execution timeout {EXECUTION_SECONDS / 3600:.0f}h; "
        f"planned execution bound ${EXECUTION_SECONDS * EXECUTION_RATE_PER_SECOND:.2f}; "
        f"budget ${USER_BUDGET_USD:.0f}."
    )
    result = train.remote(config_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result)
    print(f"Saved {output_path} ({len(result):,} bytes). The finite GPU function has returned.")
