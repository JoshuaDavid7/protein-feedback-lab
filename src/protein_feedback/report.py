"""Report every registered method and seed; never select the best-looking run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_landscape
from .experiment import content_hash, evaluate

COLORS = {
    "random": "#8a8a8a",
    "frozen": "#ad8a64",
    "weighted_sft": "#2a996b",
    "ppo": "#3076c4",
    "dpo": "#e28c31",
    "rf_ucb": "#8757a5",
}


def verify_results(output: Path, archive: Path) -> dict:
    payload = json.loads((output / "results.json").read_text())
    landscape = load_landscape(archive)
    index = {v: i for i, v in enumerate(landscape.variants)}
    for row in payload["runs"]:
        record = json.loads(
            (output / "frozen_recommendations" / f"{row['seed']}_{row['method']}.json").read_text()
        )
        fingerprint = content_hash(
            {"variants": record["evaluation_variants"], "scores": record["scores"]}
        )
        if fingerprint != row["recommendations_sha256"]:
            raise ValueError("A frozen recommendation file changed")
        ids = [x["candidate_id"] for x in record["trace"]["measurements"]]
        if len(ids) != payload["config"]["budget"] or len(ids) != len(set(ids)):
            raise ValueError("Measurement budget or uniqueness mismatch")
        acquisition = record["trace"]["candidate_variants"]
        if set(acquisition) & set(record["evaluation_variants"]):
            raise ValueError("Acquisition and evaluation overlap")
        for measurement in record["trace"]["measurements"]:
            v = acquisition[measurement["candidate_id"]]
            expected = float(landscape.fitness[index[v]])
            if expected != measurement["fitness"]:
                raise ValueError("A reported measurement disagrees with the data")
        labels = np.asarray([landscape.fitness[index[v]] for v in record["evaluation_variants"]])
        metrics = evaluate(
            np.asarray(record["scores"]),
            labels,
            record["trace"]["measurements"],
            payload["config"]["top_k"],
        )
        for key, value in metrics.items():
            if value is None:
                if row[key] is not None:
                    raise ValueError(f"Invalid null metric: {key}")
            elif not np.isclose(value, row[key], rtol=1e-9, atol=1e-10):
                raise ValueError(f"Metric does not reproduce: {key}")
    return {
        "verified_runs": len(payload["runs"]),
        "source_sha256": landscape.audit["source_sha256"],
    }


def make_report(output: Path):
    results = json.loads((output / "results.json").read_text())
    methods, seeds = results["config"]["methods"], results["config"]["seeds"]
    grouped = {m: [r for r in results["runs"] if r["method"] == m] for m in methods}
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), layout="constrained")
    for method in methods:
        best_curves, hit_curves = [], []
        for seed in seeds:
            record = json.loads(
                (output / "frozen_recommendations" / f"{seed}_{method}.json").read_text()
            )
            y = np.asarray([v["fitness"] for v in record["trace"]["measurements"]])
            best_curves.append(np.maximum.accumulate(y))
            hit_curves.append(np.cumsum(y > 1))
        x = np.arange(1, len(y) + 1)
        for ax, curves in zip(axes, [np.asarray(best_curves), np.asarray(hit_curves)]):
            ax.plot(x, curves.mean(0), label=method, color=COLORS[method], linewidth=1.7)
            ax.fill_between(x, curves.min(0), curves.max(0), color=COLORS[method], alpha=0.10)
            ax.set_xlabel("New measured variants")
    axes[0].set_ylabel("Best measured binding fitness")
    axes[1].set_ylabel("Measured variants above wild type")
    axes[0].set_title("Best discovery: mean and full seed range")
    axes[1].set_title("Hit count: mean and full seed range")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.savefig(figure_dir / "acquisition.png", dpi=170)
    fig.savefig(figure_dir / "acquisition.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4), layout="constrained")
    for i, method in enumerate(methods):
        ys = [r["heldout_top_k_mean_fitness"] for r in grouped[method]]
        ax.bar(i, np.mean(ys), color=COLORS[method], alpha=0.5)
        offsets = np.linspace(-0.1, 0.1, len(ys))
        ax.scatter(i + offsets, ys, color=COLORS[method], s=26, edgecolor="white", zorder=3)
    ax.set_xticks(range(len(methods)), methods, rotation=15)
    ax.set_ylabel("Mean measured fitness of top recommendations")
    ax.set_title(f"Unqueried evaluation pool: top {results['config']['top_k']} per seed")
    fig.savefig(figure_dir / "heldout.png", dpi=170)
    fig.savefig(figure_dir / "heldout.svg")
    plt.close(fig)

    lines = [
        "# Results",
        "",
        (
            f"Configuration: `{results['config']['name']}`. "
            f"{results['parameters']:,} parameters; seeds {seeds}."
        ),
        "",
        (
            "Each method receives the same initial observations and the same number of new "
            "measurement queries. Evaluation recommendations are frozen before held-out "
            "fitness is read. Fitness is the historical assay's relative binding enrichment; "
            "a value above 1 exceeds the assay wild type."
        ),
        "",
        (
            "| Method | Best queried fitness | Queried hits > WT | Held-out top-k mean fitness | "
            "Held-out top-k hits > WT | Run seconds |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    keys = [
        "acquisition_best_fitness",
        "acquisition_hits_above_wild_type",
        "heldout_top_k_mean_fitness",
        "heldout_top_k_hits_above_wild_type",
        "wall_seconds",
    ]
    for method in methods:
        entries = []
        for key in keys:
            values = [r[key] for r in grouped[method]]
            entries.append(
                f"{np.mean(values):.3f} +/- {np.std(values, ddof=1):.3f}"
                if len(values) > 1
                else f"{values[0]:.3f}"
            )
        lines.append("| " + " | ".join([method] + entries) + " |")
    lines += [
        "",
        (
            "Values are mean +/- sample standard deviation across the declared seeds, "
            "not confidence intervals. The number of seeds is too small for strong superiority "
            "claims. Inspect every individual run in `results.json`."
        ),
        "",
        "![Acquisition curves](figures/acquisition.png)",
        "",
        "![Held-out recommendations](figures/heldout.png)",
        "",
        "## Scope",
        "",
        *[f"- {s}" for s in results["limitations"]],
        "",
        (
            "The FLIP `keep` mask is outcome-dependent and is deliberately unused. The long "
            "fusion-construct sequence is also unused; the policy conditions on the 56-residue "
            "GB1 domain and chooses only the four assayed sites."
        ),
        "",
        "## Recompute these numbers",
        "",
        "```bash",
        f"protein-feedback verify --output {output} --data data",
        "```",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines))
