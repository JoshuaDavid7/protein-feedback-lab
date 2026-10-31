"""Frozen recommendations are written before any held-out label is evaluated."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .agents import recommendation_scores, run_agent
from .data import MeasurementOracle, fetch_data, load_landscape, partition
from .model import ModelConfig, ProteinTransformer, seed_all, train_initial
from .pretrain import pretrain_corpus


def json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def evaluate(scores: np.ndarray, fitness: np.ndarray, trace: list[dict], top_k: int):
    k = min(top_k, len(fitness))
    order = np.argsort(-scores, kind="stable")[:k]
    acquired = np.asarray([r["fitness"] for r in trace])
    correlation = float(spearmanr(scores, fitness).statistic)
    return {
        "new_measurements": len(acquired),
        "acquisition_best_fitness": float(acquired.max()),
        "acquisition_hits_above_wild_type": int((acquired > 1).sum()),
        "acquisition_mean_fitness": float(acquired.mean()),
        "heldout_top_k": k,
        "heldout_top_k_mean_fitness": float(fitness[order].mean()),
        "heldout_top_k_hits_above_wild_type": int((fitness[order] > 1).sum()),
        "heldout_spearman": correlation if np.isfinite(correlation) else None,
        "heldout_population_mean_fitness": float(fitness.mean()),
        "heldout_population_hit_rate": float((fitness > 1).mean()),
    }


def run_experiment(config: dict, output: Path, data_root: Path):
    if (output / "results.json").exists():
        raise FileExistsError("Use a new output directory; completed results are immutable")
    output.mkdir(parents=True, exist_ok=True)
    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; no silent CPU fallback")
    if config["budget"] > config["pool_size"] or config["budget"] % config["batch_size"]:
        raise ValueError("Budget must fit the pool and be divisible by batch size")
    seed_all(config["seeds"][0], config["threads"])
    source = fetch_data(data_root)
    data = load_landscape(source)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    commit = config.get("source_code_commit", commit)
    dirty = config.get("source_code_dirty", dirty)
    frozen = {
        "config": config,
        "dataset": data.audit,
        "code_commit": commit,
        "code_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "platform": platform.platform(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "protocol": "docs/PROTOCOL.md",
        "started_unix": time.time(),
    }
    json_write(output / "protocol_frozen.json", frozen)

    initial_model = ProteinTransformer(ModelConfig(**config["model"])).to(device)
    corpus_report = None
    if config.get("protein_corpus_steps", 0):
        corpus_report = pretrain_corpus(
            initial_model, config, data_root, output, config["seeds"][0], device
        )
    initial_weights = copy.deepcopy(initial_model.state_dict())
    all_frozen = []
    for seed in config["seeds"]:
        seed_all(seed, config["threads"])
        split = partition(
            data, seed, config["pool_size"], config["evaluation_size"], config["initial_size"]
        )
        model = ProteinTransformer(ModelConfig(**config["model"])).to(device)
        # CPU pilot: independent random initializations. GPU study: common
        # from-scratch corpus checkpoint, with separate seeded adaptation/runs.
        if corpus_report is not None:
            model.load_state_dict(initial_weights)
        history = train_initial(
            model,
            data.tokens[split.pretrain],
            data.tokens[split.pretrain_validation],
            config["initial_steps"],
            128,
            config["initial_lr"],
            seed,
            device,
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.save(output / "models" / f"reference_{seed}.pt")
        json_write(output / f"initial_training_{seed}.json", history)
        split_record = {
            name: data.variants[getattr(split, name)].tolist()
            for name in ["initial", "pretrain", "pretrain_validation", "acquisition", "evaluation"]
        }
        json_write(output / f"partitions_{seed}.json", split_record)

        for method in config["methods"]:
            started = time.monotonic()
            oracle = MeasurementOracle(data.fitness[split.acquisition], config["budget"])
            # The reference is immutable; the trainable agent receives its own copy.
            trainable = copy.deepcopy(model)
            for parameter in trainable.parameters():
                parameter.requires_grad_(True)
            trace, fitted, forest = run_agent(
                method,
                trainable,
                data.tokens[split.acquisition],
                data.tokens[split.initial],
                data.fitness[split.initial],
                oracle,
                config,
                seed,
                device,
            )
            scores = recommendation_scores(
                method, fitted, forest, data.tokens[split.evaluation], device, seed
            )
            if not np.isfinite(scores).all():
                raise FloatingPointError("Nonfinite recommendation score")
            trace["candidate_variants"] = data.variants[split.acquisition].tolist()
            record = {
                "seed": seed,
                "method": method,
                "trace": trace,
                "evaluation_variants": data.variants[split.evaluation].tolist(),
                "scores": scores.tolist(),
                "wall_seconds": time.monotonic() - started,
            }
            record["recommendations_sha256"] = content_hash(
                {"variants": record["evaluation_variants"], "scores": record["scores"]}
            )
            json_write(output / "frozen_recommendations" / f"{seed}_{method}.json", record)
            if fitted is not None:
                fitted.save(output / "models" / f"{seed}_{method}.pt")
            all_frozen.append((record, split.evaluation))
            print(
                json.dumps(
                    {
                        "completed": method,
                        "seed": seed,
                        "queries": len(oracle.queried),
                        "wall_seconds": round(record["wall_seconds"], 2),
                    }
                ),
                flush=True,
            )

    # All policies, acquisition trajectories and recommendations are now frozen.
    # No evaluation label was passed to run_agent or recommendation_scores.
    evaluations = []
    for record, ids in all_frozen:
        scores = np.asarray(record["scores"])
        metrics = evaluate(
            scores, data.fitness[ids], record["trace"]["measurements"], config["top_k"]
        )
        evaluations.append(
            {
                "seed": record["seed"],
                "method": record["method"],
                "recommendations_sha256": record["recommendations_sha256"],
                "wall_seconds": record["wall_seconds"],
                **metrics,
            }
        )
    results = {
        "config": config,
        "parameters": initial_model.parameter_count,
        "model_config": asdict(initial_model.config),
        "dataset": data.audit,
        "corpus_pretraining": corpus_report,
        "runs": evaluations,
        "limitations": [
            "Retrospective replay, not new wet-lab experiments.",
            "Four mutable sites in one GB1 domain.",
            "Single-machine training, not distributed frontier-model training.",
            "Public historical labels cannot provide independent blinding.",
            "No hyperparameter changes based on evaluation results.",
        ],
    }
    json_write(output / "results.json", results)
    return results
