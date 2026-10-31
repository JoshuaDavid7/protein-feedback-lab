"""Equal-budget acquisition, PPO, DPO, and supervised/search baselines."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from torch.nn import functional as F

from .data import MeasurementOracle
from .model import (
    ProteinTransformer,
    action_log_probs,
    conditional_draw_log_probs,
    conditional_kl,
    dpo_loss,
    ppo_clipped_loss,
    returns_to_go,
    score_variants,
)


def one_hot(variants: np.ndarray) -> np.ndarray:
    return np.eye(20, dtype=np.float32)[variants].reshape(len(variants), -1)


def fit_rf(variants: np.ndarray, values: np.ndarray, seed: int):
    forest = RandomForestRegressor(
        n_estimators=64,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=seed,
        n_jobs=1,
    )
    forest.fit(one_hot(variants), np.log1p(values))
    return forest


def rf_scores(forest, variants: np.ndarray, scale: float = 0):
    x = one_hot(variants)
    predictions = np.stack([tree.predict(x) for tree in forest.estimators_])
    # Tree dispersion is an acquisition heuristic, not a calibrated interval.
    return predictions.mean(0) + scale * predictions.std(0)


def preference_pairs(values: np.ndarray, rng: np.random.Generator, count: int = 256):
    if len(values) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    left = rng.integers(0, len(values), count)
    right = rng.integers(0, len(values), count)
    good = np.abs(values[left] - values[right]) > 1e-8
    left, right = left[good], right[good]
    choose_left = values[left] > values[right]
    return np.where(choose_left, left, right), np.where(choose_left, right, left)


def run_agent(
    method: str,
    reference: ProteinTransformer,
    pool: np.ndarray,
    initial_x: np.ndarray,
    initial_y: np.ndarray,
    oracle: MeasurementOracle,
    config: dict,
    seed: int,
    device: str,
) -> tuple[dict, ProteinTransformer | None, object]:
    """No evaluation inputs or evaluation labels are accepted by this function."""
    allowed = {"random", "frozen", "weighted_sft", "ppo", "dpo", "rf_ucb"}
    if method not in allowed:
        raise ValueError(f"Unknown method: {method}")
    rng = np.random.default_rng(seed)
    model = copy.deepcopy(reference).to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["posttrain_lr"], weight_decay=0.01)
    observed_x, observed_y = initial_x.copy(), initial_y.copy()
    available = np.ones(len(pool), dtype=bool)
    ref_scores = score_variants(reference, pool, device).detach()
    reward_scale = max(float(np.std(np.log1p(initial_y))), 0.1)
    baseline = float(np.mean(np.log1p(initial_y)))
    trace, diagnostics = [], []
    start_time = time.monotonic()
    forest = None

    while oracle.remaining:
        remaining_ids = np.flatnonzero(available)
        count = min(config["batch_size"], oracle.remaining, len(remaining_ids))
        if count == 0:
            raise ValueError("Candidate pool cannot satisfy the measurement budget")
        candidate_tokens = pool[remaining_ids]
        if method == "random":
            local_ids = rng.choice(len(remaining_ids), count, replace=False)
        elif method == "rf_ucb":
            forest = fit_rf(observed_x, observed_y, seed)
            scores = rf_scores(forest, candidate_tokens, config["rf_ucb_scale"])
            # Stable, seeded tie breaking does not inspect any fitness labels.
            local_ids = np.lexsort((rng.random(len(scores)), -scores))[:count]
        else:
            with torch.no_grad():
                old_logits = score_variants(model, candidate_tokens, device)
                probabilities = F.softmax(old_logits.double(), dim=0).cpu().numpy()
                local_ids = rng.choice(len(candidate_tokens), count, replace=False, p=probabilities)
                selected = torch.as_tensor(local_ids, device=device)
                old_draw = conditional_draw_log_probs(old_logits, selected).detach()

        ids = remaining_ids[local_ids]
        fitness = oracle.query(ids)
        available[ids] = False
        observed_x = np.concatenate([observed_x, pool[ids]])
        observed_y = np.concatenate([observed_y, fitness])
        rewards = np.log1p(fitness)
        step_info = {"queries": len(oracle.queried)}

        if method == "ppo":
            reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            lifetime = torch.arange(count, 0, -1, dtype=torch.float32, device=device)
            advantage = (returns_to_go(reward_tensor) - baseline * lifetime) / reward_scale
            for _ in range(config["update_epochs"]):
                new_logits = score_variants(model, candidate_tokens, device, grad=True)
                new_draw = conditional_draw_log_probs(new_logits, selected)
                kl = conditional_kl(new_logits, ref_scores[remaining_ids], selected)
                objective = ppo_clipped_loss(new_draw, old_draw, advantage)
                loss = objective + config["kl_coefficient"] * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            step_info.update(
                loss=float(loss.detach()), kl=float(kl.detach()), gradient_norm=float(grad_norm)
            )
            baseline = 0.9 * baseline + 0.1 * float(rewards.mean())

        elif method == "dpo":
            chosen, rejected = preference_pairs(observed_y, rng)
            if len(chosen):
                c = torch.as_tensor(observed_x[chosen], device=device)
                r = torch.as_tensor(observed_x[rejected], device=device)
                with torch.no_grad():
                    ref_c = action_log_probs(reference, c)
                    ref_r = action_log_probs(reference, r)
                for _ in range(config["update_epochs"]):
                    loss = dpo_loss(
                        action_log_probs(model, c),
                        action_log_probs(model, r),
                        ref_c,
                        ref_r,
                        config["dpo_beta"],
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                step_info.update(
                    loss=float(loss.detach()), gradient_norm=float(grad_norm), pairs=len(chosen)
                )

        elif method == "weighted_sft":
            weights = F.softmax(
                torch.as_tensor(
                    np.log1p(observed_y) / reward_scale, dtype=torch.float32, device=device
                ),
                dim=0,
            )
            x = torch.as_tensor(observed_x, device=device)
            for _ in range(config["update_epochs"]):
                loss = -(weights * action_log_probs(model, x)).sum() / 4
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            step_info.update(loss=float(loss.detach()), gradient_norm=float(grad_norm))

        for idx, value in zip(ids, fitness):
            trace.append({"candidate_id": int(idx), "fitness": float(value)})
        diagnostics.append(step_info)

    if method == "rf_ucb":
        forest = fit_rf(observed_x, observed_y, seed)
    return (
        {
            "method": method,
            "seed": seed,
            "measurements": trace,
            "training": diagnostics,
            "initial_observations": len(initial_y),
            "unique_new_measurements": len(oracle.queried),
            "wall_seconds": time.monotonic() - start_time,
        },
        model if method not in {"random", "rf_ucb"} else None,
        forest,
    )


def recommendation_scores(
    method: str, model, forest, tokens: np.ndarray, device: str, seed: int
) -> np.ndarray:
    """Scores depend on sequences and fitted state, never on their test labels."""
    if method == "random":
        return np.random.default_rng(seed + 100_003).random(len(tokens))
    if method == "rf_ucb":
        return rf_scores(forest, tokens, scale=0)
    return score_variants(model, tokens, device).cpu().numpy()
