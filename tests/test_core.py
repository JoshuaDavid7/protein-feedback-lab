import itertools
import math

import numpy as np
import pytest
import torch

from protein_feedback.agents import preference_pairs, recommendation_scores, run_agent
from protein_feedback.data import DOMAIN, SITES, WT, Landscape, MeasurementOracle, encode, partition
from protein_feedback.model import (
    ModelConfig,
    ProteinTransformer,
    action_log_probs,
    conditional_draw_log_probs,
    conditional_kl,
    domain_tokens,
    dpo_loss,
    ppo_clipped_loss,
    returns_to_go,
    seed_all,
)
from protein_feedback.pretrain import language_batch


@pytest.fixture(autouse=True)
def seeded():
    seed_all(9, threads=1)


@pytest.fixture
def model():
    return ProteinTransformer(ModelConfig(width=8, layers=1, heads=2, context=128)).eval()


def test_reference_domain_is_the_assayed_wild_type():
    assert len(DOMAIN) == 56
    assert "".join(DOMAIN[i] for i in SITES) == WT
    x = domain_tokens(torch.tensor(encode([WT, "AAAA"])))
    assert (x[0] != x[1]).sum() == 4


def test_transformer_does_not_see_future_tokens(model):
    left = torch.tensor([[20, 1, 2, 3, 4]])
    right = torch.tensor([[20, 1, 2, 18, 17]])
    torch.testing.assert_close(model(left)[:, :3], model(right)[:, :3])


def test_action_likelihood_is_finite_and_differentiable(model):
    score = action_log_probs(model, torch.tensor(encode([WT, "AAAA"])))
    assert score.shape == (2,)
    assert torch.isfinite(score).all()
    (-score.mean()).backward()
    assert model.output.weight.grad.abs().sum() > 0


def test_without_replacement_distribution_normalizes():
    logits = torch.tensor([0.7, -0.4, 1.2, 0.0], dtype=torch.float64)
    total = sum(
        conditional_draw_log_probs(logits, torch.tensor(p)).sum().exp()
        for p in itertools.permutations(range(4), 3)
    )
    assert float(total) == pytest.approx(1.0, abs=1e-8)


def test_without_replacement_denominator_is_not_iid():
    logits = torch.log(torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64))
    result = conditional_draw_log_probs(logits, torch.tensor([0, 1]))
    torch.testing.assert_close(result.exp(), torch.tensor([0.6, 0.75], dtype=torch.float64))


def test_concentrated_policy_has_stable_remaining_draws():
    logits = torch.tensor([1000.0, 0.0, -1.0], dtype=torch.float64)
    result = conditional_draw_log_probs(logits, torch.tensor([0, 1]))
    assert torch.isfinite(result).all()
    assert float(result[1].exp()) == pytest.approx(1 / (1 + math.exp(-1)))


def test_reference_kl_is_finite_at_masked_prefixes():
    x = torch.tensor([0.7, -0.2, 0.4], dtype=torch.float64)
    selected = torch.tensor([1, 2])
    assert float(conditional_kl(x, x, selected)) == pytest.approx(0)
    assert float(conditional_kl(x, -x, selected)) > 0


def test_reward_to_go_gradient_matches_exact_batch_return_gradient():
    theta = torch.tensor([0.2, -0.5, 0.8], dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([1.0, 0.0, 3.0], dtype=torch.float64)
    expected = theta.new_zeros(())
    policy_surrogate = theta.new_zeros(())
    for order in itertools.permutations(range(3), 2):
        selected = torch.tensor(order)
        logp = conditional_draw_log_probs(theta, selected)
        probability = logp.sum().exp()
        expected = expected + probability * rewards[selected].sum()
        policy_surrogate = (
            policy_surrogate
            + probability.detach() * (logp * returns_to_go(rewards[selected])).sum()
        )
    exact = torch.autograd.grad(expected, theta, retain_graph=True)[0]
    estimated = torch.autograd.grad(policy_surrogate, theta)[0]
    torch.testing.assert_close(exact, estimated, rtol=1e-8, atol=1e-9)


def test_ppo_positive_advantage_clips_upward_ratio():
    new = torch.tensor([math.log(2)], requires_grad=True)
    loss = ppo_clipped_loss(new, torch.zeros(1), torch.ones(1))
    assert float(loss) == pytest.approx(-1.2)
    loss.backward()
    assert float(new.grad) == 0


def test_ppo_negative_advantage_clips_downward_ratio():
    new = torch.tensor([math.log(0.5)], requires_grad=True)
    loss = ppo_clipped_loss(new, torch.zeros(1), -torch.ones(1))
    assert float(loss) == pytest.approx(0.8)
    loss.backward()
    assert float(new.grad) == 0


def test_dpo_gradient_favors_chosen_sequence():
    c, r = torch.tensor([-4.0], requires_grad=True), torch.tensor([-4.0], requires_grad=True)
    loss = dpo_loss(c, r, c.detach(), r.detach(), beta=0.1)
    assert float(loss) == pytest.approx(math.log(2))
    loss.backward()
    assert float(c.grad) < 0 < float(r.grad)


def test_dpo_common_normalizer_cancels():
    x = torch.tensor([-3.0, -5.0])
    y = torch.tensor([-6.0, -4.0])
    ref = torch.tensor([-4.0, -4.0])
    a = dpo_loss(x, y, ref, ref)
    b = dpo_loss(x + 7, y + 7, ref + 11, ref + 11)
    torch.testing.assert_close(a, b)


def test_tied_measurements_do_not_become_preferences():
    a, b = preference_pairs(np.ones(10), np.random.default_rng(2))
    assert len(a) == len(b) == 0


def test_oracle_budget_uniqueness_and_bounds():
    oracle = MeasurementOracle(np.arange(10), 3)
    np.testing.assert_array_equal(oracle.query([2, 4]), [2, 4])
    with pytest.raises(ValueError, match="Duplicate"):
        oracle.query([2])
    with pytest.raises(ValueError, match="budget"):
        oracle.query([1, 3])
    with pytest.raises(IndexError):
        oracle.query([-1])
    assert oracle.remaining == 1


def test_oracle_does_not_expose_mutable_measurements():
    source = np.array([1.0, 2.0])
    oracle = MeasurementOracle(source, 2)
    source[0] = 999
    assert oracle.query([0])[0] == 1


def test_partitions_are_label_independent_and_distance_separated():
    variants = np.asarray(["".join(v) for v in itertools.product("AVDG", repeat=4)])
    distances = np.asarray([sum(a != b for a, b in zip(v, WT)) for v in variants])
    a = Landscape(variants, encode(variants), np.arange(len(variants)), distances, {})
    b = Landscape(variants, encode(variants), -a.fitness, distances, {})
    one, two = partition(a, 7, 32, 32, 8), partition(b, 7, 32, 32, 8)
    assert not set(one.acquisition) & set(one.evaluation)
    assert not set(one.pretrain) & set(one.pretrain_validation)
    assert not set(one.initial) & set(one.pretrain_validation)
    for name in ["initial", "pretrain", "pretrain_validation", "acquisition", "evaluation"]:
        np.testing.assert_array_equal(getattr(one, name), getattr(two, name))
    assert (distances[one.pretrain] <= 2).all()
    assert (distances[one.acquisition] >= 3).all()


def test_language_batch_masks_padding_and_preserves_causality():
    x, y = language_batch(["ACDE"], 2, 8, np.random.default_rng(1), "cpu")
    assert (x[:, 0] == 20).all()
    torch.testing.assert_close(x[:, 1:4], y[:, :3])
    assert (y[:, 4:] == -100).all()


@pytest.mark.parametrize("method", ["random", "frozen", "weighted_sft", "ppo", "dpo", "rf_ucb"])
def test_each_agent_respects_budget_and_leaves_reference_unchanged(method, model):
    rng = np.random.default_rng(12)
    pool = rng.integers(0, 20, (16, 4))
    initial = rng.integers(0, 20, (8, 4))
    initial_y = rng.random(8)
    oracle = MeasurementOracle(rng.random(16), 4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    config = {
        "batch_size": 2,
        "posttrain_lr": 0.001,
        "update_epochs": 1,
        "rf_ucb_scale": 1.0,
        "kl_coefficient": 0.02,
        "dpo_beta": 0.1,
    }
    trace, fitted, forest = run_agent(
        method, model, pool, initial, initial_y, oracle, config, seed=4, device="cpu"
    )
    assert len(oracle.queried) == len(set(oracle.queried)) == 4
    assert len(trace["measurements"]) == 4
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    scores = recommendation_scores(method, fitted, forest, pool[:3], "cpu", 4)
    assert scores.shape == (3,) and np.isfinite(scores).all()
