"""A causal transformer initialized from random weights, plus a constrained policy.

All 56 GB1 domain residues provide context. Only the four experimentally varied
sites are actions. The score is the sum of four conditional amino-acid log
probabilities, not the likelihood of the fixed residues or the fusion construct.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import AMINO_ACIDS, DOMAIN, SITES

BOS = 20
PAD = 21


@dataclass
class ModelConfig:
    width: int = 32
    layers: int = 2
    heads: int = 4
    context: int = 128


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.width % config.heads:
            raise ValueError("Width must be divisible by head count")
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.proj = nn.Linear(config.width, config.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        result = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(result.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = Attention(config)
        self.norm2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ProteinTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(22, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, 20, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.config.context:
            raise ValueError("Sequence exceeds the declared model context")
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token(tokens) + self.position(pos)[None]
        return self.output(self.norm(self.blocks(x)))

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> ProteinTransformer:
        payload = torch.load(path, map_location=device, weights_only=True)
        model = cls(ModelConfig(**payload["config"])).to(device)
        model.load_state_dict(payload["state"])
        return model


def domain_tokens(variants: torch.Tensor) -> torch.Tensor:
    wt = torch.tensor([AMINO_ACIDS.index(a) for a in DOMAIN], device=variants.device)
    sequence = wt[None].expand(len(variants), -1).clone()
    sequence[:, list(SITES)] = variants
    return sequence


def action_log_probs(model: ProteinTransformer, variants: torch.Tensor) -> torch.Tensor:
    sequence = domain_tokens(variants)
    inputs = torch.cat([torch.full_like(sequence[:, :1], BOS), sequence[:, :-1]], dim=1)
    logits = model(inputs)[:, list(SITES)]
    return F.log_softmax(logits, dim=-1).gather(-1, variants[..., None]).squeeze(-1).sum(-1)


def score_variants(
    model: ProteinTransformer,
    variants: np.ndarray,
    device: str,
    batch_size: int = 256,
    grad: bool = False,
) -> torch.Tensor:
    pieces = []
    with torch.set_grad_enabled(grad):
        for start in range(0, len(variants), batch_size):
            v = torch.as_tensor(variants[start : start + batch_size], device=device)
            pieces.append(action_log_probs(model, v))
    return torch.cat(pieces)


def conditional_draw_log_probs(logits: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Exact ordered, without-replacement log probabilities (Plackett-Luce).

    Removing earlier draws changes the denominator for each subsequent draw.
    Keeping these denominators is important for a correct PPO likelihood ratio.
    """
    logp = F.log_softmax(logits, dim=0)
    chosen = logp[selected]
    previous_mass = torch.cat([chosen.new_zeros(1), chosen.exp().cumsum(0)[:-1]])
    return chosen - torch.log1p(-previous_mass.clamp(max=1 - 1e-7))


def ppo_clipped_loss(
    new: torch.Tensor, old: torch.Tensor, advantage: torch.Tensor, clip: float = 0.2
) -> torch.Tensor:
    ratio = (new - old).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()


def dpo_loss(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    margin = beta * ((chosen - rejected) - (reference_chosen - reference_rejected))
    return -F.logsigmoid(margin).mean()


def seed_all(seed: int, threads: int = 4):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_initial(
    model: ProteinTransformer,
    train: np.ndarray,
    validation: np.ndarray,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best = math.inf
    best_state = None
    history = []
    for step in range(steps):
        ids = rng.integers(0, len(train), min(batch_size, len(train)))
        v = torch.as_tensor(train[ids], device=device)
        model.train()
        loss = -action_log_probs(model, v).mean() / 4
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            model.eval()
            val = -score_variants(model, validation, device).mean().item() / 4
            history.append(
                {
                    "step": step,
                    "train_action_nll": loss.item(),
                    "validation_action_nll": val,
                    "gradient_norm": float(norm),
                }
            )
            if val < best:
                best = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history
