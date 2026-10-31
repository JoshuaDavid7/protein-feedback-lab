"""Optional from-scratch language-model pretraining on a sampled UniProt corpus."""

from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import AMINO_ACIDS, sha256_file
from .model import BOS, PAD, ProteinTransformer

CORPUS_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete/uniprot_sprot.fasta.gz"
)


def read_fasta(path: Path):
    with gzip.open(path, "rt") as stream:
        identifier, pieces = None, []
        for line in stream:
            if line.startswith(">"):
                if identifier is not None:
                    yield identifier, "".join(pieces)
                identifier, pieces = line[1:].split()[0], []
            else:
                pieces.append(line.strip())
        if identifier is not None:
            yield identifier, "".join(pieces)


def corpus_sample(path: Path, count: int, seed: int):
    rng = np.random.default_rng(seed)
    reservoir, valid = [], 0
    for identifier, sequence in read_fasta(path):
        if not 32 <= len(sequence) <= 4096 or any(a not in AMINO_ACIDS for a in sequence):
            continue
        item = (identifier, sequence)
        valid += 1
        if len(reservoir) < count:
            reservoir.append(item)
        else:
            replace = int(rng.integers(valid))
            if replace < count:
                reservoir[replace] = item
    if len(reservoir) < count:
        raise ValueError("Corpus contains fewer valid proteins than requested")
    train, validation = [], []
    for identifier, sequence in reservoir:
        key = hashlib.sha256(identifier.encode()).hexdigest()
        (validation if int(key[:8], 16) % 10 == 0 else train).append(sequence)
    return train, validation, valid


def language_batch(
    sequences: list[str], count: int, context: int, rng: np.random.Generator, device: str
):
    aa_index = {a: i for i, a in enumerate(AMINO_ACIDS)}
    inputs = np.full((count, context), PAD, dtype=np.int64)
    targets = np.full((count, context), -100, dtype=np.int64)
    for row, idx in enumerate(rng.integers(len(sequences), size=count)):
        seq = sequences[idx]
        start = int(rng.integers(max(1, len(seq) - context + 1)))
        seq = seq[start : start + context]
        ids = [aa_index[a] for a in seq]
        inputs[row, : len(ids)] = [BOS] + ids[:-1]
        targets[row, : len(ids)] = ids
    return torch.as_tensor(inputs, device=device), torch.as_tensor(targets, device=device)


def pretrain_corpus(
    model: ProteinTransformer,
    config: dict,
    data_dir: Path,
    output_dir: Path,
    seed: int,
    device: str,
) -> dict:
    path = data_dir / "raw" / "uniprot_sprot.fasta.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(CORPUS_URL, timeout=120) as response, path.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
    train, validation, valid = corpus_sample(path, config["protein_corpus_sequences"], seed)
    if not validation:
        raise ValueError("Empty protein validation partition")
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    history, token_count = [], 0
    started = time.monotonic()
    use_amp = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    for step in range(config["protein_corpus_steps"]):
        model.train()
        x, y = language_batch(
            train, config["protein_corpus_batch_size"], model.config.context, rng, device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 20), y.reshape(-1), ignore_index=-100)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        token_count += int((y != -100).sum())
        if step % 100 == 0 or step == config["protein_corpus_steps"] - 1:
            model.eval()
            # Fixed validation crops permit a meaningful learning curve.
            vx, vy = language_batch(
                validation,
                min(64, len(validation)),
                model.config.context,
                np.random.default_rng(seed + 111),
                device,
            )
            with torch.no_grad():
                val_loss = F.cross_entropy(
                    model(vx).reshape(-1, 20), vy.reshape(-1), ignore_index=-100
                ).item()
            row = {
                "step": step,
                "train_nll": float(loss.detach()),
                "validation_nll": val_loss,
                "tokens": token_count,
                "gradient_norm": float(norm),
            }
            history.append(row)
            print(json.dumps({"protein_pretraining": row}), flush=True)
    payload = {
        "source_url": CORPUS_URL,
        "source_sha256": sha256_file(path),
        "source_version_note": "Current-release download; exact bytes pinned by this run's SHA256.",
        "license": "CC BY 4.0",
        "valid_source_proteins": valid,
        "training_proteins": len(train),
        "validation_proteins": len(validation),
        "split_caveat": "Accession split, not a homology-disjoint generalization benchmark.",
        "parameters": model.parameter_count,
        "tokens_seen": token_count,
        "wall_seconds": time.monotonic() - started,
        "history": history,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protein_pretraining.json").write_text(json.dumps(payload, indent=2) + "\n")
    model.save(output_dir / "models" / "protein_initialization.pt")
    return payload
