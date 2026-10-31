"""GB1 provenance, label-independent partitions, and budgeted measurement access."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
WT = "VDGV"
DOMAIN = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
SITES = (38, 39, 40, 53)  # zero-based: V39, D40, G41, V54
SOURCE_COMMIT = "9f58a97455d4133669c04b0a723f9d6e49e29eb2"
SOURCE_URL = (
    "https://raw.githubusercontent.com/J-SNACKKB/FLIP/"
    f"{SOURCE_COMMIT}/splits/gb1/four_mutations_full_data.csv.zip"
)
SOURCE_SHA256 = "85692d808dcd3ae54fa2ac31f4e590858d4582369b6c7b05df299b9b6c383bff"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(variants: list[str] | np.ndarray) -> np.ndarray:
    index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    return np.asarray([[index[aa] for aa in v] for v in variants], dtype=np.int64)


def stable_order(variants: np.ndarray, salt: str) -> np.ndarray:
    """Ordering depends on sequences and a declared salt, never on fitness."""
    keys = [hashlib.sha256(f"{salt}:{v}".encode()).hexdigest() for v in variants]
    return np.argsort(keys)


@dataclass(frozen=True)
class Landscape:
    variants: np.ndarray
    tokens: np.ndarray
    fitness: np.ndarray
    distance: np.ndarray
    audit: dict


def load_landscape(path: Path) -> Landscape:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("GB1 archive checksum differs from the pinned source")
    with zipfile.ZipFile(path) as archive:
        rows = list(csv.DictReader(io.TextIOWrapper(archive.open("four_mutations_full_data.csv"))))
    variants, fitness, distances = [], [], []
    low_input = 0
    for row in rows:
        variant = row["Variants"]
        value = float(row["Fitness"])
        if len(variant) != 4 or any(aa not in AMINO_ACIDS for aa in variant):
            raise ValueError(f"Unexpected variant: {variant!r}")
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid measurement for {variant}")
        hd = sum(a != b for a, b in zip(variant, WT))
        if hd != int(row["HD"]):
            raise ValueError("Hamming-distance annotation mismatch")
        variants.append(variant)
        fitness.append(value)
        distances.append(hd)
        low_input += int(int(row["Count input"]) < 100)
    if len(set(variants)) != len(variants):
        raise ValueError("Duplicate variant measurements require explicit aggregation")
    if "".join(DOMAIN[i] for i in SITES) != WT:
        raise ValueError("Domain reconstruction does not match assay wild type")
    audit = {
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "rows": len(rows),
        "low_input_count_below_100": low_input,
        "quality_caveat": "Low-count measurements are retained and can be noisy.",
        "outcome_dependent_keep_column_used": False,
        "full_fusion_sequence_column_used": False,
        "domain_length": len(DOMAIN),
        "sites_one_based": [i + 1 for i in SITES],
        "distance_counts": {str(i): distances.count(i) for i in range(5)},
    }
    return Landscape(
        np.asarray(variants),
        encode(variants),
        np.asarray(fitness, dtype=np.float64),
        np.asarray(distances),
        audit,
    )


def fetch_data(root: Path) -> Path:
    path = root / "raw" / "gb1.csv.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(
            SOURCE_URL, headers={"User-Agent": "protein-feedback-lab/0.1"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            data = response.read(16_000_001)
        if len(data) > 16_000_000:
            raise ValueError("Unexpectedly large GB1 archive")
        temporary = path.with_suffix(".download")
        temporary.write_bytes(data)
        if sha256_file(temporary) != SOURCE_SHA256:
            temporary.unlink()
            raise ValueError("Downloaded data failed checksum validation")
        temporary.replace(path)
    landscape = load_landscape(path)
    (root / "audit.json").write_text(json.dumps(landscape.audit, indent=2) + "\n")
    return path


@dataclass(frozen=True)
class Partition:
    initial: np.ndarray
    pretrain: np.ndarray
    pretrain_validation: np.ndarray
    acquisition: np.ndarray
    evaluation: np.ndarray


def partition(
    data: Landscape, seed: int, pool_size: int, eval_size: int, initial_size: int
) -> Partition:
    low = np.flatnonzero(data.distance <= 2)
    high = np.flatnonzero(data.distance >= 3)
    low = low[stable_order(data.variants[low], f"low-{seed}")]
    high = high[stable_order(data.variants[high], f"high-{seed}")]
    n_val = max(1, len(low) // 10)
    if initial_size > len(low) - n_val or pool_size + eval_size > len(high):
        raise ValueError("Requested partitions exceed the measured landscape")
    return Partition(
        initial=low[n_val : n_val + initial_size],
        pretrain=low[n_val:],
        pretrain_validation=low[:n_val],
        acquisition=high[:pool_size],
        evaluation=high[pool_size : pool_size + eval_size],
    )


class MeasurementOracle:
    """Retrospective measured values, exposed only when a paid query is made.

    This is an access boundary for an offline benchmark, not a real wet lab or a
    security sandbox. Agents never receive the evaluation labels.
    """

    def __init__(self, fitness: np.ndarray, budget: int):
        self.__fitness = np.asarray(fitness).copy()
        self.budget = budget
        self.queried: list[int] = []
        self._seen: set[int] = set()

    @property
    def remaining(self) -> int:
        return self.budget - len(self.queried)

    def query(self, indices: np.ndarray | list[int]) -> np.ndarray:
        ids = [int(i) for i in indices]
        if len(ids) != len(set(ids)) or any(i in self._seen for i in ids):
            raise ValueError("Duplicate queries violate the unique-measurement protocol")
        if len(ids) > self.remaining:
            raise ValueError("Measurement budget exhausted")
        if any(i < 0 or i >= len(self.__fitness) for i in ids):
            raise IndexError("Candidate outside the acquisition pool")
        self.queried.extend(ids)
        self._seen.update(ids)
        return self.__fitness[ids].copy()
