"""SST-2 TSV loading only. Split policies are explicit engineering choices."""
import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    source_id: str
    text: str
    label: int


def load_tsv(path, source):
    """Read labeled GLUE-format train/dev TSV; preserve sentence text exactly."""
    if source not in {"train", "validation"}:
        raise ValueError("source must be train or validation (official dev)")
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"sentence", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Expected sentence and label columns; unlabeled test is unsupported")
        for index, row in enumerate(reader):
            if row.get("label") not in {"0", "1"} or not row.get("sentence", "").strip():
                raise ValueError(f"Invalid labeled sentence at row {index + 2}")
            rows.append(Example(f"{source}:{index}", row["sentence"], int(row["label"])))
    if not rows:
        raise ValueError("Empty dataset")
    return rows


def _sample(rows, count, rng):
    """Proportional largest-remainder allocation; not an author-specified algorithm."""
    if not 0 < count <= len(rows):
        raise ValueError("Sample count must be positive and within available rows")
    groups = [[r for r in rows if r.label == label] for label in (0, 1)]
    if any(not group for group in groups):
        raise ValueError("Both labels are required")
    quotas = [count * len(group) / len(rows) for group in groups]
    sizes = [int(q) for q in quotas]
    for label in sorted((0, 1), key=lambda i: (-(quotas[i] - sizes[i]), i))[:count - sum(sizes)]:
        sizes[label] += 1
    selected, rest = [], []
    for group, size in zip(groups, sizes):
        rng.shuffle(group)
        selected.extend(group[:size])
        rest.extend(group[size:])
    rng.shuffle(selected)
    rng.shuffle(rest)
    return selected, rest


def prepare(train, official_validation, *, policy, seed, cap=7666):
    """Select one explicitly named interpretation; never silently pick a paper split.

    cap_then_holdout: cap to N, use ceil(10% N) internally, dev as test.
    holdout_then_cap: hold out ceil(10% source train), cap remainder, dev as test.
    capped_70_15_15: cap source train, allocate floor(70%)/floor(15%)/remainder.
    Rounding and RNG are engineering choices, not claimed paper details.
    """
    if policy not in {"cap_then_holdout", "holdout_then_cap", "capped_70_15_15"}:
        raise ValueError("An explicit supported split policy is required")
    if seed not in {0, 1, 2}:
        raise ValueError("Paper seeds are 0, 1, 2")
    all_rows = list(train) + list(official_validation)
    if len({r.source_id for r in all_rows}) != len(all_rows):
        raise ValueError("Duplicate source IDs")
    if any(r.label not in (0, 1) or not r.text.strip() for r in all_rows):
        raise ValueError("Invalid example")
    rng = random.Random(seed)
    if policy == "holdout_then_cap":
        val, pool = _sample(train, (len(train) + 9) // 10, rng)
        fit, _ = _sample(pool, cap, rng)
        test = list(official_validation)
    else:
        pool, _ = _sample(train, cap, rng)
        if policy == "cap_then_holdout":
            val, fit = _sample(pool, (cap + 9) // 10, rng)
            test = list(official_validation)
        else:
            fit, rest = _sample(pool, cap * 70 // 100, rng)
            val, test = _sample(rest, cap * 15 // 100, rng)
    splits = {"train": fit, "validation": val, "test": test}
    if any({r.label for r in rows} != {0, 1} for rows in splits.values()):
        raise ValueError("Every output split must contain both labels")
    fingerprints = [{hashlib.sha256(r.text.encode()).hexdigest() for r in rows} for rows in splits.values()]
    overlaps = sum(len(fingerprints[i] & fingerprints[j]) for i in range(3) for j in range(i + 1, 3))
    manifest = {
        "policy": policy, "seed": seed, "cap": cap,
        "exact_author_split": False,
        "sampling": "Python random.Random; proportional largest remainder; label-order tie break",
        "cross_split_text_overlap_pairs": overlaps,
        "text_processing": "Preserved verbatim; no tokenization or lexical normalization",
        "counts": {k: len(v) for k, v in splits.items()},
        "ids": {k: [r.source_id for r in v] for k, v in splits.items()},
    }
    return splits, manifest
