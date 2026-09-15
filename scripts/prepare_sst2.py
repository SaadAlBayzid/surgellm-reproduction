"""Run from the project root: python -m scripts.prepare_sst2 --help."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from src.sst2 import load_tsv, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--policy", required=True, choices=["cap_then_holdout", "holdout_then_cap", "capped_70_15_15"])
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--cap", type=int, default=7666)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits, manifest = prepare(load_tsv(args.train, "train"), load_tsv(args.dev, "validation"), policy=args.policy, seed=args.seed, cap=args.cap)
    manifest["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [("train", args.train), ("official_validation", args.dev)]}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in splits.items():
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
