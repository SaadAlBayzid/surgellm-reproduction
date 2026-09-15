"""Read-only scientific readiness check. No model imports or training."""
import argparse
import json
from pathlib import Path


def unresolved(value, path=""):
    if value is None:
        return [path]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in unresolved(v, f"{path}.{k}".strip("."))]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in unresolved(v, f"{path}[{i}]" )]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cuda-certificate", type=Path,
                        default=Path('results/kaggle-t4x2-smoke/cuda_smoke_pass.json'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    missing = unresolved(cfg)
    data_status = 'NOT_CONFIGURED'
    if cfg.get('data_lock'):
        from src.task_data import sha256, read_frozen
        root = args.config.resolve().parents[1]
        lock_path = root / cfg['data_lock']
        lock = json.loads(lock_path.read_text())
        if set(lock['tasks']) != {'D1','D2','D3','D4'}:
            raise ValueError('Four frozen tasks required')
        for task, info in lock['tasks'].items():
            manifest_path = lock_path.parent / info['manifest']
            if sha256(manifest_path) != info['sha256']:
                raise ValueError('Frozen manifest checksum mismatch')
            splits = read_frozen(manifest_path.parent)
            if {k:len(v) for k,v in splits.items()} != cfg['tasks'][task]['partition_counts']:
                raise ValueError('Config split counts differ from frozen data')
        data_status = 'FROZEN_VERIFIED'
    from src.training_gate import verified_data, fingerprint, gate_errors
    root = args.config.resolve().parents[1]
    _, hashes = verified_data(root, cfg)
    errors = gate_errors(args.cuda_certificate, fingerprint(root, cfg, hashes))
    if missing:
        errors.append('Unresolved config values')
    report = {"seed": args.seed, "training_started": False,
                      "status": "BLOCKED" if errors else "CUDA_SMOKE_VERIFIED", "unresolved_fields": missing,
                      "data_status": data_status,
                      "assumptions": cfg.get('assumptions', []),
                      "implementation_blockers": cfg.get('implementation_blockers', []),
                      "full_run_enabled": cfg.get("full_run_enabled", False),
                      "gate_errors": errors,
                      "runner_status": "trainer implemented; full CLI also checks current two-GPU runtime/software"}
    print(json.dumps(report, indent=2))
    if args.output:
        if Path('results').resolve() not in args.output.resolve().parents:
            raise ValueError('Reports must be under results/')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2))
    raise SystemExit(2 if errors else 0)


if __name__ == "__main__":
    main()
