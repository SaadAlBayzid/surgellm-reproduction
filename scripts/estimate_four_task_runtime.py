"""Extrapolate from the matching two-T4 trainer smoke; never starts training."""
import argparse
import json
from pathlib import Path
from src.multitask_baseline import TASKS, batch_plan, batch_semantics
from src.training_gate import verified_data, fingerprint, gate_errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/baseline_roberta_four_task.json'))
    p.add_argument('--smoke', type=Path, default=Path('results/kaggle-t4x2-smoke'))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(args.config.read_text())
    _, hashes = verified_data(root, cfg)
    errors = gate_errors(args.smoke/'cuda_smoke_pass.json', fingerprint(root, cfg, hashes))
    if errors:
        raise RuntimeError('; '.join(errors))
    log = json.loads((args.smoke/'epochs.jsonl').read_text().splitlines()[-1])
    b = cfg['training']['batch']
    sizes = {t:cfg['tasks'][t]['partition_counts']['train'] for t in TASKS}
    plan = batch_plan(sizes, b['per_device'], 2, b['accumulation'], cfg['training']['task_schedule'], 0, 0)
    full = batch_semantics(plan, b['per_device'], 2, b['accumulation'], b['global_effective'])
    training_seconds = log['training_seconds'] * full['microbatches_per_epoch'] / log['batch_semantics']['microbatches_per_epoch']
    report = {'estimated_training_seconds_per_epoch':training_seconds,
        'observed_validation_and_checkpoint_seconds': log['evaluation_checkpoint_seconds'],
        'estimated_max_budget_seconds_excluding_final_test_and_setup': cfg['training']['max_epochs'] * (
            training_seconds + log['evaluation_checkpoint_seconds']),
        'caution':'Four-update warmup smoke includes initialization/allocation overhead; extrapolation is approximate. Final test and initial tokenization/download are extra.'}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
