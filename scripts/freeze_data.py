"""Freeze data once with seed 0, independently of model variants and training seeds."""
import json
from pathlib import Path
from src.task_data import load_d1, load_d2, load_d3, load_d4, select, partition, proportional_counts, freeze_task, sha256


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / 'configs/data_freeze.json').read_text())
    output = root / cfg['output']
    results = {}
    for task in ('D1', 'D2', 'D3', 'D4'):
        paths = [root / p for p in cfg['raw'][task]]
        if not all(p.exists() for p in paths):
            print(f'{task}: MISSING raw files; not frozen')
            continue
        if task == 'D1':
            train, dev = load_d1(paths[0], 'train'), load_d1(paths[1], 'validation')
            previous = json.loads((root / cfg['d1_debug_manifest']).read_text())
            splits = {name: [(dev if name == 'test' else train)[int(sid.split(':')[-1])]
                             for sid in previous['ids'][name]] for name in ('train','validation','test')}
        elif task == 'D2':
            rows = load_d2(paths)
            splits = partition(select(rows, {0: 980, 1: 1020}, 0, 'D2-cap'))
        elif task == 'D3':
            rows = load_d3(paths[0])
            splits = partition(select(rows, proportional_counts(rows, 3164), 0, 'D3-cap'))
        else:
            rows = load_d4(paths[0])
            splits = partition(select(rows, {0: 2500, 1: 2500}, 0, 'D4-cap'))
        freeze_task(task, splits, output / task, paths, cfg['policies'][task])
        results[task] = {name: len(rows) for name, rows in splits.items()}
        print(task, results[task])
    if len(results) == 4:
        lock = {'data_seed': 0, 'training_seeds': [0,1,2], 'model_independent_splits': True,
                'tasks': {t: {'manifest': f'{t}/manifest.json', 'sha256': sha256(output/t/'manifest.json')} for t in results}}
        p = output / 'experiment.lock.json'
        data = json.dumps(lock, indent=2) + '\n'
        if p.exists() and p.read_text() != data:
            raise ValueError('Experiment lock changed; refusing overwrite')
        if not p.exists():
            p.write_text(data)
        print('ALL FOUR TASKS FROZEN')
    else:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
