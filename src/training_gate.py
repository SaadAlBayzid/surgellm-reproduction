"""Read-only frozen-data verification and config/code-bound CUDA smoke gate."""
import hashlib
import json
import subprocess
from pathlib import Path
from src.task_data import read_frozen, sha256


def verified_data(root, cfg):
    lock_path = root / cfg['data_lock']
    lock = json.loads(lock_path.read_text())
    if set(lock['tasks']) != {'D1', 'D2', 'D3', 'D4'}:
        raise ValueError('Four locked tasks required')
    hashes = {'experiment.lock.json': sha256(lock_path)}
    data = {}
    for task, info in lock['tasks'].items():
        path = lock_path.parent / info['manifest']
        if sha256(path) != info['sha256']:
            raise ValueError('Manifest hash mismatch')
        data[task] = read_frozen(path.parent)
        if {s: len(rows) for s, rows in data[task].items()} != cfg['tasks'][task]['partition_counts']:
            raise ValueError('Frozen/config count mismatch')
        if any(row.task_id != task for rows in data[task].values() for row in rows):
            raise ValueError('Frozen task routing mismatch')
        hashes[task] = info['sha256']
    return data, hashes


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity(root, cfg, hashes):
    root = Path(root)
    files = sorted({p for directory in ('src', 'scripts', 'tests', 'configs', 'data')
                    for p in (root/directory).rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts and
                    not any(part.startswith('tmp_') for part in p.relative_to(root).parts)} |
                   set(root.glob('environment*.yml')) | set(root.glob('requirements*.txt')))
    try:
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True).stdout.strip()
        git = {'available':True, 'head':commit}
    except (OSError, subprocess.CalledProcessError):
        git = {'available':False, 'head':'NOT_A_GIT_CHECKOUT; content hashes are authoritative'}
    return {'config': cfg, 'dataset_hashes': hashes, 'git':git,
            'files':{p.relative_to(root).as_posix():sha256(p) for p in files}}


def fingerprint(root, cfg, hashes):
    return object_hash(identity(root, cfg, hashes))


def gate_errors(path, signature, environment_hash=None):
    if not Path(path).is_file():
        return ['Matching two-T4 CUDA smoke certificate is missing']
    try:
        report = json.loads(Path(path).read_text())
    except (ValueError, OSError):
        return ['CUDA smoke certificate is unreadable']
    errors = []
    if report.get('fingerprint') != signature:
        errors.append('CUDA smoke certificate is stale: config, code or dataset hashes changed')
    if report.get('status') != 'PASSED' or report.get('world_size') != 2 or report.get('precision') != 'fp16':
        errors.append('A successful two-rank FP16 smoke is required')
    if len(report.get('devices', [])) != 2 or any('T4' not in name for name in report.get('devices', [])):
        errors.append('Certificate must come from two T4 GPUs')
    if not report.get('all_heads_updated') or not report.get('complete_validation') or not report.get('best_restored'):
        errors.append('Required CUDA smoke checks did not pass')
    environment = report.get('environment')
    if not environment or report.get('environment_hash') != object_hash(environment):
        errors.append('Missing or inconsistent environment hash')
    elif environment_hash is not None and report['environment_hash'] != environment_hash:
        errors.append('Environment changed since CUDA smoke')
    if report.get('mode') != 'smoke' or report.get('full_training_started') is not False:
        errors.append('Only smoke evidence can authorize a full run')
    return errors
