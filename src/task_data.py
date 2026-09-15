"""Strict, text-preserving loaders and model-independent deterministic splits."""
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskExample:
    text: str
    label: int
    task_id: str
    source_id: str
    row_index: int


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate(rows, count=None, class_counts=None):
    if not rows or any(not isinstance(r.text, str) or not r.text.strip() for r in rows):
        raise ValueError('Empty/missing text; do not silently filter raw data')
    if any(type(r.label) is not int or r.label not in (0, 1) for r in rows):
        raise ValueError('Expected binary labels 0/1')
    if len({r.source_id for r in rows}) != len(rows):
        raise ValueError('Duplicate source IDs')
    if count is not None and len(rows) != count:
        raise ValueError(f'Wrong dataset version/count: expected {count}, got {len(rows)}')
    actual = dict(Counter(r.label for r in rows))
    if class_counts is not None and actual != class_counts:
        raise ValueError(f'Wrong raw class counts: expected {class_counts}, got {actual}')
    return rows


def load_binary_csv(path, task_id, expected_count, expected_classes):
    csv.field_size_limit(10_000_000)
    digest = sha256(path)
    rows = []
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'text', 'label'}.issubset(reader.fieldnames or []):
            raise ValueError('Required CSV columns: text,label')
        for i, r in enumerate(reader):
            if r['label'] not in {'0', '1'}:
                raise ValueError(f'Invalid label at data row {i}')
            if task_id == 'D4' and 'source' in r and ((r['source'] == 'Human') != (r['label'] == '0')):
                raise ValueError(f'D4 source/label contradiction at data row {i}')
            rows.append(TaskExample(r['text'], int(r['label']), task_id, f'{task_id}:{digest}:{i}', i))
    return validate(rows, expected_count, expected_classes)


def load_d3(path):
    # Deliberately no permissive/latest-version fallback.
    return load_binary_csv(path, 'D3', 14877, {0: 13712, 1: 1165})


def load_d4(path):
    return load_binary_csv(path, 'D4', 10000, {0: 4452, 1: 5548})


def load_d1(path, split):
    from src.sst2 import load_tsv
    if split not in ('train', 'validation'):
        raise ValueError('D1 requires labeled train or official validation')
    digest = sha256(path)
    rows = [TaskExample(r.text, r.label, 'D1', f'D1:{digest}:{i}', i)
            for i, r in enumerate(load_tsv(path, split))]
    return validate(rows, 67349 if split == 'train' else 872,
                    {0:29780,1:37569} if split == 'train' else {0:428,1:444})


def hotpot_example(raw, row_index):
    mapping = {'easy': 0, 'medium': 1, 'hard': 1}
    if raw.get('level') not in mapping:
        raise ValueError('Unexpected HotPotQA difficulty')
    context = raw['context']
    facts = raw['supporting_facts']
    pairs = list(zip(context['title'], context['sentences'])) if isinstance(context, dict) else context
    titles = set(facts['title']) if isinstance(facts, dict) else {x[0] for x in facts}
    # Documented choice: all sentences of gold-supporting paragraphs, original order, no titles.
    sentences = [s for title, para in pairs if title in titles for s in para]
    if not sentences or not raw['question'].strip():
        raise ValueError('Missing question/supporting context')
    context_text = ' '.join(' '.join(sentences).split()[:300])
    sid = raw.get('id', raw.get('_id'))
    if not sid:
        raise ValueError('Missing HotPotQA source ID')
    return TaskExample('[Q]' + raw['question'] + '[CTX]' + context_text,
                       mapping[raw['level']], 'D2', 'D2:' + sid, row_index)


def load_d2(paths):
    import pyarrow.parquet as pq
    rows = []
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            for raw in batch.to_pylist():
                rows.append(hotpot_example(raw, len(rows)))
    # Pinned HF revision documents 90447; paper reports 90564 and calls it validation.
    return validate(rows, 90447, {0:17972,1:72475})


def select(rows, counts, seed=0, namespace='subset'):
    """SHA256 ordering avoids RNG/library-version drift; no replacement."""
    selected = []
    for label, count in sorted(counts.items()):
        group = [r for r in rows if r.label == label]
        if count < 0 or count > len(group):
            raise ValueError('Requested class quota unavailable without replacement')
        group.sort(key=lambda r: hashlib.sha256(f'{seed}|{namespace}|{r.source_id}'.encode()).hexdigest())
        selected.extend(group[:count])
    return sorted(selected, key=lambda r: r.source_id)


def proportional_counts(rows, n):
    if not 0 < n <= len(rows):
        raise ValueError('Invalid sample size')
    counts = Counter(r.label for r in rows)
    quotas = {label: n * counts[label] // len(rows) for label in (0, 1)}
    order = sorted((0, 1), key=lambda x: (-(n * counts[x] % len(rows)), x))
    for label in order[:n - sum(quotas.values())]:
        quotas[label] += 1
    return quotas


def partition(rows, seed=0):
    # Global floor(70%), floor(15%), remainder; each selection is class-stratified.
    train = select(rows, proportional_counts(rows, len(rows) * 70 // 100), seed, 'train')
    ids = {r.source_id for r in train}
    rest = [r for r in rows if r.source_id not in ids]
    val = select(rest, proportional_counts(rest, len(rows) * 15 // 100), seed, 'validation')
    ids.update(r.source_id for r in val)
    return {'train': train, 'validation': val, 'test': [r for r in rows if r.source_id not in ids]}


def freeze_task(task, splits, output, raw_artifacts, policy):
    """Create once; an existing bundle is verified, never overwritten or resampled."""
    output = Path(output)
    combined = [r for rows in splits.values() for r in rows]
    validate(combined)
    manifest = {'task_id': task, 'data_seed': 0, 'policy': policy,
                'raw_artifacts': {str(Path(p).name): sha256(p) for p in raw_artifacts},
                'splits': {}, 'cross_split_exact_text_overlap': {}}
    payloads = {}
    for name, rows in splits.items():
        validate(rows)
        payload = ''.join(json.dumps(asdict(r), ensure_ascii=False) + '\n' for r in rows).encode('utf-8')
        payloads[name + '.jsonl'] = payload
        manifest['splits'][name] = {'count': len(rows), 'class_counts': dict(Counter(r.label for r in rows)),
              'source_ids': [r.source_id for r in rows], 'row_indices': [r.row_index for r in rows],
              'file': name + '.jsonl', 'sha256': hashlib.sha256(payload).hexdigest()}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        manifest['cross_split_exact_text_overlap'][a+'_'+b] = len({r.text for r in splits[a]} & {r.text for r in splits[b]})
    payloads['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    if output.exists():
        if any(not (output / n).exists() or (output / n).read_bytes() != data for n, data in payloads.items()):
            raise ValueError('Frozen bundle differs; refusing to overwrite')
        return json.loads((output / 'manifest.json').read_text())
    output.mkdir(parents=True)
    for name, data in payloads.items():
        (output / name).write_bytes(data)
    return manifest


def read_frozen(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    splits = {}
    for name, info in manifest['splits'].items():
        p = directory / info['file']
        if p.parent.resolve() != directory.resolve() or sha256(p) != info['sha256']:
            raise ValueError('Frozen data hash/path validation failed')
        rows = [TaskExample(**json.loads(s)) for s in p.read_text(encoding='utf-8').splitlines()]
        validate(rows, info['count'], {int(k): v for k, v in info['class_counts'].items()})
        if [r.source_id for r in rows] != info['source_ids'] or [r.row_index for r in rows] != info['row_indices']:
            raise ValueError('Frozen IDs changed')
        splits[name] = rows
    validate([r for rows in splits.values() for r in rows])
    return splits
