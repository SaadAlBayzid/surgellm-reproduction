"""Shared RoBERTa baseline, deterministic batch plans and complete-split metrics."""
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from src.device_runtime import autocast_context

TASKS = ('D1', 'D2', 'D3', 'D4')


class EarlyStopping:
    def __init__(self, patience):
        if patience < 1:
            raise ValueError('Patience must be positive')
        self.patience, self.best, self.stale = patience, -float('inf'), 0

    def observe(self, score):
        if not math.isfinite(score):
            raise ValueError('Nonfinite stopping score')
        improved = score > self.best
        if improved:
            self.best, self.stale = score, 0
        else:
            self.stale += 1
        return improved, self.stale >= self.patience


class FourTaskRoberta(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        if encoder.config.hidden_size != head['hidden_size']:
            raise ValueError('Encoder/head hidden sizes disagree')
        self.encoder = encoder
        self.heads = nn.ModuleDict({task: nn.Sequential(
            nn.Dropout(head['dropout_before_first']),
            nn.Linear(head['hidden_size'], head['intermediate_size']), nn.GELU(),
            nn.Dropout(head['dropout_before_second']),
            nn.Linear(head['intermediate_size'], 2)) for task in TASKS})
        for module in self.heads.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=head.get('initializer_std', .02))
                nn.init.zeros_(module.bias)

    def forward(self, task_id, input_ids, attention_mask, labels=None):
        if task_id not in self.heads:
            raise ValueError('Unknown task: ' + task_id)
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.heads[task_id](hidden)
        return {'logits': logits, 'loss': None if labels is None else F.cross_entropy(logits.float(), labels)}


def batch_plan(sizes, per_device, world_size, accumulation, policy, seed, epoch):
    """Global task-homogeneous microbatches. Full windows via documented padding.

    Proportional: visit every row once before each task's final batch is padded;
    shuffle task batch order. Balanced: equal rounds in D1..D4 order, cycling
    shorter tasks with reshuffles. A trailing accumulation window is padded by
    replaying leading microbatches. Padding affects exposures, never split files.
    """
    if set(sizes) != set(TASKS) or any(n <= 0 for n in sizes.values()):
        raise ValueError('Four nonempty tasks required')
    if min(per_device, world_size, accumulation) < 1 or policy not in ('proportional', 'task_balanced'):
        raise ValueError('Invalid batch plan settings')
    width = per_device * world_size
    counts = {t: math.ceil(sizes[t] / width) for t in TASKS}
    rng = random.Random(seed + 1000003 * epoch)
    if policy == 'task_balanced':
        order = list(TASKS) * max(counts.values())
    else:
        order = [t for t in TASKS for _ in range(counts[t])]
        rng.shuffle(order)
    pools, cursors = {}, {t: 0 for t in TASKS}
    for t in TASKS:
        pools[t] = list(range(sizes[t]))
        rng.shuffle(pools[t])
    plan = []
    for t in order:
        ids = []
        while len(ids) < width:
            if cursors[t] == sizes[t]:
                rng.shuffle(pools[t])
                cursors[t] = 0
            take = min(width - len(ids), sizes[t] - cursors[t])
            ids.extend(pools[t][cursors[t]:cursors[t] + take])
            cursors[t] += take
        plan.append((t, ids))
    original = len(plan)
    for i in range((-original) % accumulation):
        task, ids = plan[i % original]
        plan.append((task, list(ids)))
    return plan


def batch_semantics(plan, per_device, world_size, accumulation, expected):
    actual = per_device * world_size * accumulation
    if actual != expected or len(plan) % accumulation:
        raise ValueError('Effective global batch / accumulation window mismatch')
    if any(len(ids) != per_device * world_size for _, ids in plan):
        raise ValueError('Incomplete global microbatch')
    return {'per_device': per_device, 'world_size': world_size, 'accumulation': accumulation,
            'effective_global_batch': actual, 'microbatches_per_epoch': len(plan),
            'updates_per_epoch': len(plan) // accumulation,
            'example_exposures_per_epoch': sum(len(ids) for _, ids in plan)}


def optimizer_groups(model, decay, exclusions):
    groups = {True: [], False: []}
    for name, param in model.named_parameters():
        excluded = any(name.endswith(suffix) for suffix in exclusions)
        groups[excluded].append(param)
    return [{'params': groups[False], 'weight_decay': decay},
            {'params': groups[True], 'weight_decay': 0.0}]


def binary_metrics(labels, probabilities, loss_sum):
    # AUC ranks ALL predictions for this split, never averages batch AUCs.
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    if not labels or len(labels) != len(probabilities):
        raise ValueError('Nonempty matching predictions required')
    preds = [int(p > .5) for p in probabilities]  # argmax ties choose class 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average='macro', zero_division=0)
    single_class = len(set(labels)) < 2
    return {'examples': len(labels), 'loss': loss_sum / len(labels),
            'accuracy': float(accuracy_score(labels, preds)), 'macro_f1': float(f1),
            'macro_precision': float(precision), 'macro_recall': float(recall),
            'roc_auc': None if single_class else float(roc_auc_score(labels, probabilities)),
            'roc_auc_undefined_reason': 'single_class_split' if single_class else None}


def tensor_batch(encoded, indices, device):
    return {key: value[indices].to(device) for key, value in encoded.items()}


@torch.no_grad()
def evaluate_task(model, task, encoded, device, runtime, batch_size):
    """Rank zero uses the unwrapped model and visits every row exactly once."""
    model.eval()
    labels, probabilities, loss_sum = [], [], 0.0
    n = len(encoded['labels'])
    for start in range(0, n, batch_size):
        batch = tensor_batch(encoded, list(range(start, min(start + batch_size, n))), device)
        with autocast_context(runtime):
            output = model(task_id=task, **batch)
        if not torch.isfinite(output['loss']) or not torch.isfinite(output['logits']).all():
            raise RuntimeError('Nonfinite evaluation output')
        loss_sum += output['loss'].item() * len(batch['labels'])
        labels.extend(batch['labels'].cpu().tolist())
        probabilities.extend(output['logits'].float().softmax(-1)[:, 1].cpu().tolist())
    result = binary_metrics(labels, probabilities, loss_sum)
    if result['examples'] != n:
        raise RuntimeError('Incomplete split evaluation')
    return result


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, score, metadata):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'epoch': epoch, 'score': score, 'metadata': metadata}, temporary)
    temporary.replace(path)


def load_checkpoint(path, model):
    # Only locally generated checkpoints; never consume untrusted pickle files.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    return checkpoint
