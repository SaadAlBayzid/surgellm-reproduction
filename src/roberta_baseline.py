"""Plain single-task RoBERTa sequence classification; no feature channels."""
import math
import random
import numpy as np
import torch
import os
from torch.utils.data import Dataset


def seed_everything(seed, threads):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class EncodedSST2(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.encoding = tokenizer([r.text for r in rows], truncation=True,
                                  padding="max_length", max_length=max_length,
                                  return_tensors="pt")
        self.labels = torch.tensor([r.label for r in rows], dtype=torch.long)
        expected = (len(rows), max_length)
        assert self.encoding["input_ids"].shape == expected
        assert self.encoding["attention_mask"].shape == expected
        assert self.labels.shape == (len(rows),)
        assert self.encoding["input_ids"].dtype == torch.long

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.encoding.items()}, "labels": self.labels[i]}


def classification_metrics(labels, predictions):
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Metrics require equally sized nonempty inputs")
    if any(x not in (0, 1) for x in labels + predictions):
        raise ValueError("Binary labels required")
    f1 = []
    for label in (0, 1):
        tp = sum(y == label and p == label for y, p in zip(labels, predictions))
        fp = sum(y != label and p == label for y, p in zip(labels, predictions))
        fn = sum(y == label and p != label for y, p in zip(labels, predictions))
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {"accuracy": sum(y == p for y, p in zip(labels, predictions)) / len(labels),
            "macro_f1": sum(f1) / 2}


def learning_rate_factor(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


@torch.no_grad()
def evaluate(model, loader, device, runtime=None):
    from src.device_runtime import autocast_context
    runtime = runtime or {"device": device, "precision": "fp32"}
    model.eval()
    total_loss, labels, predictions = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_context(runtime):
            output = model(**batch)
        if not torch.isfinite(output.loss) or not torch.isfinite(output.logits).all():
            raise RuntimeError("Nonfinite evaluation output")
        total_loss += output.loss.item() * len(batch["labels"])
        labels.extend(batch["labels"].cpu().tolist())
        predictions.extend(output.logits.argmax(-1).cpu().tolist())
    return {"loss": total_loss / len(labels), **classification_metrics(labels, predictions),
            "examples": len(labels)}
