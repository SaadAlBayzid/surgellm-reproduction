"""At most two debug optimizer steps; records SKIPPED when CUDA is unavailable."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.device_runtime import resolve_runtime, autocast_context, grad_scaler, optimizer_update
from src.roberta_baseline import EncodedSST2, seed_everything, evaluate
from src.sst2 import load_tsv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if Path("results").resolve() not in a.output.resolve().parents:
        raise ValueError("Output must be under results/")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(0, 4)
    runtime = resolve_runtime()
    report = {"runtime": runtime, "purpose": "tiny debugging CUDA check, not four-task reproduction", "full_training_started": False}
    if runtime["device"] != "cuda":
        report.update(status="SKIPPED", reason="torch.cuda.is_available() is false; CUDA path not validated")
        a.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        return
    start = time.time()
    try:
        torch.cuda.reset_peak_memory_stats()
        rows = load_tsv(a.assets / "train.tsv", "train")[:4]
        tokenizer = AutoTokenizer.from_pretrained(a.assets / "model", local_files_only=True, use_fast=True)
        data = EncodedSST2(rows, tokenizer, 128)
        model = AutoModelForSequenceClassification.from_pretrained(a.assets / "model", num_labels=2, local_files_only=True, use_safetensors=True).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        scaler = grad_scaler(runtime)
        initial = model.classifier.out_proj.weight.detach().clone()
        updates = []
        model.train()
        for batch in list(DataLoader(data, batch_size=1))[:2]:
            batch = {k: v.cuda() for k, v in batch.items()}
            with autocast_context(runtime):
                out = model(**batch)
            assert out.logits.shape == (1, 2) and torch.isfinite(out.loss)
            scaler.scale(out.loss).backward()
            norm, skipped = optimizer_update(model, optimizer, scaler, 1.0)
            updates.append({"loss": out.loss.item(), "gradient_norm": norm, "skipped": skipped})
        metrics = evaluate(model, DataLoader(data, batch_size=1), "cuda", runtime)
        changed = not torch.equal(initial, model.classifier.out_proj.weight.detach())
        assert changed, "No parameter update; inspect AMP overflow/skips"
        torch.cuda.synchronize()
        report.update(status="PASSED", updates=updates, parameter_update=changed, metrics=metrics,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.time()-start)
    except Exception as exc:
        report.update(status="FAILED", error=repr(exc))
        a.output.write_text(json.dumps(report, indent=2))
        raise
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
