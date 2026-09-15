"""Only smoke mode is enabled in this development stage."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.sst2 import load_tsv, prepare, _sample
from src.roberta_baseline import EncodedSST2, seed_everything, evaluate, learning_rate_factor
from src.device_runtime import resolve_runtime, autocast_context, grad_scaler, optimizer_update


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/debug_sst2_roberta.json"))
    p.add_argument("--device", choices=["auto", "cpu", "cuda"])
    p.add_argument("--precision", choices=["auto", "fp32", "fp16"])
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=[0, 1, 2])
    p.add_argument("--smoke", action="store_true", required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.seed is not None:
        cfg["seed"] = a.seed
    assert cfg["model_name"] == "roberta-base" and cfg["num_labels"] == 2
    assert cfg["smoke_epochs"] == 1 and 100 <= cfg["smoke_train_examples"] <= 200
    device_info = resolve_runtime(a.device or cfg["device"], a.precision or cfg["precision"])
    cfg.update({k: device_info[k] for k in ["device", "precision"]})
    seed_everything(cfg["seed"], cfg["cpu_threads"])
    output = a.output.resolve()
    results = Path("results").resolve()
    if results not in output.parents:
        raise ValueError("Run output must be a new directory under results/")
    output.mkdir(parents=True, exist_ok=False)
    cfg["mode"] = "smoke"
    cfg["epochs_executed_max"] = cfg["smoke_epochs"]
    (output / "resolved_config.json").write_text(json.dumps(cfg, indent=2))
    provenance = json.loads((a.assets / "provenance.json").read_text())
    for name, info in provenance["roberta-base"]["files"].items():
        actual = hashlib.sha256((a.assets / "model" / name).read_bytes()).hexdigest()
        if actual != info["sha256"]:
            raise ValueError(f"Model asset changed since download: {name}")
    (output / "asset_provenance.json").write_text(json.dumps(provenance, indent=2))
    splits, manifest = prepare(load_tsv(a.assets / "train.tsv", "train"),
                               load_tsv(a.assets / "dev.tsv", "validation"),
                               policy=cfg["split_policy"], seed=cfg["seed"], cap=cfg["training_cap"])
    rng = random.Random(cfg["seed"])
    train, _ = _sample(splits["train"], cfg["smoke_train_examples"], rng)
    val, _ = _sample(splits["validation"], cfg["smoke_validation_examples"], rng)
    manifest["smoke_ids"] = {"train": [r.source_id for r in train], "validation": [r.source_id for r in val]}
    manifest["official_dev_evaluated"] = False
    manifest["input_hashes"] = {name: hashlib.sha256((a.assets / name).read_bytes()).hexdigest() for name in ["train.tsv", "dev.tsv"]}
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(a.assets / "model", use_fast=True, local_files_only=True, add_prefix_space=False)
    tokenizer.padding_side = tokenizer.truncation_side = "right"
    train_data = EncodedSST2(train, tokenizer, cfg["max_length"])
    val_data = EncodedSST2(val, tokenizer, cfg["max_length"])
    loader = DataLoader(train_data, batch_size=cfg["micro_batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    val_loader = DataLoader(val_data, batch_size=cfg["eval_batch_size"])
    model = AutoModelForSequenceClassification.from_pretrained(a.assets / "model", num_labels=2,
              classifier_dropout=cfg["classifier_dropout"], local_files_only=True, use_safetensors=True)
    model.to(cfg["device"])
    assert all(p.requires_grad for p in model.parameters()), "Fine-tune entire encoder and head"
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                betas=tuple(cfg["adam_betas"]), eps=cfg["adam_epsilon"], weight_decay=cfg["weight_decay"])
    scaler = grad_scaler(device_info)
    ga = cfg["gradient_accumulation_steps"]
    steps = math.ceil(len(loader) / ga) * cfg["smoke_epochs"]
    warmup = math.ceil(steps * cfg["warmup_fraction"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, steps, warmup))
    checks = {"tokenization": True, "tensor_shapes": True, "forward": False, "finite_loss": False,
              "backward": False, "optimizer_updates_parameters": False, "evaluation_macro_f1": False}
    runtime = {"python": platform.python_version(), "torch": torch.__version__,
               "transformers": importlib.metadata.version("transformers"), "device": cfg["device"],
               "total_optimizer_steps": steps, "warmup_steps": warmup,
               "train_shape": list(train_data.encoding["input_ids"].shape),
               "validation_shape": list(val_data.encoding["input_ids"].shape)}
    runtime["device_detection"] = device_info
    runtime["package_versions"] = {name: importlib.metadata.version(name) for name in
        ["numpy", "tokenizers", "safetensors", "huggingface-hub", "pyarrow", "PyYAML"]}
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    initial = model.classifier.out_proj.weight.detach().clone()
    initial_encoder = model.roberta.encoder.layer[0].attention.self.query.weight.detach().clone()
    start = time.time()
    best, stale, metrics = -math.inf, 0, []
    with (output / "training_log.jsonl").open("w") as log:
        for epoch in range(cfg["smoke_epochs"]):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for index, batch in enumerate(loader):
                batch = {k: v.to(cfg["device"]) for k, v in batch.items()}
                window_start = (index // ga) * ga * cfg["micro_batch_size"]
                window_n = min(ga * cfg["micro_batch_size"], len(train_data) - window_start)
                with autocast_context(device_info):
                    out = model(**batch)
                assert out.logits.shape == (len(batch["labels"]), 2)
                assert out.loss.ndim == 0 and torch.isfinite(out.loss) and torch.isfinite(out.logits).all()
                checks["forward"] = checks["finite_loss"] = True
                scaler.scale(out.loss * len(batch["labels"]) / window_n).backward()
                assert model.classifier.out_proj.weight.grad is not None
                checks["backward"] = True
                loss_sum += out.loss.item() * len(batch["labels"])
                record = {"epoch": epoch + 1, "micro_step": index + 1, "loss": out.loss.item(), "lr": optimizer.param_groups[0]["lr"]}
                if (index + 1) % ga == 0 or index + 1 == len(loader):
                    norm, skipped = optimizer_update(model, optimizer, scaler, cfg["gradient_clip_norm"])
                    if not skipped:
                        scheduler.step()
                    record["amp_update_skipped"] = skipped
                    record["gradient_norm_before_clip"] = float(norm)
                    print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + "\n")
                log.flush()
            result = {"epoch": epoch + 1, "train_loss": loss_sum / len(train_data),
                      "validation": evaluate(model, val_loader, cfg["device"], device_info)}
            metrics.append(result)
            checks["evaluation_macro_f1"] = math.isfinite(result["validation"]["macro_f1"])
            if result["validation"]["macro_f1"] > best:
                best, stale = result["validation"]["macro_f1"], 0
                torch.save(model.state_dict(), output / "best_model.pt")
            else:
                stale += 1
            (output / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
            if stale >= cfg["early_stopping_patience"]:
                break
    checks["optimizer_updates_parameters"] = not torch.equal(initial, model.classifier.out_proj.weight.detach())
    checks["encoder_parameters_updated"] = not torch.equal(initial_encoder, model.roberta.encoder.layer[0].attention.self.query.weight.detach())
    report = {"checks": checks, "passed": all(checks.values()), "elapsed_seconds": time.time() - start,
              "epochs": len(metrics), "training_examples": len(train), "validation_examples": len(val),
              "head_max_parameter_change": (initial - model.classifier.out_proj.weight.detach()).abs().max().item()}
    (output / "smoke_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    assert report["passed"]


if __name__ == "__main__":
    main()
