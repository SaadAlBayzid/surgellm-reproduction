import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from scripts.preflight_baseline import unresolved
from src.device_runtime import resolve_runtime, autocast_context, grad_scaler, optimizer_update


class RuntimeTests(unittest.TestCase):
    def test_cpu_detection_and_reject_fp16(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_runtime()["precision"], "fp32")
            with self.assertRaises(RuntimeError):
                resolve_runtime("cuda")
            with self.assertRaises(ValueError):
                resolve_runtime("cpu", "fp16")

    def test_cuda_selection_without_allocating(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.get_device_name", return_value="mock"):
            self.assertEqual(resolve_runtime()["device"], "cuda")
            self.assertEqual(resolve_runtime()["precision"], "fp16")

    def test_cpu_scaled_backward_update(self):
        runtime = resolve_runtime("cpu", "fp32")
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scaler = grad_scaler(runtime)
        before = model.weight.detach().clone()
        with autocast_context(runtime):
            loss = model(torch.ones(2, 2)).square().mean()
        scaler.scale(loss).backward()
        norm, skipped = optimizer_update(model, optimizer, scaler, 1.0)
        self.assertFalse(skipped)
        self.assertTrue(norm >= 0)
        self.assertFalse(torch.equal(before, model.weight))

    def test_reproduction_config_cannot_run(self):
        p = Path(__file__).resolve().parents[1] / "configs/baseline_roberta_four_task.json"
        cfg = json.loads(p.read_text())
        self.assertFalse(cfg["full_run_enabled"])
        self.assertEqual(len(cfg["tasks"]), 4)
        self.assertEqual(sum(t["reported_cap"] for t in cfg["tasks"].values()), 17830)
        self.assertEqual(cfg['training']['batch']['global_effective'],64)
        self.assertTrue(cfg['implementation_blockers'])
        self.assertTrue(cfg['assumptions'])
        self.assertEqual(unresolved(cfg),[])


if __name__ == "__main__":
    unittest.main()
