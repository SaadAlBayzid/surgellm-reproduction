"""Shared CPU/CUDA detection and FP16 helpers (Torch 2.1-compatible API)."""
from contextlib import nullcontext
import torch


def resolve_runtime(device="auto", precision="auto"):
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto/cpu/cuda")
    if precision not in {"auto", "fp32", "fp16"}:
        raise ValueError("precision must be auto/fp32/fp16")
    available = torch.cuda.is_available()
    selected = ("cuda" if available else "cpu") if device == "auto" else device
    if selected == "cuda" and not available:
        raise RuntimeError("CUDA requested but unavailable to this PyTorch build/driver")
    resolved = ("fp16" if selected == "cuda" else "fp32") if precision == "auto" else precision
    if resolved == "fp16" and selected != "cuda":
        raise ValueError("FP16 requires CUDA; use CPU FP32 for debugging")
    return {"device": selected, "precision": resolved, "cuda_available": available,
            "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if available else None}


def autocast_context(runtime):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if runtime["precision"] == "fp16" else nullcontext()


def grad_scaler(runtime):
    return torch.cuda.amp.GradScaler(enabled=runtime["precision"] == "fp16")


def optimizer_update(model, optimizer, scaler, clip_norm):
    scaler.unscale_(optimizer)
    # AMP needs nonfinite gradients to reach scaler.step so it can skip/reduce scale.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=not scaler.is_enabled())
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < before
    optimizer.zero_grad(set_to_none=True)
    return float(norm), skipped
