"""GPU helpers — device selection, CUDA smoke test, deterministic seeding.

Mirrors the Kronos pattern (torch cu121 on the RTX 3050). Everything degrades to
CPU if CUDA is missing so the pipeline still runs (just slower); the smoke test
CLI reports which path is live.
"""
from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Deterministic runs — Python, NumPy, Torch (CPU + CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass



def tune_backend(verbose: bool = False) -> dict:
    """Turn on the Ampere fast paths and make VRAM exhaustion fail loudly.

    Measured on this box (RTX 3050, sm 8.6, 6 GB) 2026-09-01, 4096x4096 matmul,
    median of 3 reps on an idle card at full clock: strict fp32 1.43 TFLOPS,
    TF32 2.62 TFLOPS -- **1.84x**. (Benchmark the card idle: an earlier read
    taken while ollama held 5.9 of 6 GB and the GPU sat at P8 gave 0.85/1.16 and
    a misleading 1.36x.)

    Only `matmul.allow_tf32` actually changes anything: torch already defaults
    `cudnn.allow_tf32` to True, so conv/RNN layers have been running TF32 in
    every run ever recorded here. Setting it below is defensive, not a change --
    do not read it as "alpaca's Conv1d/GRU results just moved".

    TF32 keeps the fp32 exponent but drops the mantissa to 10 bits, so a matmul
    is NOT bit-identical to a strict-fp32 one. GPU_TF32=0 restores the previous
    precision setting -- it does NOT buy bit-exact reproduction, which would
    also need cudnn.benchmark pinned, torch.use_deterministic_algorithms(True)
    with CUBLAS_WORKSPACE_CONFIG=:4096:8, and the same torch/CUDA/driver build.

    The memory cap stands in for the driver's "sysmem fallback" policy: on
    Windows an allocation past 6 GB silently spills into system RAM and the run
    goes 10-50x slower instead of failing. A research bench should fail loudly.
    Set GPU_MEM_FRACTION=0 to lift the cap, or to a float to change it.

    Idempotent and fail-safe -- any problem leaves torch's defaults alone.
    """
    state = {"tf32": False, "mem_fraction": None}
    try:
        import torch
        if not torch.cuda.is_available():
            return state

        if os.environ.get("GPU_TF32", "1") != "0":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            state["tf32"] = True

        raw = os.environ.get("GPU_MEM_FRACTION", "0.92")
        frac = float(raw)
        if frac > 0:
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            state["mem_fraction"] = frac
    except Exception as e:  # never let tuning break a run
        state["error"] = repr(e)
    if verbose:
        print(state)
    return state


def get_device():
    """Return the torch device to train on ('cuda' if available else 'cpu').

    Applies tune_backend() on the way out, so every caller gets the Ampere fast
    paths without having to remember them.
    """
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        tune_backend()
    return dev


def device_report() -> dict:
    """Fail-safe summary of the compute backend."""
    info = {"torch": None, "cuda_available": False, "device_name": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["tuning"] = tune_backend()
    except Exception as e:  # torch not installed / import error
        info["error"] = repr(e)
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(device_report(), indent=2))
