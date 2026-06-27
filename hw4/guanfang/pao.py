"""Run the official trainer and record wall time and peak CUDA memory."""

import json
import runpy
import sys
import time
from pathlib import Path

import torch


def argument_value(*names):
    for name in names:
        if name in sys.argv:
            return sys.argv[sys.argv.index(name) + 1]
    return None


model_path = argument_value("-m", "--model_path")
if model_path is None:
    raise ValueError("pao.py requires -m/--model_path")

start = time.perf_counter()
status = "completed"
error = None
torch.cuda.reset_peak_memory_stats()

try:
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
except BaseException as exc:
    status = "failed"
    error = f"{type(exc).__name__}: {exc}"
    raise
finally:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    record = {
        "status": status,
        "error": error,
        "wall_seconds": time.perf_counter() - start,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    output = Path(model_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "jilu.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("BENCHMARK", json.dumps(record, ensure_ascii=False), flush=True)
