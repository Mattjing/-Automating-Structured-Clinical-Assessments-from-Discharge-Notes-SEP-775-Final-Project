"""Quick GPU readiness check for PyTorch/MedBERT workflows."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"[ERROR] Could not import torch: {exc}")
        print("Install PyTorch with CUDA support before running GPU workloads.")
        return 1

    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_build: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("cuda_available: False")
        print("No CUDA runtime/device detected. Workloads will run on CPU.")
        return 2

    device_count = torch.cuda.device_count()
    print("cuda_available: True")
    print(f"cuda_device_count: {device_count}")

    for idx in range(device_count):
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        print(f"cuda_device_{idx}: {name} (compute_capability={cap[0]}.{cap[1]})")

    try:
        # Validate that a real tensor operation executes on GPU.
        a = torch.randn((1024, 1024), device="cuda")
        b = torch.randn((1024, 1024), device="cuda")
        _ = (a @ b).mean().item()
        torch.cuda.synchronize()
        print("gpu_tensor_op: PASS")
    except Exception as exc:
        print(f"gpu_tensor_op: FAIL ({exc})")
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
