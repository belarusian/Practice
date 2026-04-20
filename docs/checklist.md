# Training Environment Checklist

## Overview

The checklist module (`src/industry_ml_lab/training/checklist.py`) provides live checks that verify what's actually available on your machine before training starts. This prevents "training isn't possible until VRAM is free" from being rediscovered every session.

The checklist is designed around the local lab machine split. On **Sunny (4090 GPU)**, GPU pressure from demo services is the primary constraint. On Mac and CPU-only environments, the main constraint is whether the training dependencies are actually installed.

## Commands

### Check Environment

```bash
# Human-readable output (defaults target to vision, auto-selects device)
uv run ml-lab check --target vision --verbose

# JSON output for CI/CD
uv run ml-lab check --target vision --json

# Check specific target
uv run ml-lab check --target audio --verbose

# Force specific device (cpu, cuda, mps)
uv run ml-lab check --target vision --device cuda --verbose
```

### Make Targets

```bash
# Check environment
make check

# Exit with error if not ready to train
make training-mode
```

## Live Checks

### 1. Python Version
- **Status**: `ok` | `error`
- **Checks**: Python 3.11+ is installed
- **Fix**: Install Python 3.11 or later

### 2. Target Dependencies (vision/audio)
- **Status**: `ok` | `error`
- **Checks**: Required PyTorch packages installed
  - `vision`: `torch`, `torchvision`
  - `audio`: `torch`, `torchaudio`
- **Fix**: `uv sync --extra training`

### 3. Device Backend
- **Status**: `ok` | `error`
- **Checks**: Requested device backend is available and usable
  - `cpu`: backend is always available, but target dependencies must still be installed
  - `cuda`: Requires CUDA drivers and `torch.cuda.is_available()`
  - `mps`: Requires Apple Silicon and `torch.backends.mps.is_available()`
- **Auto-selects device**: CUDA → MPS → CPU

### 4. GPU Memory (CUDA only)
- **Status**: `ok` | `warning` | `error`
- **Checks**: ACTUAL free VRAM using `torch.cuda.mem_get_info()`
- **Critical**: Checks FREE memory, not total VRAM - this is the Sunny constraint
- **Also checks**: GPU utilization via `nvidia-smi` if available
- **Fix**: Kill processes using GPU or wait for memory to free

### 5. Disk Space
- **Status**: `ok` | `error`
- **Checks**: The nearest existing parent of the output path has 5GB+ free space
- **Fix**: Free up disk space or use different output directory

### 6. Dataset Existence
- **Status**: `ok` | `warning`
- **Checks**: Dataset root directory exists
- **Note**: Datasets will be downloaded on first run if missing

## GPU Memory Check Details

The checklist uses `torch.cuda.mem_get_info()` to get **actual free VRAM**, not just total VRAM. This is critical on Sunny where:

- Total VRAM: 24 GB (4090)
- Free VRAM: May be 2-4 GB when demo services are running
- **The checklist fails if free VRAM < 4 GB**

Example output when GPU is busy:
```
[ERR] gpu_memory: Insufficient FREE VRAM: 2.1 GB
    Total VRAM: 24.0 GB | Free VRAM: 2.1 GB | Required: 4.0 GB
    The GPU may be pinned by other processes. Try: nvidia-smi to see running processes.
```

## Checklist Output

```
============================================================
TRAINING ENVIRONMENT CHECKLIST
============================================================

Target: vision
Device: mps

[OK] python_version: Python 3.13 available
[OK] dependency_torch: PyTorch installed
[OK] dependency_torchvision: torchvision installed
[OK] device_backend: Using MPS training path

STATUS: Ready for vision training
============================================================
```

With `--verbose` on Sunny with free GPU:
```
============================================================
TRAINING ENVIRONMENT CHECKLIST
============================================================

Target: vision
Device: cuda

[OK] python_version: Python 3.13 available
[OK] dependency_torch: PyTorch installed
[OK] dependency_torchvision: torchvision installed
[OK] device_backend: Using CUDA training path (NVIDIA GeForce RTX 4090)
[OK] gpu_memory: GPU has sufficient FREE VRAM: 22.3 GB
    Total VRAM: 24.0 GB | Free VRAM: 22.3 GB

STATUS: Ready for vision training
============================================================
```

With `--verbose` on Sunny with busy GPU:
```
============================================================
TRAINING ENVIRONMENT CHECKLIST
============================================================

Target: vision
Device: cuda

[OK] python_version: Python 3.13 available
[OK] dependency_torch: PyTorch installed
[OK] dependency_torchvision: torchvision installed
[OK] device_backend: Using CUDA training path (NVIDIA GeForce RTX 4090)
[ERR] gpu_memory: Insufficient FREE VRAM: 3.2 GB
    Total VRAM: 24.0 GB | Free VRAM: 3.2 GB | Required: 4.0 GB
    The GPU may be pinned by other processes. Try: nvidia-smi to see running processes.

STATUS: Not ready for vision training
============================================================
```

## JSON Output

```json
{
  "target": "vision",
  "resolved_device": "mps",
  "is_ready": true,
  "has_warnings": false,
  "checks": [
    {
      "name": "python_version",
      "status": "ok",
      "message": "Python 3.13 available",
      "details": null
    },
    {
      "name": "dependency_torch",
      "status": "ok",
      "message": "PyTorch installed",
      "details": null
    },
    {
      "name": "dependency_torchvision",
      "status": "ok",
      "message": "torchvision installed",
      "details": null
    },
    {
      "name": "device_backend",
      "status": "ok",
      "message": "Using MPS training path",
      "details": null
    }
  ]
}
```

## Integration Points

| Command | Checklist Runs? | Purpose |
|---------|-----------------|---------|
| `ml-lab check` | Manual | Verify environment |
| `ml-lab train-vision` | Automatic | Fail fast if not ready |
| `ml-lab train-audio` | Automatic | Fail fast if not ready |
| `ml-lab serve` | No | Serving doesn't need training checks |

## Guarantees

The checklist guarantees:

- ✅ No false "ready" on lite/dev environments (checks for required dependencies)
- ✅ No false "ready" on Mac/CPU/MPS when training deps are missing (checks torch, torchvision/torchaudio)
- ✅ Sunny/CUDA readiness reflects real free VRAM, not only hardware presence (uses `torch.cuda.mem_get_info()`)
- ✅ Training commands fail through checklist output before heavy training imports when required deps are missing

## Best Practices

1. **Run checklist before training**: `uv run ml-lab check --target vision --verbose`
2. **Check GPU status first**: `nvidia-smi` to see running processes
3. **If GPU is busy**: Wait or kill demo services
4. **CPU fallback**: Use `--device cpu` for non-GPU-critical work
5. **CI/CD**: Use `make training-mode` or `ml-lab check --json` with exit code
