# Quick Reference: Training Checklist

## Overview

This checklist covers the local lab split: Sunny for CUDA training, Mac Studio for large Apple Silicon inference, and CPU fallback when training deps are installed but GPU backends are not in use.

## Commands

```bash
# Defaults target to vision and auto-selects device (cuda/mps/cpu)
uv run ml-lab check --target vision --verbose

# Force CPU mode
uv run ml-lab check --target vision --device cpu --verbose

# Force CUDA mode (useful for Sunny)
uv run ml-lab check --target vision --device cuda --verbose

# Check audio target
uv run ml-lab check --target audio --verbose

# JSON for CI/CD
uv run ml-lab check --target vision --json
```

## Checklist Output Reference

| Status | Symbol | Meaning |
|--------|--------|---------|
| `ok` | `[OK]` | Check passed, ready to proceed |
| `error` | `[ERR]` | Check failed, must fix before training |
| `warning` | `[! ]` | Check passed but with caveats |

## Exit Codes

| Code | Meaning |
|------|--------|
| 0 | Environment is ready for the requested training target |
| 1 | Environment has errors, fix before training |

## GPU Memory Check Details

The checklist uses `torch.cuda.mem_get_info()` to get **actual free VRAM**, not just total VRAM.

### Sunny Scenario 1: GPU Free
```
[OK] gpu_memory: GPU has sufficient FREE VRAM: 22.3 GB
    Total VRAM: 24.0 GB | Free VRAM: 22.3 GB
```

### Sunny Scenario 2: GPU Busy (Demo Services)
```
[ERR] gpu_memory: Insufficient FREE VRAM: 2.1 GB
    Total VRAM: 24.0 GB | Free VRAM: 2.1 GB | Required: 4.0 GB
    The GPU may be pinned by other processes. Try: nvidia-smi to see running processes.
```

## Integration Points

| Command | Checklist Runs? | Purpose |
|---------|-----------------|---------|
| `ml-lab check` | Manual | Verify environment |
| `ml-lab train-vision` | Automatic | Fail fast if not ready |
| `ml-lab train-audio` | Automatic | Fail fast if not ready |
| `ml-lab serve` | No | Serving doesn't need training checks |

## Common Scenarios on Sunny

### "GPU may be pinned by other processes"
```bash
# Check what's using the GPU
nvidia-smi

# Kill processes if needed (example for container cleanup)
docker rm -f $(docker ps -aq)
```

### "Insufficient FREE VRAM"
```bash
# Check free memory
uv run ml-lab check --target vision --device cuda --verbose

# Or wait for processes to finish, or restart the GPU
nvidia-smi --gpu-reset
```

### CPU Fallback
```bash
# Train on CPU if GPU is busy
uv run ml-lab train-vision --device cpu --output-dir artifacts/vision-baseline
```

## Best Practices for Sunny

1. **Run checklist before training**: `uv run ml-lab check --target vision --verbose`
2. **Check GPU status first**: `nvidia-smi` to see running processes
3. **If GPU is busy**: Wait or kill demo services
4. **CPU fallback**: Use `--device cpu` for non-GPU-critical work
5. **CI/CD**: Use `make training-mode` or `ml-lab check --json` with exit code
