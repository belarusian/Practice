"""Training environment checklist and validation."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


SUPPORTED_DEVICES = ("cpu", "cuda", "mps")
SUPPORTED_TARGETS = ("vision", "audio", "text", "philosopher")
DEFAULT_TARGET = "vision"
DEFAULT_REQUIRED_CUDA_VRAM_GB = 4.0
DEFAULT_REQUIRED_DISK_GB = 5.0
TRAINING_INSTALL_HINT = "Install training dependencies with: uv sync --extra training"
TRANSFORMER_INSTALL_HINT = "Install transformer dependencies with: uv sync --extra transformer"


@dataclasses.dataclass(slots=True)
class CheckResult:
    """Result of a single checklist item."""

    name: str
    status: str  # "ok", "warning", "error"
    message: str
    details: str | None = None


@dataclasses.dataclass(slots=True)
class Checklist:
    """A collection of checks with their results."""

    target: str = DEFAULT_TARGET
    resolved_device: str = "cpu"
    checks: list[CheckResult] = dataclasses.field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return all(check.status != "error" for check in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(check.status == "warning" for check in self.checks)

    def add_result(self, result: CheckResult) -> None:
        self.checks.append(result)


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _import_torch():
    import torch

    return torch


def _validate_target(target: str) -> str:
    if target not in SUPPORTED_TARGETS:
        supported = ", ".join(SUPPORTED_TARGETS)
        raise ValueError(f"Unsupported training target {target!r}. Expected one of: {supported}")
    return target


def _validate_requested_device(device: str | None) -> str | None:
    if device is None:
        return None
    if device not in SUPPORTED_DEVICES:
        supported = ", ".join(SUPPORTED_DEVICES)
        raise ValueError(f"Unsupported device {device!r}. Expected one of: {supported}")
    return device


def _resolve_device(requested_device: str | None) -> str:
    if requested_device is not None:
        return requested_device

    try:
        torch = _import_torch()
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _existing_disk_usage_path(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            return Path.cwd()
        candidate = candidate.parent
    return candidate


def check_python_version() -> CheckResult:
    version = sys.version_info
    if version.major >= 3 and version.minor >= 11:
        return CheckResult(
            name="python_version",
            status="ok",
            message=f"Python {version.major}.{version.minor} available",
        )
    return CheckResult(
        name="python_version",
        status="error",
        message=f"Python {version.major}.{version.minor} is too old, need 3.11+",
    )


def check_target_dependencies(target: str) -> list[CheckResult]:
    target = _validate_target(target)

    dependencies = [("torch", "PyTorch")]
    if target == "vision":
        dependencies.append(("torchvision", "torchvision"))
    elif target == "audio":
        dependencies.append(("torchaudio", "torchaudio"))
    elif target == "text":
        dependencies.extend(
            [
                ("transformers", "Transformers"),
                ("datasets", "Hugging Face Datasets"),
            ]
        )
    elif target == "philosopher":
        dependencies.extend(
            [
                ("transformers", "Transformers"),
                ("datasets", "Hugging Face Datasets"),
            ]
        )

    results: list[CheckResult] = []
    for module_name, display_name in dependencies:
        if _has_module(module_name):
            results.append(
                CheckResult(
                    name=f"dependency_{module_name}",
                    status="ok",
                    message=f"{display_name} installed",
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"dependency_{module_name}",
                    status="error",
                    message=f"{display_name} not installed",
                    details=TRANSFORMER_INSTALL_HINT if target == "text" else TRAINING_INSTALL_HINT,
                )
            )
    return results


def check_device_backend(device: str) -> CheckResult:
    device = _validate_requested_device(device) or "cpu"

    if device == "cpu":
        return CheckResult(
            name="device_backend",
            status="ok",
            message="Using CPU training path",
        )

    try:
        torch = _import_torch()
    except ImportError:
        return CheckResult(
            name="device_backend",
            status="error",
            message=f"{device.upper()} requested but PyTorch is not installed",
            details=TRAINING_INSTALL_HINT,
        )

    if device == "cuda":
        if not torch.cuda.is_available():
            return CheckResult(
                name="device_backend",
                status="error",
                message="CUDA requested but not available to PyTorch",
                details="Check CUDA drivers or run with --device cpu",
            )
        return CheckResult(
            name="device_backend",
            status="ok",
            message=f"Using CUDA training path ({torch.cuda.get_device_name(0)})",
        )

    if device == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            return CheckResult(
                name="device_backend",
                status="error",
                message="MPS requested but not available to PyTorch",
                details="Use --device cpu or install a PyTorch build with MPS support",
            )
        return CheckResult(
            name="device_backend",
            status="ok",
            message="Using MPS training path",
        )

    raise AssertionError(f"Unhandled device: {device}")


def check_gpu_memory_free(required_gb: float = DEFAULT_REQUIRED_CUDA_VRAM_GB, device: int = 0) -> CheckResult:
    try:
        torch = _import_torch()
    except ImportError:
        return CheckResult(
            name="gpu_memory",
            status="error",
            message="PyTorch not installed, cannot check GPU memory",
            details=TRAINING_INSTALL_HINT,
        )

    if not torch.cuda.is_available():
        return CheckResult(
            name="gpu_memory",
            status="error",
            message="CUDA not available",
            details="Cannot check GPU memory without CUDA",
        )

    device_count = torch.cuda.device_count()
    if device_count <= device:
        return CheckResult(
            name="gpu_memory",
            status="error",
            message=f"GPU {device} not found",
            details=f"Only {device_count} GPU(s) available",
        )

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)

        if free_gb < required_gb:
            return CheckResult(
                name="gpu_memory",
                status="error",
                message=f"Insufficient FREE VRAM: {free_gb:.1f} GB",
                details=(
                    f"Total VRAM: {total_gb:.1f} GB | "
                    f"Free VRAM: {free_gb:.1f} GB | "
                    f"Required: {required_gb:.1f} GB\n"
                    "The GPU may be pinned by other processes. Try: nvidia-smi to see running processes."
                ),
            )

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().splitlines()
                if device < len(lines):
                    parts = [part.strip() for part in lines[device].split(",")]
                    if len(parts) >= 2:
                        gpu_util = float(parts[1])
                        if gpu_util > 90:
                            return CheckResult(
                                name="gpu_memory",
                                status="warning",
                                message=f"GPU under heavy load: {gpu_util:.0f}% utilization",
                                details=(
                                    f"Free VRAM: {free_gb:.1f} GB | GPU Utilization: {gpu_util:.0f}%\n"
                                    "Training will work but may be slower."
                                ),
                            )
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

        return CheckResult(
            name="gpu_memory",
            status="ok",
            message=f"GPU has sufficient FREE VRAM: {free_gb:.1f} GB",
            details=f"Total VRAM: {total_gb:.1f} GB | Free VRAM: {free_gb:.1f} GB",
        )
    except Exception as exc:
        return CheckResult(
            name="gpu_memory",
            status="error",
            message="Failed to check GPU memory",
            details=str(exc),
        )


def check_disk_space(path: Path, required_gb: float = DEFAULT_REQUIRED_DISK_GB) -> CheckResult:
    disk_path = _existing_disk_usage_path(path)
    stat = shutil.disk_usage(disk_path)
    free_gb = stat.free / (1024**3)

    if free_gb < required_gb:
        return CheckResult(
            name="disk_space",
            status="error",
            message=f"Insufficient disk space: {free_gb:.1f} GB free",
            details=f"Checked path: {disk_path} | Required: {required_gb:.1f} GB for datasets and checkpoints",
        )

    return CheckResult(
        name="disk_space",
        status="ok",
        message=f"Disk space available: {free_gb:.1f} GB free",
        details=f"Checked path: {disk_path}",
    )


def check_dataset_root(dataset_root: Path) -> CheckResult:
    if dataset_root.exists():
        return CheckResult(
            name=f"dataset_{dataset_root.name}",
            status="ok",
            message=f"Dataset root exists: {dataset_root}",
        )
    return CheckResult(
        name=f"dataset_{dataset_root.name}",
        status="warning",
        message=f"Dataset root does not exist: {dataset_root}",
        details="Datasets will be downloaded on first run (requires internet)",
    )


def run_training_checklist(
    *,
    target: str = DEFAULT_TARGET,
    device: str | None = None,
    output_dir: Path | None = None,
    dataset_root: Path | None = None,
) -> Checklist:
    target = _validate_target(target)
    requested_device = _validate_requested_device(device)
    resolved_device = _resolve_device(requested_device)

    checklist = Checklist(target=target, resolved_device=resolved_device)
    checklist.add_result(check_python_version())

    for result in check_target_dependencies(target):
        checklist.add_result(result)

    checklist.add_result(check_device_backend(resolved_device))

    if resolved_device == "cuda":
        checklist.add_result(check_gpu_memory_free())

    if output_dir is not None:
        checklist.add_result(check_disk_space(output_dir))

    if dataset_root is not None:
        checklist.add_result(check_dataset_root(dataset_root))

    return checklist


def format_checklist(checklist: Checklist, verbose: bool = False) -> str:
    lines = [
        "=" * 60,
        "TRAINING ENVIRONMENT CHECKLIST",
        "=" * 60,
        "",
        f"Target: {checklist.target}",
        f"Device: {checklist.resolved_device}",
        "",
    ]

    for check in checklist.checks:
        if check.status == "ok":
            prefix = "[OK]"
        elif check.status == "warning":
            prefix = "[! ]"
        else:
            prefix = "[ERR]"

        lines.append(f"{prefix} {check.name}: {check.message}")
        if verbose and check.details:
            lines.append(f"    {check.details}")

    lines.append("")
    if checklist.is_ready:
        lines.append(f"STATUS: Ready for {checklist.target} training")
    else:
        lines.append(f"STATUS: Not ready for {checklist.target} training")

    if checklist.has_warnings:
        lines.append("WARNING: Review items above for potential issues")

    lines.append("=" * 60)
    return "\n".join(lines)


def assert_training_ready(
    *,
    target: str = DEFAULT_TARGET,
    device: str | None = None,
    output_dir: Path | None = None,
    dataset_root: Path | None = None,
    verbose: bool = False,
) -> None:
    checklist = run_training_checklist(
        target=target,
        device=device,
        output_dir=output_dir,
        dataset_root=dataset_root,
    )
    output = format_checklist(checklist, verbose=verbose)

    if not checklist.is_ready:
        print(output)
        raise SystemExit(1)

    if verbose:
        print(output)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="ml-lab-check",
        description="Check training environment prerequisites.",
    )
    parser.add_argument("--target", choices=SUPPORTED_TARGETS, default=DEFAULT_TARGET)
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for artifacts")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset root directory")
    parser.add_argument("--verbose", action="store_true", help="Show details for all checks")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args(argv)

    checklist = run_training_checklist(
        target=args.target,
        device=args.device,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "target": checklist.target,
                    "resolved_device": checklist.resolved_device,
                    "is_ready": checklist.is_ready,
                    "has_warnings": checklist.has_warnings,
                    "checks": [
                        {
                            "name": check.name,
                            "status": check.status,
                            "message": check.message,
                            "details": check.details,
                        }
                        for check in checklist.checks
                    ],
                },
                indent=2,
            )
        )
    else:
        print(format_checklist(checklist, verbose=args.verbose))

    if not checklist.is_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
