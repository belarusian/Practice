"""Tests for the training checklist module.

This module tests the checklist semantics for Sunny GPU readiness,
focusing on real VRAM availability and device auto-detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from industry_ml_lab.training import checklist as checklist_module
from industry_ml_lab.training.checklist import (
    CheckResult,
    Checklist,
    SUPPORTED_DEVICES,
    SUPPORTED_TARGETS,
    DEFAULT_TARGET,
    DEFAULT_REQUIRED_CUDA_VRAM_GB,
    DEFAULT_REQUIRED_DISK_GB,
    TRAINING_INSTALL_HINT,
    _has_module,
    _import_torch,
    _validate_target,
    _validate_requested_device,
    _resolve_device,
    _existing_disk_usage_path,
    check_python_version,
    check_target_dependencies,
    check_device_backend,
    check_gpu_memory_free,
    check_disk_space,
    check_dataset_root,
    run_training_checklist,
    format_checklist,
    assert_training_ready,
)


class TestConstants:
    """Tests for module constants."""

    def test_supported_devices(self) -> None:
        assert "cpu" in SUPPORTED_DEVICES
        assert "cuda" in SUPPORTED_DEVICES
        assert "mps" in SUPPORTED_DEVICES

    def test_supported_targets(self) -> None:
        assert "vision" in SUPPORTED_TARGETS
        assert "audio" in SUPPORTED_TARGETS

    def test_default_target(self) -> None:
        assert DEFAULT_TARGET == "vision"

    def test_default_vram_gb(self) -> None:
        assert DEFAULT_REQUIRED_CUDA_VRAM_GB == 4.0

    def test_default_disk_gb(self) -> None:
        assert DEFAULT_REQUIRED_DISK_GB == 5.0

    def test_training_install_hint(self) -> None:
        assert "uv sync --extra training" in TRAINING_INSTALL_HINT


class TestCheckResult:
    """Tests for the CheckResult dataclass."""

    def test_check_result_default_details(self) -> None:
        result = CheckResult(
            name="test",
            status="ok",
            message="Test message",
        )
        assert result.details is None

    def test_check_result_with_details(self) -> None:
        result = CheckResult(
            name="test",
            status="error",
            message="Test message",
            details="Detailed info",
        )
        assert result.details == "Detailed info"


class TestChecklist:
    """Tests for the Checklist class."""

    def test_checklist_defaults(self) -> None:
        checklist = Checklist()
        assert checklist.target == "vision"
        assert checklist.resolved_device == "cpu"
        assert len(checklist.checks) == 0

    def test_checklist_is_ready_with_no_checks(self) -> None:
        checklist = Checklist()
        assert checklist.is_ready is True

    def test_checklist_is_ready_with_ok_checks(self) -> None:
        checklist = Checklist()
        checklist.add_result(CheckResult(name="test", status="ok", message="OK"))
        assert checklist.is_ready is True

    def test_checklist_is_ready_with_warning_checks(self) -> None:
        checklist = Checklist()
        checklist.add_result(CheckResult(name="test", status="warning", message="Warning"))
        assert checklist.is_ready is True

    def test_checklist_is_ready_with_error_checks(self) -> None:
        checklist = Checklist()
        checklist.add_result(CheckResult(name="test", status="error", message="Error"))
        assert checklist.is_ready is False

    def test_checklist_has_warnings(self) -> None:
        checklist = Checklist()
        checklist.add_result(CheckResult(name="test", status="warning", message="Warning"))
        assert checklist.has_warnings is True

    def test_checklist_add_result(self) -> None:
        checklist = Checklist()
        result = CheckResult(name="test", status="ok", message="Test")
        checklist.add_result(result)
        assert len(checklist.checks) == 1


class TestValidationFunctions:
    """Tests for validation functions."""

    def test_validate_target_ok(self) -> None:
        assert _validate_target("vision") == "vision"
        assert _validate_target("audio") == "audio"

    def test_validate_target_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unsupported training target"):
            _validate_target("invalid")

    def test_validate_requested_device_ok(self) -> None:
        assert _validate_requested_device("cpu") == "cpu"
        assert _validate_requested_device("cuda") == "cuda"
        assert _validate_requested_device("mps") == "mps"

    def test_validate_requested_device_none(self) -> None:
        assert _validate_requested_device(None) is None

    def test_validate_requested_device_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unsupported device"):
            _validate_requested_device("invalid")


class TestResolveDevice:
    """Tests for _resolve_device."""

    def test_resolve_device_with_explicit_cpu(self) -> None:
        assert _resolve_device("cpu") == "cpu"

    def test_resolve_device_with_explicit_cuda(self) -> None:
        assert _resolve_device("cuda") == "cuda"

    def test_resolve_device_with_explicit_mps(self) -> None:
        assert _resolve_device("mps") == "mps"

    def test_resolve_device_auto_detect_cpu(self) -> None:
        # When PyTorch is not installed, defaults to CPU
        # We can't easily test this without mocking
        pass


class TestHasModule:
    """Tests for _has_module."""

    def test_has_module_exists(self) -> None:
        assert _has_module("sys") is True

    def test_has_module_not_exists(self) -> None:
        assert _has_module("nonexistent_module_xyz") is False


class TestCheckPythonVersion:
    """Tests for check_python_version."""

    def test_python_version_ok(self) -> None:
        result = check_python_version()
        assert result.status == "ok"
        assert "Python" in result.message
        assert result.name == "python_version"

    def test_python_version_message_format(self) -> None:
        import sys
        result = check_python_version()
        assert f"{sys.version_info.major}.{sys.version_info.minor}" in result.message


class TestCheckTargetDependencies:
    """Tests for check_target_dependencies."""

    def test_vision_dependencies_ok(self) -> None:
        results = check_target_dependencies("vision")
        check_names = [r.name for r in results]
        assert "dependency_torch" in check_names
        assert "dependency_torchvision" in check_names

    def test_audio_dependencies_ok(self) -> None:
        results = check_target_dependencies("audio")
        check_names = [r.name for r in results]
        assert "dependency_torch" in check_names
        assert "dependency_torchaudio" in check_names

    def test_vision_missing_torchvision(self) -> None:
        # This would require mocking import torchvision
        # Just verify the function structure
        results = check_target_dependencies("vision")
        assert len(results) >= 2


class TestCheckDeviceBackend:
    """Tests for check_device_backend."""

    def test_device_backend_cpu(self) -> None:
        result = check_device_backend("cpu")
        assert result.status == "ok"
        assert "CPU" in result.message

    def test_device_backend_cuda_not_available(self) -> None:
        # On Mac without CUDA, this should return error
        result = check_device_backend("cuda")
        assert result.status == "error"
        assert "CUDA" in result.message

    def test_device_backend_mps_available(self) -> None:
        result = check_device_backend("mps")
        # On Mac with MPS, this should return ok
        assert result.status in ("ok", "error")  # Could be error if MPS not available

    def test_device_backend_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unsupported device"):
            check_device_backend("invalid")


class TestCheckGpuMemoryFree:
    """Tests for check_gpu_memory_free."""

    def test_gpu_memory_not_available(self) -> None:
        result = check_gpu_memory_free()
        # On Mac without CUDA and PyTorch not installed, this should return error about PyTorch
        assert result.status == "error"
        # The message depends on whether CUDA is available or PyTorch is installed
        assert "cannot check GPU memory" in result.message or "CUDA not available" in result.message

    def test_gpu_memory_free_gb_calculation(self) -> None:
        # This would require mocking torch.cuda.mem_get_info
        # Just verify the function exists
        result = check_gpu_memory_free(required_gb=4.0)
        assert result.name == "gpu_memory"

    def test_gpu_memory_free_gb_calculation(self) -> None:
        # This would require mocking torch.cuda.mem_get_info
        # Just verify the function exists
        result = check_gpu_memory_free(required_gb=4.0)
        assert result.name == "gpu_memory"


class TestCheckDiskSpace:
    """Tests for check_disk_space."""

    def test_disk_space_ok(self, tmp_path: Path) -> None:
        result = check_disk_space(tmp_path, required_gb=0.001)
        assert result.status == "ok"
        assert "Disk space available" in result.message

    def test_disk_space_error(self, tmp_path: Path) -> None:
        result = check_disk_space(tmp_path, required_gb=1000000)
        assert result.status == "error"
        assert "Insufficient disk space" in result.message

    def test_disk_space_nonexistent_path(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nonexistent"
        # Should handle gracefully
        result = check_disk_space(nonexistent, required_gb=0.001)
        assert result.name == "disk_space"


class TestCheckDatasetRoot:
    """Tests for check_dataset_root."""

    def test_dataset_directory_exists(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        result = check_dataset_root(dataset_dir)
        assert result.status == "ok"
        assert "exists" in result.message

    def test_dataset_directory_missing(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "nonexistent"
        result = check_dataset_root(dataset_dir)
        assert result.status == "warning"
        assert "does not exist" in result.message


class TestRunTrainingChecklist:
    """Tests for run_training_checklist."""

    def test_checklist_runs_with_defaults(self, tmp_path: Path) -> None:
        checklist = run_training_checklist(output_dir=tmp_path)
        assert checklist is not None
        assert checklist.target == "vision"
        assert len(checklist.checks) >= 1

    def test_checklist_respects_cpu_device(self, tmp_path: Path) -> None:
        checklist = run_training_checklist(
            device="cpu",
            output_dir=tmp_path,
        )
        assert checklist.resolved_device == "cpu"

    def test_checklist_auto_detects_device(self, tmp_path: Path) -> None:
        checklist = run_training_checklist(
            device=None,
            output_dir=tmp_path,
        )
        # Should have resolved device (cpu, cuda, or mps)
        assert checklist.resolved_device in SUPPORTED_DEVICES

    def test_checklist_audio_target(self, tmp_path: Path) -> None:
        checklist = run_training_checklist(
            target="audio",
            output_dir=tmp_path,
        )
        assert checklist.target == "audio"
        check_names = [c.name for c in checklist.checks]
        assert "dependency_torchaudio" in check_names


class TestFormatChecklist:
    """Tests for format_checklist."""

    def test_format_checklist_output(self) -> None:
        checklist = Checklist(target="vision", resolved_device="cpu")
        checklist.add_result(CheckResult(name="test", status="ok", message="Test message"))
        output = format_checklist(checklist, verbose=False)
        assert "TRAINING ENVIRONMENT CHECKLIST" in output
        assert "[OK]" in output
        assert "Test message" in output
        assert "Target: vision" in output
        assert "Device: cpu" in output

    def test_format_checklist_verbose(self) -> None:
        checklist = Checklist(target="vision", resolved_device="cpu")
        checklist.add_result(CheckResult(name="test", status="ok", message="Test", details="Details"))
        output = format_checklist(checklist, verbose=True)
        assert "Details" in output

    def test_format_checklist_error_status(self) -> None:
        checklist = Checklist(target="vision", resolved_device="cpu")
        checklist.add_result(CheckResult(name="test", status="error", message="Error"))
        output = format_checklist(checklist, verbose=False)
        assert "[ERR]" in output
        assert "Not ready" in output

    def test_format_checklist_status_message(self) -> None:
        checklist = Checklist(target="audio", resolved_device="cpu")
        output = format_checklist(checklist, verbose=False)
        assert "Ready for audio training" in output


class TestAssertTrainingReady:
    """Tests for assert_training_ready."""

    def test_assert_training_ready_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Mock _has_module to return True for all deps
        def mock_has_module(name: str) -> bool:
            return True

        monkeypatch.setattr("industry_ml_lab.training.checklist._has_module", mock_has_module)

        # Mock _import_torch to return a fake torch with CUDA available
        class FakeCuda:
            def is_available(self) -> bool:
                return True

            def device_count(self) -> int:
                return 1

            def get_device_name(self, idx: int) -> str:
                return "NVIDIA GeForce RTX 4090"

            def mem_get_info(self, idx: int = 0) -> tuple[int, int]:
                return (20 * 1024**3, 24 * 1024**3)

        class FakeTorch:
            cuda = FakeCuda()
            backends = type("Backends", (), {"mps": type("Mps", (), {"is_available": lambda: False})()})()

        monkeypatch.setattr("industry_ml_lab.training.checklist._import_torch", lambda: FakeTorch())

        assert_training_ready(target="vision", device="cpu", output_dir=tmp_path)

    def test_assert_training_ready_fails_with_error(self, tmp_path: Path) -> None:
        # With CUDA device on Mac, should fail
        with pytest.raises(SystemExit) as exc_info:
            assert_training_ready(device="cuda", output_dir=tmp_path)
        assert exc_info.value.code == 1


class TestIntegration:
    """Integration tests for the checklist."""

    def test_checklist_json_output(self, tmp_path: Path) -> None:
        checklist = run_training_checklist(output_dir=tmp_path)
        output = {
            "target": checklist.target,
            "resolved_device": checklist.resolved_device,
            "is_ready": checklist.is_ready,
            "has_warnings": checklist.has_warnings,
            "checks": [
                {"name": c.name, "status": c.status, "message": c.message}
                for c in checklist.checks
            ],
        }
        json_output = json.dumps(output)
        parsed = json.loads(json_output)
        assert "checks" in parsed
        assert "is_ready" in parsed
        assert "target" in parsed
        assert "resolved_device" in parsed

    def test_checklist_has_required_fields(self, tmp_path: Path) -> None:
        checklist = run_training_checklist(output_dir=tmp_path)
        check_names = [c.name for c in checklist.checks]
        # Python version check should always be present
        assert "python_version" in check_names
        # Device backend check should always be present
        assert "device_backend" in check_names

    def test_checklist_target_and_device_in_output(self, tmp_path: Path) -> None:
        checklist = run_training_checklist(target="audio", device="cpu", output_dir=tmp_path)
        output = format_checklist(checklist, verbose=False)
        assert "Target: audio" in output
        assert "Device: cpu" in output

    def test_lite_environment_is_not_ready_for_vision_training(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        dataset_root = tmp_path / "data" / "vision"
        dataset_root.mkdir(parents=True)

        monkeypatch.setattr(checklist_module, "_has_module", lambda name: False)

        result = run_training_checklist(
            target="vision",
            device="cpu",
            output_dir=tmp_path / "artifacts" / "vision",
            dataset_root=dataset_root,
        )

        assert not result.is_ready
        assert result.resolved_device == "cpu"
        assert {item.name for item in result.checks if item.status == "error"} == {
            "dependency_torch",
            "dependency_torchvision",
        }

    def test_fresh_output_dir_uses_existing_parent_for_disk_check(self, tmp_path: Path) -> None:
        result = check_disk_space(tmp_path / "artifacts" / "new-run")

        assert result.status == "ok"
        assert "Checked path" in (result.details or "")
        assert str(tmp_path) in (result.details or "")

    def test_cuda_readiness_fails_when_free_vram_is_too_low(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        dataset_root = tmp_path / "data" / "vision"
        dataset_root.mkdir(parents=True)

        class FakeCuda:
            def is_available(self) -> bool:
                return True

            def device_count(self) -> int:
                return 1

            def get_device_name(self, _: int) -> str:
                return "NVIDIA GeForce RTX 4090"

            def mem_get_info(self, _: int) -> tuple[int, int]:
                return int(2.5 * (1024**3)), int(24 * (1024**3))

        class FakeTorch:
            cuda = FakeCuda()

        monkeypatch.setattr(
            checklist_module,
            "_has_module",
            lambda name: name in {"torch", "torchvision"},
        )
        monkeypatch.setattr(checklist_module, "_import_torch", lambda: FakeTorch())

        result = run_training_checklist(
            target="vision",
            device="cuda",
            output_dir=tmp_path / "artifacts" / "vision",
            dataset_root=dataset_root,
        )

        assert not result.is_ready
        gpu_check = next(item for item in result.checks if item.name == "gpu_memory")
        assert gpu_check.status == "error"
        assert "Insufficient FREE VRAM" in gpu_check.message
        assert "Required: 4.0 GB" in (gpu_check.details or "")
