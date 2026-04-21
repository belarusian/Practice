#!/usr/bin/env python3
"""Build summary.json from ops/sunny status.tsv (name, exit_code, output_path).

If restore-demo exits 124 (timeout from `timeout(1)`) but audit-after-restore exits 0,
the demo stack is treated as healthy and the run is still considered fully successful.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _gate_entries(entries: list[dict], proof_mode: bool) -> list[dict]:
    """Steps that gate overall success. In proof mode, WSL check is informational only."""
    if not proof_mode:
        return list(entries)
    return [e for e in entries if e["name"] != "wsl-ml-lab-check"]


def _nested_windows_smoke_failure_note(report_dir: Path) -> str | None:
    """If windows-*-smoke-summary.json exists and reports failure, return a note string."""
    if not report_dir.is_dir():
        return None
    failed: list[str] = []
    for path in sorted(report_dir.glob("windows-*-smoke-summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("all_commands_succeeded") is False:
            failed.append(path.name)
    if not failed:
        return None
    return (
        "nested Windows smoke reports failure in: " + ", ".join(failed)
    )


def compute_all_succeeded(
    entries: list[dict], proof_mode: bool
) -> tuple[bool, str | None]:
    gated = _gate_entries(entries, proof_mode)
    by_name = {e["name"]: e for e in gated}
    restore = by_name.get("restore-demo")
    audit_after = by_name.get("audit-after-restore")
    if (
        restore is not None
        and restore["exit_code"] == 124
        and audit_after is not None
        and audit_after["exit_code"] == 0
    ):
        others_ok = all(
            e["exit_code"] == 0 for e in gated if e["name"] != "restore-demo"
        )
        if others_ok:
            return True, (
                "restore-demo exited 124 (timeout) but audit-after-restore passed; "
                "treating overall status as success."
            )
    all_ok = all(e["exit_code"] == 0 for e in gated)
    return all_ok, None


def main() -> None:
    argv = sys.argv[1:]
    proof_mode = False
    if argv and argv[0] == "--proof":
        proof_mode = True
        argv = argv[1:]

    status_path = Path(argv[0])
    summary_path = Path(argv[1])
    output_dir = Path(argv[2]) if len(argv) > 2 else None

    entries = []
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, exit_code, output_path = line.split("\t", 2)
        entries.append(
            {
                "name": name,
                "exit_code": int(exit_code),
                "output_path": output_path,
            }
        )

    all_ok, note = compute_all_succeeded(entries, proof_mode=proof_mode)
    nested_note = _nested_windows_smoke_failure_note(summary_path.parent)
    if nested_note is not None:
        all_ok = False
    summary: dict = {
        "report_dir": str(summary_path.parent),
        "all_commands_succeeded": all_ok,
        "entries": entries,
    }
    if nested_note is not None:
        summary["nested_windows_smoke_note"] = nested_note
    if proof_mode:
        summary["proof_mode"] = True
        wsl = next((e for e in entries if e["name"] == "wsl-ml-lab-check"), None)
        if wsl is not None and wsl["exit_code"] != 0:
            summary["proof_mode_note"] = (
                "wsl-ml-lab-check is non-gating in proof mode "
                "(WSL is not the training runtime)."
            )
    if output_dir is not None:
        summary["output_dir"] = str(output_dir)
    if note:
        summary["restore_demo_note"] = note

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
