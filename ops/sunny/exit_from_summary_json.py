#!/usr/bin/env python3
"""Exit 0 if summary.json has all_commands_succeeded true; else 1. Missing/invalid JSON: 2."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: exit_from_summary_json.py SUMMARY_JSON", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"exit_from_summary_json: missing {path}", file=sys.stderr)
        return 2
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"exit_from_summary_json: {exc}", file=sys.stderr)
        return 2
    if data.get("all_commands_succeeded") is True:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
