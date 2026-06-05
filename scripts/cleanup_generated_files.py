from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

SAFE_CACHE_DIR_NAMES = {".pytest_cache", "__pycache__"}
SAFE_LOG_FILE_NAMES = {"app.log", "train_now.log"}
PROTECTED_NAMES = {".env", "data", "models", "backups"}
SKIP_DIR_NAMES = {".git", ".venv", "data", "models", "backups", "node_modules", "venv"}


@dataclass(frozen=True)
class CleanupCandidate:
    path: Path
    kind: str
    reason: str


def build_cleanup_plan(root: Path | str = ROOT) -> list[CleanupCandidate]:
    root_path = Path(root).resolve()
    candidates: dict[Path, CleanupCandidate] = {}
    safe_cache_dirs: set[Path] = set()

    for path in _walk_dirs(root_path):
        if path.name in SAFE_CACHE_DIR_NAMES and _is_safe_candidate(path, root_path):
            resolved = path.resolve()
            safe_cache_dirs.add(resolved)
            candidates[resolved] = CleanupCandidate(resolved, "directory", path.name)

    for path in _walk_files(root_path):
        if not _is_safe_candidate(path, root_path):
            continue
        if _inside_any(path, safe_cache_dirs):
            continue
        reason = _safe_file_reason(path)
        if reason is not None:
            candidates[path.resolve()] = CleanupCandidate(path.resolve(), "file", reason)

    return sorted(candidates.values(), key=lambda candidate: str(candidate.path))


def cleanup_generated_files(root: Path | str = ROOT, *, apply: bool = False) -> dict[str, Any]:
    root_path = Path(root).resolve()
    candidates = build_cleanup_plan(root_path)
    report: dict[str, Any] = {
        "repository_root": str(root_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "apply" if apply else "dry_run",
        "deleted": [],
        "planned": [],
        "skipped": [],
    }

    print(f"cleanup mode: {'apply' if apply else 'dry-run'}")
    for candidate in candidates:
        relative = _relative(candidate.path, root_path)
        item = {
            "path": relative,
            "kind": candidate.kind,
            "reason": candidate.reason,
        }
        report["planned"].append(item)
        if candidate.kind == "directory":
            for contained_file in _contained_files(candidate.path):
                contained_relative = _relative(contained_file, root_path)
                prefix = "DRY-RUN would delete file in directory" if not apply else "Deleting file in directory"
                print(f"{prefix}: {contained_relative}")
        if not apply:
            print(f"DRY-RUN would delete {candidate.kind}: {relative}")
            continue

        print(f"Deleting {candidate.kind}: {relative}")
        try:
            _delete_candidate(candidate)
        except OSError as exc:
            report["skipped"].append({**item, "error": type(exc).__name__})
        else:
            report["deleted"].append(item)

    report_path = root_path / "logs" / "cleanup_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"cleanup report: {_relative(report_path, root_path)}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delete only safe generated cache/log files.")
    parser.add_argument("--root", default=str(ROOT), help="Repository root to clean. Defaults to this checkout.")
    parser.add_argument("--apply", action="store_true", help="Actually delete files. Omit for dry-run.")
    args = parser.parse_args(argv)

    cleanup_generated_files(Path(args.root), apply=bool(args.apply))
    return 0


def _walk_dirs(root: Path) -> list[Path]:
    paths: list[Path] = []
    for current_root, dirs, _ in os.walk(root):
        dirs[:] = [name for name in dirs if name not in SKIP_DIR_NAMES]
        paths.extend(Path(current_root) / name for name in dirs)
    return sorted(paths)


def _walk_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in SKIP_DIR_NAMES]
        paths.extend(Path(current_root) / name for name in files)
    return sorted(paths)


def _is_safe_candidate(path: Path, root: Path) -> bool:
    try:
        relative_parts = path.resolve().relative_to(root).parts
    except ValueError:
        return False
    if not relative_parts:
        return False
    if relative_parts[0] in PROTECTED_NAMES:
        return False
    if path.name in PROTECTED_NAMES:
        return False
    return True


def _safe_file_reason(path: Path) -> str | None:
    if path.suffix == ".pyc":
        return "pyc"
    if path.name in SAFE_LOG_FILE_NAMES:
        return path.name
    if path.name.startswith("train_") and path.suffix == ".log":
        return "train_*.log"
    return None


def _delete_candidate(candidate: CleanupCandidate) -> None:
    if candidate.kind == "directory":
        shutil.rmtree(candidate.path)
    else:
        candidate.path.unlink()


def _inside_any(path: Path, directories: set[Path]) -> bool:
    resolved = path.resolve()
    return any(_is_relative_to(resolved, directory) for directory in directories)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _contained_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(item for item in path.rglob("*") if item.is_file())


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
