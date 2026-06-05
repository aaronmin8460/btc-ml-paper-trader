from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR_NAMES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "__pycache__",
}
BACKUP_DIR_NAMES = {
    "backup",
    "backups",
}
SKIP_WALK_DIR_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "venv",
}


def build_audit_report(root: Path | str = ROOT) -> dict[str, Any]:
    root_path = Path(root).resolve()
    registry_path = root_path / "models" / "registry.json"
    registry = _read_json(registry_path)
    referenced_models = _registry_model_references(registry, root_path=root_path)
    model_files = [_model_file_payload(path, root_path=root_path, referenced_models=referenced_models) for path in sorted((root_path / "models").glob("*.joblib"))]

    return {
        "repository_root": str(root_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "untracked_files": _git_untracked_files(root_path),
        "cache_directories": [_path_payload(path, root_path) for path in _find_cache_dirs(root_path)],
        "log_files": [_file_payload(path, root_path) for path in _find_log_files(root_path)],
        "old_train_logs": [_file_payload(path, root_path) for path in _find_train_logs(root_path)],
        "backup_directories": [_path_payload(path, root_path) for path in _find_backup_dirs(root_path)],
        "models": model_files,
        "model_summary": {
            "count": len(model_files),
            "total_size_bytes": sum(int(item["size_bytes"]) for item in model_files),
        },
        "registry": {
            "exists": registry_path.exists(),
            "path": _relative(registry_path, root_path),
            "referenced_model_paths": sorted(_relative(path, root_path) for path in referenced_models),
            "referenced_model_exists": any(path.exists() for path in referenced_models),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit generated files without deleting anything.")
    parser.add_argument("--root", default=str(ROOT), help="Repository root to audit. Defaults to this checkout.")
    args = parser.parse_args(argv)

    report = build_audit_report(Path(args.root))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _git_untracked_files(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(line for line in result.stdout.splitlines() if line)


def _find_cache_dirs(root: Path) -> list[Path]:
    return [
        path
        for path in _walk_dirs(root)
        if path.name in CACHE_DIR_NAMES
    ]


def _find_log_files(root: Path) -> list[Path]:
    return [
        path
        for path in _walk_files(root)
        if path.suffix == ".log"
    ]


def _find_train_logs(root: Path) -> list[Path]:
    return [
        path
        for path in _walk_files(root)
        if path.name == "train_now.log" or (path.name.startswith("train_") and path.suffix == ".log")
    ]


def _find_backup_dirs(root: Path) -> list[Path]:
    return [
        path
        for path in _walk_dirs(root)
        if path.name.lower() in BACKUP_DIR_NAMES or path.name.lower().endswith("_backup")
    ]


def _walk_dirs(root: Path) -> list[Path]:
    paths: list[Path] = []
    for current_root, dirs, _ in os.walk(root):
        dirs[:] = [name for name in dirs if name not in SKIP_WALK_DIR_NAMES]
        paths.extend(Path(current_root) / name for name in dirs)
    return sorted(paths)


def _walk_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in SKIP_WALK_DIR_NAMES]
        paths.extend(Path(current_root) / name for name in files)
    return sorted(paths)


def _model_file_payload(path: Path, *, root_path: Path, referenced_models: set[Path]) -> dict[str, Any]:
    payload = _file_payload(path, root_path)
    payload["age_days"] = _age_days(path)
    payload["referenced_by_registry"] = path.resolve() in referenced_models
    return payload


def _file_payload(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": _relative(path, root),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }


def _path_payload(path: Path, root: Path) -> dict[str, Any]:
    return {"path": _relative(path, root)}


def _age_days(path: Path) -> float:
    age = datetime.now(UTC) - datetime.fromtimestamp(path.stat().st_mtime, UTC)
    return round(age.total_seconds() / 86_400, 3)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _registry_model_references(registry: dict[str, Any], *, root_path: Path) -> set[Path]:
    references: set[Path] = set()
    for value in _walk_json_values(registry):
        if not isinstance(value, str) or not value.endswith(".joblib"):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = root_path / path
        references.add(path.resolve())
    return references


def _walk_json_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        values: list[Any] = []
        for item in value.values():
            values.extend(_walk_json_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_walk_json_values(item))
        return values
    return [value]


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
