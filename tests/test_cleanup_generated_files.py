import json
from pathlib import Path

from scripts.audit_generated_files import build_audit_report
from scripts.cleanup_generated_files import cleanup_generated_files


def test_cleanup_dry_run_deletes_nothing(tmp_path):
    safe_file = tmp_path / "train_now.log"
    safe_file.write_text("training output\n", encoding="utf-8")
    cache_file = tmp_path / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("[]\n", encoding="utf-8")

    report = cleanup_generated_files(tmp_path, apply=False)

    assert report["mode"] == "dry_run"
    assert safe_file.exists()
    assert cache_file.exists()
    report_path = tmp_path / "logs" / "cleanup_report.json"
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["mode"] == "dry_run"


def test_cleanup_apply_deletes_only_safe_generated_files(tmp_path):
    safe_files = [
        tmp_path / "train_now.log",
        tmp_path / "train_20260605.log",
        tmp_path / "app.log",
        tmp_path / "module.pyc",
    ]
    for path in safe_files:
        path.write_text("generated\n", encoding="utf-8")
    pycache_file = tmp_path / "pkg" / "__pycache__" / "module.cpython-312.pyc"
    pycache_file.parent.mkdir(parents=True)
    pycache_file.write_text("bytecode\n", encoding="utf-8")
    pytest_cache_file = tmp_path / ".pytest_cache" / "README.md"
    pytest_cache_file.parent.mkdir(parents=True)
    pytest_cache_file.write_text("cache\n", encoding="utf-8")
    protected_files = _write_protected_files(tmp_path)
    keep_file = tmp_path / "notes.txt"
    keep_file.write_text("manual note\n", encoding="utf-8")

    report = cleanup_generated_files(tmp_path, apply=True)

    assert report["mode"] == "apply"
    for path in safe_files:
        assert not path.exists()
    assert not pycache_file.parent.exists()
    assert not pytest_cache_file.parent.exists()
    assert keep_file.exists()
    for path in protected_files:
        assert path.exists()


def test_cleanup_never_deletes_env_database_models_or_registry(tmp_path):
    protected_files = _write_protected_files(tmp_path)
    for path in [
        tmp_path / "data" / "app.log",
        tmp_path / "models" / "train_old.log",
        tmp_path / "backups" / "app.log",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("protected generated-looking file\n", encoding="utf-8")
        protected_files.append(path)

    cleanup_generated_files(tmp_path, apply=True)

    for path in protected_files:
        assert path.exists()


def test_audit_reports_models_and_registry_reference(tmp_path):
    model_path = tmp_path / "models" / "active.joblib"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("model\n", encoding="utf-8")
    (tmp_path / "models" / "registry.json").write_text(
        json.dumps({"active_model_path": "models/active.joblib"}),
        encoding="utf-8",
    )

    report = build_audit_report(tmp_path)

    assert report["registry"]["exists"] is True
    assert report["registry"]["referenced_model_exists"] is True
    assert report["registry"]["referenced_model_paths"] == ["models/active.joblib"]
    assert report["models"][0]["referenced_by_registry"] is True


def _write_protected_files(root: Path) -> list[Path]:
    protected_files = [
        root / ".env",
        root / "data" / "trading.db",
        root / "models" / "model.joblib",
        root / "models" / "registry.json",
    ]
    for path in protected_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("protected\n", encoding="utf-8")
    return protected_files
