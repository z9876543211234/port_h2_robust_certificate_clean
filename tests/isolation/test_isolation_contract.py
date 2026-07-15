from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"


def test_project_is_an_independent_git_repository() -> None:
    assert (PROJECT_ROOT / ".git").is_dir()


def test_project_contains_no_symbolic_links() -> None:
    links = [path for path in PROJECT_ROOT.rglob("*") if path.is_symlink()]
    assert links == []


def test_production_imports_do_not_reference_historical_packages() -> None:
    forbidden_roots = {
        "port_h2_robust",
        "port_h2_robust_clean",
        "shift_arrival_experiment",
    }
    violations: list[str] = []
    source_files = sorted(SOURCE_ROOT.rglob("*.py"))
    assert source_files, "production package scaffold has not been created"

    for path in source_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split(".", 1)[0] in forbidden_roots:
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}:{name}")

    assert violations == []


def test_production_code_does_not_mutate_sys_path_or_use_dynamic_loading() -> None:
    forbidden_fragments = (
        "sys.path.insert",
        "sys.path.append",
        "importlib.util.spec_from_file_location",
        "subprocess.run",
        "subprocess.Popen",
    )
    violations = {
        str(path.relative_to(PROJECT_ROOT)): fragment
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        for fragment in forbidden_fragments
        if fragment in path.read_text(encoding="utf-8")
    }
    assert violations == {}


def test_certificate_package_does_not_import_uncertainty_builders() -> None:
    certificate_root = SOURCE_ROOT / "port_h2_certificate"
    source_files = sorted(certificate_root.rglob("*.py"))
    assert source_files, "certificate package scaffold has not been created"
    violations = {
        str(path.relative_to(PROJECT_ROOT))
        for path in source_files
        if "port_h2_uncertainty_builders" in path.read_text(encoding="utf-8")
    }
    assert violations == set()

