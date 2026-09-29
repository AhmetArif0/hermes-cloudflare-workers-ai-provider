"""Packaging checks: what the catalog installs, what its docs page renders, and what the code may touch."""

from __future__ import annotations

import ast
from pathlib import Path

from conftest import PLUGIN_DIR, ROOT, plugin

SOURCES = sorted(PLUGIN_DIR.glob("*.py"))


def _manifest() -> dict:
    fields = {}
    lines = (PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line and not line.startswith(" ") and ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    fields["requires_env"] = [line.strip()[2:] for line in lines if line.startswith("  - ")]
    return fields


def _imports(path: Path) -> list:
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_the_plugin_is_one_file():
    assert [path.name for path in SOURCES] == ["__init__.py"]


def test_manifest_declares_a_model_provider():
    manifest = _manifest()
    assert manifest["name"] == "cloudflare-workers-ai-provider"
    assert manifest["kind"] == "model-provider"
    assert manifest["requires_env"] == [plugin.API_TOKEN_ENV, plugin.BASE_URL_ENV]


def test_plugin_readme_matches_repo_readme():
    """The catalog docs page renders <subdir>/README.md; GitHub renders the root one. Keep them identical."""
    assert (PLUGIN_DIR / "README.md").read_bytes() == (ROOT / "README.md").read_bytes()


def test_readme_links_are_absolute():
    """The catalog page renders the README away from the repo, so relative links would 404."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for part in readme.split("](")[1:]:
        assert part.startswith("https://"), part[:60]


def test_manifest_version_is_the_one_the_readme_documents():
    assert f"### {_manifest()['version']}" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_only_the_documented_provider_api_is_imported():
    for path in SOURCES:
        assert sorted(set(_imports(path))) == ["__future__", "providers", "providers.base", "typing"], path.name


def test_no_network_processes_files_or_environment_reads():
    banned = {"open", "exec", "eval", "compile", "__import__", "system", "popen", "getenv", "urlopen",
              "Thread", "write_text", "write_bytes", "read_text", "read_bytes", "mkdir", "remove", "unlink"}
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
                assert called not in banned, f"{path.name}:{node.lineno} calls {called}()"
            if isinstance(node, ast.Attribute):
                assert node.attr != "environ", f"{path.name}:{node.lineno} reads os.environ"


def test_registers_exactly_one_provider():
    tree = ast.parse((PLUGIN_DIR / "__init__.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("register")]
    assert [call.func.id for call in calls] == ["register_provider"]
