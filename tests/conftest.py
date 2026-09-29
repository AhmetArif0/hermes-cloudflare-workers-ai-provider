"""Shared helpers: the plugin loaded against a stand-in ``providers`` package, and the recorded data.

The unit tests run without Hermes. ``load_plugin`` imports the plugin with small stand-ins for
``providers`` and ``providers.base`` and restores ``sys.modules`` afterwards, so a real Hermes on
the path (the end-to-end job) is left untouched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / "cloudflare-workers-ai-provider"
HERE = Path(__file__).resolve().parent
# Cloudflare's Text Generation catalog (GET /accounts/<id>/ai/models/search), recorded 2026-09-29.
CATALOG = json.loads((HERE / "cloudflare_text_models.json").read_text(encoding="utf-8"))["result"]
# The HTTP 403 body Cloudflare returned to a Workers Free account for a paid-plan model (request id zeroed).
FREE_PLAN_403 = (HERE / "free_plan_403.json").read_text(encoding="utf-8")


class StubProfile:
    """Stands in for ``providers.base.ProviderProfile``."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def load_plugin():
    """Import the plugin once against the stand-ins; returns (module, registered profiles)."""
    registered: list = []
    providers = types.ModuleType("providers")
    providers.register_provider = registered.append
    base = types.ModuleType("providers.base")
    base.ProviderProfile = StubProfile
    providers.base = base
    saved = {name: sys.modules.get(name) for name in ("providers", "providers.base")}
    sys.modules.update({"providers": providers, "providers.base": base})
    try:
        spec = importlib.util.spec_from_file_location(
            "workers_ai_provider_under_test", PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module, registered


plugin, registered_profiles = load_plugin()
