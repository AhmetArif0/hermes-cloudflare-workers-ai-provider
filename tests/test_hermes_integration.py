"""End-to-end checks against a real Hermes checkout (skipped when Hermes is not importable).

Each check runs in a fresh interpreter whose HERMES_HOME holds an installed copy of the plugin
(``$HERMES_HOME/plugins/cloudflare-workers-ai-provider``) and a ``.env`` with the two variables the
README asks for, so Hermes discovers and configures it the way it does after
``hermes plugins install cloudflare-workers-ai-provider``. Nothing talks to Cloudflare: chat turns go to
Hermes' own scripted loopback LLM (``tests/fakes/fake_llm_provider.py``), which records every request
body and replays Cloudflare's recorded free-plan 403.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

hermes_cli = pytest.importorskip("hermes_cli")

from conftest import FREE_PLAN_403, PLUGIN_DIR, plugin  # noqa: E402

HERMES_ROOT = Path(hermes_cli.__file__).resolve().parents[1]
TOKEN = "cf-e2e-not-a-real-token"
ACCOUNT_URL = "https://api.cloudflare.com/client/v4/accounts/" + "0" * 32 + "/ai/v1"


def _home(tmp_path: Path, base_url: str = ACCOUNT_URL) -> Path:
    home = tmp_path / "hermes-home"
    shutil.copytree(PLUGIN_DIR, home / "plugins" / "cloudflare-workers-ai-provider",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (home / ".env").write_text(f"CLOUDFLARE_WORKERS_AI_API_TOKEN={TOKEN}\nCLOUDFLARE_WORKERS_AI_BASE_URL={base_url}\n",
                               encoding="utf-8", newline="\n")
    return home


def _env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OPENAI", "ANTHROPIC", "CLOUDFLARE", "OPENROUTER", "HERMES_"))}
    env.update(HERMES_HOME=str(home), PYTHONPATH=str(HERMES_ROOT), NO_COLOR="1")
    return env


def _python(home: Path, code: str):
    """Run *code* in a fresh interpreter and return the JSON it prints last."""
    proc = subprocess.run([sys.executable, "-c", code], env=_env(home), cwd=str(home.parent),
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_hermes_discovers_and_configures_the_installed_plugin(tmp_path):
    found = _python(_home(tmp_path), """
import json
from providers import get_provider_profile
from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.models import list_available_providers, provider_model_ids
from hermes_cli.model_switch import switch_model
profile = get_provider_profile("workers-ai")
row = PROVIDER_REGISTRY.get("workers-ai")
switch = switch_model("@cf/openai/gpt-oss-120b", current_provider="openrouter", current_model="x",
                      explicit_provider="workers-ai")
print(json.dumps({"cls": type(profile).__name__, "row_env": list(row.api_key_env_vars),
                  "row_url_env": row.base_url_env_var,
                  "picker": [p for p in list_available_providers() if p["id"] == "workers-ai"],
                  "models": provider_model_ids("workers-ai", force_refresh=True),
                  "switch": [switch.success, switch.target_provider, switch.base_url]}))
""")
    assert found["cls"] == "WorkersAIProfile"
    assert found["row_env"] == ["CLOUDFLARE_WORKERS_AI_API_TOKEN"]
    assert found["row_url_env"] == "CLOUDFLARE_WORKERS_AI_BASE_URL"
    assert found["picker"] == [{"id": "workers-ai", "label": "Cloudflare Workers AI", "aliases": [],
                                "authenticated": True}]
    assert found["models"] == [m.id for m in plugin.MODELS]
    assert found["switch"] == [True, "workers-ai", ACCOUNT_URL]


def test_hermes_uses_the_catalog_context_windows(tmp_path):
    """Hermes treats the account URL as a custom endpoint, whose /models probe would answer 405."""
    found = _python(_home(tmp_path), f"""
import json
from agent.model_metadata import get_model_context_length
from agent.models_dev import get_model_capabilities
models = ["@cf/openai/gpt-oss-120b", "@cf/qwen/qwen3.8-27b", "@cf/zai-org/glm-5.3"]
print(json.dumps({{m: [get_model_context_length(m, base_url="{ACCOUNT_URL}", api_key="x", provider="workers-ai"),
                      get_model_capabilities("workers-ai", m).supports_vision] for m in models}}))
""")
    assert found == {"@cf/openai/gpt-oss-120b": [128_000, False], "@cf/qwen/qwen3.8-27b": [262_144, True],
                     "@cf/zai-org/glm-5.3": [1_310_720, False]}


def _fake_llm_module():
    path = HERMES_ROOT / "tests" / "fakes" / "fake_llm_provider.py"
    if not path.is_file():
        pytest.skip("this Hermes checkout has no tests/fakes/fake_llm_provider.py")
    spec = importlib.util.spec_from_file_location("hermes_fake_llm_provider", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _chat(home: Path, model: str, effort: str | None) -> subprocess.CompletedProcess:
    (home / "config.yaml").write_text(
        f'model:\n  provider: workers-ai\n  default: "{model}"\n'
        "agent:\n  api_max_retries: 1\n" + (f"  reasoning_effort: {effort}\n" if effort else ""),
        encoding="utf-8", newline="\n")
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "chat", "-q", "Reply with: pong", "--oneshot", "-Q", "-t", "todo"],
        env=_env(home), cwd=str(home.parent), capture_output=True, text=True, timeout=240, check=False)


@pytest.mark.parametrize("model, effort, sent", [
    ("@cf/openai/gpt-oss-120b", None, None),
    ("@cf/openai/gpt-oss-120b", "none", "low"),
    ("@cf/qwen/qwen3.8-27b", "high", "medium"),
    ("@cf/deepseek-ai/deepseek-v4-flash-0731", "none", "none"),
    ("@cf/google/gemma-4-26b-a4b-it", "high", None),
])
def test_a_real_turn_sends_the_reasoning_value_the_model_accepts(tmp_path, model, effort, sent):
    fake = _fake_llm_module()
    with fake.FakeLLMServer([fake.Text("pong")], api_key=TOKEN) as llm:
        proc = _chat(_home(tmp_path, llm.base_url), model, effort)
        assert proc.returncode == 0, (proc.stdout[-1000:], proc.stderr[-2000:])
        assert "pong" in proc.stdout
        main = llm.main_requests()
    assert main, "the turn made no chat request"
    body = main[0]
    assert body["model"] == model
    assert body.get("reasoning_effort") == sent
    assert "reasoning" not in body  # the generic nested fallback never goes to Cloudflare


def test_a_free_plan_refusal_does_not_blame_the_token(tmp_path):
    fake = _fake_llm_module()
    with fake.FakeLLMServer([fake.Raw(body=FREE_PLAN_403, status=403)], api_key=TOKEN) as llm:
        proc = _chat(_home(tmp_path, llm.base_url), "@cf/moonshotai/kimi-k2.6", None)
    out = proc.stdout + proc.stderr
    assert "retrying won't help" in out, out[-2000:]
    assert "rejected your API key" not in out
    assert "not available on the Workers Free plan" in out
