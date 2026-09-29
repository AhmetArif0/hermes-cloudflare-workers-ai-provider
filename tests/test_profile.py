"""The profile without Hermes: the model table, reasoning values, context windows and the 403 mapping."""

from __future__ import annotations

import json

import pytest
from conftest import CATALOG, FREE_PLAN_403, plugin, registered_profiles

PROFILE = registered_profiles[0] if registered_profiles else None
GPT_OSS = "@cf/openai/gpt-oss-120b"
QWEN = "@cf/qwen/qwen3.8-27b"
DEEPSEEK = "@cf/deepseek-ai/deepseek-v4-flash-0731"
KIMI = "@cf/moonshotai/kimi-k2.6"
GLM_53 = "@cf/zai-org/glm-5.3"
GEMMA = "@cf/google/gemma-4-26b-a4b-it"


def _catalog_rows():
    """(id, context, vision, reasoning, efforts, paid) for the catalog's function-calling models with 64K+."""
    rows = {}
    for item in CATALOG:
        props = {p["property_id"]: p.get("value") for p in item.get("properties") or []}
        if props.get("function_calling") != "true" or int(props.get("context_window") or 0) < 64_000:
            continue
        effort = props.get("reasoning_effort")
        efforts = effort.get("supported_efforts") if isinstance(effort, dict) else None
        rows[item["name"]] = (int(props["context_window"]), props.get("vision") == "true",
                              props.get("reasoning") == "true",
                              tuple(sorted(efforts, key=plugin.EFFORT_LADDER.index)) if efforts else None,
                              props.get("require_workers_paid") == "true")
    return rows


def test_the_table_is_cloudflares_catalog():
    table = {m.id: (m.context_window, m.vision, m.reasoning, m.efforts, m.paid_only) for m in plugin.MODELS}
    assert table == _catalog_rows()


def test_free_plan_models_come_first():
    paid = [m.paid_only for m in plugin.MODELS]
    assert paid == sorted(paid) and paid.count(False) == 9


@pytest.mark.parametrize("model, config, expected", [
    (GPT_OSS, None, None),                                    # unset: Cloudflare's default
    (GPT_OSS, {"enabled": False}, "low"),                     # gpt-oss always reasons: its lowest level
    (GPT_OSS, {"enabled": True, "effort": "none"}, "low"),
    (GPT_OSS, {"enabled": True, "effort": "minimal"}, "low"),
    (GPT_OSS, {"enabled": True, "effort": "medium"}, "medium"),
    (GPT_OSS, {"enabled": True, "effort": "xhigh"}, "high"),
    (GPT_OSS, {"enabled": True, "effort": "ultra"}, "high"),
    (QWEN, {"enabled": True, "effort": "high"}, "medium"),    # qwen3.8 takes low/medium/xhigh
    (QWEN, {"enabled": True, "effort": "xhigh"}, "xhigh"),
    (QWEN, {"enabled": True, "effort": "max"}, "xhigh"),
    (QWEN, {"enabled": False}, "low"),                        # no "none": its lowest level, not the xhigh default
    (DEEPSEEK, {"enabled": False}, "none"),
    (DEEPSEEK, {"enabled": True, "effort": "medium"}, "low"),
    (DEEPSEEK, {"enabled": True, "effort": "ultra"}, "max"),
    (KIMI, {"enabled": True, "effort": "none"}, "none"),
    (KIMI, {"enabled": True, "effort": "medium"}, "high"),    # nothing weaker but "none": its lowest level
    (GLM_53, {"enabled": False}, "low"),                      # Cloudflare would turn "none" into "max"
    (GPT_OSS, {"enabled": True}, None),                       # no level: Cloudflare's default
    (GPT_OSS, {"enabled": True, "effort": "turbo"}, None),    # not a Hermes level: never guessed
])
def test_reasoning_goes_out_in_the_form_the_model_takes(model, config, expected):
    assert plugin.reasoning_effort_for(model, config) == expected


@pytest.mark.parametrize("model", [GEMMA, "@cf/zai-org/glm-4.7-flash", "@cf/meta/llama-4-scout-17b-16e-instruct",
                                   "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "", None])
def test_models_without_listed_levels_get_no_field(model):
    for config in ({"enabled": False}, {"enabled": True, "effort": "high"}):
        assert plugin.reasoning_effort_for(model, config) is None


def test_every_level_sent_is_one_the_model_lists():
    for model in plugin.MODELS:
        for effort in plugin.EFFORT_LADDER:
            sent = plugin.reasoning_effort_for(model.id, {"enabled": True, "effort": effort})
            assert sent is None or sent in model.efforts, (model.id, effort, sent)


def test_profile_puts_reasoning_effort_top_level_only():
    high = {"enabled": True, "effort": "high"}
    assert PROFILE.build_api_kwargs_extras(reasoning_config=high, model=QWEN) == ({}, {"reasoning_effort": "medium"})
    assert PROFILE.build_api_kwargs_extras(reasoning_config=high, model=GEMMA) == ({}, {})
    assert PROFILE.build_api_kwargs_extras(reasoning_config=None, model=GPT_OSS) == ({}, {})
    assert PROFILE.build_api_kwargs_extras(reasoning_config=high) == ({}, {})


def test_context_windows_come_from_the_table():
    assert PROFILE.get_model_context_length(GPT_OSS) == 128_000
    assert PROFILE.get_model_context_length(QWEN) == 262_144
    assert PROFILE.get_model_context_length("@cf/meta/llama-3.1-8b-instruct-fp8") is None
    for model in plugin.MODELS:
        assert PROFILE.model_capabilities[model.id] == {
            "supports_tools": True, "supports_vision": model.vision,
            "supports_reasoning": model.reasoning, "context_window": model.context_window}


def test_a_free_plan_refusal_is_a_model_the_account_cannot_use():
    message = json.loads(FREE_PLAN_403)["errors"][0]["message"].lower()
    verdict = plugin.classify_api_error(None, status_code=403, error_code=None, message=message, body={}, model=KIMI)
    assert verdict == {"reason": "model_entitlement", "retryable": False, "should_fallback": True}


@pytest.mark.parametrize("status, message", [
    (403, "authentication error"),     # a bad token stays an auth error
    (401, "is not available on the workers free plan"),
    (400, "unexpected reasoning effort high"),
])
def test_other_errors_keep_hermes_classification(status, message):
    assert plugin.classify_api_error(None, status_code=status, error_code=None, message=message,
                                     body={}, model=KIMI) is None


def test_profile_registration():
    assert len(registered_profiles) == 1
    assert PROFILE.name == "workers-ai"
    assert PROFILE.display_name == "Cloudflare Workers AI"
    assert PROFILE.env_vars == ("CLOUDFLARE_WORKERS_AI_API_TOKEN", "CLOUDFLARE_WORKERS_AI_BASE_URL")
    assert PROFILE.base_url == ""
    assert PROFILE.auth_type == "api_key"
    assert PROFILE.supports_model_listing is False and PROFILE.supports_health_check is False
    assert PROFILE.fallback_models == tuple(m.id for m in plugin.MODELS)
    assert PROFILE.default_aux_model in PROFILE.fallback_models
    assert not plugin.MODELS_BY_ID[PROFILE.default_aux_model].paid_only
    assert PROFILE.classify_api_error is plugin.classify_api_error
