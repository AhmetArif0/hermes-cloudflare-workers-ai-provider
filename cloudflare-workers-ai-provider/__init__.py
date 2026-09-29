"""Cloudflare Workers AI as a Hermes model provider: pick it in ``hermes model`` or pass ``--provider workers-ai``.

Without it Hermes cannot select Workers AI as a provider at all, and a custom endpoint pointed at it
lists no models (Cloudflare's OpenAI-compatible ``/models`` answers HTTP 405). This plugin adds:

* the Workers AI models Hermes can run (function calling and at least 64K of context), with the
  context window, vision and reasoning flags from Cloudflare's catalog, so Hermes compresses in time
  (a custom endpoint assumes 256K for the 128K gpt-oss models);
* reasoning effort in the form each model accepts (a custom endpoint's turn on qwen3.8 with
  reasoning ``high`` fails with HTTP 400);
* a clear message when a free-plan account picks a paid-plan model, instead of "rejected your API key".

The endpoint holds the account ID, so the base URL comes from ``CLOUDFLARE_WORKERS_AI_BASE_URL``,
which Hermes reads per profile, and the token from ``CLOUDFLARE_WORKERS_AI_API_TOKEN``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from providers import register_provider
from providers.base import ProviderProfile

API_TOKEN_ENV = "CLOUDFLARE_WORKERS_AI_API_TOKEN"
BASE_URL_ENV = "CLOUDFLARE_WORKERS_AI_BASE_URL"

# Hermes' reasoning-effort levels, weakest first.
EFFORT_LADDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")


class Model(NamedTuple):
    id: str
    context_window: int
    vision: bool
    reasoning: bool
    efforts: tuple[str, ...] | None  # the reasoning_effort values the model takes; None = send none
    paid_only: bool


# Cloudflare's Text Generation catalog (GET /accounts/<id>/ai/models/search) on 2026-09-29, limited to
# the models with function calling and at least 64K of context. Free-plan models first.
MODELS = (
    Model("@cf/openai/gpt-oss-120b", 128_000, vision=False, reasoning=True, efforts=("low", "medium", "high"), paid_only=False),
    Model("@cf/openai/gpt-oss-20b", 128_000, vision=False, reasoning=True, efforts=("low", "medium", "high"), paid_only=False),
    Model("@cf/qwen/qwen3.8-27b", 262_144, vision=True, reasoning=True, efforts=("low", "medium", "xhigh"), paid_only=False),
    Model("@cf/zai-org/glm-4.7-flash", 131_072, vision=False, reasoning=True, efforts=None, paid_only=False),
    Model("@cf/google/gemma-4-26b-a4b-it", 256_000, vision=True, reasoning=True, efforts=None, paid_only=False),
    Model("@cf/nvidia/nemotron-3-120b-a12b", 256_000, vision=False, reasoning=True, efforts=None, paid_only=False),
    Model("@cf/meta/llama-4-scout-17b-16e-instruct", 131_000, vision=True, reasoning=False, efforts=None, paid_only=False),
    Model("@cf/mistralai/mistral-small-3.1-24b-instruct", 128_000, vision=False, reasoning=False, efforts=None, paid_only=False),
    Model("@cf/ibm-granite/granite-4.0-h-micro", 131_000, vision=False, reasoning=False, efforts=None, paid_only=False),
    Model("@cf/deepseek-ai/deepseek-v4-flash-0731", 1_310_720, vision=False, reasoning=True, efforts=("none", "low", "high", "max"), paid_only=True),
    Model("@cf/deepseek-ai/deepseek-v4-pro-0813", 1_048_576, vision=False, reasoning=True, efforts=("none", "low", "high", "max"), paid_only=True),
    Model("@cf/moonshotai/kimi-k2.6", 262_144, vision=True, reasoning=True, efforts=("none", "high"), paid_only=True),
    Model("@cf/moonshotai/kimi-k2.7-code", 262_144, vision=True, reasoning=True, efforts=None, paid_only=True),
    Model("@cf/zai-org/glm-5.2", 262_144, vision=False, reasoning=True, efforts=("none", "high", "max"), paid_only=True),
    Model("@cf/zai-org/glm-5.3", 1_310_720, vision=False, reasoning=True, efforts=("low", "high", "max"), paid_only=True),
    Model("@cf/zai-org/glm-5.3-flash", 1_310_720, vision=True, reasoning=True, efforts=("low", "high", "max"), paid_only=True),
)
MODELS_BY_ID = {model.id: model for model in MODELS}

# Cloudflare's HTTP 403 when a Workers Free account calls a Workers Paid model.
FREE_PLAN_REFUSAL = "is not available on the workers free plan"


def reasoning_effort_for(model: Any, reasoning_config: Any) -> str | None:
    """The ``reasoning_effort`` value to send for *model*, or None to leave the field out.

    A level the model does not take becomes the nearest weaker one it does; a level below the
    model's lowest becomes that lowest one (Hermes' own ``clamp_effort`` rule). Turning reasoning
    off maps to ``none`` where the model has it and to its lowest level otherwise. Models whose
    catalog entry lists no levels get no field.
    """
    if not isinstance(reasoning_config, dict):
        return None
    entry = MODELS_BY_ID.get(str(model or ""))
    if entry is None or not entry.efforts:
        return None
    supported = entry.efforts
    floor = next(level for level in supported if level != "none")
    requested = str(reasoning_config.get("effort") or "").strip().lower()
    if reasoning_config.get("enabled") is False or requested == "none":
        return "none" if "none" in supported else floor
    if requested not in EFFORT_LADDER:
        return None
    if requested in supported:
        return requested
    rank = EFFORT_LADDER.index(requested)
    weaker = [level for level in supported if level != "none" and EFFORT_LADDER.index(level) < rank]
    return weaker[-1] if weaker else floor


def classify_api_error(error: Any, *, status_code: Any = None, message: Any = "", **_: Any) -> dict | None:
    """Report a Workers Free account calling a paid-plan model as a model the account cannot use."""
    if status_code == 403 and FREE_PLAN_REFUSAL in str(message or "").lower():
        return {"reason": "model_entitlement", "retryable": False, "should_fallback": True}
    return None


class WorkersAIProfile(ProviderProfile):
    """Cloudflare Workers AI's OpenAI-compatible endpoint with per-model reasoning values."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        effort = reasoning_effort_for(context.get("model"), reasoning_config)
        return {}, ({"reasoning_effort": effort} if effort else {})

    def get_model_context_length(self, model: str) -> int | None:
        # Hermes treats this host as a custom endpoint, whose /models probe fails here (HTTP 405), so
        # the window must come from the profile or Hermes falls back to 256K.
        entry = MODELS_BY_ID.get(str(model or ""))
        return entry.context_window if entry else None


workers_ai = WorkersAIProfile(
    name="workers-ai",
    display_name="Cloudflare Workers AI",
    description="Cloudflare Workers AI — open models on Cloudflare's network (OpenAI-compatible)",
    signup_url="https://dash.cloudflare.com/?to=/:account/ai/workers-ai",
    env_vars=(API_TOKEN_ENV, BASE_URL_ENV),
    base_url="",
    auth_type="api_key",
    # Cloudflare's OpenAI-compatible /models answers HTTP 405: the list is the catalog above.
    supports_model_listing=False,
    supports_health_check=False,
    default_aux_model="@cf/openai/gpt-oss-20b",
    fallback_models=tuple(model.id for model in MODELS),
    model_capabilities={
        model.id: {"supports_tools": True, "supports_vision": model.vision,
                   "supports_reasoning": model.reasoning, "context_window": model.context_window}
        for model in MODELS
    },
    classify_api_error=classify_api_error,
)

register_provider(workers_ai)
