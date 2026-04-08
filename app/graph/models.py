from __future__ import annotations

import vendor_bootstrap  # noqa: F401
from langchain_openai import ChatOpenAI

from config.settings import settings


def _resolve_main_provider() -> tuple[str | None, str | None, str]:
    return settings.SF_BASE_URL, settings.SF_API_KEY, settings.MAIN_MODEL_NAME


def _resolve_sub_provider() -> tuple[str | None, str | None, str]:
    if settings.AL_BAILIAN_BASE_URL and settings.AL_BAILIAN_API_KEY and settings.SUB_MODEL_NAME:
        return settings.AL_BAILIAN_BASE_URL, settings.AL_BAILIAN_API_KEY, settings.SUB_MODEL_NAME
    return _resolve_main_provider()


def build_main_chat_model(*, temperature: float = 0, streaming: bool = True) -> ChatOpenAI:
    base_url, api_key, model_name = _resolve_main_provider()
    return ChatOpenAI(
        model=model_name,
        openai_api_base=base_url,
        openai_api_key=api_key,
        temperature=temperature,
        streaming=streaming,
    )


def build_sub_chat_model(*, temperature: float = 0, streaming: bool = True) -> ChatOpenAI:
    base_url, api_key, model_name = _resolve_sub_provider()
    return ChatOpenAI(
        model=model_name,
        openai_api_base=base_url,
        openai_api_key=api_key,
        temperature=temperature,
        streaming=streaming,
    )
