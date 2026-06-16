"""Small helper for optional OpenAI-backed JSON extraction.

The parser only calls this module when an API key is available. The helper uses
the standard HTTPS API directly so we do not add a hard dependency on the
OpenAI Python package.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_MODEL = "gpt-4o-mini"


def _api_key(api_key: str | None = None) -> str | None:
    return api_key or os.environ.get("OPENAI_API_KEY")


def _model(model: str | None = None) -> str:
    return model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


def openai_chat_json(
    *,
    messages: list[dict[str, str]],
    model: str | None = None,
    api_key: str | None = None,
    timeout: int = 60,
) -> dict[str, Any] | list[Any] | None:
    """Call the OpenAI chat completions endpoint and parse a JSON response.

    Returns ``None`` when the API key is missing or when the request fails.
    """

    key = _api_key(api_key)
    if not key:
        return None

    payload = {
        "model": _model(model),
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return None

    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

    if not content:
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None
