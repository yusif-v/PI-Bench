"""Ollama API client for PI-Bench."""

from __future__ import annotations

import time

import requests

DEFAULT_URL = "http://localhost:11434/api/chat"


def query_model(
    url: str,
    model: str,
    system: str,
    payload: str,
    timeout: int = 180,
    retries: int = 2,
) -> dict:
    """Send a chat completion request to Ollama.

    Args:
        url: Ollama API endpoint.
        model: Ollama model tag.
        system: System prompt content.
        payload: User message / attack payload.
        timeout: Request timeout in seconds.
        retries: Number of retries on failure.

    Returns:
        Ollama API response dict.
    """
    body = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0, "seed": 42},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": payload},
        ],
    }
    for attempt in range(max(0, retries + 1)):
        try:
            resp = requests.post(url, json=body, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout:
            raise
        except (requests.RequestException, ValueError):
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")
