"""Ollama API client for PI-Bench."""

from __future__ import annotations

import time

import requests

DEFAULT_URL = "http://localhost:11434/api/chat"


def _post(url: str, body: dict, timeout: int, retries: int) -> dict:
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


def query_model(
    url: str,
    model: str,
    system: str,
    payload: str,
    timeout: int = 180,
    retries: int = 2,
) -> dict:
    """Send a single-turn chat completion request to Ollama."""
    body = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0, "seed": 42},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": payload},
        ],
    }
    return _post(url, body, timeout, retries)


def run_conversation(
    url: str,
    model: str,
    system: str,
    user_turns: list[str],
    timeout: int = 180,
    retries: int = 2,
) -> list[dict]:
    """Run an N-turn conversation, returning one Ollama response dict per turn.

    The assistant's reply from each turn is appended to `messages` before the
    next user turn, so context accumulates as in a real chat.
    """
    messages = [{"role": "system", "content": system}]
    responses: list[dict] = []
    for user_text in user_turns:
        messages.append({"role": "user", "content": user_text})
        body = {
            "model": model,
            "stream": False,
            "options": {"temperature": 0, "seed": 42},
            "messages": messages,
        }
        data = _post(url, body, timeout, retries)
        responses.append(data)
        msg = data.get("message") or {}
        assistant_content = msg.get("content", "")
        messages.append({"role": "assistant", "content": assistant_content})
    return responses
