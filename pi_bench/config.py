"""Config loaders for models, system prompts, and payloads."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from .constants import CATEGORY_FILES, ROOT


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_models() -> dict[str, dict]:
    """Load model registry from config/models.yaml."""
    data = load_yaml(ROOT / "config" / "models.yaml")
    if not data:
        raise ValueError("models.yaml is empty")
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def resolve_models(specs: list[str], registry: dict[str, dict]) -> list[str]:
    """Resolve model specifiers (tags or 'all') against the registry."""
    if specs == ["all"]:
        return list(registry.keys())

    resolved = []
    for s in specs:
        if s in registry:
            resolved.append(s)
        else:
            matches = [
                tag
                for tag, meta in registry.items()
                if s.lower() in meta.get("display_name", "").lower()
                or s.lower() in meta.get("family", "").lower()
            ]
            if matches:
                resolved.extend(matches)
            else:
                raise KeyError(
                    f"Model '{s}' not found in registry. "
                    f"Available: {list(registry.keys())}"
                )
    seen = set()
    return [m for m in resolved if not (m in seen or seen.add(m))]


def load_prompt(name: str) -> dict:
    """Load a system prompt configuration by name."""
    data = load_yaml(ROOT / "config" / "system_prompts.yaml")
    if name not in data:
        raise KeyError(f"Prompt '{name}' not found. Available: {list(data.keys())}")
    prompt = data[name]
    if "content" not in prompt:
        raise KeyError(f"Prompt '{name}' missing required 'content' field")
    if "secrets" not in prompt or not isinstance(prompt["secrets"], list):
        raise KeyError(f"Prompt '{name}' missing required 'secrets' list")
    return prompt


def resolve_prompts(specs: list[str]) -> list[str]:
    """Resolve prompt specifiers ('all' or explicit names)."""
    if specs == ["all"]:
        data = load_yaml(ROOT / "config" / "system_prompts.yaml")
        return list(data.keys())
    return specs


def load_payloads(categories: list[str]) -> list[tuple[str, list[str], str]]:
    """Load payloads for the given categories.

    Returns list of (payload_id, turns, category) tuples.

    Single-turn payloads use the line format `PID | text` and yield a
    1-element turns list. Multi-turn payloads use `PID >> turn text`,
    repeated once per turn with the same PID; consecutive lines with the
    same PID are grouped in order.
    """
    payloads: list[tuple[str, list[str], str]] = []
    for cat in categories:
        path = ROOT / "payloads" / CATEGORY_FILES[cat]
        if not path.exists():
            raise FileNotFoundError(
                f"Payload file not found for category '{cat}': {path}"
            )
        lines = path.read_text(encoding="utf-8").splitlines()
        order: list[str] = []
        turns_by_id: dict[str, list[str]] = {}
        for line_num, raw in enumerate(lines, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ">>" in line:
                pid, text = line.split(">>", 1)
            elif "|" in line:
                pid, text = line.split("|", 1)
            else:
                print(
                    f"  [WARN] Skipping malformed line {line_num} in {path.name}",
                    file=sys.stderr,
                )
                continue
            pid = pid.strip()
            text = text.strip()
            if not pid:
                print(
                    f"  [WARN] Skipping line {line_num} in {path.name}: empty ID",
                    file=sys.stderr,
                )
                continue
            if pid not in turns_by_id:
                turns_by_id[pid] = []
                order.append(pid)
            turns_by_id[pid].append(text)
        for pid in order:
            payloads.append((pid, turns_by_id[pid], cat))
    return payloads
