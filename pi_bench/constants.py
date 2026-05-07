"""Shared constants for PI-Bench."""

from pathlib import Path

ROOT = Path(__file__).parent.parent

CATEGORY_FILES = {
    "J": "J_jailbreak_roleplay.txt",
    "O": "O_instruction_override.txt",
    "E": "E_obfuscation_encoding.txt",
    "C": "C_context_manipulation.txt",
    "G": "G_gradient_automated.txt",
    "P": "P_indirect_pipeline.txt",
    "M": "M_indirect_misinfo.txt",
    "T": "T_multiturn.txt",
}

CATEGORY_ORDER = ["J", "O", "E", "C", "G", "P", "M", "T"]

CATEGORY_NAMES = {
    "J": "Jailbreak / Roleplay",
    "O": "Instruction Override",
    "E": "Obfuscation / Encoding",
    "C": "Context Manipulation",
    "G": "Gradient-Based / Automated",
    "P": "Indirect: Data Pipeline",
    "M": "Indirect: Misinformation",
    "T": "Multi-Turn / Conversational",
}

CSV_FIELDS = [
    "timestamp",
    "model",
    "model_family",
    "model_parameters",
    "prompt_name",
    "payload_id",
    "category",
    "payload_text",
    "response",
    "num_turns",
    "leak_turn",
    "response_per_turn",
    "prompt_tokens",
    "response_tokens",
    "total_ms",
    "eval_ms",
    "leaked_secrets",
    "attack_success",
    "error",
]
