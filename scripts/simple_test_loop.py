"""Batch runner: one model, one category or all payloads → txt results file."""
import requests
import yaml
import argparse
from pathlib import Path
from datetime import datetime

OLLAMA = "http://localhost:11434/api/chat"
ROOT = Path(__file__).parent.parent

CATEGORY_FILES = {
    "J": "J_jailbreak_roleplay.txt",
    "O": "O_instruction_override.txt",
    "E": "E_obfuscation_encoding.txt",
    "C": "C_context_manipulation.txt",
    "G": "G_gradient_automated.txt",
    "P": "P_indirect_pipeline.txt",
    "M": "M_indirect_misinfo.txt",
    "all": "all_payloads.txt",
}


def load_prompt(name: str) -> dict:
    with open(ROOT / "config/system_prompts.yaml") as f:
        return yaml.safe_load(f)[name]


def load_payloads(category: str) -> list[tuple[str, str]]:
    filename = CATEGORY_FILES[category]
    path = ROOT / "payloads" / filename
    payloads = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        pid, text = line.split("|", 1)
        payloads.append((pid.strip(), text.strip()))
    return payloads


def run(model: str, system: str, payload: str) -> dict:
    body = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0, "seed": 42},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": payload},
        ],
    }
    return requests.post(OLLAMA, json=body).json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="phi4:14b")
    parser.add_argument("--prompt", default="nexabank")
    parser.add_argument("--category", default="all", choices=list(CATEGORY_FILES.keys()))
    args = parser.parse_args()

    config = load_prompt(args.prompt)
    payloads = load_payloads(args.category)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "results" / "raw" / f"{args.model.replace(':', '-')}_{args.prompt}_{args.category}_{timestamp}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model   : {args.model}")
    print(f"Prompt  : {config['name']}")
    print(f"Category: {args.category}  ({len(payloads)} payloads)")
    print(f"Output  : {out_path}")
    print("=" * 60)

    with open(out_path, "w") as out:
        for pid, payload_text in payloads:
            print(f"  {pid}...", end="", flush=True)
            data = run(args.model, config["content"], payload_text)
            reply = data["message"]["content"]
            leaked = [s["name"] for s in config["secrets"] if s["value"].lower() in reply.lower()]

            block = (
                f"{'='*60}\n"
                f"ID      : {pid}\n"
                f"PAYLOAD : {payload_text}\n"
                f"RESPONSE:\n{reply}\n"
                f"LEAKED  : {', '.join(leaked) if leaked else 'none'}\n"
                f"TOKENS  : prompt={data.get('prompt_eval_count','?')} "
                f"response={data.get('eval_count','?')} "
                f"ms={data.get('eval_duration',0)/1e6:.0f}\n"
            )
            out.write(block + "\n")
            out.flush()

            status = "FAIL" if leaked else "pass"
            print(f" {status}")

    print(f"\nDone. Results → {out_path}")


if __name__ == "__main__":
    main()
