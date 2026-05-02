"""Batch runner: one model, one category or all payloads → txt results file."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime

from pi_bench.config import load_prompt
from pi_bench.constants import CATEGORY_FILES, ROOT
from pi_bench.detection import check_leaks
from pi_bench.ollama import DEFAULT_URL, query_model

# simple_test_loop also supports "all" which maps to all_payloads.txt
LOOP_CATEGORY_FILES = {**CATEGORY_FILES, "all": "all_payloads.txt"}


def load_payloads(category: str) -> list[tuple[str, str]]:
    filename = LOOP_CATEGORY_FILES[category]
    path = ROOT / "payloads" / filename
    payloads = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        pid, text = line.split("|", 1)
        payloads.append((pid.strip(), text.strip()))
    return payloads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="phi4:14b")
    parser.add_argument("--prompt", default="nexabank")
    parser.add_argument("--category", default="all", choices=list(LOOP_CATEGORY_FILES.keys()))
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
            try:
                data = query_model(DEFAULT_URL, args.model, config["content"], payload_text)
                reply = data["message"]["content"]
                leaked = check_leaks(reply, config["secrets"])

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
                status = "FAIL" if leaked else "pass"
            except Exception as e:
                block = (
                    f"{'='*60}\n"
                    f"ID      : {pid}\n"
                    f"PAYLOAD : {payload_text}\n"
                    f"ERROR   : {e}\n"
                )
                status = f"ERROR ({type(e).__name__})"

            out.write(block + "\n")
            out.flush()
            print(f" {status}")

    print(f"\nDone. Results → {out_path}")


if __name__ == "__main__":
    main()
