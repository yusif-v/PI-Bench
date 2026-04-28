"""
Basic statistics from a harness.py CSV output.

Usage:
    python analysis/analyze.py results/raw/my_run.csv
    python analysis/analyze.py results/raw/my_run.csv --by model category
"""
import csv
import argparse
from pathlib import Path
from collections import defaultdict

CATEGORY_NAMES = {
    "J": "Jailbreak / Roleplay",
    "O": "Instruction Override",
    "E": "Obfuscation / Encoding",
    "C": "Context Manipulation",
    "G": "Gradient-Based / Automated",
    "P": "Indirect: Data Pipeline",
    "M": "Indirect: Misinformation",
}


def load_results(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["attack_success"] != ""]


def asr_table(rows: list[dict], group_key: str) -> dict[str, tuple[int, int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        buckets[r[group_key]].append(int(r["attack_success"]))
    return {k: (sum(v), len(v)) for k, v in sorted(buckets.items())}


def print_table(title: str, data: dict[str, tuple[int, int]], label_map: dict = None):
    print(f"\n{'─'*52}")
    print(f"  {title}")
    print(f"{'─'*52}")
    print(f"  {'Group':<28}  {'Fails':>5}  {'Total':>5}  {'ASR':>6}")
    print(f"  {'─'*28}  {'─'*5}  {'─'*5}  {'─'*6}")
    for key, (fails, total) in data.items():
        asr = fails / total * 100 if total else 0
        label = label_map.get(key, key) if label_map else key
        print(f"  {label:<28}  {fails:>5}  {total:>5}  {asr:>5.1f}%")
    total_fails = sum(f for f, _ in data.values())
    total_runs = sum(t for _, t in data.values())
    overall = total_fails / total_runs * 100 if total_runs else 0
    print(f"  {'─'*28}  {'─'*5}  {'─'*5}  {'─'*6}")
    print(f"  {'TOTAL':<28}  {total_fails:>5}  {total_runs:>5}  {overall:>5.1f}%")


def secret_breakdown(rows: list[dict]):
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        for secret in r["leaked_secrets"].split("|"):
            if secret:
                counts[secret] += 1
    if not counts:
        print("\n  No secrets leaked.")
        return
    print(f"\n{'─'*40}")
    print(f"  Leaked secrets breakdown")
    print(f"{'─'*40}")
    for secret, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {secret:<30}  {count:>4}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to harness.py CSV output")
    parser.add_argument("--by", nargs="+", default=["category", "model"],
                        choices=["category", "model", "prompt_name"],
                        help="Grouping dimensions to report")
    args = parser.parse_args()

    rows = load_results(args.csv_file)
    if not rows:
        print("No valid rows found.")
        return

    print(f"\nFile  : {args.csv_file}")
    print(f"Rows  : {len(rows)}")

    if "category" in args.by:
        data = asr_table(rows, "category")
        labeled = {k: f"{k} – {CATEGORY_NAMES.get(k, k)}" for k in data}
        print_table("ASR by Attack Category", data, labeled)

    if "model" in args.by:
        print_table("ASR by Model", asr_table(rows, "model"))

    if "prompt_name" in args.by:
        print_table("ASR by System Prompt", asr_table(rows, "prompt_name"))

    secret_breakdown(rows)


if __name__ == "__main__":
    main()
