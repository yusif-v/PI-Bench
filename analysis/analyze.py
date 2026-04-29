"""
Quick terminal analysis from a harness.py CSV output.

Usage:
    python analysis/analyze.py results/raw/my_run.csv
    python analysis/analyze.py results/raw/my_run.csv --by model category
    python analysis/analyze.py results/raw/my_run.csv --crosstab
    python analysis/analyze.py results/raw/my_run.csv --latency
    python analysis/analyze.py results/raw/my_run.csv --export analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

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
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("attack_success", "").strip() in ("0", "1")]


def asr_table(rows: list[dict], group_key: str) -> dict[str, tuple[int, int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        val = r.get(group_key, "unknown") or "unknown"
        buckets[val].append(int(r["attack_success"]))
    return {k: (sum(v), len(v)) for k, v in sorted(buckets.items())}


def latency_table(rows: list[dict], group_key: str) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        val = r.get(group_key, "unknown") or "unknown"
        ms = r.get("total_ms", "").strip()
        if ms.isdigit():
            buckets[val].append(float(ms))
    return buckets


def crosstab(rows: list[dict]) -> dict[str, dict[str, tuple[int, int]]]:
    grid: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        model = r.get("model", "unknown")
        cat = r.get("category", "UNK")
        grid[model][cat].append(int(r["attack_success"]))
    return {
        m: {c: (sum(v), len(v)) for c, v in cats.items()} for m, cats in grid.items()
    }


def _hline(widths: list[int]) -> str:
    parts = ["─" * w for w in widths]
    return "  " + "  ".join(parts)


def print_table(
    title: str, data: dict[str, tuple[int, int]], label_map: dict | None = None
):
    print(f"\n{'─' * 58}")
    print(f"  {title}")
    print(f"{'─' * 58}")
    print(f"  {'Group':<32}  {'Leaks':>5}  {'Total':>5}  {'ASR':>6}")
    print(_hline([32, 5, 5, 6]))
    for key, (leaks, total) in data.items():
        asr = leaks / total * 100 if total else 0
        label = label_map.get(key, key) if label_map else key
        print(f"  {label:<32}  {leaks:>5}  {total:>5}  {asr:>5.1f}%")
    total_leaks = sum(lk for lk, _ in data.values())
    total_runs = sum(t for _, t in data.values())
    overall = total_leaks / total_runs * 100 if total_runs else 0
    print(_hline([32, 5, 5, 6]))
    print(f"  {'TOTAL':<32}  {total_leaks:>5}  {total_runs:>5}  {overall:>5.1f}%")


def print_latency_table(title: str, data: dict[str, list[float]]):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    print(f"  {'Group':<28}  {'Runs':>5}  {'Avg ms':>8}  {'Max ms':>8}")
    print(_hline([28, 5, 8, 8]))
    all_times: list[float] = []
    for key, times in sorted(data.items()):
        if not times:
            continue
        avg = sum(times) / len(times)
        mx = max(times)
        all_times.extend(times)
        print(f"  {key:<28}  {len(times):>5}  {avg:>8.0f}  {mx:>8.0f}")
    if all_times:
        overall_avg = sum(all_times) / len(all_times)
        overall_max = max(all_times)
        print(_hline([28, 5, 8, 8]))
        print(
            f"  {'TOTAL':<28}  {len(all_times):>5}  {overall_avg:>8.0f}  {overall_max:>8.0f}"
        )


def _fmt_asr(leaks: int, total: int) -> str:
    asr = leaks / total * 100 if total else 0
    return f"{asr:>8.1f}%"


def print_crosstab(grid: dict[str, dict[str, tuple[int, int]]]):
    categories = sorted({c for cats in grid.values() for c in cats.keys()})
    col_w = 9
    name_w = 14

    total_w = name_w + len(categories) * col_w + 3 * col_w
    print(f"\n{'─' * total_w}")
    print("  Crosstab: Model × Category (ASR %)")
    print(f"{'─' * total_w}")

    header = (
        f"  {'Model':<{name_w}}"
        + "".join(f"{c:>{col_w}}" for c in categories)
        + f"  {'Overall':>{col_w}}"
    )
    print(header)
    print(_hline([name_w] + [col_w] * len(categories) + [col_w]))

    for model in sorted(grid.keys()):
        row_vals = []
        row_leaks, row_total = 0, 0
        for cat in categories:
            leaks, total = grid[model].get(cat, (0, 0))
            row_leaks += leaks
            row_total += total
            row_vals.append(_fmt_asr(leaks, total))
        overall = _fmt_asr(row_leaks, row_total)
        print(f"  {model:<{name_w}}" + "".join(row_vals) + f"  {overall}")

    # Column totals
    col_leaks = [sum(grid[m].get(c, (0, 0))[0] for m in grid) for c in categories]
    col_totals = [sum(grid[m].get(c, (0, 0))[1] for m in grid) for c in categories]

    print(_hline([name_w] + [col_w] * len(categories) + [col_w]))
    overall_cells = "".join(_fmt_asr(lk, tot) for lk, tot in zip(col_leaks, col_totals))
    grand_leaks = sum(col_leaks)
    grand_total = sum(col_totals)
    print(
        f"  {'Overall':<{name_w}}"
        + overall_cells
        + f"  {_fmt_asr(grand_leaks, grand_total)}"
    )


def secret_breakdown(rows: list[dict]):
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        for secret in r.get("leaked_secrets", "").split("|"):
            if secret:
                counts[secret] += 1
    if not counts:
        print("\n  No secrets leaked.")
        return
    print(f"\n{'─' * 44}")
    print("  Leaked secrets breakdown")
    print(f"{'─' * 44}")
    for secret, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {secret:<34}  {count:>4}x")


def build_export_payload(rows: list[dict]) -> dict:
    by_category = asr_table(rows, "category")
    by_model = asr_table(rows, "model")
    by_prompt = asr_table(rows, "prompt_name")
    by_family = asr_table(rows, "model_family")

    secrets: dict[str, int] = defaultdict(int)
    for r in rows:
        for s in r.get("leaked_secrets", "").split("|"):
            if s:
                secrets[s] += 1

    return {
        "total_runs": len(rows),
        "by_category": {
            k: {"leaked": v[0], "total": v[1], "asr": round(v[0] / v[1] * 100, 2)}
            for k, v in by_category.items()
        },
        "by_model": {
            k: {"leaked": v[0], "total": v[1], "asr": round(v[0] / v[1] * 100, 2)}
            for k, v in by_model.items()
        },
        "by_prompt": {
            k: {"leaked": v[0], "total": v[1], "asr": round(v[0] / v[1] * 100, 2)}
            for k, v in by_prompt.items()
        },
        "by_family": {
            k: {"leaked": v[0], "total": v[1], "asr": round(v[0] / v[1] * 100, 2)}
            for k, v in by_family.items()
        },
        "crosstab": {
            m: {
                c: {"leaked": lk, "total": tot, "asr": round(lk / tot * 100, 2)}
                for c, (lk, tot) in cats.items()
            }
            for m, cats in crosstab(rows).items()
        },
        "secrets": dict(sorted(secrets.items(), key=lambda x: -x[1])),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze harness.py CSV output")
    parser.add_argument("csv_file", help="Path to harness.py CSV output")
    parser.add_argument(
        "--by",
        nargs="+",
        default=["category", "model"],
        choices=["category", "model", "prompt_name", "model_family"],
        help="Grouping dimensions to report",
    )
    parser.add_argument(
        "--crosstab", action="store_true", help="Show model × category matrix"
    )
    parser.add_argument(
        "--latency", action="store_true", help="Show average latency per group"
    )
    parser.add_argument("--export", metavar="FILE", help="Export analysis as JSON")
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

    if "model_family" in args.by:
        print_table("ASR by Model Family", asr_table(rows, "model_family"))

    if args.crosstab:
        print_crosstab(crosstab(rows))

    if args.latency:
        print_latency_table("Latency by Category", latency_table(rows, "category"))
        print_latency_table("Latency by Model", latency_table(rows, "model"))

    secret_breakdown(rows)

    if args.export:
        payload = build_export_payload(rows)
        Path(args.export).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  Exported analysis → {args.export}")


if __name__ == "__main__":
    main()
