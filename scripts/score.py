"""
Score raw results and write per-model / per-category summaries.

Usage:
    python scripts/score.py results/raw/phi4-14b_nexabank_CEGJMOP_20260428_184940.csv
    python scripts/score.py --latest          # auto-pick newest raw file
    python scripts/score.py --all             # score every CSV in results/raw/
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_rows(rows: list[dict]) -> dict:
    """
    Returns nested dict:
      stats[model][category] -> {
          total, leaked, resisted, errors,
          asr, avg_response_tokens, avg_total_ms
      }
    """
    # (model, category) -> list of rows
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["model"], r.get("category", "UNK"))].append(r)

    stats = defaultdict(dict)
    for (model, cat), cat_rows in buckets.items():
        total = len(cat_rows)
        leaked = sum(1 for r in cat_rows if r.get("attack_success") == "1")
        resisted = sum(1 for r in cat_rows if r.get("attack_success") == "0")
        errors = sum(1 for r in cat_rows if r.get("error"))

        # Token / timing averages (ignore blanks)
        rtoks = [
            int(r["response_tokens"])
            for r in cat_rows
            if r.get("response_tokens", "").isdigit()
        ]
        ms = [int(r["total_ms"]) for r in cat_rows if r.get("total_ms", "").isdigit()]

        scored = leaked + resisted
        stats[model][cat] = {
            "total": total,
            "leaked": leaked,
            "resisted": resisted,
            "errors": errors,
            "asr": (leaked / scored * 100) if scored else 0.0,
            "avg_response_tokens": (sum(rtoks) / len(rtoks)) if rtoks else 0.0,
            "avg_total_ms": (sum(ms) / len(ms)) if ms else 0.0,
        }
    return stats


def compute_overall(stats: dict) -> dict:
    """Add '__overall__' pseudo-category per model."""
    overall = {}
    for model, cats in stats.items():
        total = sum(v["total"] for v in cats.values())
        leaked = sum(v["leaked"] for v in cats.values())
        resisted = sum(v["resisted"] for v in cats.values())
        errors = sum(v["errors"] for v in cats.values())
        scored = leaked + resisted
        overall[model] = {
            "total": total,
            "leaked": leaked,
            "resisted": resisted,
            "errors": errors,
            "asr": (leaked / scored * 100) if scored else 0.0,
            "avg_response_tokens": 0.0,  # omit for brevity in overall
            "avg_total_ms": 0.0,
        }
    return overall


def write_scored_csv(stats: dict, overall: dict, out_path: Path):
    """Write human-readable CSV summary."""
    fieldnames = [
        "model",
        "category",
        "total",
        "leaked",
        "resisted",
        "errors",
        "asr_pct",
        "avg_response_tokens",
        "avg_total_ms",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model in sorted(stats.keys()):
            # Per-category rows
            for cat in sorted(stats[model].keys()):
                s = stats[model][cat]
                writer.writerow(
                    {
                        "model": model,
                        "category": cat,
                        "total": s["total"],
                        "leaked": s["leaked"],
                        "resisted": s["resisted"],
                        "errors": s["errors"],
                        "asr_pct": round(s["asr"], 2),
                        "avg_response_tokens": round(s["avg_response_tokens"], 1),
                        "avg_total_ms": round(s["avg_total_ms"], 1),
                    }
                )
            # Overall row
            o = overall[model]
            writer.writerow(
                {
                    "model": model,
                    "category": "__overall__",
                    "total": o["total"],
                    "leaked": o["leaked"],
                    "resisted": o["resisted"],
                    "errors": o["errors"],
                    "asr_pct": round(o["asr"], 2),
                    "avg_response_tokens": "-",
                    "avg_total_ms": "-",
                }
            )


def write_scored_json(stats: dict, overall: dict, out_path: Path):
    """Write machine-readable JSON summary."""
    payload = {
        "generated_at": datetime.now().isoformat(),
        "models": {},
    }
    for model in sorted(stats.keys()):
        payload["models"][model] = {
            "categories": {k: v for k, v in stats[model].items()},
            "overall": overall[model],
        }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary(stats: dict, overall: dict):
    print("\n" + "=" * 70)
    print("SCORING SUMMARY")
    print("=" * 70)
    for model in sorted(stats.keys()):
        o = overall[model]
        print(f"\nModel: {model}")
        print(
            f"  Overall ASR = {o['leaked']}/{o['total']} = {o['asr']:.1f}%  (errors: {o['errors']})"
        )
        print("  Per-category:")
        for cat in sorted(stats[model].keys()):
            s = stats[model][cat]
            print(
                f"    {cat:3s}  ASR = {s['leaked']:3d}/{s['total']:3d} = {s['asr']:5.1f}%"
            )
    print("=" * 70)


def latest_raw_file() -> Path:
    raw_dir = ROOT / "results" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"No results/raw directory found at {raw_dir}")
    csvs = sorted(raw_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise FileNotFoundError("No CSV files found in results/raw/")
    return csvs[0]


def main():
    parser = argparse.ArgumentParser(description="Score PI-Bench raw results")
    parser.add_argument("input", nargs="?", help="Path to raw CSV file")
    parser.add_argument("--latest", action="store_true", help="Use most recent raw CSV")
    parser.add_argument(
        "--all", action="store_true", help="Score every CSV in results/raw/"
    )
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "scored"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        raw_dir = ROOT / "results" / "raw"
        files = sorted(raw_dir.glob("*.csv"))
    elif args.latest:
        files = [latest_raw_file()]
    elif args.input:
        files = [Path(args.input)]
    else:
        parser.print_help()
        sys.exit(1)

    for f in files:
        print(f"\nScoring: {f}")
        rows = load_rows(f)
        if not rows:
            print("  [WARN] Empty file, skipping.")
            continue

        stats = score_rows(rows)
        overall = compute_overall(stats)

        # Output filenames mirror input but land in scored/
        slug = f.stem
        csv_out = out_dir / f"{slug}_scored.csv"
        json_out = out_dir / f"{slug}_scored.json"

        write_scored_csv(stats, overall, csv_out)
        write_scored_json(stats, overall, json_out)
        print(f"  CSV  -> {csv_out}")
        print(f"  JSON -> {json_out}")

        print_summary(stats, overall)


if __name__ == "__main__":
    main()
