"""
Generate figures from scored results.

Usage:
    python scripts/plot.py --latest
    python scripts/plot.py results/scored/phi4-14b_nexabank_CEGJMOP_20260428_184940_scored.json
    python scripts/plot.py --all
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
FIGURES_DIR = ROOT / "figures"

# Academic style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
    }
)

CATEGORY_ORDER = ["J", "O", "E", "C", "G", "P", "M"]
CATEGORY_LABELS = {
    "J": "Jailbreak",
    "O": "Instruction\nOverride",
    "E": "Obfuscation",
    "C": "Context\nManipulation",
    "G": "Gradient",
    "P": "Pipeline",
    "M": "Misinformation",
}
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def load_scored(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_model_data(data: dict) -> dict:
    """
    Returns dict[model_name] -> {
        categories: {cat: {asr, leaked, total, ...}},
        overall: {...}
    }
    """
    return data.get("models", {})


def save(fig: plt.Figure, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in [".png", ".pdf"]:
        out = FIGURES_DIR / f"{name}{ext}"
        fig.savefig(out)
        print(f"  Saved {out}")
    plt.close(fig)


def plot_asr_by_category(all_models: dict, slug: str):
    """
    Grouped bar chart: categories on X, ASR on Y, one bar per model.
    """
    models = list(all_models.keys())
    n_models = len(models)
    n_cats = len(CATEGORY_ORDER)
    width = 0.8 / n_models
    x = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, model in enumerate(models):
        cats = all_models[model].get("categories", {})
        asrs = [
            cats.get(cat, {}).get("asr", 0.0) if cat in cats else 0.0
            for cat in CATEGORY_ORDER
        ]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            asrs,
            width,
            label=model,
            color=COLORS[i],
            edgecolor="black",
            linewidth=0.5,
        )
        # Value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90 if height > 50 else 0,
                )

    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_xlabel("Attack Category")
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORY_ORDER])
    ax.set_ylim(0, 105)
    ax.legend(title="Model", loc="upper left", frameon=True)
    ax.set_title(f"ASR by Attack Category — {slug}")

    fig.tight_layout()
    save(fig, f"{slug}_asr_by_category")
    return fig


def plot_heatmap(all_models: dict, slug: str):
    """
    Heatmap: models (rows) vs categories (cols), cells = ASR %.
    """
    models = list(all_models.keys())
    matrix = np.zeros((len(models), len(CATEGORY_ORDER)))

    for i, model in enumerate(models):
        cats = all_models[model].get("categories", {})
        for j, cat in enumerate(CATEGORY_ORDER):
            matrix[i, j] = cats.get(cat, {}).get("asr", 0.0)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(models) + 1)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(CATEGORY_ORDER)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORY_ORDER])
    ax.set_yticklabels(models)

    # Text annotations
    for i in range(len(models)):
        for j in range(len(CATEGORY_ORDER)):
            val = matrix[i, j]
            text_color = "white" if val > 50 else "black"
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title(f"ASR Heatmap — {slug}")
    fig.colorbar(im, ax=ax, label="ASR (%)", shrink=0.6)

    fig.tight_layout()
    save(fig, f"{slug}_heatmap")
    return fig


def plot_overall_ranking(all_models: dict, slug: str):
    """
    Horizontal bar chart of overall ASR per model.
    """
    models = []
    asrs = []
    leaked = []
    totals = []

    for model in sorted(all_models.keys()):
        ov = all_models[model].get("overall", {})
        models.append(model)
        asrs.append(ov.get("asr", 0.0))
        leaked.append(ov.get("leaked", 0))
        totals.append(ov.get("total", 0))

    # Sort by ASR descending
    sorted_idx = np.argsort(asrs)[::-1]
    models = [models[i] for i in sorted_idx]
    asrs = [asrs[i] for i in sorted_idx]
    leaked = [leaked[i] for i in sorted_idx]
    totals = [totals[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(models) + 1.5))
    y = np.arange(len(models))
    bars = ax.barh(y, asrs, color=COLORS[0], edgecolor="black", linewidth=0.5)

    # Color by severity
    for bar, val in zip(bars, asrs):
        if val > 50:
            bar.set_color("#d62728")  # red
        elif val > 25:
            bar.set_color("#ff7f0e")  # orange
        else:
            bar.set_color("#2ca02c")  # green

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Overall Attack Success Rate (%)")

    for i, (val, l, t) in enumerate(zip(asrs, leaked, totals)):
        ax.text(val + 1.5, i, f"{val:.1f}% ({l}/{t})", va="center", fontsize=9)

    ax.set_title(f"Overall ASR Ranking — {slug}")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#2ca02c", label="Low (<25%)"),
            plt.Rectangle((0, 0), 1, 1, color="#ff7f0e", label="Medium (25-50%)"),
            plt.Rectangle((0, 0), 1, 1, color="#d62728", label="High (>50%)"),
        ],
        loc="lower right",
        frameon=True,
    )

    fig.tight_layout()
    save(fig, f"{slug}_overall_ranking")
    return fig


def plot_leak_resist_stacked(all_models: dict, slug: str):
    """
    Stacked horizontal bar: leaked vs resisted per category.
    """
    models = list(all_models.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        cats = all_models[model].get("categories", {})
        cats_present = [c for c in CATEGORY_ORDER if c in cats]
        leaked = [cats[c]["leaked"] for c in cats_present]
        resisted = [cats[c]["resisted"] for c in cats_present]

        y = np.arange(len(cats_present))
        ax.barh(
            y, leaked, color="#d62728", label="Leaked", edgecolor="black", linewidth=0.5
        )
        ax.barh(
            y,
            resisted,
            left=leaked,
            color="#2ca02c",
            label="Resisted",
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_yticks(y)
        ax.set_yticklabels([CATEGORY_LABELS[c] for c in cats_present])
        ax.invert_yaxis()
        ax.set_xlabel("Payload Count")
        ax.set_title(model)
        ax.legend(loc="lower right")

    fig.suptitle(f"Leak vs Resist by Category — {slug}", y=1.02)
    fig.tight_layout()
    save(fig, f"{slug}_leak_resist_stacked")
    return fig


def process_file(path: Path):
    print(f"\nPlotting: {path}")
    data = load_scored(path)
    models = extract_model_data(data)
    if not models:
        print("  [WARN] No model data found.")
        return

    slug = path.stem.replace("_scored", "")
    plot_asr_by_category(models, slug)
    plot_heatmap(models, slug)
    plot_overall_ranking(models, slug)
    plot_leak_resist_stacked(models, slug)


def latest_scored_file() -> Path:
    scored_dir = ROOT / "results" / "scored"
    if not scored_dir.exists():
        raise FileNotFoundError(f"No results/scored directory found at {scored_dir}")
    jsons = sorted(
        scored_dir.glob("*_scored.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not jsons:
        raise FileNotFoundError("No scored JSON files found in results/scored/")
    return jsons[0]


def main():
    parser = argparse.ArgumentParser(description="Generate PI-Bench figures")
    parser.add_argument("input", nargs="?", help="Path to scored JSON file")
    parser.add_argument(
        "--latest", action="store_true", help="Use most recent scored JSON"
    )
    parser.add_argument(
        "--all", action="store_true", help="Plot every scored JSON in results/scored/"
    )
    args = parser.parse_args()

    if args.all:
        scored_dir = ROOT / "results" / "scored"
        files = sorted(scored_dir.glob("*_scored.json"))
    elif args.latest:
        files = [latest_scored_file()]
    elif args.input:
        files = [Path(args.input)]
    else:
        parser.print_help()
        sys.exit(1)

    for f in files:
        try:
            process_file(f)
        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
