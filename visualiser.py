"""
utils/visualiser.py — Visualisation for GeoSentiFake results.

Generates:
    1. ESS comparison bar chart
    2. Emotion radar / heatmap
    3. VAD scatter plot
    4. Feature importance bar chart
    5. ESS timeline (for batch results)

Production: pip install matplotlib seaborn
Fallback:   ASCII charts printed to terminal
"""

import math
import json
from typing import List, Dict


# ── ASCII fallback charts (zero dependency) ───────────────────────────────────

def ascii_ess_chart(results: List[Dict], width: int = 50):
    """Print ESS bar chart to terminal."""
    print("\n  ESS COMPARISON CHART")
    print("  " + "─" * (width + 30))
    for r in results:
        score = r["ess_score"]
        bar_len = int(score / 100 * width)
        bar = "█" * bar_len + "░" * (width - bar_len)
        band  = r["ess_band"]
        band_marker = {"HIGH": "▲", "MODERATE": "●", "LOW": "▼"}.get(band, "?")
        title = r["title"][:28]
        print(f"  {title:<28} {band_marker} [{bar}] {score:5.1f}")
    print("  " + "─" * (width + 30))


def ascii_emotion_table(results: List[Dict]):
    """Print emotion profiles as aligned table."""
    emotions = ["fear", "anger", "disgust", "sadness",
                "surprise", "joy", "trust", "anticipation"]
    print("\n  EMOTION PROFILE TABLE")
    header = f"  {'Article':<20} " + " ".join(f"{e[:5]:>5}" for e in emotions)
    print(header)
    print("  " + "─" * len(header))
    for r in results:
        emos = r.get("emotions", {})
        row  = f"  {r['title'][:20]:<20} "
        row += " ".join(f"{emos.get(e, 0):.3f}" for e in emotions)
        print(row)


def ascii_vad_summary(results: List[Dict]):
    """Print VAD scores as a simple table."""
    print("\n  VAD SCORES (Valence / Arousal / Dominance)")
    print(f"  {'Article':<30} {'Valence':>8} {'Arousal':>8} {'Dominance':>10}")
    print("  " + "─" * 60)
    for r in results:
        vad = r.get("vad", {})
        print(f"  {r['title'][:30]:<30} "
              f"{vad.get('valence', 0):>+8.3f} "
              f"{vad.get('arousal', 0):>8.3f} "
              f"{vad.get('dominance', 0):>+10.3f}")


# ── Matplotlib charts (when available) ───────────────────────────────────────

def plot_ess_comparison(results: List[Dict], output_path: str = "ess_comparison.png"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        fig, ax = plt.subplots(figsize=(12, 6))
        titles = [r["title"][:45] + "…" if len(r["title"]) > 45
                  else r["title"] for r in results]
        scores = [r["ess_score"] for r in results]
        bands  = [r["ess_band"]  for r in results]

        colours = {
            "HIGH":     "#e74c3c",
            "MODERATE": "#f39c12",
            "LOW":      "#2ecc71",
        }
        bar_colours = [colours.get(b, "#95a5a6") for b in bands]

        bars = ax.barh(range(len(results)), scores, color=bar_colours,
                       edgecolor="white", linewidth=0.8, height=0.6)

        # ESS band lines
        ax.axvline(33,  color="#2ecc71", linestyle="--", linewidth=1, alpha=0.6, label="LOW/MOD boundary")
        ax.axvline(66,  color="#f39c12", linestyle="--", linewidth=1, alpha=0.6, label="MOD/HIGH boundary")

        # Value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 1, i, f"{score:.1f}", va="center", fontsize=9, color="#2c3e50")

        ax.set_yticks(range(len(results)))
        ax.set_yticklabels(titles, fontsize=9)
        ax.set_xlabel("Emotional Skew Score (ESS)", fontsize=11)
        ax.set_title("GeoSentiFake — Emotional Skew Score Comparison", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 115)
        ax.invert_yaxis()

        patches = [mpatches.Patch(color=c, label=b) for b, c in colours.items()]
        ax.legend(handles=patches, loc="lower right", fontsize=9)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ ESS chart saved: {output_path}")
    except ImportError:
        ascii_ess_chart(results)


def plot_emotion_heatmap(results: List[Dict], output_path: str = "emotion_heatmap.png"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np

        emotions = ["fear", "anger", "disgust", "sadness",
                    "surprise", "joy", "trust", "anticipation"]
        titles = [r["title"][:35] + "…" if len(r["title"]) > 35
                  else r["title"] for r in results]
        data = np.array([
            [r["emotions"].get(e, 0) for e in emotions]
            for r in results
        ])

        fig, ax = plt.subplots(figsize=(12, max(4, len(results) * 0.9)))
        cmap = plt.cm.YlOrRd
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=0.5)

        ax.set_xticks(range(len(emotions)))
        ax.set_xticklabels([e.capitalize() for e in emotions], fontsize=10)
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels(titles, fontsize=9)

        # Cell annotations
        for i in range(len(results)):
            for j in range(len(emotions)):
                val = data[i, j]
                text_colour = "white" if val > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_colour)

        plt.colorbar(im, ax=ax, label="Emotion Probability", shrink=0.8)
        ax.set_title("GeoSentiFake — Plutchik Emotion Heatmap", fontsize=13, fontweight="bold")
        ax.set_xlabel("Plutchik Emotion Dimension", fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Emotion heatmap saved: {output_path}")
    except ImportError:
        ascii_emotion_table(results)


def plot_vad_scatter(results: List[Dict], output_path: str = "vad_scatter.png"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 7))

        valences  = [r["vad"]["valence"]   for r in results]
        arousals  = [r["vad"]["arousal"]   for r in results]
        ess_scores = [r["ess_score"]       for r in results]
        labels_   = [r["label"]            for r in results]

        colour_map = {"REAL": "#2ecc71", "FAKE": "#e74c3c", "PARTIAL": "#f39c12"}
        colours_   = [colour_map.get(l, "#95a5a6") for l in labels_]

        scatter = ax.scatter(
            valences, arousals,
            c=ess_scores, cmap="YlOrRd",
            s=[sz * 4 + 100 for sz in ess_scores],
            alpha=0.85, edgecolors="white", linewidths=1.5,
            vmin=0, vmax=100
        )

        for i, r in enumerate(results):
            ax.annotate(
                r["title"][:25],
                (valences[i], arousals[i]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=8, color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
            )

        plt.colorbar(scatter, label="ESS Score")
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
        ax.axvline(0,   color="grey", linestyle=":", alpha=0.5)
        ax.set_xlabel("Valence  (−1 = Negative  →  +1 = Positive)", fontsize=11)
        ax.set_ylabel("Arousal  (0 = Calm  →  1 = Activated)", fontsize=11)
        ax.set_title("GeoSentiFake — VAD Emotional Space (size = ESS)", fontsize=13, fontweight="bold")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.05, 1.1)

        # Quadrant labels
        ax.text(-1.0, 0.97, "High Arousal\nNegative", fontsize=8, color="grey", alpha=0.7)
        ax.text( 0.5, 0.97, "High Arousal\nPositive", fontsize=8, color="grey", alpha=0.7)
        ax.text(-1.0, 0.02, "Low Arousal\nNegative", fontsize=8, color="grey", alpha=0.7)
        ax.text( 0.5, 0.02, "Low Arousal\nPositive", fontsize=8, color="grey", alpha=0.7)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ VAD scatter plot saved: {output_path}")
    except ImportError:
        ascii_vad_summary(results)


def plot_feature_importance(results: List[Dict], output_path: str = "feature_importance.png"):
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Average SHAP across all results
        feature_totals: Dict[str, float] = {}
        feature_counts: Dict[str, int]   = {}
        for r in results:
            for feat, imp in r.get("shap_top_features", []):
                feature_totals[feat] = feature_totals.get(feat, 0) + imp
                feature_counts[feat] = feature_counts.get(feat, 0) + 1

        avg_importance = {
            f: feature_totals[f] / feature_counts[f]
            for f in feature_totals
        }
        sorted_feats = sorted(avg_importance.items(), key=lambda x: x[1])

        fig, ax = plt.subplots(figsize=(10, 5))
        feats  = [f[0] for f in sorted_feats]
        values = [f[1] for f in sorted_feats]

        palette = plt.cm.Blues(np.linspace(0.4, 0.9, len(feats)))
        bars = ax.barh(feats, values, color=palette, edgecolor="white", height=0.6)

        for bar, val in zip(bars, values):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

        ax.set_xlabel("Mean SHAP Importance", fontsize=11)
        ax.set_title("GeoSentiFake — Feature Importance (Averaged SHAP)", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Feature importance chart saved: {output_path}")
    except ImportError:
        print("  matplotlib not installed — skipping feature importance chart.")


def generate_all_charts(results_path: str = "results.json", output_dir: str = "."):
    """Generate all visualisation charts from a results JSON file."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    print(f"\n  Generating visualisations from {results_path} ({len(results)} articles)…\n")

    plot_ess_comparison(results,
        output_path=os.path.join(output_dir, "ess_comparison.png"))
    plot_emotion_heatmap(results,
        output_path=os.path.join(output_dir, "emotion_heatmap.png"))
    plot_vad_scatter(results,
        output_path=os.path.join(output_dir, "vad_scatter.png"))
    plot_feature_importance(results,
        output_path=os.path.join(output_dir, "feature_importance.png"))

    ascii_ess_chart(results)
    ascii_vad_summary(results)
    print()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    generate_all_charts(results_path=path)
