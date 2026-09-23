from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "figures" / "07_pipeline_overview.png"


def box(ax, xy, width, height, title, body, color):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1,
        edgecolor=color,
        facecolor=color + "18",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height * 0.72, title, ha="center", va="center", fontsize=8.3, weight="bold", color=color)
    ax.text(xy[0] + width / 2, xy[1] + height * 0.36, body, ha="center", va="center", fontsize=6.5, color="#222222", linespacing=1.18)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color="#555555"))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 220})
    fig, ax = plt.subplots(figsize=(7.15, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.015, 0.56), 0.18, 0.30, "Input views", "PPG | z-score | detrended\nPPG+VPG+APG | multiview\n2, 5, 10, 15 s", "#2457A6")
    box(ax, (0.215, 0.56), 0.22, 0.30, "Representations", "adaptive/morphology ResNet\nlong-conv | Conv-Transformer\nattention and pooling", "#1B7F79")
    box(ax, (0.455, 0.56), 0.19, 0.30, "Objectives", "direct | MAP/PP multitask\nclinical | ordinal | CCC\nasymmetric | focal", "#7551A8")
    box(ax, (0.665, 0.56), 0.15, 0.30, "Locked screen", "200 configurations\n80+40+40+40\npatient-disjoint", "#B56820")
    box(ax, (0.84, 0.56), 0.145, 0.30, "Confirmation", "prior + 4 finalists\n5 folds x 3 seeds\n75 fits; paired gate", "#A3333D")
    for left, right in [(0.195, 0.215), (0.435, 0.455), (0.645, 0.665), (0.82, 0.84)]:
        arrow(ax, (left, 0.71), (right, 0.71))

    box(ax, (0.19, 0.10), 0.27, 0.25, "Internal outcome", "No confirmed new improvement\nretained MAE: 12.87/8.18 mmHg\nhigh-BP sensitivity: 3.9%", "#206A5D")
    box(ax, (0.54, 0.10), 0.27, 0.25, "External stress test", "Three cohorts; synchronized VitalDB\nzero-shot: 18.20/11.24 mmHg\n0% direct high-BP sensitivity", "#A3333D")
    arrow(ax, (0.91, 0.56), (0.455, 0.36))
    arrow(ax, (0.91, 0.56), (0.675, 0.36))

    ax.text(0.5, 0.96, "Leakage-aware component-wise evaluation workflow", ha="center", va="center", fontsize=11, weight="bold")
    fig.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
