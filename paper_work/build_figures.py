"""Build two compact, source-traceable figures for the four-question manuscript."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def framework():
    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    colors = ["#dbeafe", "#dcfce7", "#fef3c7", "#fce7f3"]
    labels = [
        ("Q1: data", "Q_A proxy + mixture", "A1-A15"),
        ("Q2: loss", "N-D-Q_B law", "B1-B8"),
        ("Q3: decision", "Cost-constrained optimum", "C7 + Q2"),
        ("Q4: ability", "Benchmark frontier", "C1-C8"),
    ]
    for i, (head, body, source) in enumerate(labels):
        x = 0.2 + i * 3.0
        ax.add_patch(plt.Rectangle((x, 0.95), 2.6, 1.25, facecolor=colors[i], edgecolor="#334155", linewidth=1.1))
        ax.text(x + 1.3, 1.89, head, ha="center", va="center", weight="bold", fontsize=11)
        ax.text(x + 1.3, 1.51, body, ha="center", va="center", fontsize=8.5)
        ax.text(x + 1.3, 1.16, source, ha="center", va="center", fontsize=8, color="#475569")
        if i < 3:
            ax.annotate("", xy=(x + 2.99, 1.58), xytext=(x + 2.65, 1.58), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#334155"})
    ax.plot([1.5, 4.5], [0.67, 0.67], color="#b91c1c", lw=1.1, ls="--")
    ax.text(3.0, 0.37, "Q_A to Q_B scale: no shared anchor", ha="center", va="center", fontsize=8.5, color="#991b1b")
    ax.plot([7.5, 10.5], [0.67, 0.67], color="#b91c1c", lw=1.1, ls="--")
    ax.text(9.0, 0.37, "Loss to score: bridge RMSE 11.86", ha="center", va="center", fontsize=8.5, color="#991b1b")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "four_question_framework.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def mixture_rmse():
    names = ["Nested OOF\n1M", "Held-out\n1M", "Held-out\n60M", "Held-out\n1B"]
    v2 = np.array([0.3340, 0.2702, 1.5959, 2.7021])
    v3 = np.array([0.2522, 0.2082, 1.5614, 2.7357])
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})
    for idx, ax in enumerate(axes):
        subset = [0, 1] if idx == 0 else [2, 3]
        xs = np.arange(2)
        width = .31
        ax.bar(xs - width / 2, v2[subset], width, label="ALR Ridge V2", color="#64748b")
        ax.bar(xs + width / 2, v3[subset], width, label="CLR spline V3", color="#0f766e")
        ax.set_xticks(xs, [names[k] for k in subset])
        ax.set_ylabel("RMSE (Loss)")
        ax.set_ylim(0, max(v2[subset].max(), v3[subset].max()) * 1.20)
        ax.grid(axis="y", color="#e2e8f0", linewidth=.7)
        ax.set_axisbelow(True)
        for j, k in enumerate(subset):
            ax.text(j - width / 2, v2[k] + 0.015, f"{v2[k]:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(j + width / 2, v3[k] + 0.015, f"{v3[k]:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_title("In-scale mixture response")
    axes[1].set_title("Scale-transfer stress test")
    fig.tight_layout()
    fig.savefig(OUT / "q1_v2_v3_by_scale.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    framework()
    mixture_rmse()
