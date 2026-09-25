"""Export two static figures from the already computed optimization studies."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def p3() -> None:
    d = pd.read_csv(HERE / "p3_cap_profile.csv")
    d = d[d.budget == 1e24].sort_values("cap_factor", na_position="last")
    x = np.arange(len(d))
    labels = [f"{int(z)}x" if np.isfinite(z) else "free" for z in d.cap_factor]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.plot(x, d.loss, "o-", color="#155e75", label="Optimal loss")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Multiplier of B1 parameter and token limits")
    ax.set_ylabel("Conditional optimal loss", color="#155e75")
    ax.tick_params(axis="y", labelcolor="#155e75")
    ay = ax.twinx()
    ay.plot(x, 100 * d.budget_ratio, "s--", color="#b45309", label="Budget used")
    ay.set_ylabel("Budget used (%)", color="#b45309")
    ay.tick_params(axis="y", labelcolor="#b45309")
    ay.set_ylim(0, 110)
    ax.set_title(r"P3 cap sensitivity at $10^{24}$ FLOPs")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(FIG / "p3_cap_sensitivity.png", dpi=180)
    plt.close(fig)


def p4() -> None:
    d = pd.read_csv(HERE / "q4/frontier_variants.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for subset, color, name in [
        ("strict_W1", "#155e75", "Strict open weights"),
        ("inherited_C2_aligned_cutoff", "#b45309", "C2 inherited sensitivity"),
    ]:
        z = d[(d.subset == subset) & (d.window_quarters == 1) & (d.metric == "p95")]
        z = z.sort_values("end_quarter")
        ax.plot(z.end_quarter, z.value, "o-", color=color, label=name)
        for row in z.itertuples():
            offset = (3, -16) if subset == "inherited_C2_aligned_cutoff" else (3, 6)
            ax.annotate(f"n={row.n}", (row.end_quarter, row.value), xytext=offset,
                        textcoords="offset points", fontsize=8, color=color)
    ax.set_ylabel("Quarterly p95 benchmark score")
    ax.set_xlabel("Submission quarter")
    ax.set_title("P4 frontier direction depends on sample definition")
    ax.grid(alpha=.2)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "p4_frontier_sample_sensitivity.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p3()
    p4()
    print(FIG)
