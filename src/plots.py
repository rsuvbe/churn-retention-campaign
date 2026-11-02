"""Shared plotting style so every figure in the report looks consistent."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.config import path

# One accent for "churn / risk", one neutral for "retained / baseline".
RISK = "#C8102E"
NEUTRAL = "#4A5A6A"
ACCENT = "#1F77B4"
GOOD = "#2E8B57"
MUTED = "#B0B8C0"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (8, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.frameon": False,
    })


def save(fig, name: str) -> Path:
    """Save a figure into reports/figures/<name>.png."""
    out = path("figures") / f"{name}.png"
    fig.savefig(out)
    return out


def save_table(df, name: str) -> Path:
    """Save a result table into reports/tables/<name>.csv."""
    out = path("tables") / f"{name}.csv"
    df.to_csv(out, index=False)
    return out
