#!/usr/bin/env python3
"""Sinusoid vs step-like season vs LF AM, and nested T=12 reconstruction.

Shows why a nested-month loading vector is not a phenological calendar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jtech_palette import label_panels  # noqa: E402

from nested_cseof import demodulate  # noqa: E402

FIG = Path(__file__).resolve().parent / "figs"
FIG.mkdir(exist_ok=True)
CLR = ["#111111", "#c0392b", "#2471a3", "#1e8449", "#7d3c98"]


def sinusoid(t_year: np.ndarray, lag_month: float = 0.0) -> np.ndarray:
    """Peak in July when lag_month=0 (phase so max at month 7)."""
    # month 7 = 6.5/12 yr from Jan 1 if t=0 is 1 Jan
    return np.cos(2 * np.pi * (t_year - (6.5 - lag_month) / 12.0))


def step_season(t_year: np.ndarray, width_month: float = 0.7) -> np.ndarray:
    """Summer plateau, winter plateau; spring/autumn jumps (tanh)."""
    month = (t_year * 12.0) % 12.0
    # high from ~May (4.5) through ~Sep (8.5)
    w = width_month
    up = 0.5 * (1.0 + np.tanh((month - 4.2) / w))
    down = 0.5 * (1.0 + np.tanh((month - 9.0) / w))
    # wrap: month 0-4 still winter
    y = up - down
    y = 2.0 * y - 1.0
    return y


def fig_season_shapes() -> Path:
    n = 624
    t = np.arange(n) / 12.0
    month = (np.arange(12) + 1).astype(float)
    t12 = (month - 0.5) / 12.0
    sine = sinusoid(t12, 0.0)
    step = step_season(t12)
    # one-year nested reconstruction of a long step series
    y_step = step_season(t)
    d = demodulate(y_step, icyc=12, npts=6)
    rec = d["tcpy"]
    rec12 = rec[:12]
    rec12 = rec12 / (np.max(np.abs(rec12)) + 1e-12)
    step_n = step / (np.max(np.abs(step)) + 1e-12)
    sine_n = sine / (np.max(np.abs(sine)) + 1e-12)

    env = 1.0 + 0.55 * np.cos(2 * np.pi * t / 8.0)
    am = env * sinusoid(t, 0.0)

    fig, ax = plt.subplots(2, 2, figsize=(8.8, 6.4))
    ax[0, 0].plot(month, sine_n, color=CLR[0], lw=1.8, marker="o", ms=4, label="sinusoid")
    ax[0, 0].axvline(7.0, color="0.6", ls=":", lw=0.9)
    ax[0, 0].set_xlim(1, 12)
    ax[0, 0].set_xticks([1, 4, 7, 10, 12])
    ax[0, 0].set_xlabel("Month")
    ax[0, 0].set_ylabel("Normalized amplitude")
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(month, step_n, color=CLR[1], lw=1.8, marker="o", ms=4, label="step-like season")
    ax[0, 1].plot(month, sine_n, color=CLR[0], lw=1.2, ls="--", label="sinusoid (July peak)")
    ax[0, 1].axvline(4.2, color=CLR[1], ls=":", lw=0.9)
    ax[0, 1].set_xlim(1, 12)
    ax[0, 1].set_xticks([1, 4, 7, 10, 12])
    ax[0, 1].set_xlabel("Month")
    ax[0, 1].set_ylabel("Normalized amplitude")
    ax[0, 1].legend(fontsize=8)

    sl = slice(0, 96)
    ax[1, 0].plot(t[sl], sinusoid(t[sl], 0.0), color="0.7", lw=0.8, label="carrier")
    ax[1, 0].plot(t[sl], am[sl], color=CLR[2], lw=1.2, label=r"LF $\times$ sinusoid")
    ax[1, 0].plot(t[sl], env[sl], color=CLR[3], lw=1.4, label="envelope")
    ax[1, 0].set_xlim(0, 8)
    ax[1, 0].set_xlabel("Time (year)")
    ax[1, 0].set_ylabel("Amplitude")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(month, step_n, color=CLR[1], lw=1.6, marker="o", ms=4, label="true step")
    ax[1, 1].plot(month, rec12, color=CLR[2], lw=1.8, marker="s", ms=4, label=r"nested rec. $T=12$")
    ax[1, 1].plot(month, sine_n, color=CLR[0], lw=1.0, ls="--", label="sinusoid")
    pk = int(np.argmax(rec12)) + 1
    ax[1, 1].axvline(pk, color=CLR[2], ls=":", lw=0.9)
    ax[1, 1].set_xlim(1, 12)
    ax[1, 1].set_xticks([1, 4, 7, 10, 12])
    ax[1, 1].set_xlabel("Month")
    ax[1, 1].set_ylabel("Normalized amplitude")
    ax[1, 1].legend(fontsize=8)

    label_panels(
        ax,
        titles=[
            "Sinusoid",
            "Step-like season",
            r"LF $\times$ sinusoid",
            r"Nested $T=12$ reconstruction",
        ],
    )
    fig.tight_layout()
    p = FIG / "16_season_step_am_nested.png"
    fig.savefig(p, dpi=140)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


if __name__ == "__main__":
    print(fig_season_shapes())
