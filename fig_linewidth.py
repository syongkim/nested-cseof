#!/usr/bin/env python3
"""Periodograms at 1 cpy: finite-record line vs AM envelope vs white vs red.

Discussion 1: a peak no broader than a pure sinusoid of length N is a line,
not an envelope to recover.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jtech_palette import label_panels  # noqa: E402

from nested_cseof import periodogram  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figs"
FIG.mkdir(exist_ok=True)

NTOT = 624
SEED = 7
CLR = ["#111111", "#c0392b", "#2471a3", "#1e8449"]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    }
)


def fwhm_cpy(freq: np.ndarray, spec: np.ndarray, f0: float = 1.0) -> float:
    """Half-power width of the nearest peak to f0 (cpy)."""
    i0 = int(np.argmin(np.abs(freq - f0)))
    peak = spec[i0]
    if peak <= 0:
        return float("nan")
    half = 0.5 * peak
    left = i0
    while left > 0 and spec[left] >= half:
        left -= 1
    right = i0
    while right < spec.size - 1 and spec[right] >= half:
        right += 1
    return float(freq[right] - freq[left])


def main() -> None:
    t = np.arange(NTOT) / 12.0
    rng = np.random.default_rng(SEED)
    season = np.cos(2.0 * np.pi * t)
    env = 1.0 + 0.25 * np.cos(2.0 * np.pi * t / 8.0)
    am = env * season
    white = rng.normal(0.0, 1.0, NTOT)
    red = t / (t.max() - t.min()) + rng.normal(0.0, 0.05, NTOT)

    series = {
        "season": season,
        "am": am,
        "white": white,
        "red": red,
    }
    specs = {k: periodogram(v) for k, v in series.items()}
    t_rec = NTOT / 12.0
    df = 1.0 / t_rec
    stats = {
        "N": NTOT,
        "T_rec_yr": t_rec,
        "df_cpy": df,
        "fwhm_season": fwhm_cpy(*specs["season"]),
        "fwhm_am": fwhm_cpy(*specs["am"]),
        "fwhm_white": fwhm_cpy(*specs["white"]),
        "fwhm_red": fwhm_cpy(*specs["red"]),
    }

    fig, ax = plt.subplots(2, 2, figsize=(8.8, 6.2), sharex=True)
    panels = [
        (ax[0, 0], "season", CLR[0], r"Unit sinusoid at $1$ cpy"),
        (ax[0, 1], "am", CLR[1], r"$8$-yr envelope $\times$ sinusoid"),
        (ax[1, 0], "white", CLR[2], r"White noise"),
        (ax[1, 1], "red", CLR[3], r"Linear trend plus weak white"),
    ]
    for axi, key, color, _lab in panels:
        f, s = specs[key]
        axi.semilogy(f, np.clip(s, 1e-8, None), color=color, lw=1.3)
        axi.axvline(1.0, color="0.55", ls=":", lw=0.9)
        axi.axvspan(1.0 - 0.5 * df, 1.0 + 0.5 * df, color="0.85", zorder=0)
        axi.set_xlim(0.0, 2.5)
        axi.set_ylabel(r"Power (cpy$^{-1}$)")
    ax[1, 0].set_xlabel("Frequency (cpy)")
    ax[1, 1].set_xlabel("Frequency (cpy)")
    fig.tight_layout()
    label_panels(
        ax,
        titles=[
            r"Sinusoid, $N=624$",
            r"AM, $8$-yr envelope",
            "White",
            "Red (trend)",
        ],
    )
    png = FIG / "19_linewidth_line_vs_envelope.png"
    fig.savefig(png)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)
    out = ROOT / "linewidth_summary.json"
    stats["figure"] = str(png)
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("wrote", png)


if __name__ == "__main__":
    main()
