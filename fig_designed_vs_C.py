#!/usr/bin/env python3
"""Designed cyclostationary field: nested CSEOF vs eigenanalysis of C.

Also writes the white vs designed space–time nested figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jtech_palette import label_panels  # noqa: E402

from nested_cseof import (  # noqa: E402
    align_sign,
    cseof_of_C,
    nested_two_stage,
    pattern_corr,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figs"
FIG.mkdir(exist_ok=True)

NSPACE = 40
NTIME = 624
ICYC = 12
NKEEP = 12
SEED_WHITE = 7
SEED_SEASON = 11
NOISE = 0.20
CLR = ["#111111", "#c0392b", "#2471a3", "#1e8449"]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
    }
)


def generating_B(nspace: int = NSPACE, icyc: int = ICYC) -> np.ndarray:
    """Gaussian bump whose center walks along the grid through nested month."""
    x = np.arange(nspace, dtype=float)
    h = np.arange(icyc, dtype=float)
    x0 = 8.0 + 24.0 * h / (icyc - 1)
    w = 4.5
    return np.exp(-0.5 * ((x[:, None] - x0[None, :]) / w) ** 2)


def designed_field(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B = generating_B()
    t = np.arange(NTIME)
    a = 1.0 + 0.25 * np.cos(2.0 * np.pi * t / (ICYC * 8.0))
    D = B[:, t % ICYC] * a[None, :]
    D = D + NOISE * rng.normal(size=D.shape)
    D = D - D.mean(axis=1, keepdims=True)
    return D, B, a


def white_field(rng: np.random.Generator) -> np.ndarray:
    D = rng.normal(0.0, 1.0, (NSPACE, NTIME))
    return D - D.mean(axis=1, keepdims=True)


def pmesh(ax, Z, *, normalize: bool = True, **kw):
    Z = np.asarray(Z, dtype=float)
    if normalize:
        Z = Z / (np.max(np.abs(Z)) + 1e-12)
    kw.setdefault("cmap", "RdBu_r")
    kw.setdefault("vmin", -1.0)
    kw.setdefault("vmax", 1.0)
    im = ax.pcolormesh(
        np.arange(1, ICYC + 1),
        np.arange(NSPACE),
        Z,
        shading="auto",
        **kw,
    )
    ax.set_xlabel("Nested month")
    ax.set_ylabel("Grid index")
    return im


def savefig(fig, stem: str) -> Path:
    png = FIG / f"{stem}.png"
    fig.savefig(png)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)
    return png


def fig_nested_vs_C(D: np.ndarray, B_true: np.ndarray, a: np.ndarray) -> tuple[Path, dict]:
    C = cseof_of_C(D, icyc=ICYC)
    nest = nested_two_stage(D, nkeep=NKEEP, icyc=ICYC)
    B_C = align_sign(C["B"], B_true)
    B_mean = align_sign(C["mean_map"], B_true)
    B_N = align_sign(nest["spatial"], B_true)
    if pattern_corr(C["B"], B_true) < 0:
        pc_C = -C["pc_year"]
    else:
        pc_C = C["pc_year"]
    if pattern_corr(nest["spatial"], B_true) < 0:
        pc_N = -nest["pcs"][:, 0]
    else:
        pc_N = nest["pcs"][:, 0]
    rec_C = C["rec"]
    rec_N = np.empty_like(D)
    for t in range(NTIME):
        rec_N[:, t] = pc_N[t] * B_N[:, t % ICYC]
    rms = float(np.sqrt(np.mean(D**2)))
    stats = {
        "r_true_C": pattern_corr(B_true, B_C),
        "r_true_mean": pattern_corr(B_true, B_mean),
        "r_true_nested": pattern_corr(B_true, B_N),
        "r_C_nested": pattern_corr(B_C, B_N),
        "frac_C": C["frac"],
        "frac_nested": nest["mode1_frac"],
        "rmse_C_over_rms": float(np.sqrt(np.mean((D - rec_C) ** 2)) / (rms + 1e-12)),
        "rmse_nested_over_rms": float(np.sqrt(np.mean((D - rec_N) ** 2)) / (rms + 1e-12)),
        "ratio_tcof_pc1": nest["ratio_tcof_pc1"],
        "noise": NOISE,
        "nspace": NSPACE,
        "ntime": NTIME,
    }
    t_year = np.arange(NTIME) / 12.0
    n_years = C["n_years"]
    t_year_C = 0.5 + np.arange(n_years)
    fig = plt.figure(figsize=(9.4, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.045])
    ax = np.empty((2, 2), dtype=object)
    ax[0, 0] = fig.add_subplot(gs[0, 0])
    ax[0, 1] = fig.add_subplot(gs[0, 1])
    ax[1, 0] = fig.add_subplot(gs[1, 0])
    ax[1, 1] = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 2])
    im = pmesh(ax[0, 0], B_true)
    pmesh(ax[0, 1], B_C)
    pmesh(ax[1, 0], B_N)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.set_label("Normalized amplitude")
    sl = slice(0, 96)
    ax[1, 1].plot(t_year[sl], a[sl], color=CLR[0], lw=1.4, label=r"generating $a(t)$")
    ax[1, 1].plot(
        t_year_C[t_year_C <= 8],
        pc_C[: int(np.sum(t_year_C <= 8))] / (np.max(np.abs(pc_C)) + 1e-12) * np.max(np.abs(a)),
        color=CLR[1],
        lw=1.2,
        marker="o",
        ms=3,
        label="C yearly PC (scaled)",
    )
    ax[1, 1].plot(
        t_year[sl],
        pc_N[sl] / (np.max(np.abs(pc_N)) + 1e-12) * np.max(np.abs(a)),
        color=CLR[2],
        lw=1.1,
        label="nested PC (scaled)",
    )
    ax[1, 1].set_xlim(0, 8)
    ax[1, 1].set_xlabel("Time (year)")
    ax[1, 1].set_ylabel("Amplitude")
    ax[1, 1].legend(fontsize=7)
    label_panels(
        ax,
        titles=[
            r"Generating $B(x,h)$",
            r"Leading evec of $C$ (anomaly)",
            "Nested CSEOF Bloch",
            r"$a(t)$; scaled coefficients",
        ],
    )
    path = savefig(fig, "17_nested_vs_C")
    return path, stats


def fig_white_and_season(Dw: np.ndarray, Ds: np.ndarray) -> Path:
    nest_w = nested_two_stage(Dw, nkeep=NKEEP, icyc=ICYC)
    nest_s = nested_two_stage(Ds, nkeep=NKEEP, icyc=ICYC)
    B_true = generating_B()
    spatial_s = align_sign(nest_s["spatial"], B_true)
    pc_s = nest_s["pcs"][:, 0]
    if pattern_corr(nest_s["spatial"], B_true) < 0:
        pc_s = -pc_s
    lam_w = nest_w["lam"]
    lam_s = nest_s["lam"]
    t_year = np.arange(NTIME) / 12.0
    fig = plt.figure(figsize=(10.8, 5.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.045])
    ax = np.empty((2, 3), dtype=object)
    for i in range(2):
        for j in range(3):
            ax[i, j] = fig.add_subplot(gs[i, j])
    cax_w = fig.add_subplot(gs[0, 3])
    cax_s = fig.add_subplot(gs[1, 3])
    ax[0, 0].plot(lam_w[:NKEEP] / lam_w.sum() * 100, "o-", color=CLR[0], ms=4)
    ax[0, 0].set_xlabel("EOF mode")
    ax[0, 0].set_ylabel("% variance")
    ax[0, 1].plot(t_year, nest_w["pcs"][:, 0], color=CLR[2], lw=0.8)
    ax[0, 1].set_xlabel("Time (year)")
    ax[0, 1].set_ylabel("CSEOF PC1")
    im_w = pmesh(ax[0, 2], nest_w["spatial"])
    fig.colorbar(im_w, cax=cax_w)
    ax[1, 0].plot(lam_s[:NKEEP] / lam_s.sum() * 100, "o-", color=CLR[0], ms=4)
    ax[1, 0].set_xlabel("EOF mode")
    ax[1, 0].set_ylabel("% variance")
    ax[1, 1].plot(t_year, pc_s, color=CLR[2], lw=0.8)
    ax[1, 1].set_xlabel("Time (year)")
    ax[1, 1].set_ylabel("CSEOF PC1")
    im_s = pmesh(ax[1, 2], spatial_s)
    fig.colorbar(im_s, cax=cax_s)
    label_panels(
        ax,
        titles=[
            "White, EOF variance",
            "White, nested PC",
            "White, Bloch",
            "Season, EOF variance",
            "Season, nested PC",
            "Season, Bloch",
        ],
    )
    return savefig(fig, "18_spacetime_white_season")


def main() -> None:
    rng_w = np.random.default_rng(SEED_WHITE)
    rng_s = np.random.default_rng(SEED_SEASON)
    Dw = white_field(rng_w)
    Ds, B, a = designed_field(rng_s)
    p_c, stats = fig_nested_vs_C(Ds, B, a)
    p_st = fig_white_and_season(Dw, Ds)
    stats["figures"] = {"nested_vs_C": str(p_c), "white_season": str(p_st)}
    out = ROOT / "designed_vs_C_summary.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
