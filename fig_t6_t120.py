#!/usr/bin/env python3
"""Nested kernel at T=6 months and T=120 months (10 yr) vs T=12, monthly sampling."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jtech_palette import label_panels  # noqa: E402

from nested_cseof import carrier_piece, demodulate, periodogram, weight  # noqa: E402

FIG = Path(__file__).resolve().parent / "figs"
FIG.mkdir(exist_ok=True)

NTOT = 624
SEED = 7
DT_CPY = 1.0 / 12.0  # monthly samples → frequency in cpy
CLR = ["#111111", "#c0392b", "#2471a3", "#1e8449", "#7d3c98"]


def kernel_spectrum(icyc: int, k: int):
    nwgt = 2 * icyc
    wgts = weight(icyc, nwgt, 1)
    t = np.arange(-nwgt, nwgt + 1)
    wfull = np.array([wgts[abs(j)] for j in t])
    wk = wfull * np.cos(2.0 * np.pi * k * t / icyc)
    n = wk.size
    fr = np.fft.fftshift(np.fft.fftfreq(n, d=DT_CPY))
    sp = np.abs(np.fft.fftshift(np.fft.fft(wk))) / n
    return fr, sp


def ar1_red(n=NTOT, phi: float = 0.92) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    e = rng.normal(0.0, 1.0, n)
    y = np.empty(n)
    y[0] = e[0]
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y


def fig_kernel_t6_t12_t120() -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.55), sharey=True)
    specs = (
        (6, 3.2, r"$T=6$ months; $k=1$ at 2 cpy"),
        (12, 3.2, r"$T=12$ months; $k=1$ at 1 cpy"),
        (120, 0.55, r"$T=120$ months; $k=1$ at 0.1 cpy"),
    )
    for col, (icyc, xmax, title) in enumerate(specs):
        for k, c in zip(range(4), CLR[1:]):
            fr, sp = kernel_spectrum(icyc, k)
            ax[col].semilogy(
                fr, np.clip(sp, 1e-8, None), color=c, lw=1.4, label=rf"$k={k}$"
            )
        ax[col].axvline(12.0 / icyc, color="0.45", ls="--", lw=0.9, zorder=0)
        ax[col].axvline(1.0, color="0.35", ls=":", lw=0.9, zorder=0)
        ax[col].axvline(0.0, color="0.75", ls=":", lw=0.8, zorder=0)
        ax[col].set_xlim(-xmax, xmax)
        ax[col].set_xlabel("Frequency (cpy)")
        ax[col].set_ylim(1e-5, 2e-1)
        ax[col].legend(fontsize=7, ncol=2)
    ax[0].set_ylabel(r"$|W(\sigma_k,\omega)|$")
    label_panels(ax, titles=[s[2] for s in specs])
    fig.tight_layout()
    p = FIG / "19_kernel_T6_T12_T120.png"
    fig.savefig(p, dpi=150)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def fig_carve_t6_t120() -> Path:
    y = ar1_red()
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=False)
    cases = (
        (6, 3, 3.2, r"$T=6$: lines at 2, 4, 6 cpy"),
        (120, 3, 0.55, r"$T=120$: lines at 0.1, 0.2, 0.3 cpy"),
    )
    f0, s0 = periodogram(y)
    for col, (icyc, npts, xmax, title) in enumerate(cases):
        d = demodulate(y, icyc=icyc, npts=npts)
        ax[col].semilogy(
            f0, np.clip(s0, 1e-8, None), color=CLR[0], lw=1.5, label=r"$S$ (AR1)"
        )
        for k, c in zip((1, 2, 3), CLR[1:4]):
            piece = carrier_piece(d["cf"], d["sf"], k, icyc)
            fk, sk = periodogram(piece)
            ax[col].semilogy(
                fk,
                np.clip(sk, 1e-8, None),
                color=c,
                lw=1.3,
                label=rf"$k={k}$ $\times$ carrier",
            )
            ax[col].axvline(k * 12.0 / icyc, color="0.75", ls=":", lw=0.8, zorder=0)
        ax[col].axvline(1.0, color="0.35", ls=":", lw=0.9, zorder=0)
        ax[col].set_xlim(0, xmax)
        ax[col].set_xlabel("Frequency (cpy)")
        ax[col].legend(fontsize=7)
    ax[0].set_ylabel("Power (cpy$^{-1}$)")
    label_panels(ax, titles=[c[3] for c in cases])
    fig.tight_layout()
    p = FIG / "20_carve_T6_T120.png"
    fig.savefig(p, dpi=150)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def fig_paper_t6_t120() -> Path:
    """Condensed 2×2 for JTECH: kernel and AR(1) isolation at T=6 and T=120."""
    y = ar1_red()
    fig, ax = plt.subplots(2, 2, figsize=(8.8, 6.4))
    xlim_k = {0: 3.2, 1: 0.55}
    for col, icyc in enumerate((6, 120)):
        for k, c in zip(range(4), CLR[1:]):
            fr, sp = kernel_spectrum(icyc, k)
            ax[0, col].semilogy(
                fr, np.clip(sp, 1e-8, None), color=c, lw=1.5, label=rf"$k={k}$"
            )
        ax[0, col].axvline(12.0 / icyc, color="0.45", ls="--", lw=0.9, zorder=0)
        ax[0, col].axvline(1.0, color="0.35", ls=":", lw=0.9, zorder=0)
        ax[0, col].axvline(0.0, color="0.75", ls=":", lw=0.8, zorder=0)
        ax[0, col].set_xlim(-xlim_k[col], xlim_k[col])
        ax[0, col].set_ylim(1e-5, 2e-1)
        ax[0, col].legend(fontsize=7, ncol=2)
        ax[0, col].set_xlabel("Frequency (cpy)")
    ax[0, 0].set_ylabel(r"$|W(\sigma_k,\omega)|$")

    xlim_c = {0: 3.2, 1: 0.55}
    f0, s0 = periodogram(y)
    for col, (icyc, npts) in enumerate(((6, 3), (120, 3))):
        d = demodulate(y, icyc=icyc, npts=npts)
        ax[1, col].semilogy(
            f0, np.clip(s0, 1e-8, None), color=CLR[0], lw=1.5, label=r"$S$ (AR1)"
        )
        for k, c in zip((1, 2, 3), CLR[1:4]):
            piece = carrier_piece(d["cf"], d["sf"], k, icyc)
            fk, sk = periodogram(piece)
            ax[1, col].semilogy(
                fk,
                np.clip(sk, 1e-8, None),
                color=c,
                lw=1.3,
                label=rf"$k={k}$ $\times$ carrier",
            )
            ax[1, col].axvline(k * 12.0 / icyc, color="0.75", ls=":", lw=0.8, zorder=0)
        ax[1, col].axvline(1.0, color="0.35", ls=":", lw=0.9, zorder=0)
        ax[1, col].set_xlim(0, xlim_c[col])
        ax[1, col].set_xlabel("Frequency (cpy)")
        ax[1, col].legend(fontsize=7)
    ax[1, 0].set_ylabel("Power (cpy$^{-1}$)")
    label_panels(
        ax,
        titles=[
            r"Kernel, $T=6$ months",
            r"Kernel, $T=120$ months",
            r"AR(1), $T=6$ months",
            r"AR(1), $T=120$ months",
        ],
    )
    fig.tight_layout()
    p = FIG / "10_kernel_T6_T120.png"
    fig.savefig(p, dpi=160)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def print_table() -> None:
    n = NTOT
    print(
        f"{'T (month)':>10} {'T (yr)':>7} {'k=1 (cpy)':>10} "
        f"{'~width':>8} {'2T win':>8} {'wrap %':>8} {'NPTS':>6} {'NYR':>7} {'1 cpy k':>8}"
    )
    for icyc in (6, 12, 24, 120):
        f1 = 12.0 / icyc
        width = 12.0 / (2.0 * icyc)
        nwgt = 2 * icyc
        wrap = 100.0 * (2.0 * nwgt) / n
        npts = icyc // 2
        nyr = n / icyc
        k_ann = icyc / 12.0
        k_ann_s = f"{k_ann:.0f}" if abs(k_ann - round(k_ann)) < 1e-9 else "—"
        print(
            f"{icyc:10d} {icyc/12:7.1f} {f1:10.3f} {width:8.3f} "
            f"{nwgt:8d} {wrap:8.1f} {npts:6d} {nyr:7.2f} {k_ann_s:>8}"
        )


if __name__ == "__main__":
    print_table()
    for p in (fig_kernel_t6_t12_t120(), fig_carve_t6_t120(), fig_paper_t6_t120()):
        print(p)
