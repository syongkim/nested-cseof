#!/usr/bin/env python3
"""T=12 vs T=24 nested kernels and red-continuum carving (Cheon 2021 T=24)."""
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
CLR = ["#111111", "#c0392b", "#2471a3", "#1e8449", "#7d3c98"]


def kernel_spectrum(icyc: int, k: int):
    nwgt = 2 * icyc
    wgts = weight(icyc, nwgt, 1)
    t = np.arange(-nwgt, nwgt + 1)
    wfull = np.array([wgts[abs(j)] for j in t])
    wk = wfull * np.cos(2 * np.pi * k * t / icyc)
    n = wk.size
    fr = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / 12.0))
    sp = np.abs(np.fft.fftshift(np.fft.fft(wk))) / n
    return fr, sp


def fig_kernel_t12_t24() -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.5), sharey=True)
    for col, icyc in enumerate((12, 24)):
        for k, c in zip(range(4), CLR[1:]):
            fr, sp = kernel_spectrum(icyc, k)
            ax[col].semilogy(
                fr, np.clip(sp, 1e-8, None), color=c, lw=1.5, label=rf"$k={k}$"
            )
        f1 = 12.0 / icyc
        ax[col].axvline(f1, color="0.45", ls="--", lw=0.9, zorder=0)
        ax[col].axvline(0.0, color="0.75", ls=":", lw=0.8, zorder=0)
        ax[col].set_xlim(-3.2, 3.2)
        ax[col].set_xlabel("Frequency (cpy)")
        ax[col].set_ylim(1e-5, 2e-1)
    ax[0].set_ylabel(r"$|W(\sigma_k,\omega)|$")
    ax[0].legend(fontsize=8, ncol=2)
    ax[1].legend(fontsize=8, ncol=2)
    label_panels(
        ax,
        titles=[
            r"$T=12$ months; $k=1$ at 1 cpy",
            r"$T=24$ months; $k=1$ at 0.5 cpy",
        ],
    )
    fig.tight_layout()
    p = FIG / "13_kernel_T12_T24.png"
    fig.savefig(p, dpi=140)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def ar1_red(n=NTOT, phi: float = 0.92) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    e = rng.normal(0.0, 1.0, n)
    y = np.empty(n)
    y[0] = e[0]
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y


def fig_carve_red() -> Path:
    y = ar1_red()
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=True)
    f0, s0 = periodogram(y)
    for col, (icyc, npts) in enumerate(((12, 6), (24, 12))):
        d = demodulate(y, icyc=icyc, npts=npts)
        ax[col].semilogy(
            f0, np.clip(s0, 1e-8, None), color=CLR[0], lw=1.6, label=r"$S$ (AR1)"
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
        ax[col].set_xlim(0, 3.2)
        ax[col].set_xlabel("Frequency (cpy)")
        ax[col].legend(fontsize=8)
    ax[0].set_ylabel("Power (cpy$^{-1}$)")
    label_panels(
        ax,
        titles=[
            r"$T=12$: carved lines at 1, 2, 3 cpy",
            r"$T=24$: carved lines at 0.5, 1, 1.5 cpy",
        ],
    )
    fig.tight_layout()
    p = FIG / "14_carve_red_T12_T24.png"
    fig.savefig(p, dpi=140)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def fig_white_carrier_t24() -> Path:
    rng = np.random.default_rng(SEED)
    y = rng.normal(0.0, 1.0, NTOT)
    d = demodulate(y, icyc=24, npts=12)
    t = np.arange(NTOT) / 12.0
    k = 1
    frq = 2.0 * np.pi * k / 24.0
    tt = np.arange(NTOT, dtype=float)
    # MATLAB TCPY: extra sqrt(2) on packed cos/sin → factor 2 on unpacked
    piece = 2.0 * (
        d["cf"][:, k] * np.cos(frq * tt) + d["sf"][:, k] * np.sin(frq * tt)
    )
    f_y, s_y = periodogram(y)
    f_e, s_e = periodogram(d["cf"][:, k])
    f_p, s_p = periodogram(piece)
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.6))
    ax[0].plot(t, y, color="0.75", lw=0.5, label=r"white $S$")
    ax[0].plot(t, d["cf"][:, 1], color=CLR[2], lw=1.2, label=r"$k=1$ envelope")
    ax[0].plot(t, piece, color=CLR[1], lw=0.9, label=r"$k=1$ $\times$ 2-yr carrier")
    ax[0].set_xlim(0, 20)
    ax[0].set_xlabel("Time (year)")
    ax[0].set_ylabel("Amplitude")
    ax[0].legend(fontsize=8)
    ax[1].semilogy(f_y, np.clip(s_y, 1e-6, None), color="0.55", lw=1.1, label=r"white $S$")
    ax[1].semilogy(
        f_e, np.clip(s_e, 1e-6, None), color=CLR[2], lw=1.4, label=r"$k=1$ envelope"
    )
    ax[1].semilogy(
        f_p, np.clip(s_p, 1e-6, None), color=CLR[1], lw=1.4, label=r"after $\times$ carrier"
    )
    ax[1].axvline(0.5, color="0.45", ls="--", lw=0.9)
    ax[1].set_xlim(0, 3.2)
    ax[1].set_xlabel("Frequency (cpy)")
    ax[1].set_ylabel("Power (cpy$^{-1}$)")
    ax[1].legend(fontsize=8)
    label_panels(
        ax,
        titles=[
            r"White input, $T=24$ months",
            r"Envelope is slow; 0.5 cpy is the carrier",
        ],
    )
    fig.tight_layout()
    p = FIG / "15_white_T24_carrier.png"
    fig.savefig(p, dpi=140)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def fig_paper_t12_t24() -> Path:
    """Condensed 2×2 for the JTECH figure: kernel and AR(1) isolation at T=12, 24."""
    y = ar1_red()
    fig, ax = plt.subplots(2, 2, figsize=(8.8, 6.4))
    for col, icyc in enumerate((12, 24)):
        for k, c in zip(range(4), CLR[1:]):
            fr, sp = kernel_spectrum(icyc, k)
            ax[0, col].semilogy(
                fr, np.clip(sp, 1e-8, None), color=c, lw=1.5, label=rf"$k={k}$"
            )
        ax[0, col].axvline(12.0 / icyc, color="0.45", ls="--", lw=0.9, zorder=0)
        ax[0, col].axvline(0.0, color="0.75", ls=":", lw=0.8, zorder=0)
        ax[0, col].set_xlim(-3.2, 3.2)
        ax[0, col].set_ylim(1e-5, 2e-1)
        ax[0, col].legend(fontsize=7, ncol=2)
    ax[0, 0].set_ylabel(r"$|W(\sigma_k,\omega)|$")
    ax[0, 0].set_xlabel("Frequency (cpy)")
    ax[0, 1].set_xlabel("Frequency (cpy)")

    for col, (icyc, npts) in enumerate(((12, 6), (24, 12))):
        d = demodulate(y, icyc=icyc, npts=npts)
        f0, s0 = periodogram(y)
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
        ax[1, col].set_xlim(0, 3.2)
        ax[1, col].set_xlabel("Frequency (cpy)")
        ax[1, col].legend(fontsize=7)
    ax[1, 0].set_ylabel("Power (cpy$^{-1}$)")
    label_panels(
        ax,
        titles=[
            r"Kernel, $T=12$ months",
            r"Kernel, $T=24$ months",
            r"AR(1), $T=12$ months",
            r"AR(1), $T=24$ months",
        ],
    )
    fig.tight_layout()
    p = FIG / "09_kernel_T12_T24.png"
    fig.savefig(p, dpi=160)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


if __name__ == "__main__":
    paths = [
        fig_kernel_t12_t24(),
        fig_carve_red(),
        fig_white_carrier_t24(),
        fig_paper_t12_t24(),
    ]
    for p in paths:
        print(p)
