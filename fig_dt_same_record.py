#!/usr/bin/env python3
"""Same high-rate record, different averaging: nested CSEOF at the same physical T.

360-day years so that a 30-day month (T=12) and a 6-day mean (T=60) both
impose a 1-yr nested period. 6-day is the integer analogue of weekly.
13-day analogue: 12-hourly T=26 vs daily T=13 (Na et al. 2012 nested period).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jtech_palette import label_panels  # noqa: E402

from nested_cseof import carrier_piece, demodulate, periodogram, weight  # noqa: E402

FIG = Path(__file__).resolve().parent / "figs"
FIG.mkdir(exist_ok=True)
CLR = ["#111111", "#c0392b", "#2471a3", "#1e8449", "#7d3c98"]
SEED = 7
NYR = 52
DAY_YR = 360  # 12 x 30-day months = 60 x 6-day blocks


def ar1(n: int, phi: float, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 1.0, n)
    y = np.empty(n)
    y[0] = e[0]
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y


def block_mean(y: np.ndarray, w: int) -> np.ndarray:
    n = (y.size // w) * w
    return y[:n].reshape(-1, w).mean(axis=1)


def fig_month30_vs_hexad6() -> Path:
    """Daily red + 1 cpy + 7 cpy; 30-d T=12 vs 6-d T=60."""
    n_day = NYR * DAY_YR
    t = np.arange(n_day, dtype=float)
    # Daily AR(1) whose 30-day mean is near the paper's monthly phi=0.92.
    y = ar1(n_day, phi=0.92 ** (1.0 / 30.0))
    y = y + 2.2 * np.cos(2.0 * np.pi * t / DAY_YR)
    y = y + 1.4 * np.cos(2.0 * np.pi * 7.0 * t / DAY_YR)

    month = block_mean(y, 30)
    hexad = block_mean(y, 6)
    specs = (
        (month, 12, 6, 1.0 / 12.0, r"30-day mean, $T=12$"),
        (hexad, 60, 30, 1.0 / 60.0, r"6-day mean, $T=60$"),
    )
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=True)
    for col, (ser, icyc, npts, dt, title) in enumerate(specs):
        f0, s0 = periodogram(ser, dt_year=dt)
        ax[col].semilogy(
            f0, np.clip(s0, 1e-8, None), color=CLR[0], lw=1.5, label=r"averaged $S$"
        )
        d = demodulate(ser, icyc=icyc, npts=npts)
        for k, c in zip((1, 2, 7), (CLR[1], CLR[2], CLR[3])):
            if k > npts:
                continue
            piece = carrier_piece(d["cf"], d["sf"], k, icyc)
            fk, sk = periodogram(piece, dt_year=dt)
            ax[col].semilogy(
                fk,
                np.clip(sk, 1e-8, None),
                color=c,
                lw=1.3,
                label=rf"$k={k}$ $\times$ carrier",
            )
            ax[col].axvline(float(k), color="0.7", ls=":", lw=0.8, zorder=0)
        ax[col].axvline(6.0, color="0.45", ls="--", lw=0.8, zorder=0)
        ax[col].set_xlim(0, 8.5)
        ax[col].set_xlabel("Frequency (cpy)")
        ax[col].legend(fontsize=8, loc="lower left")
        _ = title
    ax[0].text(
        7.25,
        3e1,
        r"no $k=7$ box",
        ha="center",
        va="center",
        fontsize=9,
        color="0.35",
    )
    ax[0].set_ylabel(r"Power (cpy$^{-1}$)")
    label_panels(ax, titles=[s[4] for s in specs])
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 1.0))
    cap_lines = [
        r"Same daily series (AR(1) + 1 cpy + 7 cpy; 52 $\times$ 360-d years).",
        r"(a) 30-day mean, nested $T=12$ (1 yr). Nyquist 6 cpy (dashed); no nested box at 7 cpy.",
        r"(b) 6-day mean, nested $T=60$ (1 yr), Nyquist 30 cpy. Green: $k=7\times$ carrier at 7 cpy.",
        r"Dotted lines at 1, 2, 7 cpy. Red $k=1$ is the shared annual box.",
    ]
    fig.text(
        0.02,
        0.01,
        "\n".join(cap_lines),
        ha="left",
        va="bottom",
        fontsize=8.5,
        linespacing=1.35,
    )
    p = FIG / "23_same_record_month12_hexad60.png"
    fig.savefig(p, dpi=150)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def white(n: int, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, n)


def fig_white_month_vs_hexad() -> Path:
    """Same daily white noise; 30-d T=12 vs 6-d T=60. No planted lines."""
    n_day = NYR * DAY_YR
    y = white(n_day)
    month = block_mean(y, 30)
    hexad = block_mean(y, 6)
    specs = (
        (month, 12, 6, 1.0 / 12.0, r"30-day mean, $T=12$"),
        (hexad, 60, 30, 1.0 / 60.0, r"6-day mean, $T=60$"),
    )
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=True)
    for col, (ser, icyc, npts, dt, title) in enumerate(specs):
        f0, s0 = periodogram(ser, dt_year=dt)
        ax[col].semilogy(
            f0, np.clip(s0, 1e-8, None), color=CLR[0], lw=1.5, label=r"averaged $S$"
        )
        d = demodulate(ser, icyc=icyc, npts=npts)
        for k, c in zip((1, 2, 7), (CLR[1], CLR[2], CLR[3])):
            if k > npts:
                continue
            piece = carrier_piece(d["cf"], d["sf"], k, icyc)
            fk, sk = periodogram(piece, dt_year=dt)
            ax[col].semilogy(
                fk,
                np.clip(sk, 1e-8, None),
                color=c,
                lw=1.3,
                label=rf"$k={k}$ $\times$ carrier",
            )
            ax[col].axvline(float(k), color="0.7", ls=":", lw=0.8, zorder=0)
        ax[col].axvline(6.0, color="0.45", ls="--", lw=0.8, zorder=0)
        ax[col].set_xlim(0, 8.5)
        ax[col].set_xlabel("Frequency (cpy)")
        ax[col].legend(fontsize=8, loc="lower left")
        _ = title
    ax[0].text(
        7.25,
        3e-2,
        r"no $k=7$ box",
        ha="center",
        va="center",
        fontsize=9,
        color="0.35",
    )
    ax[0].set_ylabel(r"Power (cpy$^{-1}$)")
    label_panels(ax, titles=[s[4] for s in specs])
    fig.tight_layout(rect=(0.0, 0.24, 1.0, 1.0))
    cap_lines = [
        r"Same daily white noise (no planted line; 52 $\times$ 360-d years). "
        r"Block averaging makes $S$ a sinc$^2$ spectrum, not white.",
        r"(a) 30-day mean, nested $T=12$ (1 yr). Nyquist 6 cpy (dashed); $k=1,2$ are carved from the continuum.",
        r"(b) 6-day mean, nested $T=60$ (1 yr). Green $k=7$ is still isolated at 7 cpy with no 7 cpy signal in the record.",
        r"Dotted lines at 1, 2, 7 cpy.",
    ]
    fig.text(
        0.02,
        0.01,
        "\n".join(cap_lines),
        ha="left",
        va="bottom",
        fontsize=8.5,
        linespacing=1.35,
    )
    p = FIG / "27_white_month12_hexad60.png"
    fig.savefig(p, dpi=150)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def fig_white_envelope() -> Path:
    """White-noise k=1 envelopes after 30-d vs 6-d averaging of the same daily series."""
    n_day = NYR * DAY_YR
    y = white(n_day)
    month = block_mean(y, 30)
    hexad = block_mean(y, 6)
    dm = demodulate(month, icyc=12, npts=6)
    dh = demodulate(hexad, icyc=60, npts=30)
    env_m = dm["cf"][:, 1]
    env_h30 = block_mean(dh["cf"][:, 1], 5)
    n = min(env_m.size, env_h30.size)
    env_m = env_m[:n]
    env_h30 = env_h30[:n]
    c = float(np.corrcoef(env_m, env_h30)[0, 1])
    rms_m = float(np.sqrt(np.mean(env_m**2)))
    rms_h = float(np.sqrt(np.mean(env_h30**2)))
    yr = np.arange(n) / 12.0

    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.6))
    ax[0].plot(yr, env_m, color=CLR[1], lw=1.2, label=r"30-day $T=12$, $k=1$")
    ax[0].plot(
        yr, env_h30, color=CLR[2], lw=1.1, label=r"6-day $T=60$, $k=1$ (5-block mean)"
    )
    ax[0].set_xlim(0, 20)
    ax[0].set_xlabel("Time (year)")
    ax[0].set_ylabel("Envelope")
    ax[0].legend(fontsize=8)
    ax[1].scatter(env_m, env_h30, s=8, c=CLR[2], alpha=0.7, linewidths=0)
    lim = float(np.max(np.abs(np.concatenate([env_m, env_h30]))))
    ax[1].plot([-lim, lim], [-lim, lim], color="0.5", ls="--", lw=0.8)
    ax[1].set_xlabel(r"30-day $T=12$ $k=1$")
    ax[1].set_ylabel(r"6-day $T=60$ $k=1$ (monthly mean)")
    ax[1].set_aspect("equal", adjustable="box")
    label_panels(
        ax,
        titles=[
            rf"white $k=1$ envelopes; $r={c:.2f}$",
            rf"rms ratio (6-day/30-day) $= {rms_h / rms_m:.2f}$",
        ],
    )
    fig.tight_layout()
    p = FIG / "28_white_k1_envelope.png"
    fig.savefig(p, dpi=150)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    print(f"white k=1 envelope corr (monthly vs 6-day→month) r={c:.3f}")
    print(f"white k=1 rms monthly={rms_m:.4f}  6-day-to-month={rms_h:.4f}")
    return p


def fig_envelope_mismatch() -> Path:
    """k=1 envelopes after 30-d vs 6-d averaging of the same daily series."""
    n_day = NYR * DAY_YR
    t = np.arange(n_day, dtype=float)
    y = ar1(n_day, phi=0.92 ** (1.0 / 30.0))
    y = y + 2.2 * np.cos(2.0 * np.pi * t / DAY_YR)
    y = y + 1.4 * np.cos(2.0 * np.pi * 7.0 * t / DAY_YR)
    month = block_mean(y, 30)
    hexad = block_mean(y, 6)
    dm = demodulate(month, icyc=12, npts=6)
    dh = demodulate(hexad, icyc=60, npts=30)
    env_m = dm["cf"][:, 1]
    env_h = dh["cf"][:, 1]
    env_h30 = block_mean(env_h, 5)  # five 6-day blocks = one 30-day month
    n = min(env_m.size, env_h30.size)
    env_m = env_m[:n]
    env_h30 = env_h30[:n]
    c = float(np.corrcoef(env_m, env_h30)[0, 1])
    rms_m = float(np.sqrt(np.mean(env_m**2)))
    rms_h = float(np.sqrt(np.mean(env_h30**2)))
    yr = np.arange(n) / 12.0

    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.6))
    ax[0].plot(yr, env_m, color=CLR[1], lw=1.2, label=r"30-day $T=12$, $k=1$")
    ax[0].plot(yr, env_h30, color=CLR[2], lw=1.1, label=r"6-day $T=60$, $k=1$ (5-block mean)")
    ax[0].set_xlim(0, 20)
    ax[0].set_xlabel("Time (year)")
    ax[0].set_ylabel("Envelope")
    ax[0].legend(fontsize=8)
    ax[1].scatter(env_m, env_h30, s=8, c=CLR[2], alpha=0.7, linewidths=0)
    lim = float(np.max(np.abs(np.concatenate([env_m, env_h30]))))
    ax[1].plot([-lim, lim], [-lim, lim], color="0.5", ls="--", lw=0.8)
    ax[1].set_xlabel(r"30-day $T=12$ $k=1$")
    ax[1].set_ylabel(r"6-day $T=60$ $k=1$ (monthly mean)")
    ax[1].set_aspect("equal", adjustable="box")
    label_panels(
        ax,
        titles=[
            rf"$k=1$ envelopes; $r={c:.2f}$",
            rf"rms ratio (6-day/30-day) $= {rms_h / rms_m:.2f}$",
        ],
    )
    fig.tight_layout()
    p = FIG / "24_k1_envelope_month_vs_hexad.png"
    fig.savefig(p, dpi=150)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    print(f"k=1 envelope corr (monthly vs 6-day→month) r={c:.3f}")
    print(f"k=1 rms monthly={rms_m:.4f}  6-day-to-month={rms_h:.4f}")
    return p


def fig_13day_dt() -> Path:
    """Same 12-hourly series: nested T=13 d as daily ICYC=13 vs 12-hourly ICYC=26."""
    n_cyc = 56
    n_12h = n_cyc * 26  # 728 d at 12 h
    t_h = np.arange(n_12h, dtype=float)
    # 13-day cosine + red continuum (12-hourly AR1).
    y12 = ar1(n_12h, phi=0.85, seed=11)
    y12 = y12 + 2.0 * np.cos(2.0 * np.pi * t_h / 26.0)
    daily = block_mean(y12, 2)
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.7), sharey=True)
    specs = (
        (daily, 13, 6, 1.0, r"Daily mean, $T=13$ d; 13 nested maps"),
        (y12, 26, 13, 0.5, r"12-hourly, $T=26$; 26 nested maps"),
    )
    for col, (ser, icyc, npts, dt_day, title) in enumerate(specs):
        f0, s0 = periodogram(ser, dt_year=dt_day / 365.25)
        # Plot vs cycles per day for this figure.
        f_cpd = f0 / 365.25
        ax[col].semilogy(
            f_cpd, np.clip(s0, 1e-8, None), color=CLR[0], lw=1.5, label=r"$S$"
        )
        d = demodulate(ser, icyc=icyc, npts=npts)
        for k, c in zip((1, 2, 3), CLR[1:4]):
            piece = carrier_piece(d["cf"], d["sf"], k, icyc)
            fk, sk = periodogram(piece, dt_year=dt_day / 365.25)
            ax[col].semilogy(
                fk / 365.25,
                np.clip(sk, 1e-8, None),
                color=c,
                lw=1.3,
                label=rf"$k={k}$ $\times$ carrier",
            )
            ax[col].axvline(k / 13.0, color="0.7", ls=":", lw=0.8, zorder=0)
        ax[col].set_xlim(0, 0.35)
        ax[col].set_xlabel("Frequency (cpd)")
        ax[col].legend(fontsize=7)
        _ = title
    ax[0].set_ylabel(r"Power (cpy$^{-1}$)")
    label_panels(ax, titles=[s[4] for s in specs])
    fig.tight_layout()
    p = FIG / "25_kernel_13d_daily_vs_12h.png"
    fig.savefig(p, dpi=150)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def psd_onesided_cpd(y: np.ndarray, dt_day: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD vs cycles per day, scaled so ∫ S df_cpd = variance."""
    y = np.asarray(y, dtype=float) - np.mean(y)
    n = y.size
    yf = np.fft.rfft(y)
    freq = np.fft.rfftfreq(n, d=dt_day)
    df = freq[1] - freq[0] if n > 1 else 1.0
    spec = (np.abs(yf) ** 2) / (n**2) / df
    spec[1:-1] *= 2.0
    return freq, spec


def fig_13day_psd_nyquist() -> Path:
    """True PSD of the [12] parent series: Nyquist vs density height."""
    n_cyc = 56
    n_12h = n_cyc * 26
    t_h = np.arange(n_12h, dtype=float)
    y12 = ar1(n_12h, phi=0.85, seed=11)
    y12 = y12 + 2.0 * np.cos(2.0 * np.pi * t_h / 26.0)
    daily = block_mean(y12, 2)
    series = (
        (y12, 0.5, "12-hourly", CLR[2], "-"),
        (daily, 1.0, "Daily mean", CLR[0], "-"),
    )
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.7), sharey=True)
    for ser, dt_day, lab, c, ls in series:
        f, s = psd_onesided_cpd(ser, dt_day)
        for a in ax:
            a.semilogy(
                f,
                np.clip(s, 1e-8, None),
                color=c,
                lw=1.8 if dt_day == 1.0 else 1.2,
                ls=ls,
                label=lab,
            )
    for a in ax:
        a.axvline(1.0 / 13.0, color="0.7", ls=":", lw=0.8, zorder=0)
        a.axvline(0.5, color="0.35", ls="--", lw=0.9, zorder=0)
        a.axvline(1.0, color="0.35", ls="--", lw=0.9, zorder=0)
        a.set_xlabel("Frequency (cpd)")
    ax[0].set_xlim(0, 1.05)
    ax[1].set_xlim(0, 0.35)
    ax[0].set_ylabel(r"PSD (cpd$^{-1}$)")
    ax[0].legend(fontsize=7)
    label_panels(
        ax,
        titles=[
            r"One-sided PSD, to 12-hourly Nyquist",
            r"Same PSD, [12] window",
        ],
    )
    fig.tight_layout()
    p = FIG / "26_psd_13d_nyquist_density.png"
    fig.savefig(p, dpi=150)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def _carve_panel(ax, ser, icyc: int, npts: int, dt: float, show_k7: bool) -> None:
    f0, s0 = periodogram(ser, dt_year=dt)
    ax.semilogy(f0, np.clip(s0, 1e-8, None), color=CLR[0], lw=1.5, label=r"averaged $S$")
    d = demodulate(ser, icyc=icyc, npts=npts)
    ks = (1, 2, 7) if show_k7 else (1, 2)
    cols = (CLR[1], CLR[2], CLR[3]) if show_k7 else (CLR[1], CLR[2])
    for k, c in zip(ks, cols):
        piece = carrier_piece(d["cf"], d["sf"], k, icyc)
        fk, sk = periodogram(piece, dt_year=dt)
        ax.semilogy(
            fk,
            np.clip(sk, 1e-8, None),
            color=c,
            lw=1.3,
            label=rf"$k={k}$ $\times$ carrier",
        )
        ax.axvline(float(k), color="0.7", ls=":", lw=0.8, zorder=0)
    if not show_k7:
        ax.axvline(7.0, color="0.7", ls=":", lw=0.8, zorder=0)
    ax.axvline(6.0, color="0.45", ls="--", lw=0.8, zorder=0)
    ax.set_xlim(0, 8.5)
    ax.set_xlabel("Frequency (cpy)")


def _carve_panel_13d(ax, ser, icyc: int, npts: int, dt_day: float, show_k7: bool) -> None:
    f0, s0 = psd_onesided_cpd(ser, dt_day)
    ax.semilogy(f0, np.clip(s0, 1e-8, None), color=CLR[0], lw=1.5, label=r"averaged $S$")
    d = demodulate(ser, icyc=icyc, npts=npts)
    ks = (1, 2, 7) if show_k7 else (1, 2)
    cols = (CLR[1], CLR[2], CLR[3]) if show_k7 else (CLR[1], CLR[2])
    for k, c in zip(ks, cols):
        piece = carrier_piece(d["cf"], d["sf"], k, icyc)
        fk, sk = psd_onesided_cpd(piece, dt_day)
        ax.semilogy(
            fk,
            np.clip(sk, 1e-8, None),
            color=c,
            lw=1.3,
            label=rf"$k={k}$ $\times$ carrier",
        )
        ax.axvline(k / 13.0, color="0.7", ls=":", lw=0.8, zorder=0)
    if not show_k7:
        ax.axvline(7.0 / 13.0, color="0.7", ls=":", lw=0.8, zorder=0)
    ax.axvline(0.5, color="0.45", ls="--", lw=0.8, zorder=0)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Frequency (cpd)")


def fig_paper_13d() -> Path:
    """Paper figure: [12] condensed with Nyquist shown. Daily T=13 vs 12-hourly T=26."""
    n_cyc = 56
    n_12h = n_cyc * 26
    t_h = np.arange(n_12h, dtype=float)
    y_line = ar1(n_12h, phi=0.85, seed=11)
    y_line = y_line + 2.0 * np.cos(2.0 * np.pi * t_h / 26.0)
    y_line = y_line + 1.4 * np.cos(2.0 * np.pi * 7.0 * t_h / 26.0)
    y_w = white(n_12h, seed=11)
    fig, ax = plt.subplots(2, 2, figsize=(8.8, 6.6), sharex=True)
    for row, y in enumerate((y_line, y_w)):
        daily = block_mean(y, 2)
        _carve_panel_13d(ax[row, 0], daily, 13, 6, 1.0, show_k7=False)
        _carve_panel_13d(ax[row, 1], y, 26, 13, 0.5, show_k7=True)
        ax[row, 0].set_ylabel(r"PSD (cpd$^{-1}$)")
        ax[row, 0].text(
            0.72,
            3e1 if row == 0 else 8e-1,
            r"no $k=7$ box",
            ha="center",
            va="center",
            fontsize=9,
            color="0.35",
        )
    ax[0, 1].legend(fontsize=7, loc="lower left")
    ax[1, 1].legend(fontsize=7, loc="lower left")
    ax[0, 1].axvline(1.0, color="0.45", ls="--", lw=0.8, zorder=0)
    ax[1, 1].axvline(1.0, color="0.45", ls="--", lw=0.8, zorder=0)
    label_panels(
        ax,
        titles=[
            r"Planted $1/13$ and $7/13$ cpd, daily $T=13$",
            r"Planted $1/13$ and $7/13$ cpd, 12-hourly $T=26$",
            r"White, daily $T=13$",
            r"White, 12-hourly $T=26$",
        ],
    )
    fig.tight_layout()
    p = FIG / "12_dt_T13d_daily13_12h26.png"
    fig.savefig(p, dpi=150)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def fig_paper_dt_t1yr() -> Path:
    """Paper figure C: [10] planted + [14] white, one 2x2."""
    n_day = NYR * DAY_YR
    t = np.arange(n_day, dtype=float)
    y_line = ar1(n_day, phi=0.92 ** (1.0 / 30.0))
    y_line = y_line + 2.2 * np.cos(2.0 * np.pi * t / DAY_YR)
    y_line = y_line + 1.4 * np.cos(2.0 * np.pi * 7.0 * t / DAY_YR)
    y_w = white(n_day)
    fig, ax = plt.subplots(2, 2, figsize=(8.8, 6.6), sharex=True)
    for row, y in enumerate((y_line, y_w)):
        month = block_mean(y, 30)
        hexad = block_mean(y, 6)
        _carve_panel(ax[row, 0], month, 12, 6, 1.0 / 12.0, show_k7=False)
        _carve_panel(ax[row, 1], hexad, 60, 30, 1.0 / 60.0, show_k7=True)
        ax[row, 0].set_ylabel(r"Power (cpy$^{-1}$)")
        ax[row, 0].text(
            7.25,
            3e1 if row == 0 else 3e0,
            r"no $k=7$ box",
            ha="center",
            va="center",
            fontsize=9,
            color="0.35",
        )
    ax[0, 1].legend(fontsize=7, loc="lower left")
    ax[1, 1].legend(fontsize=7, loc="lower left")
    label_panels(
        ax,
        titles=[
            r"Planted 1 and 7 cpy, 30-day $T=12$",
            r"Planted 1 and 7 cpy, 6-day $T=60$",
            r"White, 30-day $T=12$",
            r"White, 6-day $T=60$",
        ],
    )
    fig.tight_layout()
    p = FIG / "11_dt_T1yr_month12_hexad60.png"
    fig.savefig(p, dpi=150)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def _kernel_mag(icyc: int, k: int, dt_year: float) -> tuple[np.ndarray, np.ndarray]:
    nwgt = 2 * icyc
    wgts = weight(icyc, nwgt, 1)
    t = np.arange(-nwgt, nwgt + 1)
    wfull = np.array([wgts[abs(int(j))] for j in t])
    wk = wfull * np.cos(2.0 * np.pi * k * t / icyc)
    n = wk.size
    fr = np.fft.fftshift(np.fft.fftfreq(n, d=dt_year))
    sp = np.abs(np.fft.fftshift(np.fft.fft(wk))) / n
    return fr, sp


def _k1_envelopes(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    month = block_mean(y, 30)
    hexad = block_mean(y, 6)
    env_m = demodulate(month, icyc=12, npts=6)["cf"][:, 1]
    env_h30 = block_mean(demodulate(hexad, icyc=60, npts=30)["cf"][:, 1], 5)
    n = min(env_m.size, env_h30.size)
    env_m = env_m[:n]
    env_h30 = env_h30[:n]
    c = float(np.corrcoef(env_m, env_h30)[0, 1])
    return env_m, env_h30, c


def fig_paper_dt_kernel_envelope() -> Path:
    """Paper companion to fig:dt: kernels (ICYC) and collapsed k=1 envelopes."""
    n_day = NYR * DAY_YR
    t = np.arange(n_day, dtype=float)
    y_line = ar1(n_day, phi=0.92 ** (1.0 / 30.0))
    y_line = y_line + 2.2 * np.cos(2.0 * np.pi * t / DAY_YR)
    y_line = y_line + 1.4 * np.cos(2.0 * np.pi * 7.0 * t / DAY_YR)
    env_m, env_h, r_line = _k1_envelopes(y_line)
    env_mw, env_hw, r_w = _k1_envelopes(white(n_day))
    yr = np.arange(env_m.size) / 12.0

    fig, ax = plt.subplots(2, 2, figsize=(8.8, 6.6))
    specs = (
        (12, 1.0 / 12.0, (1, 2), (CLR[1], CLR[2]), False),
        (60, 1.0 / 60.0, (1, 2, 7), (CLR[1], CLR[2], CLR[3]), True),
    )
    for col, (icyc, dt, ks, cols, show_k7) in enumerate(specs):
        a = ax[0, col]
        for k, c in zip(ks, cols):
            fr, sp = _kernel_mag(icyc, k, dt)
            a.semilogy(
                fr, np.clip(sp, 1e-8, None), color=c, lw=1.5, label=rf"$k={k}$"
            )
        a.axvline(6.0, color="0.45", ls="--", lw=0.8, zorder=0)
        for fmark in (1.0, 2.0, 7.0):
            a.axvline(fmark, color="0.7", ls=":", lw=0.8, zorder=0)
        a.set_xlim(0, 8.5)
        a.set_ylim(1e-5, 4e-1)
        a.set_xlabel("Frequency (cpy)")
        a.legend(fontsize=7, loc="upper right")
        if not show_k7:
            a.text(
                7.25,
                4e-3,
                r"no $k=7$ box",
                ha="center",
                va="center",
                fontsize=8,
                color="0.35",
            )
    ax[0, 0].set_ylabel(r"$|W(\sigma_k,\omega)|$")

    pairs = (
        (env_m, env_h, r_line, r"planted $k=1$"),
        (env_mw, env_hw, r_w, r"white $k=1$"),
    )
    for col, (em, eh, r, _lab) in enumerate(pairs):
        a = ax[1, col]
        a.plot(yr, em, color=CLR[1], lw=1.2, label=r"30-day $T=12$")
        a.plot(yr, eh, color=CLR[2], lw=1.1, label=r"6-day $T=60$, 5-block mean")
        a.set_xlim(0, 20)
        a.set_xlabel("Time (year)")
        a.legend(fontsize=7, loc="upper right")
        _ = r
    ax[1, 0].set_ylabel(r"$k=1$ coefficient")

    label_panels(
        ax,
        titles=[
            r"30-day mean, $T=12$ kernel",
            r"6-day mean, $T=60$ kernel",
            rf"planted $k=1$ envelopes, $r={r_line:.2f}$",
            rf"white $k=1$ envelopes, $r={r_w:.2f}$",
        ],
    )
    fig.tight_layout()
    p = FIG / "12_dt_kernel_envelope.png"
    fig.savefig(p, dpi=150)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    print(f"paper kernel+envelope r_planted={r_line:.3f}  r_white={r_w:.3f}")
    return p


if __name__ == "__main__":
    for fn in (
        fig_month30_vs_hexad6,
        fig_envelope_mismatch,
        fig_white_month_vs_hexad,
        fig_white_envelope,
        fig_13day_dt,
        fig_13day_psd_nyquist,
        fig_paper_dt_t1yr,
        fig_paper_13d,
        fig_paper_dt_kernel_envelope,
    ):
        print(fn())
