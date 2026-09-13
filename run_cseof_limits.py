#!/usr/bin/env python3
"""Diagnostic experiments for nested CSEOF limitations.

Reproduces the PC-level operator in Kim/Hamlington MATLAB and the synthetic
cases from cseof_integr.m (white, red, linear sum/product, cosine product).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jtech_palette import label_panels  # noqa: E402

from nested_cseof import (
    band_power,
    bloch_from_W,
    carrier_piece,
    demodulate,
    first_eof,
    harmonic_ls,
    periodogram,
    preprocess,
    second_eof_tcof,
    weight,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figs"
FIG.mkdir(exist_ok=True)

NTOT = 624
ICYC = 12
NPTS = 6
SEED = 7
N_ENS = 40

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
CLR = ["#111111", "#c0392b", "#2471a3", "#1e8449", "#7d3c98"]


def t_year(n=NTOT):
    return np.arange(n) / 12.0


def seasonal(n=NTOT):
    t = t_year(n)
    return np.cos(2 * np.pi * t) + 0.1 * np.cos(4 * np.pi * t) + 0.2 * np.sin(4 * np.pi * t)


def cases(rng: np.random.Generator) -> dict[str, np.ndarray]:
    t = t_year()
    g = seasonal()
    g = g / np.max(np.abs(g))
    white = rng.normal(0.0, 1.0, NTOT)
    red = t / (t.max() - t.min()) + rng.normal(0.0, 0.05, NTOT)
    return {
        "white": white,
        "red": red,
        "season": g,
        "linearsum": 0.5 * (t - t.mean()) / (t.max() - t.min()) * 2 + g,
        "linearprod": ((t - t.mean()) * g),
        "cosineprod": np.cos(2 * np.pi * t / 50.0) * g,
    }


def spec_metrics(y: np.ndarray) -> dict:
    f, s = periodogram(y)
    tot = band_power(f, s, f0=s.size, half=1e9) if False else float(np.trapezoid(s, f))
    p1 = band_power(f, s, 1.0, 0.15)
    p0 = band_power(f, s, 0.0, 0.25)
    return {"f": f, "s": s, "tot": tot, "p1": p1, "p0": p0, "frac1": p1 / tot if tot > 0 else np.nan}


def summarize_series(name: str, y: np.ndarray, prep: str = "raw") -> dict:
    y = preprocess(y, prep)
    d = demodulate(y)
    ls = harmonic_ls(y)
    m0 = spec_metrics(y)
    m_hat = spec_metrics(d["cf"][:, 1])
    rms = float(np.sqrt(np.mean(y**2)))
    rms_k1 = float(np.sqrt(np.mean(d["cf"][:, 1] ** 2 + d["sf"][:, 1] ** 2)))
    rec_err = float(np.sqrt(np.mean((y - d["tcpy"]) ** 2))) / (rms + 1e-12)
    ls_err = float(ls["rmse"]) / (rms + 1e-12)
    return {
        "name": name,
        "prep": prep,
        "e0": d["e0"],
        "ratio_tcof": d["ratio_tcof"],
        "ratio_tcpy": d["ratio_tcpy"],
        "frac1_raw": m0["frac1"],
        "frac1_k1": m_hat["frac1"],
        "rms_k1_over_rms": rms_k1 / (rms + 1e-12),
        "ls_ratio": ls["ratio"],
        "rec_rmse_over_rms": rec_err,
        "ls_rmse_over_rms": ls_err,
        "y": y,
        "d": d,
        "ls": ls,
        "m0": m0,
    }


def savefig(fig, stem: str) -> Path:
    png = FIG / f"{stem}.png"
    pdf = FIG / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def fig_kernel():
    icyc, npts, nwgt = 12, 6, 24
    wgts = weight(icyc, nwgt, 1)
    # two-sided window as used in integr (j = -MWGT..MWGT)
    t = np.arange(-nwgt, nwgt + 1)
    wfull = np.array([wgts[abs(j)] for j in t])
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for k, c in zip(range(4), CLR[1:]):
        wk = wfull * np.cos(2 * np.pi * k * t / icyc)
        ax[0].plot(t / icyc, wk, color=c, lw=1.4, label=rf"$k={k}$")
        n = wk.size
        fr = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / 12.0))
        sp = np.abs(np.fft.fftshift(np.fft.fft(wk))) / n
        ax[1].semilogy(fr, np.clip(sp, 1e-8, None), color=c, lw=1.4, label=rf"$k={k}$")
    ax[0].set_xlabel("Time (year)")
    ax[0].set_ylabel(r"$W(\sigma_k,t)$")
    ax[0].set_xlim(-2.1, 2.1)
    ax[1].set_xlabel("Frequency (cpy)")
    ax[1].set_ylabel(r"$|W(\sigma_k,\omega)|$")
    ax[1].set_xlim(-6, 6)
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    label_panels(ax, titles=["Time domain", r"$|W(\sigma_k,\omega)|$"])
    return savefig(fig, "01_kernel")


def fig_null(rows: dict):
    fig, ax = plt.subplots(2, 2, figsize=(8.6, 6.2))
    for col, key in enumerate(("red", "white")):
        r = rows[key]
        t = t_year()
        ax[0, col].plot(t, r["y"], color=CLR[0], lw=0.7, label=r"$S(t)$")
        for k, c in zip(range(4), CLR[1:]):
            ax[0, col].plot(t, r["d"]["cf"][:, k], color=c, lw=1.1, label=rf"$\hat S(\sigma_{k},t)$")
        ax[0, col].set_xlabel("Time (year)")
        ax[0, col].set_ylabel("Amplitude")
        ax[0, col].legend(fontsize=7, ncol=2)
        f, s = r["m0"]["f"], r["m0"]["s"]
        ax[1, col].semilogy(f, np.clip(s, 1e-12, None), color=CLR[0], lw=1.6, label=r"$Z(\omega)$")
        for k, c in zip(range(4), CLR[1:]):
            fk, sk = periodogram(r["d"]["cf"][:, k])
            ax[1, col].semilogy(fk, np.clip(sk, 1e-12, None), color=c, lw=1.1, label=rf"$\hat Z(\sigma_{k})$")
        ax[1, col].set_xlabel("Frequency (cpy)")
        ax[1, col].set_ylabel("Power (cpy$^{-1}$)")
        ax[1, col].set_xlim(0, 6)
        ax[1, col].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    label_panels(
        ax,
        titles=["Red, $S(t)$", "White, $S(t)$", "Red, spectra", "White, spectra"],
    )
    return savefig(fig, "02_null_red_white")


def fig_white_carrier(row: dict):
    """White noise: demodulate is a slow envelope; the carrier piece looks annual."""
    y = row["y"]
    cf = row["d"]["cf"]
    sf = row["d"]["sf"]
    t = t_year()
    bp1 = carrier_piece(cf, sf, 1, ICYC)
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
    sl = slice(0, 96)  # first 8 years
    ax[0].plot(t[sl], y[sl], color="#bbbbbb", lw=0.8, label=r"$S(t)$ white")
    ax[0].plot(t[sl], cf[sl, 1], color=CLR[1], lw=1.6, label=r"$\hat S(\sigma_1,t)$ envelope")
    ax[0].plot(t[sl], bp1[sl], color=CLR[2], lw=1.1, label=r"$\hat S(\sigma_1,t)\cos(\sigma_1 t)$")
    ax[0].set_xlabel("Time (year)")
    ax[0].set_ylabel("Amplitude")
    ax[0].legend(fontsize=7)
    f0, s0 = periodogram(y)
    f1, s1 = periodogram(cf[:, 1])
    fb, sb = periodogram(bp1)
    ax[1].semilogy(f0, np.clip(s0, 1e-8, None), color=CLR[0], lw=1.3, label=r"$Z(\omega)$ white")
    ax[1].semilogy(f1, np.clip(s1, 1e-8, None), color=CLR[1], lw=1.4, label=r"envelope $\hat Z(\sigma_1)$")
    ax[1].semilogy(fb, np.clip(sb, 1e-8, None), color=CLR[2], lw=1.4, label="carrier restored")
    ax[1].axvline(1.0, color=CLR[0], lw=0.6, ls=":")
    ax[1].set_xlim(0, 3)
    ax[1].set_xlabel("Frequency (cpy)")
    ax[1].set_ylabel("Power (cpy$^{-1}$)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    label_panels(ax, titles=[r"First eight years", r"Periodograms"])
    return savefig(fig, "08_white_carrier")


def fig_energy(table: list[dict]):
    names = [r["name"] for r in table]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    w = 0.25
    ax.bar(x - w, [r["ratio_tcof"] for r in table], w, color=CLR[1], label=r"$\sum TCOF^2 / \sum S^2$")
    ax.bar(x, [r["ratio_tcpy"] for r in table], w, color=CLR[2], label=r"$\sum TCPY^2 / \sum S^2$")
    ax.bar(x + w, [r["ls_ratio"] for r in table], w, color=CLR[3], label=r"LS harmonics $\sum \hat S^2 / \sum S^2$")
    ax.axhline(1.0, color=CLR[0], lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Energy ratio vs original")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return savefig(fig, "03_energy_ratio")


def fig_modulation(rows: dict):
    fig, ax = plt.subplots(2, 2, figsize=(8.6, 6.2))
    keys = ("season", "linearsum", "linearprod", "cosineprod")
    titles = ("Season only", "Linear sum", "Linear product", "Cosine product")
    for i, (key, title) in enumerate(zip(keys, titles)):
        a = ax[i // 2, i % 2]
        r = rows[key]
        f, s = r["m0"]["f"], r["m0"]["s"]
        a.semilogy(f, np.clip(s, 1e-12, None), color=CLR[0], lw=1.5, label="data")
        fl, sl = periodogram(r["ls"]["rec"])
        a.semilogy(fl, np.clip(sl, 1e-12, None), color="#e67e22", lw=1.3, ls="--", label="LS rec")
        for k, c in zip(range(3), (CLR[1], CLR[2], CLR[4])):
            fk, sk = periodogram(r["d"]["cf"][:, k])
            a.semilogy(fk, np.clip(sk, 1e-12, None), color=c, lw=1.0, label=rf"CSEOF $k={k}$")
        a.set_xlim(0, 4)
        a.set_xlabel("Frequency (cpy)")
        a.set_ylabel("Power (cpy$^{-1}$)")
        a.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    label_panels(ax, titles=titles)
    return savefig(fig, "04_modulation_vs_ls")


def fig_preprocess(white: np.ndarray):
    fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.2), sharey=True)
    for i, mode in enumerate(("raw", "demean", "detrend")):
        r = summarize_series("white", white, prep=mode)
        f, s = r["m0"]["f"], r["m0"]["s"]
        ax[i].semilogy(f, np.clip(s, 1e-12, None), color=CLR[0], lw=1.5, label="data")
        for k, c in zip(range(4), CLR[1:]):
            fk, sk = periodogram(r["d"]["cf"][:, k])
            ax[i].semilogy(fk, np.clip(sk, 1e-12, None), color=c, lw=1.0, label=rf"$k={k}$")
        ax[i].set_xlim(0, 6)
        ax[i].set_xlabel("Frequency (cpy)")
    ax[0].set_ylabel("Power (cpy$^{-1}$)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    label_panels(ax, titles=["raw", "demean", "detrend"])
    return savefig(fig, "05_white_preprocess")


def fig_ensemble(ens: dict):
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.3))
    ax[0].hist(ens["ratio_tcof"], bins=12, color=CLR[2], edgecolor=CLR[0], lw=0.4)
    ax[0].axvline(np.median(ens["ratio_tcof"]), color=CLR[1], lw=1.4, label="median")
    ax[0].set_xlabel(r"$\sum TCOF^2 / \sum S^2$")
    ax[0].set_ylabel("Count")
    ax[0].legend(fontsize=8)
    ax[1].hist(ens["rms_k1"], bins=12, color=CLR[3], edgecolor=CLR[0], lw=0.4)
    ax[1].axvline(np.median(ens["rms_k1"]), color=CLR[1], lw=1.4, label="median")
    ax[1].set_xlabel(r"rms of 1-cpy channel / rms$(S)$")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    label_panels(ax, titles=["TCOF energy ratio", r"$k=1$ rms / rms$(S)$"])
    return savefig(fig, "06_white_ensemble")


def fig_spacetime(rng: np.random.Generator):
    nspace, ntime, nkeep = 40, NTOT, 12
    D = rng.normal(0.0, 1.0, (nspace, ntime))
    D = D - D.mean(axis=1, keepdims=True)
    U, pcts, lam = first_eof(D, nkeep)
    tcof_list = []
    for m in range(nkeep):
        tcof_list.append(demodulate(pcts[:, m])["tcof"])
    pcs, W, evals, tvar, X = second_eof_tcof(tcof_list, ncor=ntime, nkeep=6)
    bloch = bloch_from_W(W[:, 0], nkeep, tcof_list[0].shape[1], ICYC, NPTS)
    spatial = U @ bloch
    fig, ax = plt.subplots(1, 3, figsize=(9.4, 3.3))
    ax[0].plot(lam[:nkeep] / lam.sum() * 100, "o-", color=CLR[0], ms=4)
    ax[0].set_xlabel("EOF mode")
    ax[0].set_ylabel("% variance")
    ax[1].plot(t_year(ntime), pcs[:, 0], color=CLR[2], lw=0.9)
    ax[1].set_xlabel("Time (year)")
    ax[1].set_ylabel("CSEOF PC1")
    im = ax[2].pcolormesh(np.arange(1, 13), np.arange(nspace), spatial, cmap="gray", shading="auto")
    ax[2].set_xlabel("Nested month")
    ax[2].set_ylabel("Grid index")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    label_panels(
        ax,
        titles=["EOF variance", "Leading nested PC", "Mode-1 Bloch map"],
    )
    frac = float(evals[0] / tvar) if tvar else np.nan
    return savefig(fig, "07_spacetime_white"), {
        "pc1_std": float(pcs[:, 0].std()),
        "mode1_frac": frac,
        "eof1_frac": float(lam[0] / lam.sum()),
    }


def jsonable(row: dict) -> dict:
    skip = {"y", "d", "ls", "m0"}
    out = {}
    for k, v in row.items():
        if k in skip:
            continue
        if isinstance(v, (float, np.floating)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main():
    rng = np.random.default_rng(SEED)
    C = cases(rng)
    rows = {k: summarize_series(k, v, "raw") for k, v in C.items()}

    table = [rows[k] for k in ("white", "red", "season", "linearsum", "linearprod", "cosineprod")]
    prep_rows = [summarize_series("white", C["white"], m) for m in ("raw", "demean", "detrend")]
    prep_rows += [summarize_series("red", C["red"], m) for m in ("raw", "demean", "detrend")]

    ens = {"ratio_tcof": [], "rms_k1": [], "ls_ratio": []}
    for i in range(N_ENS):
        w = np.random.default_rng(1000 + i).normal(0.0, 1.0, NTOT)
        r = summarize_series("white", w, "raw")
        ens["ratio_tcof"].append(r["ratio_tcof"])
        ens["rms_k1"].append(r["rms_k1_over_rms"])
        ens["ls_ratio"].append(r["ls_ratio"])
    ens_sum = {k: {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))} for k, v in ens.items()}

    paths = {
        "kernel": str(fig_kernel()),
        "null": str(fig_null(rows)),
        "carrier": str(fig_white_carrier(rows["white"])),
        "energy": str(fig_energy(table)),
        "mod": str(fig_modulation(rows)),
        "prep": str(fig_preprocess(C["white"])),
        "ens": str(fig_ensemble(ens)),
    }
    st_path, st_stats = fig_spacetime(rng)
    paths["spacetime"] = str(st_path)

    summary = {
        "ntot": NTOT,
        "icyc": ICYC,
        "npts": NPTS,
        "n_ens": N_ENS,
        "seed": SEED,
        "cases": [jsonable(r) for r in table],
        "preprocess": [jsonable(r) for r in prep_rows],
        "ensemble_white": ens_sum,
        "spacetime_white": st_stats,
        "figures": paths,
    }
    out = ROOT / "limits_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
