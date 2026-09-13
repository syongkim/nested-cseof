"""Nested CSEOF operators matching Kim/Hamlington MATLAB (cseof/weight.m, integr.m, cseof.m).

Indexing follows the MATLAB 1-based loops, including edge wrapping by ICYC
and the sqrt(2) packing of sine/cosine channels.
"""
from __future__ import annotations

import numpy as np


def weight(icyc: int, nwgt: int, nsub: int = 1) -> np.ndarray:
    """Modified sinc window. MATLAB weight.m (NPTS unused)."""
    n = nwgt * nsub + 1
    wgts = np.empty(n, dtype=float)
    wgts[0] = 1.0 / icyc
    for i in range(1, n):
        T = (i + 1) / nsub
        wgts[i] = np.sin(np.pi * T / icyc) / (np.pi * T)
    wgts /= nsub
    return wgts


def _sample_index(i0: np.ndarray, j: int, ntot: int, icyc: int, nsub: int) -> np.ndarray:
    """MATLAB integr.m sample index (returned 0-based)."""
    J1 = (i0 + 1) + j / nsub
    out = np.empty(i0.shape, dtype=int)
    left = J1 < 1
    right = J1 >= ntot
    mid = ~left & ~right
    if np.any(left):
        J1l = J1[left]
        JJ = np.mod(J1l + 2 * icyc - 1, icyc) + 1
        out[left] = np.clip(JJ.astype(int) - 1, 0, ntot - 1)
    if np.any(right):
        J1r = J1[right]
        JJ = ntot - np.mod(ntot - J1r + 2 * icyc, icyc)
        out[right] = np.clip(JJ.astype(int) - 1, 0, ntot - 1)
    if np.any(mid):
        if nsub != 1:
            raise NotImplementedError("nsub!=1 interpolation is unused in the distributed run")
        out[mid] = J1[mid].astype(int) - 1
    return out


def integr(
    tser: np.ndarray,
    wgts: np.ndarray,
    icyc: int,
    npts: int,
    nwgt: int,
    nsub: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Windowed harmonic demodulation. MATLAB integr.m.

    Returns CF, SF with shape (ntot, npts+1) for frequencies k = 0 .. npts.
    """
    tser = np.asarray(tser, dtype=float).ravel()
    ntot = tser.size
    mwgt = nwgt * nsub
    cf = np.zeros((ntot, npts + 1))
    sf = np.zeros((ntot, npts + 1))
    i0 = np.arange(ntot)
    jvals = np.arange(-mwgt, mwgt + 1)
    for k in range(npts + 1):
        frq = 2.0 * np.pi * k / icyc
        acc_c = np.zeros(ntot)
        acc_s = np.zeros(ntot)
        for j in jvals:
            idx = _sample_index(i0, int(j), ntot, icyc, nsub)
            tp = i0 + j / nsub
            val = tser[idx]
            w = wgts[abs(int(j))]
            acc_c += w * val * np.cos(frq * tp)
            acc_s += -w * val * np.sin(frq * tp)
        cf[:, k] = acc_c
        sf[:, k] = acc_s
    return cf, sf


def pack_tcof(cf: np.ndarray, sf: np.ndarray, icyc: int, npts: int) -> np.ndarray:
    """Pack cosine/sine coefficients as in cseof.m (sqrt(2) on non-Nyquist)."""
    ntot = cf.shape[0]
    mpts = 2 * npts
    if 2 * npts != icyc:
        mpts = 2 * npts + 1
    tcof = np.zeros((ntot, mpts))
    tcof[:, 0] = cf[:, 0]
    for k_m in range(2, mpts + 1):
        k = k_m - 1
        KK = k_m // 2
        if k_m % 2 == 0:
            tcof[:, k] = np.sqrt(2.0) * cf[:, KK]
        else:
            tcof[:, k] = -np.sqrt(2.0) * sf[:, KK]
    if mpts == icyc:
        tcof[:, mpts - 1] = cf[:, npts]
    return tcof


def reconstruct_tcpy(tcof: np.ndarray, icyc: int, npts: int) -> np.ndarray:
    """Local nested-cycle reconstruction from TCOF (commented TCPY loop in cseof.m)."""
    ntot, mpts = tcof.shape
    tcpy = tcof[:, 0].copy()
    t = np.arange(ntot, dtype=float)
    for k_m in range(2, mpts + 1):
        k = k_m - 1
        fact = 1.0 if k_m == icyc else np.sqrt(2.0)
        KK = k_m // 2
        frq = 2.0 * np.pi * KK / icyc
        if k_m % 2 == 0:
            tcpy += fact * tcof[:, k] * np.cos(frq * t)
        else:
            tcpy += fact * tcof[:, k] * np.sin(frq * t)
    return tcpy


def demodulate(tser: np.ndarray, icyc: int = 12, npts: int = 6) -> dict:
    """Full nested demodulation of one series (the PC-level CSEOF operator)."""
    nwgt = 2 * icyc
    wgts = weight(icyc, nwgt, nsub=1)
    cf, sf = integr(tser, wgts, icyc, npts, nwgt, nsub=1)
    tcof = pack_tcof(cf, sf, icyc, npts)
    tcpy = reconstruct_tcpy(tcof, icyc, npts)
    e0 = float(np.sum(tser**2))
    e_tcof = float(np.sum(tcof**2))
    e_tcpy = float(np.sum(tcpy**2))
    return {
        "wgts": wgts,
        "cf": cf,
        "sf": sf,
        "tcof": tcof,
        "tcpy": tcpy,
        "e0": e0,
        "e_tcof": e_tcof,
        "e_tcpy": e_tcpy,
        "ratio_tcof": e_tcof / e0 if e0 > 0 else np.nan,
        "ratio_tcpy": e_tcpy / e0 if e0 > 0 else np.nan,
    }


def harmonic_ls(tser: np.ndarray, icyc: int = 12, npts: int = 6) -> dict:
    """Least-squares fit at nested harmonics (same K=12 bases as the CSEOF pack)."""
    ntot = tser.size
    t = np.arange(ntot, dtype=float)
    cols = [np.ones(ntot)]
    for k in range(1, npts):
        cols.append(np.cos(2 * np.pi * k * t / icyc))
        cols.append(np.sin(2 * np.pi * k * t / icyc))
    cols.append(np.cos(2 * np.pi * npts * t / icyc))
    G = np.column_stack(cols)
    m, *_ = np.linalg.lstsq(G, tser, rcond=None)
    rec = G @ m
    e0 = float(np.sum(tser**2))
    e_rec = float(np.sum(rec**2))
    resid = tser - rec
    return {
        "G": G,
        "m": m,
        "rec": rec,
        "resid": resid,
        "e0": e0,
        "e_rec": e_rec,
        "ratio": e_rec / e0 if e0 > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean(resid**2))),
    }


def periodogram(y: np.ndarray, dt_year: float = 1.0 / 12.0):
    """One-sided power spectrum vs frequency in cycles per year."""
    y = np.asarray(y, dtype=float) - np.mean(y)
    n = y.size
    yf = np.fft.rfft(y)
    freq = np.fft.rfftfreq(n, d=dt_year)
    df = freq[1] - freq[0] if n > 1 else 1.0
    spec = (np.abs(yf) ** 2) / n / df
    spec[1:-1] *= 2.0
    return freq, spec


def band_power(freq: np.ndarray, spec: np.ndarray, f0: float, half: float = 0.15) -> float:
    m = (freq >= f0 - half) & (freq <= f0 + half)
    if not np.any(m):
        return 0.0
    df = freq[1] - freq[0]
    return float(np.sum(spec[m]) * df)


def preprocess(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    t = np.arange(y.size, dtype=float)
    if mode == "raw":
        return y
    if mode == "demean":
        return y - y.mean()
    if mode == "detrend":
        p = np.polyfit(t, y, 1)
        return y - np.polyval(p, t)
    raise ValueError(mode)


def first_eof(D: np.ndarray, nkeep: int):
    """D is space x time. Temporal-covariance EOF as in eigenx.m (no /N)."""
    cov = D.T @ D
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    nkeep = min(nkeep, D.shape[1], D.shape[0])
    V = evecs[:, :nkeep]
    lam = np.clip(evals[:nkeep], 0, None)
    std = np.sqrt(lam)
    pcts = V * std[np.newaxis, :]
    U = D @ V
    nrm = np.linalg.norm(U, axis=0)
    nrm[nrm == 0] = 1.0
    U = U / nrm
    return U, pcts, lam


def second_eof_tcof(tcof_list: list[np.ndarray], ncor: int | None = None, nkeep: int = 8):
    """Stack per-mode TCOF (ntot, mpts) into MATS x MATS covariance, as cseof.m."""
    ntot, mpts = tcof_list[0].shape
    nst = len(tcof_list)
    mats = nst * mpts
    X = np.zeros((ntot, mats))
    for j, tcof in enumerate(tcof_list):
        X[:, j * mpts : (j + 1) * mpts] = tcof
    if ncor is None:
        ncor = ntot
    cov = (X[:ncor].T @ X[:ncor]) / ncor
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    nkeep = min(nkeep, mats)
    W = evecs[:, :nkeep]
    pcs = X @ W
    tvar = float(np.trace(cov))
    return pcs, W, evals[:nkeep], tvar, X


def carrier_piece(cf: np.ndarray, sf: np.ndarray, k: int, icyc: int) -> np.ndarray:
    """Restore the nested harmonic: Ŝ_k(t) cos(σ_k t) + cosine-quadrature sine term.

    The distributed CF/SF are moving-window Fourier coefficients (slow).
    Multiplying by the carrier puts a line at k cycles per nested period
    even when the parent series is white.
    """
    t = np.arange(cf.shape[0], dtype=float)
    frq = 2.0 * np.pi * k / icyc
    return cf[:, k] * np.cos(frq * t) + sf[:, k] * np.sin(frq * t)


def pattern_corr(A: np.ndarray, B: np.ndarray) -> float:
    """Pearson correlation of flattened maps (sign-sensitive)."""
    a = np.asarray(A, dtype=float).ravel()
    b = np.asarray(B, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def align_sign(map_est: np.ndarray, map_ref: np.ndarray) -> np.ndarray:
    """Flip map_est if its correlation with map_ref is negative."""
    out = np.asarray(map_est, dtype=float).copy()
    if pattern_corr(out, map_ref) < 0:
        out *= -1.0
    return out


def cseof_of_C(D: np.ndarray, icyc: int = 12):
    """Eigenanalysis of the nested-phase sample covariance of D (space × time).

    Years are stacked as length-(M T) vectors. The leading eigenvector is
    reshaped to B(x, h). Reconstruction is rank-1 in that yearly stacking.
    This is the theoretical CSEOF writing, not the nested operator.
    """
    D = np.asarray(D, dtype=float)
    nspace, ntime = D.shape
    if ntime % icyc != 0:
        raise ValueError("ntime must be a multiple of icyc")
    n_years = ntime // icyc
    Y = np.empty((n_years, nspace * icyc))
    for n in range(n_years):
        Y[n] = D[:, n * icyc : (n + 1) * icyc].ravel(order="C")
    mu = Y.mean(axis=0)
    Yc = Y - mu
    cov = (Yc.T @ Yc) / n_years
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    B = evecs[:, 0].reshape(nspace, icyc, order="C")
    mean_map = mu.reshape(nspace, icyc, order="C")
    pc_year = Yc @ evecs[:, 0]
    rec = np.empty_like(D)
    for n in range(n_years):
        rec[:, n * icyc : (n + 1) * icyc] = mean_map + pc_year[n] * B
    tvar = float(np.sum(evals))
    frac = float(evals[0] / tvar) if tvar else float("nan")
    return {
        "B": B,
        "mean_map": mean_map,
        "pc_year": pc_year,
        "rec": rec,
        "evals": evals,
        "frac": frac,
        "n_years": n_years,
    }


def nested_two_stage(D: np.ndarray, nkeep: int = 12, nkeep2: int = 6, icyc: int = 12, npts: int = 6):
    """Distributed nested CSEOF of a space–time field (demean is caller's)."""
    U, pcts, lam = first_eof(D, nkeep)
    tcof_list = [demodulate(pcts[:, m], icyc=icyc, npts=npts)["tcof"] for m in range(nkeep)]
    pcs, W, evals, tvar, X = second_eof_tcof(tcof_list, ncor=D.shape[1], nkeep=nkeep2)
    bloch = bloch_from_W(W[:, 0], nkeep, tcof_list[0].shape[1], icyc, npts)
    spatial = U @ bloch
    rec = np.empty_like(D)
    ntime = D.shape[1]
    for t in range(ntime):
        rec[:, t] = pcs[t, 0] * spatial[:, t % icyc]
    ratio_tcof = float(np.sum(tcof_list[0] ** 2) / np.sum(pcts[:, 0] ** 2))
    return {
        "U": U,
        "pcts": pcts,
        "lam": lam,
        "pcs": pcs,
        "spatial": spatial,
        "bloch": bloch,
        "rec": rec,
        "evals": evals,
        "tvar": tvar,
        "mode1_frac": float(evals[0] / tvar) if tvar else float("nan"),
        "eof1_frac": float(lam[0] / lam.sum()),
        "ratio_tcof_pc1": ratio_tcof,
    }


def bloch_from_W(W_col: np.ndarray, nst: int, mpts: int, icyc: int, npts: int) -> np.ndarray:
    """Synthesize nested-cycle pattern for one CSEOF mode (cseof.m EOF1 loop)."""
    W = W_col.reshape(nst, mpts)
    eof = np.zeros((nst, icyc))
    t = np.arange(icyc, dtype=float)
    for L in range(nst):
        eof[L, :] = W[L, 0]
        for k_m in range(2, mpts + 1):
            k = k_m - 1
            fact = 1.0 if k_m == icyc else np.sqrt(2.0)
            KK = k_m // 2
            frq = 2.0 * np.pi * KK / icyc
            if k_m % 2 == 0:
                eof[L, :] += fact * W[L, k] * np.cos(frq * t)
            else:
                eof[L, :] += fact * W[L, k] * np.sin(frq * t)
    return eof
