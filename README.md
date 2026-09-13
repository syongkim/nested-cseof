# Nested CSEOF Python (this paper)

Python implementation of the nested cyclostationary EOF operator written in
Kim (submitted to *J. Atmos. Oceanic Technol.*), and the scripts that draw
Table 1 and Figs. 1–15 of that manuscript from synthetic series.

This is **not** the FORTRAN of K.-Y. Kim, **not** the MATLAB of
B. D. Hamlington, and **not** eigenanalysis of a cyclostationary covariance
\(C(t+T,s+T)=C(t,s)\). See `NOTICE`. Those originals are not in this
repository. They were obtained privately and are not redistributed.

## Run

```bash
python -m pip install -r requirements.txt
python run_cseof_limits.py          # Table 1, kernel, null, carrier, energy, ensemble
python fig_t24_carving.py           # T = 12 vs 24 months
python fig_t6_t120.py               # T = 6 vs 120 months
python fig_dt_same_record.py        # ICYC identity (1 yr; 13 d)
python fig_designed_vs_C.py         # Figs. 12–13: white vs designed season; nested vs C
python fig_linewidth.py             # Fig. 14: line vs AM sidebands vs white vs red
python fig_season_shapes.py         # Fig. 15: sinusoid / step / AM / nested rec.
```

Needs NumPy and Matplotlib. MATLAB is not required. Default experiment:
`N = 624` monthly samples, nested period `ICYC = 12`, `NPTS = 6`.

## What the operator is

Ordinary EOF of the Gram \(\mathbf{D}^\dagger\mathbf{D}\); modified-sinc
demodulation of the PCs at frequencies \(k/T\); packing of sine and cosine
coefficients (TCOF, extra \(\sqrt{2}\)); second EOF; harmonic synthesis onto
the nested calendar. Reconstruction is of packed \(\hat S\), not of
\(\mathbf{D}\). Discrete samples of the window follow 2010 MATLAB `weight.m`
as written in the paper.

## License

MIT (`LICENSE`) for this Python. Method literature remains with its authors.
If you need the original FORTRAN or MATLAB, ask those authors; do not take
them from this archive — they are not here.

## Cite

Kim, S. Y. (2026). Nested CSEOF Python (this paper) (v1.1.0).
https://github.com/syongkim/nested-cseof

The concept DOI `10.5281/zenodo.22735764` resolves to the latest Zenodo
version. Tag `v1.0.0` (`10.5281/zenodo.22735765`) does not include Fig. 14.
Cite the version DOI of tag `v1.1.0` in the paper once Zenodo has minted it.
