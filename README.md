# Nested CSEOF operator as distributed

Python implementation of the nested cyclostationary EOF operator recorded in
Kim (submitted to *J. Atmos. Oceanic Technol.*). It reproduces Table 1 and
Figs. kernel–space of that manuscript from synthetic series.

This is **not** the FORTRAN of K.-Y. Kim, **not** the MATLAB of
B. D. Hamlington, and **not** eigenanalysis of a cyclostationary covariance
\(C(t+T,s+T)=C(t,s)\). See `NOTICE`. Those originals are not in this
repository.

## Run

```bash
python -m pip install -r requirements.txt
python run_cseof_limits.py          # Table 1, kernel, null, carrier, energy, space
python fig_t24_carving.py           # T = 12 vs 24 months
python fig_t6_t120.py               # T = 6 vs 120 months
python fig_dt_same_record.py        # ICYC identity (1 yr; 13 d)
python fig_designed_vs_C.py         # designed nested season; nested vs C
python fig_season_shapes.py         # sinusoid / step / AM / nested rec.
```

Needs NumPy and Matplotlib. MATLAB is not required. Default experiment:
`N = 624` monthly samples, nested period `ICYC = 12`, `NPTS = 6`.

## What the operator is

Ordinary EOF of the Gram \(\mathbf{D}^\dagger\mathbf{D}\); modified-sinc
demodulation of the PCs at frequencies \(k/T\); packing of sine and cosine
coefficients (TCOF, extra \(\sqrt{2}\)); second EOF; harmonic synthesis onto
the nested calendar. Reconstruction is of packed \(\hat S\), not of
\(\mathbf{D}\). Discrete samples of the window follow 2010 MATLAB `weight.m`
as recorded in the paper.

## License

MIT (`LICENSE`) for this Python. Method literature remains with its authors.
If you need the original FORTRAN or MATLAB, ask those authors; do not take
them from this archive — they are not here.

## Cite

Until a Zenodo DOI exists, cite the paper. After the first GitHub release is
harvested by Zenodo, replace the identifier in `CITATION.cff` and in the
paper's data availability statement with the **version DOI** of that release
(not only the GitHub URL).
