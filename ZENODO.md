# GitHub + Zenodo (DOI)

Do this after the deposit folder is the GitHub root. MATLAB/FORTRAN are
not in this folder. Do not add them.

## 0. GitHub repo (account syongkim)

On the machine with git and a browser:

```bash
cd /pao1/work/papers/cycloEOF/deposit
git init
git add LICENSE NOTICE README.md CITATION.cff requirements.txt .gitignore \
  jtech_palette.py nested_cseof.py run_cseof_limits.py \
  fig_t24_carving.py fig_t6_t120.py fig_dt_same_record.py \
  fig_designed_vs_C.py fig_season_shapes.py
git commit -m "Nested CSEOF Python: operator as distributed."
```

Create an empty public repository `nested-cseof` under https://github.com/syongkim
(no README on GitHub, so the first push is clean). Then:

```bash
git branch -M main
git remote add origin git@github.com:syongkim/nested-cseof.git
git push -u origin main
```

Zenodo's GitHub harvest needs a **public** repo (or a paid Zenodo/GitHub
connection). Keep the repo public if the paper's availability statement
should resolve without a login.

## 1. Zenodo account

1. Open https://zenodo.org and **Log in with GitHub** (same `syongkim`).
   That is the least error-prone link. ORCID login also works; then
   connect GitHub under the account menu.
2. Confirm the email Zenodo sends.
3. You do not need to share a password. After login, the GitHub
   applications list will show Zenodo.

## 2. Flip the repo on

1. Zenodo: GitHub icon (top right) → **GitHub**.
2. Find `syongkim/nested-cseof` and switch it **on**.
3. Zenodo installs a webhook on that repository.

If the repo does not appear, grant Zenodo access to that repository in
GitHub → Settings → Applications → Zenodo → Repository access.

## 3. Mint the DOI (a GitHub Release)

Zenodo does not mint a DOI from an ordinary push. It mints one when
GitHub publishes a **Release**.

1. GitHub → `nested-cseof` → **Releases** → **Draft a new release**.
2. Tag: `v1.0.0` (create tag on `main`).
3. Title: `nested-cseof v1.0.0`.
4. Publish release.

Wait one to several minutes. Zenodo harvests the tag, builds a tarball,
and assigns:

- a **version DOI** for `v1.0.0` (cite this in the paper), e.g.
  `10.5281/zenodo.1234567`
- a **concept DOI** that always points at the latest version (use in
  README badges).

Record both. Put the version DOI in `CITATION.cff` (`doi:`) and in
`\\datastatement` of `sdcseof_jtech.tex`. Later tagged releases get new
version DOIs; the concept DOI stays.

## 4. What not to upload

- `papers/cycloEOF/matlab/` and `cseof/`, `cseof2/`, zip files
- Fortran `WEIGHT` / `cseofx.f`
- `sd.cseof/`
- Unpublished correspondence

If a file is not needed to reproduce Table 1 and the manuscript figures,
leave it out.
