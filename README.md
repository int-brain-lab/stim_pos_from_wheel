# stim_pos_from_wheel

Can the on-screen stimulus azimuth be decoded from wheel position + trial timing? (2026-11-18)

**Yes, in closed form.** `x(t) = x0 + G·[w(min(t−τ, t_resp)) − w(stimOn−τ)]`,
|G| = 4°/mm × 31 mm = 124°/rad (the Bonsai log steps by 0.761° = one 1024-tick encoder step),
x0 = ±35°, τ = display lag (55–110 ms, fitted per session).

Ground truth: `raw_behavior_data/_iblrig_stimPositionScreen.raw.csv`
(contrast shown [0 = off], azimuth deg, Bonsai wall-clock; written on change only), aligned to
photodiode `stimOn_times` (residual SD 6–11 ms).

8 BWM sessions, one per lab, iblrig 6.2–6.5. Mean |error| per session, median across sessions:

| model | stimOn→response | response→stimOff | no-go |
|---|---|---|---|
| physics, side known | 0.9° (R² 0.99) | 2.5° | 0.8° |
| physics, side from wheel+timing | 0.9° | 2.5° | 35° |
| GBM wheel + all event times (T2) | 2.7° | 2.2° | 47° |
| GBM wheel + response time (T1) | 27° | 17° | 37° |
| GBM wheel only (T0) | 30° | 22° | 38° |

- Side (x0) is the only thing the wheel cannot give you. With event times it is inferable:
  choice = wheel direction at response; outcome = stimOff − feedback (≈1.07 s correct, ≈2.07 s
  error, ≈0.06 s no-go). 99% of trials correct; misses are no-go trials only (side unknowable).
- Post-response residual ≈2°: Bonsai freezes on the last frame before the threshold event.

Report: [`report/report.pdf`](report/report.pdf) (methods, data, results; source `report/report.tex`,
tables from `report/make_tables.py`, build with `tectonic report.tex`).

Files: `load.py` (loading + clock alignment), `decode.py <eids…>` (models, metrics; writes
`results/`, per-sample `.pqt` files are not tracked), `plot.py` (figure + LaTeX caption), `results/`, `legacy/` (2021 keras wheel-velocity script,
recovered from int-brain-lab/paper-brain-wide-map-OLD @ e413e93^).
Requires the IBL environment (ONE-api, ibllib, brainwidemap, scikit-learn, matplotlib) and Alyx access.

Sessions: `6713a4a7 034e726f 7dd27c2f 0f25376f 64977c74 66d98e6e c4432264 4d8c7767` (full eids in `results/sessions.csv`).
