"""Figure: on-screen stimulus azimuth decoded from wheel + timing. Saves PDF + LaTeX caption."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

R = Path(__file__).parent / 'results'
plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
MODELS = ['physics_side', 'physics_infer', 'hgb_T2', 'hgb_T1', 'hgb_T0']
LABEL = {'physics_side': 'physics, side known', 'physics_infer': 'physics, side from wheel+timing',
         'hgb_T2': 'GBM: wheel + all event times', 'hgb_T1': 'GBM: wheel + response time',
         'hgb_T0': 'GBM: wheel only'}
COL = dict(zip(MODELS, ['k', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']))

met = pd.read_csv(R / 'metrics.csv')
ses = pd.read_csv(R / 'sessions.csv')
LAG = dict(zip(ses.eid.str[:8], ses.lag_ms / 1e3))
WHEEL_COL = '#9467bd'


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


# per-trial R2 over all answered trials of all sessions (trials with a flat truth have no R2)
samples, rows = {}, []
for e in ses.eid.str[:8]:
    d = pd.read_parquet(R / f'samples_{e}.pqt')
    d = d[d.choice != 0]
    samples[e] = d
    for k, s in d.groupby('trial'):
        y = s.truth.to_numpy()
        if y.std() < 0.5:
            continue
        rows.append(dict(ses=e, trial=k, r2_phys=r2(y, s.pred_physics_infer.to_numpy()),
                         r2_t0=r2(y, s.pred_hgb_T0.to_numpy())))
tr2 = pd.DataFrame(rows).sort_values('r2_phys', ignore_index=True)
n_flat = sum(d.trial.nunique() for d in samples.values()) - len(tr2)
imed = int(np.argmin(np.abs(tr2.r2_phys - tr2.r2_phys.median())))
EX = [('best', tr2.iloc[-1]), ('best', tr2.iloc[-2]), ('median', tr2.iloc[imed]),
      ('median', tr2.iloc[imed + 1]), ('worst', tr2.iloc[0]), ('worst', tr2.iloc[1])]

fig = plt.figure(figsize=(7.2, 6.6))
outer = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.45)
gex = outer[0].subgridspec(2, 3, hspace=0.75, wspace=0.75)
gsum = outer[1].subgridspec(1, 4, wspace=0.6)
for i, (kind, r) in enumerate(EX):
    ax = fig.add_subplot(gex[i % 2, i // 2])
    s = samples[r.ses][samples[r.ses].trial == r.trial]
    ax.plot(s.t_on, s.truth, color='0.6', lw=3, alpha=0.5, zorder=0, label='screen position (Bonsai log, truth)')
    ax.plot(s.t_on, s.pred_physics_infer, color=COL['physics_infer'], lw=1,
            label='physics decoder (side from wheel + timing)')
    ax.plot(s.t_on, s.pred_hgb_T0, color=COL['hgb_T0'], lw=0.7, label='GBM decoder, wheel only (T0)')
    ax.axvline(s.t_on.iloc[0] - s.t_resp.iloc[0], color='0.8', lw=0.5)
    ax.set_title(f"{kind}: $R^2$ phys {r.r2_phys:.3f}, T0 {r.r2_t0:.2f}\n"
                 f"{r.ses} trial {r.trial}, {'correct' if s.fbtype.iloc[0] == 1 else 'error'}, "
                 f"contrast {s.contrast.iloc[0]:g}", fontsize=5.5)
    ax.set_xlabel('time from stimOn (s)')
    if i // 2 == 0:
        ax.set_ylabel('azimuth (deg)')
    ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
    # wheel: displacement since stimOn, at its own (un-lagged) time
    aw = ax.twinx()
    aw._is_twin = True
    aw.spines['right'].set_visible(True); aw.spines['right'].set_color(WHEEL_COL)
    aw.plot(s.t_on - LAG[r.ses], s.dw, color=WHEEL_COL, lw=0.8, ls='--', label='wheel displacement')
    aw.tick_params(axis='y', colors=WHEEL_COL, labelsize=6)
    aw.yaxis.set_major_locator(MaxNLocator(3))
    if i // 2 == 2:
        aw.set_ylabel('wheel (rad)', color=WHEEL_COL)
    if i == 0:
        h = ax.get_legend_handles_labels(); hw = aw.get_legend_handles_labels()
        fig.legend(h[0] + hw[0], h[1] + hw[1], fontsize=5.5, frameon=False, ncol=4,
                   loc='upper center', bbox_to_anchor=(0.5, 0.955))

# b-d: per-session error by model and epoch
for j, (ep, ttl) in enumerate([('pre_resp', 'stimOn → response'), ('post_resp', 'response → stimOff'),
                               ('nogo', 'no-go trials')]):
    ax = fig.add_subplot(gsum[0, j])
    sub = met[met.epoch == ep]
    for x, m in enumerate(MODELS):
        v = sub[sub.model == m].mae.to_numpy()
        ax.scatter(np.full(len(v), x) + np.random.default_rng(0).uniform(-.15, .15, len(v)), v,
                   s=6, color=COL[m])
        ax.hlines(np.median(v), x - .3, x + .3, color=COL[m], lw=1.5)
    ax.set_yscale('log')
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(['phys\nside', 'phys\ninfer', 'GBM\nT2', 'GBM\nT1', 'GBM\nT0'], fontsize=5)
    ax.axhline(0.761, color='0.6', lw=0.5, ls=':')
    ax.set_title(ttl, fontsize=6)
    if j == 0:
        ax.set_ylabel('mean |error| (deg)')
ax = fig.add_subplot(gsum[0, 3])
ax.scatter(ses.lag_ms, -ses.gain_free_fit if ses.gain_free_fit.mean() < 0 else ses.gain_free_fit,
           s=8, color='k')
ax.axhline(124, color='0.6', lw=0.5, ls=':')
ax.set_xlabel('fitted display lag (ms)'); ax.set_ylabel('|free-fit gain| (deg/rad)')
ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
main_axes = [a for a in fig.axes if not hasattr(a, '_is_twin')]
for a, letter in zip(main_axes, 'abcdefghij'):  # 6 examples (column-wise), 3 epochs, lag/gain
    a.text(-0.32, 1.12, letter, transform=a.transAxes, fontsize=9, fontweight='bold', va='bottom')
fig.savefig(R / 'fig_stim_pos_decoding.pdf', bbox_inches='tight')
fig.savefig(R / 'fig_stim_pos_decoding.png', dpi=200, bbox_inches='tight')

n = len(ses)
cap = (r"\caption{\textbf{On-screen stimulus azimuth is recoverable from wheel position and trial "
       r"timing.} Ground truth is the Bonsai stimulus-position log (\texttt{\_iblrig\_stimPositionScreen}), "
       r"aligned to photodiode stimOn (residual SD " + f"{ses.resid_sd_ms.median():.0f}" +
       r" ms). "
       r"\textbf{a--f}, Example trials chosen by per-trial $R^2$ of the physics decoder (side inferred "
       r"from wheel and timing), computed over all samples from stimOn to stimOff, among the "
       + f"{len(tr2):,}".replace(',', '{,}') + r" answered trials of all sessions (" + str(n_flat) +
       r" trials whose logged position never moved have no $R^2$; median per-trial $R^2$ = "
       + f"{tr2.r2_phys.median():.2f}" + r"). Panel titles give $R^2$ for the physics decoder and for the "
       r"gradient-boosting (GBM) decoder given only the wheel and time since stimOn (T0), with session, "
       r"trial, outcome and contrast. Left axis, stimulus azimuth: logged screen position (truth, thick "
       r"grey), physics decoder $x_0 + G\,\Delta w(t-\tau)$ with $|G| = 124^\circ$/rad and displacement "
       r"clipped at the $\pm35^\circ$ threshold (blue), GBM T0 (red). Right axis (purple, dashed), wheel "
       r"displacement since stimOn in radians, at its own time (not shifted by the display lag). Vertical "
       r"line, response time. "
       r"\textbf{a, b}, Best trials: the decoder follows the truth to within an encoder step, while T0, "
       r"which cannot know the starting side, has a strongly negative $R^2$. "
       r"\textbf{c, d}, Trials at the median: errors come mainly from Bonsai freezing a few degrees short "
       r"of the target after the response (c), which within a single trial weighs more than in the "
       r"pooled score. "
       r"\textbf{e, f}, Worst trials. \textbf{e}, The trials table has a corrupt stimOff time (28\,s before "
       r"feedback), so the outcome, and with it the starting side, is inferred wrongly. \textbf{f}, Very "
       r"fast response with the wheel already turning at stimulus onset; the logged stimulus stopped at "
       r"$\approx60^\circ$ instead of the $70^\circ$ threshold. "
       r"\textbf{g--i}, Mean absolute error per session (dots, $n=" + str(n) +
       r"$ sessions, one per lab; bars, median across sessions; log scale) for each model: physics "
       r"with side known or inferred, and GBM with features T2 (wheel + all event times), T1 (wheel + "
       r"response time) or T0 (wheel only), cross-validated in 5 folds grouped by trial. The dotted "
       r"line marks one encoder step ($0.76^\circ$). "
       r"\textbf{g}, Samples from stimOn to response on answered trials. "
       r"\textbf{h}, Samples from response to stimOff on answered trials; the error of $\approx2^\circ$ "
       r"comes from Bonsai freezing on the last frame before the threshold event. "
       r"\textbf{i}, Samples from no-go trials (only sessions with no-go trials). Without the side, no "
       r"decoder can place the stimulus. "
       r"\textbf{j}, Display lag $\tau$ fitted per session against the magnitude of an unconstrained "
       r"least-squares gain (dotted line: nominal 124$^\circ$/rad).}")
(R / 'fig_stim_pos_decoding.tex').write_text(cap + '\n')
