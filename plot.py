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
eid = ses.eid.iloc[0]
df = pd.read_parquet(R / f'samples_{eid[:8]}.pqt')

fig = plt.figure(figsize=(7.2, 4.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.6, wspace=0.6)
# a: example trials (answered, one correct each side, one error)
go = df[df.choice != 0].groupby('trial').agg(fb=('fbtype', 'first'), side=('side', 'first'),
                                             c=('contrast', 'first'), n=('truth', 'size'))
ex = [go[(go.fb == 1) & (go.side == -1) & (go.n.between(80, 300))].index[3],
      go[(go.fb == 1) & (go.side == 1) & (go.n.between(80, 300))].index[3],
      go[(go.fb == -1) & (go.n.between(80, 400))].index[0]]
for i, k in enumerate(ex):
    ax = fig.add_subplot(gs[0, i])
    s = df[df.trial == k]
    for m in ['hgb_T0', 'physics_infer', 'physics_side']:
        ax.plot(s.t_on, s['pred_' + m], color=COL[m], lw=1, label=LABEL[m],
                ls='--' if m == 'physics_side' else '-')
    ax.plot(s.t_on, s.truth, color='0.6', lw=3, alpha=0.5, zorder=0, label='Bonsai log (truth)')
    ax.axvline(s.t_on.iloc[0] - s.t_resp.iloc[0], color='0.8', lw=0.5)
    ax.set_title(f"trial {k}: {'correct' if s.fbtype.iloc[0] == 1 else 'error'}, "
                 f"contrast {s.contrast.iloc[0]:g}", fontsize=6)
    ax.set_xlabel('time from stimOn (s)')
    if i == 0:
        ax.set_ylabel('azimuth (deg)')
    ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
ax.legend(fontsize=5, frameon=False, loc='center left', bbox_to_anchor=(1.05, 0.5))

# b-d: per-session error by model and epoch
for j, (ep, ttl) in enumerate([('pre_resp', 'stimOn → response'), ('post_resp', 'response → stimOff'),
                               ('nogo', 'no-go trials')]):
    ax = fig.add_subplot(gs[1, j])
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
ax = fig.add_subplot(gs[1, 3])
ax.scatter(ses.lag_ms, -ses.gain_free_fit if ses.gain_free_fit.mean() < 0 else ses.gain_free_fit,
           s=8, color='k')
ax.axhline(124, color='0.6', lw=0.5, ls=':')
ax.set_xlabel('fitted display lag (ms)'); ax.set_ylabel('|free-fit gain| (deg/rad)')
ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
for a, letter in zip(fig.axes, 'abcdefg'):  # axes in creation order: 3 examples, 3 epochs, lag/gain
    a.text(-0.32, 1.12, letter, transform=a.transAxes, fontsize=9, fontweight='bold', va='bottom')
fig.savefig(R / 'fig_stim_pos_decoding.pdf', bbox_inches='tight')
fig.savefig(R / 'fig_stim_pos_decoding.png', dpi=200, bbox_inches='tight')

n = len(ses)
cap = (r"\caption{\textbf{On-screen stimulus azimuth is recoverable from wheel position and trial "
       r"timing.} Ground truth is the Bonsai stimulus-position log (\texttt{\_iblrig\_stimPositionScreen}), "
       r"aligned to photodiode stimOn (residual SD " + f"{ses.resid_sd_ms.median():.0f}" +
       r" ms). "
       r"\textbf{a--c}, Example trials from session \texttt{" + eid[:8] + r"}, showing stimulus azimuth "
       r"against time from stimOn. Lines: logged truth (thick grey); physics model "
       r"$x_0 + G\,\Delta w(t-\tau)$, $|G|=124^\circ$/rad, with the side known (dashed black); the same "
       r"model with the side inferred from the wheel direction at response and the outcome-specific "
       r"stimOff delay (blue); a gradient-boosting model (GBM) given only wheel displacement and time "
       r"since stimOn (T0, red). The vertical line marks the response time. "
       r"\textbf{a}, Correct trial with the stimulus starting on the left ($-35^\circ$) and brought to the "
       r"centre. \textbf{b}, Correct trial with the stimulus starting on the right ($+35^\circ$). "
       r"\textbf{c}, Error trial: the stimulus is driven away from the centre, to $\approx70^\circ$. "
       r"The physics models overlie the truth. The T0 model cannot know the starting side, so it "
       r"wanders near the average of the two possibilities. "
       r"\textbf{d--f}, Mean absolute error per session (dots, $n=" + str(n) +
       r"$ sessions, one per lab; bars, median across sessions; log scale) for each model: physics "
       r"with side known or inferred, and GBM with features T2 (wheel + all event times), T1 (wheel + "
       r"response time) or T0 (wheel only), cross-validated in 5 folds grouped by trial. The dotted "
       r"line marks one encoder step ($0.76^\circ$). "
       r"\textbf{d}, Samples from stimOn to response on answered trials. "
       r"\textbf{e}, Samples from response to stimOff on answered trials; the error of $\approx2^\circ$ "
       r"comes from Bonsai freezing on the last frame before the threshold event. "
       r"\textbf{f}, Samples from no-go trials (only sessions with no-go trials). Without the side, no "
       r"decoder can place the stimulus. "
       r"\textbf{g}, Display lag $\tau$ fitted per session against the magnitude of an unconstrained "
       r"least-squares gain (dotted line: nominal 124$^\circ$/rad).}")
(R / 'fig_stim_pos_decoding.tex').write_text(cap + '\n')
