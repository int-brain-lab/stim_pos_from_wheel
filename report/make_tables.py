"""Write LaTeX tables for the report from ../results (metrics.csv, sessions.csv)."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
R = HERE.parent / 'results'
met = pd.read_csv(R / 'metrics.csv')
ses = pd.read_csv(R / 'sessions.csv')

# session details (lab, subject, date) are cached here so the report builds offline
INFO = HERE / 'session_info.csv'
if not INFO.exists():
    from one.api import ONE
    one = ONE()
    rows = []
    for e in ses.eid:
        d = one.get_details(e)
        rows.append(dict(eid=e, lab=d['lab'], subject=d['subject'], date=d['start_time'][:10]))
    pd.DataFrame(rows).to_csv(INFO, index=False)
ses = ses.merge(pd.read_csv(INFO), on='eid')

MODELS = [('physics_side', r'Physics, side known'),
          ('physics_infer', r'Physics, side inferred'),
          ('hgb_T2', r'GBM T2: wheel + all event times'),
          ('hgb_T1', r'GBM T1: wheel + response time'),
          ('hgb_T0', r'GBM T0: wheel only')]
EPOCHS = [('pre_resp', 'stimOn--response'), ('post_resp', 'response--stimOff'), ('nogo', 'no-go')]

med = met.groupby(['model', 'epoch'])[['r2', 'mae', 'medae', 'within_1tick']].median()
lines = [r'\begin{tabular}{l' + 'rrr' * len(EPOCHS) + '}', r'\toprule',
         ' & ' + ' & '.join(rf'\multicolumn{{3}}{{c}}{{{t}}}' for _, t in EPOCHS) + r' \\',
         ' '.join(rf'\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(EPOCHS))),
         'Model' + r' & MAE & $R^2$ & $\le$1 tick' * len(EPOCHS) + r' \\', r'\midrule']
for m, lab in MODELS:
    cells = []
    for ep, _ in EPOCHS:
        r = med.loc[(m, ep)]
        r2 = f'{r.r2:.2f}' if r.r2 > -1 else r'$<{-1}$'
        cells += [f'{r.mae:.1f}', r2, f'{100 * r.within_1tick:.0f}\\%']
    lines.append(lab + ' & ' + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
(HERE / 'table_summary.tex').write_text('\n'.join(lines) + '\n')

lines = [r'\begin{tabular}{llllrrrrr}', r'\toprule',
         r'Session & Lab & Subject & Date & iblrig & Trials & No-go & $\tau$ (ms) & $|G_\mathrm{free}|$ \\',
         r'\midrule']
for _, s in ses.iterrows():
    rig = s.rig if isinstance(s.rig, str) else '--'
    lab, subj = s.lab.replace('_', r'\_'), s.subject.replace('_', r'\_')
    lines.append(rf'\texttt{{{s.eid[:8]}}} & {lab} & {subj} & '
                 rf'{s.date} & {rig} & {s.n_trials} & {s.n_nogo} & {s.lag_ms:.0f} & '
                 rf'{abs(s.gain_free_fit):.1f} \\')
lines += [r'\bottomrule', r'\end{tabular}']
(HERE / 'table_sessions.tex').write_text('\n'.join(lines) + '\n')

# scalar macros used in the text
mac = {
    'NSessions': len(ses), 'NTrials': f'{ses.n_trials.sum():,}'.replace(',', '{,}'),
    'NSamples': f'{ses.n_samples.sum() / 1e6:.1f}', 'NNogo': ses.n_nogo.sum(),
    'AlignSD': f'{ses.resid_sd_ms.median():.0f}',
    'AlignSDmin': f'{ses.resid_sd_ms.min():.0f}', 'AlignSDmax': f'{ses.resid_sd_ms.max():.0f}',
    'LagMed': f'{ses.lag_ms.median():.0f}', 'LagMin': f'{ses.lag_ms.min():.0f}',
    'LagMax': f'{ses.lag_ms.max():.0f}',
    'GainMin': f'{ses.gain_free_fit.abs().min():.0f}', 'GainMax': f'{ses.gain_free_fit.abs().max():.0f}',
    'PreMAE': f"{med.loc[('physics_side', 'pre_resp')].mae:.1f}",
    'PreMedAE': f"{med.loc[('physics_side', 'pre_resp')].medae:.2f}",
    'PreRtwo': f"{med.loc[('physics_side', 'pre_resp')].r2:.3f}",
    'PreTick': f"{100 * med.loc[('physics_side', 'pre_resp')].within_1tick:.0f}",
    'PostMAE': f"{med.loc[('physics_side', 'post_resp')].mae:.1f}",
    'TzeroMAE': f"{med.loc[('hgb_T0', 'pre_resp')].mae:.0f}",
    'TtwoMAE': f"{med.loc[('hgb_T2', 'pre_resp')].mae:.1f}",
}
(HERE / 'macros.tex').write_text(''.join(rf'\newcommand{{\{k}}}{{{v}}}' + '\n' for k, v in mac.items()))
print('wrote tables + macros')
