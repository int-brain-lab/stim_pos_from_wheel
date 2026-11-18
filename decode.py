"""Can the on-screen stimulus azimuth be recovered from wheel position + trial timing?

Ground truth: Bonsai stimulus-position log, aligned to photodiode stimOn (load.py).
Samples: 10 ms grid from stimOn to stimOff, every trial.

Decoders (all cross-validated by trial where fitted):
  physics_side   x0 (+-35 from stimulus side) + G * (wheel(t-lag) - wheel(stimOn-lag)), frozen at
                 response_times + lag (screen shows w(t-lag)); |G| = 4 deg/mm * 31 mm = 124 deg/rad
                 (= 0.761 deg per 1024-tick encoder step in the Bonsai log); sign of G and display
                 lag fitted per session.
  physics_infer  same, but side inferred from wheel + timing only:
                 choice = sign of wheel displacement at response, outcome = stimOff-feedback > 1.5 s
                 (error timeout 2 s vs correct 1 s); no-go trials -> x0 = 0.  Uses future info.
  No-go trials (60 s timeouts, side unknowable from wheel) are scored separately.
  hgb_T0 .. T2   HistGradientBoosting, GroupKFold by trial, feature tiers:
                 T0 wheel displacement since stimOn + time since stimOn
                 T1 + response time (time to response, displacement at response)
                 T2 + feedback/stimOff timing
"""
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from load import load_session, WHEEL_RADIUS_MM

warnings.filterwarnings('ignore')
OUT = Path(__file__).parent / 'results'
OUT.mkdir(exist_ok=True)
DT = 0.01


def build_samples(d, lag=0.0):
    tr, sp, wh = d['trials'], d['sp'], d['wheel']
    st, sx = sp.t.to_numpy(), sp.pos.to_numpy()
    wt, wp = wh['timestamps'], wh['position']
    rows = []
    for k in range(len(tr['stimOn_times'])):
        on, off = tr['stimOn_times'][k], tr['stimOff_times'][k]
        resp, fb = tr['response_times'][k], tr['feedback_times'][k]
        if not np.all(np.isfinite([on, off, resp, fb])) or off <= on:
            continue
        t = np.arange(on, off, DT)
        i = np.searchsorted(st, t, side='right') - 1
        truth = sx[i]
        w0 = np.interp(on - lag, wt, wp)
        w = np.interp(t - lag, wt, wp) - w0
        w_resp = np.interp(resp, wt, wp) - w0  # displayed lag s after the response
        side = 1 if np.isnan(tr['contrastLeft'][k]) else -1  # +35 right, -35 left
        rows.append(pd.DataFrame(dict(
            trial=k, truth=truth, t_on=t - on, dw=w, dw_frozen=np.where(t - lag < resp, w, w_resp),
            t_resp=t - resp, dw_resp=w_resp, t_fb=t - fb, t_off=t - off,
            off_minus_fb=off - fb, side=side, fbtype=tr['feedbackType'][k],
            choice=tr['choice'][k],
            contrast=np.nansum([tr['contrastLeft'][k], tr['contrastRight'][k]]))))
    return pd.concat(rows, ignore_index=True)


def metrics(y, yhat):
    e = np.abs(y - yhat)
    return dict(r2=1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2),
                mae=e.mean(), medae=np.median(e), within_1tick=np.mean(e <= 0.77))


def run(eid):
    d = load_session(eid)
    # display lag: scan on pre-response samples of answered trials, side known, |G| = 124
    scan = []
    for lag in np.arange(0, 0.155, 0.005):
        df = build_samples(d, lag)
        m = ((df.t_resp < 0) & (df.choice != 0)).to_numpy()
        y = df.truth.to_numpy()
        sgn = np.sign(np.sum(df.dw[m] * (y - 35 * df.side)[m]))
        scan.append((np.mean(np.abs(y - 35 * df.side - sgn * 124 * df.dw)[m]), lag, sgn))
    _, lag, sgn = min(scan)
    df = build_samples(d, lag)
    y = df.truth.to_numpy()
    pre = ((df.t_resp < 0) & (df.choice != 0)).to_numpy()
    G_free = np.linalg.lstsq(df.dw[pre].to_numpy()[:, None],
                             (y - 35 * df.side)[pre], rcond=None)[0][0]
    G = sgn * 4 * WHEEL_RADIUS_MM
    pred = {}
    pred['physics_side'] = 35 * df.side + G * df.dw_frozen
    choice_w = -np.sign(G * df.dw_resp)  # stimulus moved toward -> side it came from
    err = df.off_minus_fb > 1.5
    nogo = df.off_minus_fb < 0.5
    side_hat = np.where(err, -choice_w, choice_w)
    side_hat = np.where(nogo, 0, side_hat)
    pred['physics_infer'] = 35 * side_hat + G * df.dw_frozen
    tiers = {'hgb_T0': ['dw', 't_on'],
             'hgb_T1': ['dw', 't_on', 'dw_frozen', 't_resp', 'dw_resp'],
             'hgb_T2': ['dw', 't_on', 'dw_frozen', 't_resp', 'dw_resp', 't_fb', 't_off', 'off_minus_fb']}
    for name, feats in tiers.items():
        p = np.zeros(len(df))
        for tr_i, te_i in GroupKFold(5).split(df, groups=df.trial):
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1).fit(
                df.iloc[tr_i][feats], y[tr_i])
            p[te_i] = m.predict(df.iloc[te_i][feats])
        pred[name] = p
    go = (df.choice != 0).to_numpy()
    epochs = {'all': go, 'pre_resp': go & (df.t_resp < 0).to_numpy(),
              'post_resp': go & (df.t_resp >= 0).to_numpy(), 'nogo': ~go}
    res = []
    for name, p in pred.items():
        for ep, m in epochs.items():
            if m.sum() > 100:
                res.append(dict(eid=eid, model=name, epoch=ep, **metrics(y[m], np.asarray(p)[m])))
    for k, p in pred.items():
        df['pred_' + k] = np.asarray(p)
    df.to_parquet(OUT / f'samples_{eid[:8]}.pqt')
    info = dict(eid=eid, lag_ms=lag * 1e3, gain_used=G, gain_free_fit=G_free,
                n_nogo=int(df[~go].trial.nunique()),
                n_trials=int(df.trial.nunique()), n_samples=len(df), **d['align'],
                rig=d['settings'].get('IBLRIG_VERSION_TAG'))
    return pd.DataFrame(res), info


if __name__ == '__main__':
    eids = sys.argv[1:]
    allres, infos = [], []
    for eid in eids:
        try:
            r, info = run(eid)
        except Exception as e:  # noqa
            print('FAIL', eid, repr(e))
            continue
        print(json.dumps(info, default=float))
        allres.append(r); infos.append(info)
    res = pd.concat(allres)
    res.to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame(infos).to_csv(OUT / 'sessions.csv', index=False)
    pd.set_option('display.width', 200)
    print(res.groupby(['model', 'epoch'])[['r2', 'mae', 'medae', 'within_1tick']].median().round(3))
