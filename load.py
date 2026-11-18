"""Load ground-truth on-screen stimulus position (Bonsai log) aligned to ALF trials + wheel.

_iblrig_stimPositionScreen.raw.csv columns: contrast shown (0 = off), azimuth (deg), Bonsai
wall-clock timestamp. Rows are written only when something changes (zero-order hold).

Clock alignment: Bonsai wall clock -> ALF session clock by a linear fit of Bonsai stimulus-onset
rows (contrast 0 -> >0) against photodiode stimOn_times, so "time" means time the frame was
actually on screen. A coarse fit on encoderTrialInfo trial stamps vs trial starts seeds the match.
"""
import numpy as np
import pandas as pd
from one.api import ONE

WHEEL_RADIUS_MM = 31.0


def _wallclock(s):
    return pd.to_datetime(s.str.strip(), format='ISO8601', utc=True).astype('int64').to_numpy() / 1e9


def load_session(eid, one=None):
    one = one or ONE()
    trials = one.load_object(eid, 'trials', collection='alf')
    wheel = one.load_object(eid, 'wheel', collection='alf')
    raw = 'raw_behavior_data'
    sp = pd.read_csv(one.load_dataset(eid, '_iblrig_stimPositionScreen.raw.csv', collection=raw,
                                      download_only=True), header=None, names=['contrast', 'pos', 'ts'])
    ti = pd.read_csv(one.load_dataset(eid, '_iblrig_encoderTrialInfo.raw.ssv', collection=raw,
                                      download_only=True), sep=' ', header=None)
    settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json', collection=raw)
    sp['wall'] = _wallclock(sp.ts.astype(str))
    w0 = sp.wall.iloc[0]
    # coarse: encoderTrialInfo wall stamp (trial k, 1-based) <-> trials.intervals[k-1, 0]
    ti_wall = _wallclock(ti[8].astype(str)) - w0
    tstart = trials['intervals'][:, 0]
    n = min(len(ti_wall), len(tstart))
    ok = np.isfinite(tstart[:n])
    p0 = np.polyfit(ti_wall[:n][ok], tstart[:n][ok], 1)
    # fine: Bonsai onset rows <-> nearest photodiode stimOn
    c = sp.contrast.to_numpy()
    on_rows = np.where((c[1:] > 0) & (c[:-1] == 0))[0] + 1
    on_coarse = np.polyval(p0, sp.wall.to_numpy()[on_rows] - w0)
    so = trials['stimOn_times']
    j = np.nanargmin(np.abs(on_coarse[:, None] - so[None, :]), axis=1)
    lag = so[j] - on_coarse
    good = np.abs(lag - np.median(lag)) < 0.05
    p = np.polyfit(sp.wall.to_numpy()[on_rows][good] - w0, so[j][good], 1)
    resid = so[j][good] - np.polyval(p, sp.wall.to_numpy()[on_rows][good] - w0)
    sp['t'] = np.polyval(p, sp.wall.to_numpy() - w0)
    return dict(trials=trials, wheel=wheel, sp=sp, settings=settings,
                align=dict(n_onsets=int(good.sum()), coarse_lag_med=float(np.median(lag)),
                           resid_sd_ms=float(resid.std() * 1e3),
                           resid_p99_ms=float(np.percentile(np.abs(resid), 99) * 1e3)))
