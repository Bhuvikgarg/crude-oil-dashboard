"""
Forward test (true out-of-sample) for the PetroSync candidates.

Three commands:

  python -m evaluation.forward_test freeze
      Train the candidate models on ALL backtest history, save them with SHA-256
      fingerprints and write forward_test/manifest.json + PREREGISTRATION.md.
      Run ONCE, commit, and never change these files during the test.

  python -m evaluation.forward_test run [--log-dir DIR]
      Download live data, record every completed hourly bar not yet logged
      (calls are written BEFORE their outcome exists), then score every call
      whose outcome is now known. Designed to run hourly from GitHub Actions,
      which commits DIR to the `forward-test-log` branch (timestamped evidence).

  python -m evaluation.forward_test score [--log-dir DIR]
      Re-score the log without recording anything.

What is tested (pre-registered in the manifest):
  A. Engine baseline        : live 7-pillar sentiment call (+/-15) and the engine's own playbook trades
  B. Calibrated probability : engine `calibrated_p_up_4h` (Brier skill vs frozen climatology)
  C. Confident ML direction : gradient boosting P(up in 4h); call only when today's confidence |p-0.5| is in the
                              top 30 % of the model's own last 500 hourly confidences (rolling rule, primary).
                              The original fixed threshold from the lockbox experiment is logged as C_fixed (secondary).
  D. Improved playbook      : engine setups re-sized to the frozen stop/target geometry, taken only
                              when the frozen meta-label model says p(win) >= frozen threshold

Train/serve consistency: the ML models were trained mostly on the 1h proxy for the 15-minute
timeframe (real 15m data only covers ~60 days), so C and D are computed on a proxy replay.
A and B use the real live 15m data, exactly like the dashboard.
"""
import argparse
import hashlib
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluation import metrics as M
from evaluation.backtest import _cl_frame, attach_outcomes, replay, simulate_trades
from evaluation.data_loader import CACHE_DIR, MACRO_TICKERS, load_history
from evaluation.improve import extended_features, regeom
from evaluation.ml_models import build_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FT_DIR = os.path.join(ROOT, 'forward_test')
MODEL_DIR = os.path.join(FT_DIR, 'models')
DEFAULT_LOG_DIR = os.path.join(FT_DIR, 'log')
HORIZON = 4
TEST_DAYS = 91


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def _gb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=42)


def _meta_matrix(X, setup_types, columns=None):
    Xm = X.copy()
    for st in sorted(setup_types.unique()):
        Xm['setup_' + st[:12]] = (setup_types == st).astype(float)
    return Xm if columns is None else Xm.reindex(columns=columns, fill_value=0.0)


# ====================================================================== freeze
def freeze():
    logging.info('Freezing forward-test models ...')
    hist = load_history()
    cl = _cl_frame(hist['macro_1h'])
    df = attach_outcomes(pd.read_pickle(os.path.join(CACHE_DIR, 'signals.pkl')), cl)
    imp = json.load(open(os.path.join(ROOT, 'reports', 'improvement_results.json')))
    E = imp['direction']['h4']['E_best_dev_model_top30pct_confident']
    assert E['chosen_model'] == 'B_engine_features__gradient_boosting', E['chosen_model']
    conf_thr = float(E['confidence_threshold'])
    meta_thr = float(imp['trades']['T5_meta_labeling']['threshold_chosen_on_dev'])
    geom = imp['trades']['T4_geometry_grid']['chosen_on_dev']           # e.g. stop1.5ATR_target1.5R
    stop_atr, tgt_r = [float(x) for x in geom.replace('stop', '').replace('ATR_target', ' ').replace('R', '').split()]

    # C. direction model on the engine feature set
    X_dir = build_features(df, cl)
    y = (df.fwd_ret_4 > 0).astype(float).where(df.fwd_ret_4.notna())
    ok = y.notna()
    dir_model = _gb().fit(X_dir[ok].to_numpy(), y[ok].astype(int).to_numpy())

    # D. meta-label model (labels = would the ORIGINAL engine setup hit TP1 before the stop?)
    X_ext = extended_features(df, cl)
    lab = simulate_trades(df, cl, overlapping=True).set_index('signal_bar')
    y_meta = lab.outcome.map({'TARGET1': 1.0, 'STOP': 0.0})
    y_meta = y_meta.where(lab.outcome != 'TIMEOUT', (lab.r_multiple > 0).astype(float)).reindex(df.index)
    Xm = _meta_matrix(X_ext, df.setup_type)
    okm = y_meta.notna()
    meta_model = _gb().fit(Xm[okm].to_numpy(), y_meta[okm].astype(int).to_numpy())

    os.makedirs(MODEL_DIR, exist_ok=True)
    p_dir, p_meta = os.path.join(MODEL_DIR, 'direction_gb_h4.joblib'), os.path.join(MODEL_DIR, 'meta_label_gb.joblib')
    joblib.dump(dir_model, p_dir)
    joblib.dump(meta_model, p_meta)
    calib_path = os.path.join(ROOT, 'reports', 'engine_calibration.json')

    # In-sample coverage of the frozen threshold (for reference only)
    conf_in = np.abs(dir_model.predict_proba(X_dir[ok].to_numpy())[:, 1] - 0.5)
    base_rate = float(y[ok].mean())

    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    bars_expected = int(TEST_DAYS * 23 * 5 / 7)
    manifest = {
        'frozen_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'start_utc': start.strftime('%Y-%m-%d %H:%M'), 'planned_end_utc': (start + timedelta(days=TEST_DAYS)).strftime('%Y-%m-%d %H:%M'),
        'horizon_bars': HORIZON, 'training_data': [str(df.time.iloc[0]), str(df.time.iloc[-1])], 'training_rows': int(ok.sum()),
        'frozen_climatology_p_up_4h': round(base_rate, 4),
        'direction_model': {'file': 'models/direction_gb_h4.joblib', 'sha256': _sha256(p_dir), 'features': list(X_dir.columns),
                            'confidence_rule': {'type': 'rolling_quantile', 'window_bars': 500, 'quantile': 0.70},
                            'confidence_threshold_fixed_secondary': conf_thr,
                            'in_sample_coverage_at_fixed_threshold': round(float((conf_in >= conf_thr).mean()), 4)},
        'meta_model': {'file': 'models/meta_label_gb.joblib', 'sha256': _sha256(p_meta), 'features': list(Xm.columns),
                       'take_threshold': meta_thr, 'stop_atr': stop_atr, 'target_r': tgt_r},
        'calibration': {'file': 'reports/engine_calibration.json', 'sha256': _sha256(calib_path)},
        'success_criteria': {
            'B_calibration': 'PASS if Brier skill vs frozen climatology >= -0.005 (the probability is at least as good as knowing the base rate)',
            'C_confident_ml': 'Primary = rolling rule. CONFIRMED if hit rate > 50% AND Pesaran-Timmermann p < 0.05 on '
                              'non-overlapping calls; REJECTED if hit rate <= 50%; otherwise INCONCLUSIVE. '
                              'C_fixed is reported with the same criteria as a secondary result.',
            'D_improved_playbook': 'BETTER if profit factor > engine-playbook profit factor AND mean $/trade > engine mean $/trade; '
                                   'PROFITABLE only if additionally mean $/trade > 0 with one-sided t-test p < 0.05',
            'A_engine': 'Reference only (expected ~50% hit rate based on the backtest)',
        },
        'amendment_before_start': (
            'The lockbox experiment used a fixed confidence threshold (0.1402). The final model trained on all data is less '
            'confident (its 70th-percentile confidence is ~0.065 live vs 0.140 in DEV), so the fixed threshold would call ~5% of '
            'hours. The primary rule was therefore changed, before any forward data was logged, to the top 30% relative to the '
            'model\'s own previous 500 hourly confidences. Walk-forward backtest of this rule: DEV 55.2% hit (PT p 0.008, coverage '
            '23.6%), LOCKBOX 54.2% (PT p 0.34, coverage 29.4%). Because its lockbox result was seen while designing the fix, the '
            'forward test is its first clean evaluation.'),
        'power_note': {
            'bars_expected': bars_expected,
            'confident_calls_expected': [int(bars_expected * 0.24), int(bars_expected * 0.30)],
            'nonoverlap_calls_expected': [int(bars_expected * 0.24 / HORIZON), int(bars_expected * 0.30 / HORIZON)],
            'calls_needed_to_detect_54.5pct_at_80pct_power': int(math.ceil(((1.645 + 0.842) ** 2 * 0.25) / 0.045 ** 2)),
        },
    }
    json.dump(manifest, open(os.path.join(FT_DIR, 'manifest.json'), 'w'), indent=2)
    _write_prereg(manifest)
    logging.info('Frozen. Start %s, end %s. Commit forward_test/ now.', manifest['start_utc'], manifest['planned_end_utc'])


def _write_prereg(m):
    pn = m['power_note']
    txt = f"""# Forward-test pre-registration

Frozen {m['frozen_at_utc']} UTC. **Test window: {m['start_utc']} → {m['planned_end_utc']} UTC ({TEST_DAYS} days).**
These files must not change during the test. The SHA-256 fingerprints below let anyone check that.

| Artifact | SHA-256 |
|---|---|
| `{m['direction_model']['file']}` | `{m['direction_model']['sha256']}` |
| `{m['meta_model']['file']}` | `{m['meta_model']['sha256']}` |
| `{m['calibration']['file']}` | `{m['calibration']['sha256']}` |

Models were trained on {m['training_data'][0][:10]} → {m['training_data'][1][:10]} ({m['training_rows']:,} hourly rows).
Settings come from the lockbox experiment (`reports/IMPROVEMENT_EXPERIMENTS.md`) and are not tuned again.

## What is tested

| Track | Rule | Success criterion |
|---|---|---|
| A. Engine baseline | Live sentiment call (≥ +15 bull, ≤ −15 bear) and the engine's own playbook | Reference only |
| B. Calibrated probability | Engine `calibrated_p_up_4h` | {m['success_criteria']['B_calibration']} |
| C. Confident ML direction | Gradient boosting P(up in 4h); call only if \\|p − 0.5\\| is in the top 30 % of the model's previous 500 hourly confidences (secondary: fixed threshold {m['direction_model']['confidence_threshold_fixed_secondary']}) | {m['success_criteria']['C_confident_ml']} |
| D. Improved playbook | Engine setups with stop {m['meta_model']['stop_atr']}×ATR and target {m['meta_model']['target_r']}R, taken only if meta-model p(win) ≥ {m['meta_model']['take_threshold']} | {m['success_criteria']['D_improved_playbook']} |

Frozen climatology for Brier skill: P(up in 4h) = {m['frozen_climatology_p_up_4h']}.

## Amendment made before the test started

{m['amendment_before_start']}

## How evidence is recorded
- A GitHub Actions job runs hourly, records every completed hourly bar **before its outcome exists**, and commits the log to the `forward-test-log` branch. The commit timestamps are the proof.
- Bars recorded more than 75 minutes after they closed (scheduler delays, outages) are marked `backfilled`. They are still computed point-in-time, and the report shows results with and without them.

## Statistical power (read before judging the result)
About {pn['bars_expected']:,} hourly bars are expected. With 24–30 % coverage, track C will make roughly
{pn['confident_calls_expected'][0]}–{pn['confident_calls_expected'][1]} calls, or {pn['nonoverlap_calls_expected'][0]}–{pn['nonoverlap_calls_expected'][1]} non-overlapping 4-hour calls.
Detecting a true 54.5 % hit rate with 80 % power needs about **{pn['calls_needed_to_detect_54.5pct_at_80pct_power']} non-overlapping calls**.
So {TEST_DAYS} days can **reject** the candidate (if it lands near 50 %) or show it is **consistent** with the backtest,
but it is unlikely to **prove** it. Plan to extend the test if the result is promising.
"""
    open(os.path.join(FT_DIR, 'PREREGISTRATION.md'), 'w').write(txt)


# ====================================================================== live data
def _download(days):
    days = int(min(729, max(60, days)))
    macro = yf.download(MACRO_TICKERS, period=f'{days}d', interval='1h', progress=False, auto_adjust=False)
    cl15 = yf.Ticker('CL=F').history(period='5d', interval='15m')
    cld = yf.Ticker('CL=F').history(period='2y', interval='1d')
    for df in (macro, cl15, cld):
        df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
    if len(macro) == 0 or len(cl15) == 0:
        raise RuntimeError('Yahoo returned no data')
    return {'macro_1h': macro.sort_index(), 'cl_15m': cl15.sort_index(), 'cl_daily': cld.sort_index()}


def _load_manifest():
    path = os.path.join(FT_DIR, 'manifest.json')
    if not os.path.exists(path):
        raise SystemExit('No forward_test/manifest.json. Run: python -m evaluation.forward_test freeze')
    m = json.load(open(path))
    for key in ('direction_model', 'meta_model'):
        f = os.path.join(FT_DIR, m[key]['file'])
        if _sha256(f) != m[key]['sha256']:
            raise SystemExit(f'{f} does not match its pre-registered SHA-256. The test is invalid if models change.')
    return m


def _append_run(log_dir, status, detail=''):
    path = os.path.join(log_dir, 'runs.csv')
    new = not os.path.exists(path)
    with open(path, 'a') as f:
        if new:
            f.write('run_utc,status,detail\n')
        f.write(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')},{status},\"{detail}\"\n")


# ====================================================================== record
def record(hist, m, log_dir, backfill_days=0):
    now = pd.Timestamp.now(tz='UTC')
    start = pd.Timestamp(m['start_utc'], tz='UTC')
    if backfill_days:
        start = now - pd.Timedelta(days=backfill_days)           # testing only: rows are marked backfilled
    calls_path = os.path.join(log_dir, 'calls.csv')
    old = pd.read_csv(calls_path, parse_dates=['bar_close_utc']) if os.path.exists(calls_path) else None
    last = old.bar_close_utc.max() if old is not None and len(old) else None
    if last is not None and last.tzinfo is None:
        last = last.tz_localize('UTC')

    cl = _cl_frame(hist['macro_1h'])
    # Catch up every bar since the last logged one (or since the start), e.g. after scheduler outages.
    since = last if last is not None else start
    n_new_max = int(min(2000, max(2, (now - since).total_seconds() / 3600 + 5)))
    # Track A/B: real live 15m data (what the dashboard shows)
    real, _ = replay(hist, max_bars=n_new_max + 2)
    # Track C/D: 15m proxy, consistent with how the models were trained; 700 bars for rolling z-scores
    prox, _ = replay(hist, max_bars=max(700, n_new_max + 600), force_15m_proxy=True)

    def keep(df):
        df = df[(df.time <= now) & (df.time >= start)]
        return df if last is None else df[df.time > last]
    real_new = keep(real)
    if real_new.empty:
        return 0

    dir_model = joblib.load(os.path.join(FT_DIR, m['direction_model']['file']))
    meta_model = joblib.load(os.path.join(FT_DIR, m['meta_model']['file']))
    X_dir = build_features(prox, cl).reindex(columns=m['direction_model']['features'], fill_value=0.0)
    p_up = pd.Series(dir_model.predict_proba(X_dir.to_numpy())[:, 1], index=prox.index)
    Xm = _meta_matrix(extended_features(prox, cl), prox.setup_type, m['meta_model']['features'])
    p_win = pd.Series(meta_model.predict_proba(Xm.to_numpy())[:, 1], index=prox.index)
    g = regeom(prox, m['meta_model']['stop_atr'], m['meta_model']['target_r'])
    rule = m['direction_model']['confidence_rule']
    conf = (p_up - 0.5).abs()
    roll_thr = conf.rolling(rule['window_bars'], min_periods=200).quantile(rule['quantile']).shift(1)   # past bars only
    thr_fixed, mthr = m['direction_model']['confidence_threshold_fixed_secondary'], m['meta_model']['take_threshold']

    rows = []
    for i, r in real_new.iterrows():
        rec = {'recorded_at_utc': now.strftime('%Y-%m-%d %H:%M:%S'), 'bar_close_utc': r.time.strftime('%Y-%m-%d %H:%M'),
               'backfilled': bool(backfill_days) or (now - r.time) > pd.Timedelta(minutes=75),
               'price': r.price, 'atr': r.atr,
               'A_sent_score': r.sent_score,
               'A_call': 'BULL' if r.sent_score >= 15 else 'BEAR' if r.sent_score <= -15 else 'NEUTRAL',
               'A_setup': r.setup_type, 'A_side': r.setup_bias, 'A_entry_min': r.entry_min, 'A_entry_max': r.entry_max,
               'A_stop': r.stop, 'A_target1': r.target1, 'A_claimed_win': r.claimed_win_prob,
               'B_cal_p_up': r.get('sent_cal_p')}
        if i in prox.index:
            pu, pw, gp = float(p_up[i]), float(p_win[i]), g.loc[i]
            side = 'BULL' if pu > 0.5 else 'BEAR'
            rt = roll_thr[i]
            rec.update({'C_p_up': round(pu, 4), 'C_rolling_threshold': None if pd.isna(rt) else round(float(rt), 4),
                        'C_call': side if (not pd.isna(rt) and abs(pu - 0.5) >= rt) else 'NONE',
                        'C_fixed_call': side if abs(pu - 0.5) >= thr_fixed else 'NONE',
                        'D_setup': gp.setup_type, 'D_side': gp.setup_bias, 'D_entry_min': gp.entry_min, 'D_entry_max': gp.entry_max,
                        'D_stop': round(float(gp.stop), 2), 'D_target1': round(float(gp.target1), 2),
                        'D_p_win': round(pw, 4), 'D_take': pw >= mthr})
        rows.append(rec)
    new = pd.DataFrame(rows)
    out = new if old is None else pd.concat([old.assign(bar_close_utc=old.bar_close_utc.dt.strftime('%Y-%m-%d %H:%M')), new], ignore_index=True)
    os.makedirs(log_dir, exist_ok=True)
    out.to_csv(calls_path, index=False)
    return len(new)


# ====================================================================== score
def _bars_index(cl, times):
    """Map each decision time (bar close) to the position of that bar in cl."""
    pos = pd.Series(np.arange(len(cl)), index=cl.index + pd.Timedelta(hours=1))
    return pos.reindex(times).to_numpy()


def score(hist, m, log_dir, include_backfilled=True):
    calls_path = os.path.join(log_dir, 'calls.csv')
    if not os.path.exists(calls_path):
        return None
    log = pd.read_csv(calls_path, parse_dates=['bar_close_utc'])
    log['bar_close_utc'] = pd.to_datetime(log.bar_close_utc, utc=True)
    if not include_backfilled:
        log = log[~log.backfilled.astype(bool)]
    cl = _cl_frame(hist['macro_1h'])
    log['bar'] = _bars_index(cl, log.bar_close_utc)
    log = log.dropna(subset=['bar'])
    log['bar'] = log.bar.astype(int)
    close = cl.Close.to_numpy()
    fut = np.full(len(close), np.nan); fut[:-HORIZON] = close[HORIZON:]
    log['fwd_ret'] = fut[log.bar] / close[log.bar] - 1
    known = log[log.fwd_ret.notna()]
    res = {'n_logged': int(len(log)), 'n_scored': int(len(known)),
           'first_bar': str(log.bar_close_utc.min()) if len(log) else None, 'last_bar': str(log.bar_close_utc.max()) if len(log) else None,
           'backfilled_share': M._f(log.backfilled.astype(bool).mean()) if len(log) else None}
    if len(known) < 10:
        return res
    band = 0.25 * known.atr * np.sqrt(HORIZON) / known.price

    def direction(call_col):
        s = known[call_col].map({'BULL': 1.0, 'BEAR': -1.0}).fillna(0.0)
        d = M.directional_report(s, known.fwd_ret, 0.5, -0.5, band, HORIZON)
        return {k: d[k] for k in ('n_calls', 'coverage', 'hit_rate', 'pct_bull_calls', 'base_rate_up',
                                  'nonoverlap_calls', 'nonoverlap_hit_rate', 'binomial_p_vs_50pct', 'pesaran_timmermann_p')}
    res['A_engine_direction'] = direction('A_call')
    res['C_confident_ml'] = direction('C_call') if 'C_call' in known else {}
    res['C_fixed_threshold'] = direction('C_fixed_call') if 'C_fixed_call' in known else {}
    y = (known.fwd_ret > 0).astype(float)
    clim = m['frozen_climatology_p_up_4h']
    if known.B_cal_p_up.notna().sum() >= 10:
        kb = known.B_cal_p_up.notna()
        p = known.B_cal_p_up[kb] / 100
        brier = float(np.mean((p - y[kb]) ** 2)); brier_ref = float(np.mean((clim - y[kb]) ** 2))
        res['B_calibration'] = {'n': int(kb.sum()), 'brier': M._f(brier), 'brier_frozen_climatology': M._f(brier_ref),
                                'brier_skill': M._f(1 - brier / brier_ref), 'mean_p': M._f(p.mean()), 'observed_up': M._f(y[kb].mean())}

    def trades(prefix, mask):
        d = log[mask].copy()
        if d.empty:
            return {'n': 0}
        sig = pd.DataFrame({'time': d.bar_close_utc.values, 'setup_type': d[f'{prefix}_setup'].values,
                            'setup_bias': d[f'{prefix}_side'].values, 'entry_min': d[f'{prefix}_entry_min'].values,
                            'entry_max': d[f'{prefix}_entry_max'].values, 'stop': d[f'{prefix}_stop'].values,
                            'target1': d[f'{prefix}_target1'].values,
                            'claimed_win_prob': d['A_claimed_win'].values}, index=d.bar.values).sort_index()
        sig = sig[~sig.index.duplicated()]
        tr = simulate_trades(sig, cl)
        tr = tr[tr.outcome.isin(['TARGET1', 'STOP', 'TIMEOUT'])]
        # only count trades whose 30-bar window has fully elapsed
        tr = tr[tr.signal_bar + 30 < len(cl)]
        if tr.empty:
            return {'n': 0}
        gp, gl = tr.pnl_usd[tr.pnl_usd > 0].sum(), -tr.pnl_usd[tr.pnl_usd < 0].sum()
        t = stats.ttest_1samp(tr.pnl_usd, 0) if len(tr) > 2 else None
        return {'n': int(len(tr)), 'win_rate': M._f((tr.pnl_usd > 0).mean()), 'profit_factor': M._f(gp / gl if gl else np.nan, 3),
                'usd_per_trade': M._f(tr.pnl_usd.mean(), 2), 'total_usd': M._f(tr.pnl_usd.sum(), 0),
                'p_mean_gt_0': M._f((t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2), 4) if t is not None and not np.isnan(t.statistic) else None}
    res['A_engine_playbook'] = trades('A', pd.Series(True, index=log.index))
    res['D_improved_playbook'] = trades('D', log.get('D_take', pd.Series(False, index=log.index)).fillna(False).astype(bool))
    res['verdicts'] = _verdicts(res)
    return res


def _verdicts(r):
    v = {}
    b = r.get('B_calibration')
    if b:
        v['B_calibration'] = 'PASS' if b['brier_skill'] is not None and b['brier_skill'] >= -0.005 else 'FAIL'
    for key in ('C_confident_ml', 'C_fixed_threshold'):
        c = r.get(key) or {}
        if c.get('n_calls'):
            hr, pt = c.get('hit_rate'), c.get('pesaran_timmermann_p')
            v[key] = ('REJECTED' if hr is not None and hr <= 0.5 else
                      'CONFIRMED' if pt is not None and pt < 0.05 else 'INCONCLUSIVE')
    a, d = r.get('A_engine_playbook', {}), r.get('D_improved_playbook', {})
    if a.get('n') and d.get('n'):
        better = (d['profit_factor'] or 0) > (a['profit_factor'] or 0) and (d['usd_per_trade'] or -1e9) > (a['usd_per_trade'] or -1e9)
        prof = better and (d['usd_per_trade'] or 0) > 0 and (d.get('p_mean_gt_0') or 1) < 0.05
        v['D_improved_playbook'] = 'PROFITABLE' if prof else 'BETTER THAN ENGINE' if better else 'NOT BETTER'
    return v


def write_report(m, res_all, res_live, log_dir):
    P = lambda x: '—' if x is None else f'{x*100:.1f}%'
    N = lambda x, d=3: '—' if x is None else f'{x:.{d}f}'
    start, end = pd.Timestamp(m['start_utc'], tz='UTC'), pd.Timestamp(m['planned_end_utc'], tz='UTC')
    now = pd.Timestamp.now(tz='UTC')
    prog = max(0.0, min(1.0, (now - start) / (end - start)))
    L = [f"# Forward-test report", '',
         f"_Updated {now.strftime('%Y-%m-%d %H:%M')} UTC · window {m['start_utc']} → {m['planned_end_utc']} UTC · "
         f"**{prog*100:.0f}% complete** · pre-registration: `forward_test/PREREGISTRATION.md`_", '']
    if not res_all or res_all.get('n_scored', 0) < 10:
        L += [f"Logged {res_all.get('n_logged', 0) if res_all else 0} bars so far. Results appear once at least 10 calls have a known 4-hour outcome."]
    else:
        L += [f"Logged **{res_all['n_logged']:,}** hourly bars, **{res_all['n_scored']:,}** with a known outcome. "
              f"Backfilled share: {P(res_all['backfilled_share'])}.", '',
              '## Verdicts so far (pre-registered criteria)', '',
              '| Track | All bars | Live-recorded bars only |', '|---|---|---|']
        for k in ('B_calibration', 'C_confident_ml', 'C_fixed_threshold', 'D_improved_playbook'):
            L.append(f"| {k} | {res_all.get('verdicts', {}).get(k, '—')} | {(res_live or {}).get('verdicts', {}).get(k, '—')} |")
        L += ['', '## Direction (4-hour horizon)', '',
              '| Track | Calls | Coverage | % up | Hit rate | Base rate | Non-overlap calls | p vs 50% | Pesaran–Timmermann p |',
              '|---|---|---|---|---|---|---|---|---|']
        for k, nm in (('A_engine_direction', 'A. Engine sentiment'), ('C_confident_ml', 'C. Confident ML (rolling, primary)'),
                      ('C_fixed_threshold', 'C. Confident ML (fixed threshold, secondary)')):
            d = res_all.get(k) or {}
            L.append(f"| {nm} | {d.get('n_calls', 0)} | {P(d.get('coverage'))} | {P(d.get('pct_bull_calls'))} | {P(d.get('hit_rate'))} | "
                     f"{P(d.get('base_rate_up'))} | {d.get('nonoverlap_calls', 0)} | {N(d.get('binomial_p_vs_50pct'), 4)} | {N(d.get('pesaran_timmermann_p'), 4)} |")
        b = res_all.get('B_calibration')
        if b:
            L += ['', '## B. Calibrated probability', '',
                  f"Brier {N(b['brier'], 4)} vs frozen climatology {N(b['brier_frozen_climatology'], 4)} → **Brier skill {N(b['brier_skill'], 4)}** "
                  f"(n = {b['n']}; mean forecast {P(b['mean_p'])}, observed up-rate {P(b['observed_up'])})."]
        L += ['', '## Playbooks', '', '| Track | Trades | Win rate | Profit factor | $/trade | Total $ | p($ > 0) |', '|---|---|---|---|---|---|---|']
        for k, nm in (('A_engine_playbook', 'A. Engine playbook'), ('D_improved_playbook', 'D. Improved playbook')):
            d = res_all.get(k) or {}
            L.append(f"| {nm} | {d.get('n', 0)} | {P(d.get('win_rate'))} | {N(d.get('profit_factor'))} | {d.get('usd_per_trade', '—')} | {d.get('total_usd', '—')} | {N(d.get('p_mean_gt_0'), 4)} |")
        need = m['power_note']['calls_needed_to_detect_54.5pct_at_80pct_power']
        have = (res_all.get('C_confident_ml') or {}).get('nonoverlap_calls', 0)
        L += ['', f"_Power check: track C has {have} of the ~{need} non-overlapping calls needed to detect a 54.5 % hit rate with 80 % power._"]
    open(os.path.join(log_dir, 'FORWARD_TEST_REPORT.md'), 'w').write('\n'.join(L) + '\n')
    json.dump({'all_bars': res_all, 'live_recorded_only': res_live}, open(os.path.join(log_dir, 'forward_results.json'), 'w'), indent=2, default=str)


# ====================================================================== CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['freeze', 'run', 'score'])
    ap.add_argument('--log-dir', default=DEFAULT_LOG_DIR)
    ap.add_argument('--backfill-days', type=int, default=0, help='TESTING ONLY: record past days (marked backfilled)')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.command == 'freeze':
        return freeze()
    m = _load_manifest()
    os.makedirs(args.log_dir, exist_ok=True)
    start = pd.Timestamp(m['start_utc'], tz='UTC')
    days = (pd.Timestamp.now(tz='UTC') - start).days + 45 + args.backfill_days
    try:
        hist = _download(days)
    except Exception as e:
        _append_run(args.log_dir, 'download_error', str(e)[:200])
        raise
    n = 0
    if args.command == 'run':
        n = record(hist, m, args.log_dir, args.backfill_days)
    res_all = score(hist, m, args.log_dir, include_backfilled=True)
    res_live = score(hist, m, args.log_dir, include_backfilled=False)
    write_report(m, res_all, res_live, args.log_dir)
    if args.command == 'run':
        _append_run(args.log_dir, 'ok', f'{n} new bars')
    logging.info('%s: %d new bars recorded; report written to %s', args.command, n, args.log_dir)


if __name__ == '__main__':
    main()
