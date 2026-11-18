"""
Can PetroSync's accuracy be improved?  A pre-registered, lockbox-protected experiment.

Protocol (decided BEFORE looking at any lockbox number):
  * DEV period     : everything before 2026-04-01  -> all choices (features, thresholds, geometry) are made here
  * LOCKBOX period : 2026-04-01 onwards            -> each final candidate is scored ONCE, never tuned on
  * All model predictions are walk-forward (trained on the past only, purged), so even DEV numbers are out-of-sample.
  * Every variant tried is listed, so the reader can judge multiple-testing risk.

Run:  python -m evaluation.improve      (needs evaluation/data_cache/signals.pkl from run_evaluation)
"""
import json
import logging
import os
import numpy as np
import pandas as pd
from scipy import stats

from evaluation.data_loader import load_history, CACHE_DIR
from evaluation.backtest import _cl_frame, attach_outcomes, simulate_trades
from evaluation.ml_models import build_features, walk_forward
from evaluation import metrics as M
from evaluation.run_evaluation import flat_band, REPORT_DIR

SPLIT = pd.Timestamp('2026-04-01', tz='UTC')
VARIANTS_TRIED = []


def zscore_past(s, win=500):
    """Rolling z-score using only past values (shifted by 1 bar)."""
    m = s.rolling(win, min_periods=100).mean().shift(1)
    sd = s.rolling(win, min_periods=100).std().shift(1)
    return ((s - m) / sd).clip(-4, 4)


def extended_features(df, cl):
    X = build_features(df, cl)
    close = cl['Close']
    for lag in (120, 480):                             # ~1 week and ~1 month of trading hours (time-series momentum)
        X[f'past_ret_{lag}'] = (close / close.shift(lag) - 1).to_numpy()[df.index]
    for col in ('crack', 'brent_spread', 'ovx', 'dxy_chg', 'es_chg', 'gold_oil'):
        X[f'z_{col}'] = zscore_past(df[col].astype(float)).to_numpy()
    X['z_dist_vwap'] = zscore_past((df.price - df.vwap) / df.atr).to_numpy()
    return X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


def dir_metrics(score, r, band, h, bull=1e-9, bear=-1e-9):
    d = M.directional_report(score, r, bull, bear, band, h)
    return {k: d[k] for k in ('n_calls', 'coverage', 'hit_rate', 'pct_bull_calls', 'base_rate_up',
                              'nonoverlap_calls', 'nonoverlap_hit_rate', 'binomial_p_vs_50pct', 'pesaran_timmermann_p')}


def prob_metrics(p, y):
    d = M.probability_report(p, y)
    return {k: d[k] for k in ('n', 'brier', 'brier_skill_score', 'roc_auc', 'expected_calibration_error')}


def split_eval(fn, df):
    dev, lock = df.time < SPLIT, df.time >= SPLIT
    return {'dev': fn(dev), 'lockbox': fn(lock)}


def trade_metrics(tr):
    done = tr[tr.outcome.isin(['TARGET1', 'STOP', 'TIMEOUT'])]
    if len(done) < 5:
        return {'n': int(len(done))}
    gp, gl = done.pnl_usd[done.pnl_usd > 0].sum(), -done.pnl_usd[done.pnl_usd < 0].sum()
    t = stats.ttest_1samp(done.pnl_usd, 0)
    return {'n': int(len(done)), 'win_rate': M._f((done.pnl_usd > 0).mean()), 'tp1_rate': M._f((done.outcome == 'TARGET1').mean()),
            'profit_factor': M._f(gp / gl if gl else np.nan, 3), 'usd_per_trade': M._f(done.pnl_usd.mean(), 2),
            'total_usd': M._f(done.pnl_usd.sum(), 0), 't_stat_mean_pnl': M._f(t.statistic, 3), 'p_mean_pnl_gt_0': M._f(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2, 4)}


def trades_split(tr):
    tt = pd.to_datetime(tr.time, utc=True)
    return {'dev': trade_metrics(tr[tt < SPLIT]), 'lockbox': trade_metrics(tr[tt >= SPLIT])}


def regeom(df, stop_atr, tgt_r):
    """Rebuild stop / target as ATR multiples around the engine's own entry price."""
    g = df.copy()
    short = g.setup_bias == 'SHORT'
    entry = np.where(short, g.entry_min, g.entry_max)
    risk = stop_atr * g.atr
    g['stop'] = np.where(short, entry + risk, entry - risk)
    g['target1'] = np.where(short, entry - tgt_r * risk, entry + tgt_r * risk)
    return g


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    hist = load_history()
    cl = _cl_frame(hist['macro_1h'])
    df = attach_outcomes(pd.read_pickle(os.path.join(CACHE_DIR, 'signals.pkl')), cl)
    X = extended_features(df, cl)
    dev = df.time < SPLIT
    res = {'protocol': {'split': str(SPLIT), 'dev_bars': int(dev.sum()), 'lockbox_bars': int((~dev).sum()),
                        'dev_range': [str(df.time.iloc[0]), str(df.time[dev].iloc[-1])],
                        'lockbox_range': [str(df.time[~dev].iloc[0]), str(df.time.iloc[-1])]},
           'direction': {}, 'calibration': {}, 'trades': {}}

    # ------------------------------------------------------------- 1. DIRECTION
    for h in (4, 24):
        r, band = df[f'fwd_ret_{h}'], flat_band(df, h)
        y = (r > 0).astype(float).where(r.notna())
        sec = {}
        sec['A_engine_sentiment'] = split_eval(lambda m: {**dir_metrics(df.sent_score[m], r[m], band[m], h, 15, -15),
                                                         **prob_metrics(df.sent_bull_pct[m] / 100, y[m])}, df)
        base_cols = list(build_features(df.iloc[:5], cl).columns)
        z_cols = [c for c in X.columns if c.startswith('z_')] + ['past_ret_1', 'past_ret_4', 'past_ret_24', 'past_ret_120', 'past_ret_480',
                                                                 'tf15_score', 'tf1h_score', 'tfD_score', 'rsi_15m', 'rsi_1h', 'atr_pct']
        feature_sets = {'B_engine_features': base_cols, 'C_zscore_plus_momentum': z_cols, 'D_all_features': list(X.columns)}
        preds = {}
        for name, cols in feature_sets.items():
            p = walk_forward(X[cols], y, h, 'clf')
            for model in p.columns:
                key = f'{name}__{model}'
                VARIANTS_TRIED.append(f'direction h{h}: {key}')
                pr = p[model]
                preds[key] = pr
                ok = pr.notna() & y.notna()
                sec[key] = split_eval(lambda m: {**dir_metrics(((pr - .5) * 200)[m & ok], r[m & ok], band[m & ok], h),
                                                 **prob_metrics(pr[m & ok], y[m & ok])}, df)
        # choose the best model by DEV AUC only, then add confidence abstention (threshold from DEV)
        best = max(preds, key=lambda k: sec[k]['dev'].get('roc_auc') or 0)
        pr = preds[best]
        conf = (pr - .5).abs()
        thr = float(conf[dev & pr.notna()].quantile(0.70))          # keep the 30 % most confident calls
        score = np.where(conf >= thr, (pr - .5) * 200, 0.0)
        score = pd.Series(score, index=df.index).where(pr.notna())
        ok = score.notna()
        VARIANTS_TRIED.append(f'direction h{h}: E_best({best})_top30pct_confidence')
        sec['E_best_dev_model_top30pct_confident'] = {'chosen_model': best, 'confidence_threshold': round(thr, 4),
            **split_eval(lambda m: dir_metrics(score[m & ok], r[m & ok], band[m & ok], h, 1e-9, -1e-9), df)}
        res['direction'][f'h{h}'] = sec
        logging.info('direction h%d done (best by dev AUC: %s)', h, best)

    # ------------------------------------------------------------- 2. CALIBRATION of "bullish %"
    for h in (4, 24):
        r = df[f'fwd_ret_{h}']; y = (r > 0).astype(float).where(r.notna())
        p_cal = walk_forward(df[['sent_score']].astype(float), y, h, 'clf')['logistic_regression']   # walk-forward Platt scaling
        ok = p_cal.notna() & y.notna()
        base_past = y.expanding(min_periods=500).mean().shift(h)                                  # climatology known at the time
        okb = base_past.notna() & y.notna()
        VARIANTS_TRIED.extend([f'calibration h{h}: platt', f'calibration h{h}: past base rate'])
        res['calibration'][f'h{h}'] = {
            'engine_bullish_pct': split_eval(lambda m: prob_metrics(df.sent_bull_pct[m & ok] / 100, y[m & ok]), df),
            'platt_calibrated_sentiment': split_eval(lambda m: prob_metrics(p_cal[m & ok], y[m & ok]), df),
            'past_base_rate_only': split_eval(lambda m: prob_metrics(base_past[m & okb], y[m & okb]), df),
        }

    # ------------------------------------------------------------- 3. TRADES
    T = res['trades']
    T['T0_engine_as_is'] = trades_split(simulate_trades(df, cl))
    T['T1_drop_range_rotation'] = trades_split(simulate_trades(df[~df.setup_type.str.startswith('VALUE AREA')], cl))
    T['T2_only_trend_continuation'] = trades_split(simulate_trades(df[df.setup_type.str.contains('TREND CONTINUATION')], cl))
    T['T3_ovx_below_30_only'] = trades_split(simulate_trades(df[df.ovx < 30], cl))
    VARIANTS_TRIED.extend(['trades: T1 drop range', 'trades: T2 trend only', 'trades: T3 OVX<30'])

    # geometry grid -> choose on DEV profit factor, then score lockbox once
    grid = {}
    for s_atr in (0.7, 1.0, 1.5):
        for t_r in (0.5, 1.0, 1.5, 2.0):
            key = f'stop{s_atr}ATR_target{t_r}R'
            grid[key] = trades_split(simulate_trades(regeom(df, s_atr, t_r), cl))
            VARIANTS_TRIED.append(f'trades: geometry {key}')
    best_g = max(grid, key=lambda k: grid[k]['dev'].get('profit_factor') or 0)
    T['T4_geometry_grid'] = {'chosen_on_dev': best_g, 'grid': grid}

    # meta-labeling: ML decides whether to take each engine setup
    lab = simulate_trades(df, cl, overlapping=True).set_index('signal_bar')
    y_meta = lab.outcome.map({'TARGET1': 1.0, 'STOP': 0.0})
    y_meta = y_meta.where(lab.outcome != 'TIMEOUT', (lab.r_multiple > 0).astype(float)).reindex(df.index)
    Xm = X.copy()
    for st in df.setup_type.unique():
        Xm['setup_' + st[:12]] = (df.setup_type == st).astype(float)
    pm = walk_forward(Xm, y_meta, 30, 'clf')['gradient_boosting']
    best_thr, best_pf = None, -1
    for thr in (0.45, 0.5, 0.55, 0.6):
        tr = simulate_trades(df[(pm >= thr).fillna(False)], cl)
        m_dev = trades_split(tr)['dev']
        VARIANTS_TRIED.append(f'trades: meta-label thr {thr}')
        if (m_dev.get('profit_factor') or 0) > best_pf and m_dev.get('n', 0) >= 100:
            best_thr, best_pf = thr, m_dev['profit_factor']
    T['T5_meta_labeling'] = {'threshold_chosen_on_dev': best_thr,
                             'label_auc_oos': split_eval(lambda m: prob_metrics(pm[m & pm.notna() & y_meta.notna()], y_meta[m & pm.notna() & y_meta.notna()]), df),
                             **trades_split(simulate_trades(df[(pm >= best_thr).fillna(False)], cl))}
    # combined: best DEV geometry + meta filter
    g = regeom(df, *[float(x) for x in best_g.replace('stop', '').replace('ATR_target', ' ').replace('R', '').split()])
    T['T6_geometry_plus_meta'] = trades_split(simulate_trades(g[(pm >= best_thr).fillna(False)], cl))
    VARIANTS_TRIED.append('trades: T6 geometry + meta')

    # ------------------------------------------------------------- 4. deployable calibration (fit on ALL history)
    from sklearn.linear_model import LogisticRegression
    y4 = (df.fwd_ret_4 > 0).astype(float); okc = df.fwd_ret_4.notna()
    lr = LogisticRegression(C=1.0).fit(df.loc[okc, ['sent_score']].astype(float).to_numpy(), y4[okc].astype(int))
    calib = {'model': 'platt', 'target': 'P(WTI higher 4 hourly bars later)', 'feature': 'sentiment composite score',
             'coef': float(lr.coef_[0][0]), 'intercept': float(lr.intercept_[0]), 'n': int(okc.sum()),
             'fitted_on': [str(df.time.iloc[0]), str(df.time.iloc[-1])],
             'lockbox_brier_skill_walkforward': res['calibration']['h4']['platt_calibrated_sentiment']['lockbox']['brier_skill_score']}
    json.dump(calib, open(os.path.join(REPORT_DIR, 'engine_calibration.json'), 'w'), indent=2)
    res['deployed_calibration'] = calib

    res['variants_tried'] = VARIANTS_TRIED
    res['n_variants'] = len(VARIANTS_TRIED)
    os.makedirs(REPORT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(REPORT_DIR, 'improvement_results.json'), 'w'), indent=2, default=str)
    write_markdown(res, os.path.join(REPORT_DIR, 'IMPROVEMENT_EXPERIMENTS.md'))
    logging.info('wrote reports/improvement_results.json + IMPROVEMENT_EXPERIMENTS.md (%d variants)', len(VARIANTS_TRIED))


def write_markdown(res, path):
    P = lambda x: '—' if x is None else f'{x*100:.1f}%'
    N = lambda x, d=3: '—' if x is None else f'{x:.{d}f}'
    L = []; a = L.append
    pr = res['protocol']
    a('# Can the accuracy be improved? — Lockbox experiment\n')
    a(f"_DEV {pr['dev_range'][0][:10]} → {pr['dev_range'][1][:10]} ({pr['dev_bars']:,} bars) · "
      f"LOCKBOX {pr['lockbox_range'][0][:10]} → {pr['lockbox_range'][1][:10]} ({pr['lockbox_bars']:,} bars) · "
      f"{res['n_variants']} variants tried. All choices made on DEV; LOCKBOX scored once. "
      'All model predictions are walk-forward (trained on the past only)._\n')
    a('> Read lockbox numbers *relative to the baseline row in the same period*. The lockbox months happened to suit the strategy, so even the unchanged engine looks better there.\n')
    for h, sec in res['direction'].items():
        a(f'## Direction, {h[1:]}-hour horizon\n')
        a('| Variant | DEV hit | DEV % up | DEV PT p | DEV AUC | LOCK hit | LOCK % up | LOCK base | LOCK PT p | LOCK AUC | LOCK coverage |')
        a('|---|---|---|---|---|---|---|---|---|---|---|')
        for k, v in sec.items():
            d, l = v['dev'], v['lockbox']
            nm = k + (f" (= {v['chosen_model']})" if 'chosen_model' in v else '')
            a(f"| {nm} | {P(d.get('hit_rate'))} | {P(d.get('pct_bull_calls'))} | {N(d.get('pesaran_timmermann_p'))} | {N(d.get('roc_auc'))} | "
              f"{P(l.get('hit_rate'))} | {P(l.get('pct_bull_calls'))} | {P(l.get('base_rate_up'))} | {N(l.get('pesaran_timmermann_p'))} | {N(l.get('roc_auc'))} | {P(l.get('coverage'))} |")
        a('')
    a('## Probability calibration of "bullish %"\n')
    a('| Horizon | Method | DEV Brier skill | DEV ECE | LOCK Brier skill | LOCK ECE |')
    a('|---|---|---|---|---|---|')
    for h, sec in res['calibration'].items():
        for k, v in sec.items():
            a(f"| {h[1:]}h | {k} | {N(v['dev']['brier_skill_score'],4)} | {N(v['dev']['expected_calibration_error'])} | {N(v['lockbox']['brier_skill_score'],4)} | {N(v['lockbox']['expected_calibration_error'])} |")
    a('\n## Trade playbook variants\n')
    a('| Variant | DEV n | DEV win | DEV PF | DEV $/trade | LOCK n | LOCK win | LOCK PF | LOCK $/trade | LOCK p($>0) |')
    a('|---|---|---|---|---|---|---|---|---|---|')
    T = res['trades']
    def row(name, v):
        d, l = v['dev'], v['lockbox']
        a(f"| {name} | {d.get('n')} | {P(d.get('win_rate'))} | {N(d.get('profit_factor'))} | {d.get('usd_per_trade')} | "
          f"{l.get('n')} | {P(l.get('win_rate'))} | {N(l.get('profit_factor'))} | {l.get('usd_per_trade')} | {N(l.get('p_mean_pnl_gt_0'),4)} |")
    for k, v in T.items():
        if k == 'T4_geometry_grid':
            for gk, gv in v['grid'].items():
                row(f"T4 {gk}{' **(chosen on DEV)**' if gk == v['chosen_on_dev'] else ''}", gv)
        elif k == 'T5_meta_labeling':
            row(f"T5 meta-labeling (threshold {v['threshold_chosen_on_dev']})", v)
        else:
            row(k, v)
    m = T['T5_meta_labeling']['label_auc_oos']
    a(f"\nMeta-label model AUC (does ML predict whether a setup hits TP1 before the stop?): DEV {N(m['dev']['roc_auc'])}, LOCK {N(m['lockbox']['roc_auc'])}. "
      'Part of this is mechanical: targets that are close relative to ATR are hit more often.\n')
    a('## Every variant tried\n')
    a(', '.join(f'`{v}`' for v in res['variants_tried']))
    open(path, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
