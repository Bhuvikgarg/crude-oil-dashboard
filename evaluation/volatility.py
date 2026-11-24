"""
Volatility forecasting, and does volatility help predict direction?

Direction of WTI is almost unpredictable (return autocorrelation ~ -0.008), but the
SIZE of moves clusters (autocorrelation of |returns| ~ 0.2): volatility clustering
(Engle 1982; Bollerslev 1986). This module:

  1. forecasts realised volatility over the next h bars (h = 4, 24) with
       * a naive persistence forecast (next = last 24 bars),
       * HAR (Corsi 2009): ridge regression on log realised vol over 1, 4, 24, 96 bars
         plus log OVX (implied vol) and hour of day,
       * gradient boosting on the HAR inputs plus all 31 project features;
  2. tests whether volatility improves DIRECTION prediction
       * gradient boosting on 31 features vs 31 features + volatility inputs,
       * direction hit rate of the baseline model in predicted calm vs stormy hours.

Same protocol as the rest of the project: walk-forward training (past only, purged),
DEV before 2026-04-01 for judgement, LOCKBOX Apr-Nov 2026 scored once.
Also exports the HAR model fitted on all history for the live dashboard.

Run: python -m evaluation.volatility
"""
import json
import logging
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import metrics as M
from evaluation.backtest import _cl_frame, attach_outcomes
from evaluation.data_loader import CACHE_DIR, load_history
from evaluation.ml_models import build_features, walk_forward
from evaluation.run_evaluation import REPORT_DIR, flat_band

SPLIT = pd.Timestamp('2026-04-01', tz='UTC')
BURN_IN = 2000
LAGS = (1, 4, 24, 96)
EPS = 1e-5


def log_rv_past(r, k):
    """log of per-bar realised vol over the last k bars (inclusive), known at the bar's close."""
    return np.log(np.sqrt((r ** 2).rolling(k, min_periods=k).mean()) + EPS)


def log_rv_future(r, h):
    """log of per-bar realised vol over the NEXT h bars (the target)."""
    fut = (r ** 2).rolling(h, min_periods=h).mean().shift(-h)
    return np.log(np.sqrt(fut) + EPS)


def har_features(df, cl):
    r = np.log(cl['Close']).diff()
    pos = df.index
    X = pd.DataFrame(index=df.index)
    for k in LAGS:
        X[f'log_rv_{k}'] = log_rv_past(r, k).to_numpy()[pos]
    X['log_ovx'] = np.log(df.ovx.astype(float).clip(lower=5))
    hour = pd.DatetimeIndex(df.time).tz_convert('America/New_York').hour
    X['hour_sin'], X['hour_cos'] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    return X.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def vol_report(pred, truth, naive, past_median, mask):
    m = mask & pred.notna() & truth.notna() & naive.notna() & past_median.notna()
    p, t, n, med = pred[m], truth[m], naive[m], past_median[m]
    sse, sse_n = ((t - p) ** 2).sum(), ((t - n) ** 2).sum()
    sst = ((t - t.mean()) ** 2).sum()
    hi_true, hi_pred = (t > med), (p > med)
    auc = M.probability_report(stats.rankdata(p) / len(p), hi_true.astype(float))['roc_auc']
    return {'n': int(m.sum()), 'r2': M._f(1 - sse / sst), 'r2_naive': M._f(1 - sse_n / sst),
            'mae_log': M._f((t - p).abs().mean()), 'mae_log_naive': M._f((t - n).abs().mean()),
            'spearman': M._f(stats.spearmanr(p, t).statistic),
            'stormy_vs_calm_accuracy': M._f((hi_true == hi_pred).mean()), 'stormy_base_rate': M._f(hi_true.mean()),
            'stormy_auc': auc}


def direction_report(prob, df, mask, h=4):
    okm = mask & prob.notna() & df[f'fwd_ret_{h}'].notna()
    score = (prob[okm] - 0.5) * 200
    d = M.directional_report(score, df[f'fwd_ret_{h}'][okm], 1e-9, -1e-9, flat_band(df[okm], h), h)
    pr = M.probability_report(prob[okm], (df[f'fwd_ret_{h}'][okm] > 0).astype(float))
    return {k: d[k] for k in ('n_calls', 'hit_rate', 'pct_bull_calls', 'base_rate_up', 'pesaran_timmermann_p')} | {'roc_auc': pr['roc_auc']}


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    cl = _cl_frame(load_history()['macro_1h'])
    df = attach_outcomes(pd.read_pickle(os.path.join(CACHE_DIR, 'signals.pkl')), cl)
    r = np.log(cl['Close']).diff()
    pos = np.arange(len(df))
    dev = pd.Series((df.time < SPLIT).to_numpy() & (pos >= BURN_IN), index=df.index)
    lock = pd.Series((df.time >= SPLIT).to_numpy(), index=df.index)

    H = har_features(df, cl)
    F = build_features(df, cl)
    FULL = pd.concat([F, H], axis=1)
    res = {'protocol': {'split': str(SPLIT), 'burn_in': BURN_IN, 'har_lags_bars': list(LAGS)},
           'autocorrelation': {'returns_lag1': M._f(r.autocorr(1)), 'abs_returns_lag1': M._f(r.abs().autocorr(1)),
                               'abs_returns_lag4': M._f(r.abs().autocorr(4)), 'abs_returns_lag24': M._f(r.abs().autocorr(24))},
           'volatility': {}, 'direction': {}}

    # ---------------- 1. volatility forecasts
    vol_preds = {}
    for h in (4, 24):
        y = pd.Series(log_rv_future(r, h).to_numpy()[df.index], index=df.index)
        naive = H['log_rv_24']
        past_med = pd.Series(log_rv_past(r, h).rolling(500, min_periods=200).median().to_numpy()[df.index], index=df.index)
        har = walk_forward(H, y, h, 'reg')['ridge_regression']
        gb = walk_forward(FULL, y, h, 'reg')['gradient_boosting_reg']
        vol_preds[h] = har
        # How often does the realised h-bar move stay inside the forecast 1-sigma / 2-sigma range?
        band = (np.exp(har) - EPS) * np.sqrt(h)
        fwd = np.log(1 + df[f'fwd_ret_{h}'])
        cov = {}
        for nm, msk in (('dev', dev), ('lockbox', lock)):
            okc = msk & har.notna() & fwd.notna()
            cov[nm] = {'inside_1sigma': M._f((fwd[okc].abs() <= band[okc]).mean()),
                       'inside_2sigma': M._f((fwd[okc].abs() <= 2 * band[okc]).mean()), 'n': int(okc.sum())}
        res.setdefault('range_coverage', {})[f'h{h}'] = cov
        res['volatility'][f'h{h}'] = {}
        for name, p in (('Naive (last 24 bars)', naive), ('HAR ridge', har), ('Gradient boosting, all features', gb)):
            res['volatility'][f'h{h}'][name] = {'dev': vol_report(p, y, naive, past_med, dev), 'lockbox': vol_report(p, y, naive, past_med, lock)}
        logging.info('volatility h%d done', h)

    # ---------------- 2. does volatility help direction?
    y4 = (df.fwd_ret_4 > 0).astype(float).where(df.fwd_ret_4.notna())
    base = walk_forward(F, y4, 4, 'clf')['gradient_boosting']
    Fv = FULL.copy()
    Fv['har_vol_forecast_4h'] = vol_preds[4].ffill().fillna(H['log_rv_24'])
    withvol = walk_forward(Fv, y4, 4, 'clf')['gradient_boosting']
    res['direction']['gb_31_features'] = {'dev': direction_report(base, df, dev), 'lockbox': direction_report(base, df, lock)}
    res['direction']['gb_31_plus_volatility'] = {'dev': direction_report(withvol, df, dev), 'lockbox': direction_report(withvol, df, lock)}
    # regime split using the HAR forecast vs its own past median (known at decision time)
    vf = vol_preds[4]
    vf_med = vf.rolling(500, min_periods=200).median().shift(1)
    stormy = vf > vf_med
    res['direction']['by_predicted_regime'] = {}
    for nm, reg in (('predicted calm', ~stormy & vf_med.notna()), ('predicted stormy', stormy & vf_med.notna())):
        res['direction']['by_predicted_regime'][nm] = {'dev': direction_report(base, df, dev & reg), 'lockbox': direction_report(base, df, lock & reg)}

    # ---------------- 3. live model: HAR fitted on all history (4h horizon)
    y_all = pd.Series(log_rv_future(r, 4).to_numpy()[df.index], index=df.index)
    ok = y_all.notna() & H.notna().all(axis=1)
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(H[ok].to_numpy(), y_all[ok].to_numpy())
    sc, rg = pipe[0], pipe[1]
    hist_pred = pd.Series(pipe.predict(H[ok].to_numpy()))
    live = {'model': 'HAR ridge on log realised volatility', 'horizon_bars': 4, 'features': list(H.columns),
            'scaler_mean': sc.mean_.tolist(), 'scaler_scale': sc.scale_.tolist(), 'coef': rg.coef_.tolist(), 'intercept': float(rg.intercept_),
            'eps': EPS, 'lags': list(LAGS),
            'percentiles_log_rv_forecast': [float(hist_pred.quantile(q)) for q in (0.2, 0.4, 0.6, 0.8)],
            'lockbox_r2_walkforward': res['volatility']['h4']['HAR ridge']['lockbox']['r2'],
            'fitted_on': [str(df.time.iloc[0]), str(df.time.iloc[-1])]}
    json.dump(live, open(os.path.join(REPORT_DIR, 'vol_model.json'), 'w'), indent=1)
    json.dump(res, open(os.path.join(REPORT_DIR, 'volatility_results.json'), 'w'), indent=2, default=str)
    write_markdown(res, os.path.join(REPORT_DIR, 'VOLATILITY.md'))
    logging.info('wrote reports/VOLATILITY.md, volatility_results.json, vol_model.json')


def write_markdown(res, path):
    P = lambda x: '—' if x is None else f'{x*100:.1f}%'
    N = lambda x, d=3: '—' if x is None else f'{x:.{d}f}'
    a = res['autocorrelation']
    L = ['# Volatility forecasting, and does volatility help direction?', '',
         f"Return autocorrelation (lag 1): {a['returns_lag1']} · |return| autocorrelation: lag 1 {a['abs_returns_lag1']}, lag 4 {a['abs_returns_lag4']}, lag 24 {a['abs_returns_lag24']}. "
         'Direction is nearly random; the size of moves clusters.', '',
         'Target: log realised volatility over the next h hourly bars. Walk-forward, DEV before 2026-04-01, LOCKBOX Apr–Nov 2026.', '']
    for h, sec in res['volatility'].items():
        L += [f'## Volatility, next {h[1:]} bars', '',
              '| Model | DEV R² | LOCK R² | LOCK Spearman | LOCK stormy-vs-calm accuracy | LOCK stormy AUC |', '|---|---|---|---|---|---|']
        for nm, v in sec.items():
            d, l = v['dev'], v['lockbox']
            L.append(f"| {nm} | {N(d['r2'])} | {N(l['r2'])} | {N(l['spearman'])} | {P(l['stormy_vs_calm_accuracy'])} | {N(l['stormy_auc'])} |")
        L.append('')
    L += ['## Is the forecast range honest? (HAR, share of real moves inside the range)', '',
          '| Horizon | DEV inside ±1σ | DEV inside ±2σ | LOCK inside ±1σ | LOCK inside ±2σ |', '|---|---|---|---|---|']
    for h, c in res['range_coverage'].items():
        L.append(f"| {h[1:]} bars | {P(c['dev']['inside_1sigma'])} | {P(c['dev']['inside_2sigma'])} | {P(c['lockbox']['inside_1sigma'])} | {P(c['lockbox']['inside_2sigma'])} |")
    L += ['', 'A normal distribution would put 68 % / 95 % inside. Oil has fat tails, so the real shares are lower; the dashboard states the measured share.', '']
    L += ['## Does volatility help predict direction? (4h)', '', '| Model | DEV hit | DEV AUC | LOCK hit | LOCK AUC | LOCK PT p |', '|---|---|---|---|---|---|']
    for nm in ('gb_31_features', 'gb_31_plus_volatility'):
        d, l = res['direction'][nm]['dev'], res['direction'][nm]['lockbox']
        L.append(f"| {nm} | {P(d['hit_rate'])} | {N(d['roc_auc'])} | {P(l['hit_rate'])} | {N(l['roc_auc'])} | {N(l['pesaran_timmermann_p'])} |")
    L += ['', '| Baseline direction model in… | DEV hit | DEV AUC | LOCK hit | LOCK AUC | LOCK calls |', '|---|---|---|---|---|---|']
    for nm, v in res['direction']['by_predicted_regime'].items():
        d, l = v['dev'], v['lockbox']
        L.append(f"| {nm} hours | {P(d['hit_rate'])} | {N(d['roc_auc'])} | {P(l['hit_rate'])} | {N(l['roc_auc'])} | {l['n_calls']} |")
    open(path, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
