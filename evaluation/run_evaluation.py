"""
End-to-end accuracy evaluation of the PetroSync engine.

Usage (from the repo root):
    python -m evaluation.run_evaluation            # uses cached data / signals if present
    python -m evaluation.run_evaluation --refresh  # re-download data and re-replay

Outputs (in reports/):
    evaluation_results.json   machine-readable metrics (also served at /api/metrics)
    EVALUATION_RESULTS.md     human-readable tables
    predictions_vs_actual.csv every hourly call mapped to the realised outcome
    trades.csv                every simulated TT Playbook trade
"""
import argparse
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from evaluation.data_loader import load_history, CACHE_DIR
from evaluation.backtest import replay, attach_outcomes, simulate_trades
from evaluation import metrics as M
from evaluation.ml_models import build_features, walk_forward, feature_importance_logit

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
HORIZONS = (1, 4, 24)
PRIMARY_H = 4


def flat_band(sig, h):
    """Volatility-scaled 'no-move' band: |return| < 0.25 * ATR * sqrt(h) counts as FLAT."""
    return 0.25 * sig.atr * np.sqrt(h) / sig.price


def evaluate_signals(df, label=''):
    res = {}
    for h in HORIZONS:
        r, band = df[f'fwd_ret_{h}'], flat_band(df, h)
        past = (df.price / df.price.shift(h) - 1)  # persistence baseline uses only past prices
        res[f'h{h}'] = {
            'sentiment_7pillar': {
                **M.directional_report(df.sent_score, r, 15, -15, band, h),
                'probability': M.probability_report(df.sent_bull_pct / 100, (r > 0).astype(float)),
                'ic': M.information_coefficient(df.sent_score, r, h)},
            'mtf_confluence': {
                **M.directional_report(df.mtf_score, r, 65, 35, band, h),
                'probability': M.probability_report(df.mtf_score / 100, (r > 0).astype(float)),
                'ic': M.information_coefficient(df.mtf_score, r, h)},
            'tf_1h_bias': M.directional_report(df.tf1h_score, r, 70, 30, band, h),
            'baseline_persistence': M.directional_report(past, r, 1e-12, -1e-12, band, h),
            'baseline_always_long': M.directional_report(pd.Series(1.0, index=df.index), r, 0.5, -0.5, band, h),
            # Sanity checks of the metric pipeline itself (NOT models):
            'sanity_oracle_future_sign': M.directional_report(np.sign(r), r, 0.5, -0.5, band, h),
            'sanity_random_coin': M.directional_report(pd.Series(np.random.default_rng(42).choice([-1.0, 1.0], len(df)), index=df.index), r, 0.5, -0.5, band, h),
        }
    return res


def signal_strategy(df, cost_ticks=1):
    """Hold +1/-1/0 for one bar according to the sentiment label; pay 1 tick per unit traded."""
    pos = np.where(df.sent_score >= 15, 1, np.where(df.sent_score <= -15, -1, 0))
    turnover = np.abs(np.diff(np.r_[0, pos]))
    strat = pos * df.fwd_ret_1.fillna(0) - turnover * (0.01 * cost_ticks / df.price)
    return {'sentiment_long_short': M.equity_stats(strat),
            'buy_and_hold': M.equity_stats(df.fwd_ret_1.fillna(0))}


def regime_breakdown(df, h=PRIMARY_H):
    out = {}
    buckets = {'OVX<30 (low)': df.ovx < 30, 'OVX 30-45 (normal)': (df.ovx >= 30) & (df.ovx <= 45), 'OVX>45 (stress)': df.ovx > 45}
    for name, m in buckets.items():
        if m.sum() > 50:
            d = M.directional_report(df.sent_score[m], df[f'fwd_ret_{h}'][m], 15, -15, flat_band(df[m], h), h)
            out[name] = {k: d[k] for k in ('n_bars', 'coverage', 'hit_rate', 'nonoverlap_hit_rate', 'binomial_p_vs_50pct')}
    return out


def sweep_report(df, h=PRIMARY_H):
    from scipy import stats
    r = df[f'fwd_ret_{h}']
    base_down = float((r < 0).mean())
    out = {}
    for col, want, base in (('swept_high', r < 0, base_down), ('swept_low', r > 0, 1 - base_down)):
        m = df[col].astype(bool) & r.notna()
        k, n = int(want[m].sum()), int(m.sum())
        out[col] = {'n_events': n, 'reversal_rate': M._f(k / n if n else np.nan), 'base_rate': M._f(base),
                    'binomial_p_vs_base': M._f(stats.binomtest(k, n, base, alternative='greater').pvalue, 6) if n else None}
    return out


def band_report(df):
    nxt = df.next_close
    fut24 = df.fwd_close_24
    return {
        'vwap_1sigma_next_close': M.band_coverage(nxt, df.vwap_l1, df.vwap_u1, 0.6827),
        'vwap_2sigma_next_close': M.band_coverage(nxt, df.vwap_l2, df.vwap_u2, 0.9545),
        'value_area_next_close': M.band_coverage(nxt, df.val, df.vah, 0.70),
        'value_area_close_24h_later': M.band_coverage(fut24, df.val, df.vah, 0.70),
    }


def ml_section(df, cl):
    X = build_features(df, cl)
    out, oos_frames = {}, {}
    for h in HORIZONS:
        y_cls = (df[f'fwd_ret_{h}'] > 0).astype(float).where(df[f'fwd_ret_{h}'].notna())
        p = walk_forward(X, y_cls, h, 'clf')
        mask = p.notna().all(axis=1) & y_cls.notna()
        d, r = df[mask], df[f'fwd_ret_{h}'][mask]
        band = flat_band(d, h)
        sec = {'oos_start': str(d.time.iloc[0]) if len(d) else None, 'oos_bars': int(mask.sum())}
        # Rule-based model on the SAME out-of-sample window -> apples-to-apples
        sec['rule_based_sentiment_same_window'] = {
            **{k: v for k, v in M.directional_report(d.sent_score, r, 15, -15, band, h).items() if k != 'three_class'},
            'probability': {k: v for k, v in M.probability_report(d.sent_bull_pct / 100, (r > 0).astype(float)).items() if k != 'reliability'}}
        for name in p.columns:
            pr = p[name][mask]
            score = (pr - 0.5) * 200      # map prob -> -100..100 so the same thresholds apply
            sec[name] = {
                **{k: v for k, v in M.directional_report(score, r, 1e-9, -1e-9, band, h).items() if k != 'three_class'},
                'probability': M.probability_report(pr, (r > 0).astype(float)),
                'ic': M.information_coefficient(pr, r, h)}
            oos_frames[(h, name)] = pr
        # Regression: predict the h-bar return, map to a price, compare with the random walk
        y_reg = df[f'fwd_ret_{h}']
        pr_reg = walk_forward(X, y_reg, h, 'reg')
        regm = pr_reg.notna().all(axis=1) & y_reg.notna()
        sec['price_forecast'] = {}
        for name in pr_reg.columns:
            pred_price = df.price[regm] * (1 + pr_reg[name][regm])
            sec['price_forecast'][name] = M.forecast_accuracy(df[f'fwd_close_{h}'][regm], pred_price, df.price[regm])
        out[f'h{h}'] = sec
    y4 = (df[f'fwd_ret_{PRIMARY_H}'] > 0).astype(float).where(df[f'fwd_ret_{PRIMARY_H}'].notna())
    out['logit_standardised_coefficients_h4'] = {k: round(v, 4) for k, v in list(feature_importance_logit(X, y4).items())[:15]}
    return out, oos_frames


def write_markdown(res, path):
    L = []
    a = L.append
    meta = res['meta']
    a('# PetroSync Pro — Backtested Accuracy Report\n')
    a(f"_Generated {meta['generated_utc']} · sample {meta['sample_start']} → {meta['sample_end']} · "
      f"{meta['n_signals']:,} hourly decisions (real 15m data for the last {meta['n_real_15m']:,})._\n")
    a('> Auto-generated by `python -m evaluation.run_evaluation`. Methodology: `docs/EVALUATION_METHODOLOGY.md`.\n')
    a('## 1. Directional accuracy (out-of-sample replay of the live engine)\n')
    a('| Horizon | Model | Coverage | Hit rate (calls) | Non-overlap hit rate | p (binomial vs 50%) | Pesaran–Timmermann p | 3-class balanced acc. | MCC |')
    a('|---|---|---|---|---|---|---|---|---|')
    names = {'sentiment_7pillar': '7-Pillar Sentiment', 'mtf_confluence': 'MTF Confluence', 'tf_1h_bias': '1h TF bias',
             'baseline_persistence': 'Baseline: persistence', 'baseline_always_long': 'Baseline: always long',
             'sanity_oracle_future_sign': 'Sanity: oracle (cheats)', 'sanity_random_coin': 'Sanity: random coin'}
    pct = lambda x: '—' if x is None else f'{x*100:.1f}%'
    num = lambda x, d=3: '—' if x is None else f'{x:.{d}f}'
    for h in HORIZONS:
        for k, nm in names.items():
            d = res['signals'][f'h{h}'][k]
            a(f"| {h}h | {nm} | {pct(d['coverage'])} | {pct(d['hit_rate'])} | {pct(d['nonoverlap_hit_rate'])} | "
              f"{num(d['binomial_p_vs_50pct'],4)} | {num(d['pesaran_timmermann_p'],4)} | "
              f"{pct(d['three_class']['balanced_accuracy'])} | {num(d['three_class']['mcc'])} |")
    a('\n## 2. Probability quality (is "bullish %" a real probability?)\n')
    a('| Horizon | Model | Brier | Brier (climatology) | Brier Skill | Log-loss | ROC-AUC | ECE | Spearman IC |')
    a('|---|---|---|---|---|---|---|---|---|')
    for h in HORIZONS:
        for k in ('sentiment_7pillar', 'mtf_confluence'):
            p, ic = res['signals'][f'h{h}'][k]['probability'], res['signals'][f'h{h}'][k]['ic']
            a(f"| {h}h | {names[k]} | {num(p['brier'],4)} | {num(p['brier_climatology'],4)} | {num(p['brier_skill_score'],4)} | "
              f"{num(p['log_loss'],4)} | {num(p['roc_auc'])} | {num(p['expected_calibration_error'])} | {num(ic['ic_all_bars'])} |")
    s3 = res['signals'][f'h{PRIMARY_H}']['sentiment_7pillar']['three_class']
    a(f'\n### Confusion matrix — 7-Pillar Sentiment, {PRIMARY_H}h horizon (rows = actual, cols = predicted)\n')
    a('| actual \\ predicted | ' + ' | '.join(s3['labels']) + ' |')
    a('|---|' + '---|' * len(s3['labels']))
    for lab, row in zip(s3['labels'], s3['confusion_matrix']):
        a(f'| **{lab}** | ' + ' | '.join(f'{v:,}' for v in row) + ' |')
    t = res['trades']
    a('\n## 3. TT Playbook trade setups — claimed vs realised\n')
    if t.get('filled_trades'):
        a(f"Filled trades: **{t['filled_trades']}** · Target-1 hit rate **{pct(t['target1_hit_rate'])}** vs claimed "
          f"**{pct(t['claimed_win_prob_avg'])}** · net win rate {pct(t['win_rate_net'])} · avg R {num(t['avg_r_multiple'])} · "
          f"expectancy ${t['expectancy_usd_per_trade']}/contract · profit factor {num(t['profit_factor'])} · "
          f"total ${t['total_pnl_usd']:,} · max DD ${t['max_drawdown_usd']:,}\n")
        a(f"Outcome counts: `{t['outcome_counts']}`\n")
        a('| Setup | n | Claimed win % | Realised TP1 % | Net win % | Avg R | $/trade | p (claim true) |')
        a('|---|---|---|---|---|---|---|---|')
        for st, g in t['by_setup'].items():
            a(f"| {st} | {g['n']} | {pct(g['claimed_win_prob'])} | {pct(g['target1_hit_rate'])} | {pct(g['win_rate_net'])} | "
              f"{num(g['avg_r'])} | {g['expectancy_usd']} | {num(g['binomial_p_claim'],4)} |")
    a('\n## 4. Signal-following equity curve (1-bar hold, 1 tick cost per unit traded)\n')
    a('| Strategy | Total return | Ann. Sharpe | Max drawdown | Time in market |')
    a('|---|---|---|---|---|')
    for k, v in res['strategy'].items():
        a(f"| {k} | {v['total_return_pct']}% | {v['ann_sharpe']} | {v['max_drawdown_pct']}% | {v['time_in_market_pct']}% |")
    a('\n## 5. Statistical envelopes — observed vs nominal coverage\n')
    a('| Band | Observed | Nominal | Mean width ($) |')
    a('|---|---|---|---|')
    for k, v in res['bands'].items():
        a(f"| {k} | {pct(v['observed_coverage'])} | {pct(v['nominal_coverage'])} | {v['mean_width']} |")
    a('\n## 6. Liquidity-sweep detector (reversal within 4h)\n')
    a('| Event | n | Reversal rate | Base rate | p |')
    a('|---|---|---|---|---|')
    for k, v in res['sweeps'].items():
        a(f"| {k} | {v['n_events']} | {pct(v['reversal_rate'])} | {pct(v['base_rate'])} | {num(v['binomial_p_vs_base'],4)} |")
    a(f'\n## 7. Sentiment hit rate by volatility regime ({PRIMARY_H}h)\n')
    a('| Regime | Bars | Coverage | Hit rate | Non-overlap hit rate | p |')
    a('|---|---|---|---|---|---|')
    for k, v in res['regimes'].items():
        a(f"| {k} | {v['n_bars']} | {pct(v['coverage'])} | {pct(v['hit_rate'])} | {pct(v['nonoverlap_hit_rate'])} | {num(v['binomial_p_vs_50pct'],4)} |")
    a('\n## 8. Rule-based engine vs Machine-Learning baselines (same out-of-sample window)\n')
    a('| Horizon | Model | Coverage | % calls UP | Hit rate | Base rate UP | p vs 50% | Pesaran–Timmermann p | ROC-AUC | Brier Skill |')
    a('|---|---|---|---|---|---|---|---|---|---|')
    for h in HORIZONS:
        sec = res['ml'][f'h{h}']
        for k in ('rule_based_sentiment_same_window', 'logistic_regression', 'gradient_boosting'):
            d = sec[k]
            a(f"| {h}h | {k} | {pct(d['coverage'])} | {pct(d['pct_bull_calls'])} | {pct(d['hit_rate'])} | {pct(d['base_rate_up'])} | "
              f"{num(d['binomial_p_vs_50pct'],4)} | {num(d['pesaran_timmermann_p'],4)} | "
              f"{num(d['probability']['roc_auc'])} | {num(d['probability']['brier_skill_score'],4)} |")
    a('\n_If "% calls UP" is high and the hit rate ≈ base rate, the model has only learned the upward drift. '
      'Pesaran–Timmermann corrects for that; a small binomial p with a large PT p means "no real timing skill"._\n')
    a('\n### Price forecasts mapped to actual prices (vs random walk)\n')
    a('| Horizon | Model | MAE ($) | RMSE ($) | MAPE | Naive MAE | Theil U2 | MASE | Direction acc. |')
    a('|---|---|---|---|---|---|---|---|---|')
    for h in HORIZONS:
        for k, v in res['ml'][f'h{h}']['price_forecast'].items():
            a(f"| {h}h | {k} | {v['mae']} | {v['rmse']} | {v['mape_pct']}% | {v['naive_mae']} | {v['theils_u2']} | {v['mase']} | {pct(v['directional_accuracy'])} |")
    a('\n_Theil U2 / MASE < 1 would mean the model beats "tomorrow = today". Values ≈ 1 are the norm for liquid futures._\n')
    a('\n### Most influential features (standardised logistic coefficients, 4h)\n')
    for k, v in res['ml']['logit_standardised_coefficients_h4'].items():
        a(f'- `{k}`: {v:+.4f}')
    sub = res['subperiods']
    a('\n## 9. Robustness: real-15m window vs 1h-proxy window (sentiment, 4h)\n')
    a('| Window | Calls | Hit rate | p |')
    a('|---|---|---|---|')
    for k, v in sub.items():
        a(f"| {k} | {v['n_calls']} | {pct(v['hit_rate'])} | {num(v['binomial_p_vs_50pct'],4)} |")
    open(path, 'w').write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help='re-download data and re-run the replay')
    ap.add_argument('--no-ml', action='store_true')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    hist = load_history(refresh=args.refresh)
    sig_path = os.path.join(CACHE_DIR, 'signals.pkl')
    if os.path.exists(sig_path) and not args.refresh:
        sig = pd.read_pickle(sig_path)
        from evaluation.backtest import _cl_frame
        cl = _cl_frame(hist['macro_1h'])
    else:
        sig, cl = replay(hist)
        sig.to_pickle(sig_path)
    df = attach_outcomes(sig, cl)
    logging.info('signals: %d', len(df))

    trades = simulate_trades(df, cl)
    res = {
        'meta': {'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                 'sample_start': str(df.time.iloc[0]), 'sample_end': str(df.time.iloc[-1]),
                 'n_signals': int(len(df)), 'n_real_15m': int(df.tf15_real.sum()),
                 'horizons_bars': list(HORIZONS), 'primary_horizon': PRIMARY_H},
        'signals': evaluate_signals(df),
        'trades': M.trade_report(trades),
        'trades_real15m_window': M.trade_report(trades[trades.signal_bar.isin(df.index[df.tf15_real])]),
        'strategy': signal_strategy(df),
        'bands': band_report(df),
        'sweeps': sweep_report(df),
        'regimes': regime_breakdown(df),
        'subperiods': {},
    }
    for nm, m in (('real 15m data', df.tf15_real), ('1h proxy for 15m', ~df.tf15_real)):
        d = M.directional_report(df.sent_score[m], df.fwd_ret_4[m], 15, -15, flat_band(df[m], 4), 4)
        res['subperiods'][nm] = {k: d[k] for k in ('n_calls', 'hit_rate', 'binomial_p_vs_50pct')}

    oos = {}
    if not args.no_ml:
        res['ml'], oos = ml_section(df, cl)

    os.makedirs(REPORT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(REPORT_DIR, 'evaluation_results.json'), 'w'), indent=2, default=str)
    if not args.no_ml:
        write_markdown(res, os.path.join(REPORT_DIR, 'EVALUATION_RESULTS.md'))

    # Prediction-vs-actual mapping table (the "answer key")
    call = np.where(df.sent_score >= 15, 'BULL', np.where(df.sent_score <= -15, 'BEAR', 'NEUTRAL'))
    pva = pd.DataFrame({
        'time_utc': df.time, 'price_at_signal': df.price, 'sentiment_score': df.sent_score,
        'bullish_pct': df.sent_bull_pct, 'call': call, 'mtf_score': df.mtf_score, 'setup': df.setup_type,
        'actual_price_4h': df.fwd_close_4.round(2), 'actual_ret_4h_pct': (df.fwd_ret_4 * 100).round(3),
        'actual_direction_4h': np.where(df.fwd_ret_4 > 0, 'UP', np.where(df.fwd_ret_4 < 0, 'DOWN', 'UNCH')),
        'correct_4h': np.where(call == 'NEUTRAL', '', np.where((call == 'BULL') == (df.fwd_ret_4 > 0), 'YES', 'NO')),
        'actual_price_24h': df.fwd_close_24.round(2),
    })
    if (4, 'gradient_boosting') in oos:
        pva['ml_gb_prob_up_4h'] = oos[(4, 'gradient_boosting')].round(4)
    pva.to_csv(os.path.join(REPORT_DIR, 'predictions_vs_actual.csv'), index=False)
    trades.to_csv(os.path.join(REPORT_DIR, 'trades.csv'), index=False)
    # compact per-bar series for charts in the learning report
    keep = df[['time', 'price', 'sent_score', 'mtf_score', 'fwd_ret_4']].iloc[::4].copy()
    keep['time'] = keep.time.astype(str)
    keep.to_json(os.path.join(REPORT_DIR, 'chart_series.json'), orient='records')
    logging.info('Reports written to %s', REPORT_DIR)


if __name__ == '__main__':
    main()
