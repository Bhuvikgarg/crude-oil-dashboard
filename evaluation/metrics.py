"""
Evaluation metrics for directional signals, probability forecasts, price/return
forecasts, trading performance and statistical bands.

Every function returns plain Python types so results can be dumped to JSON.
References for each metric are listed in docs/REFERENCES.md.
"""
import math
import numpy as np
import pandas as pd
from scipy import stats

HOURS_PER_YEAR = 252 * 23  # CME Globex trades ~23h/day


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


# ---------------------------------------------------------------- classification
def confusion_and_scores(y_true, y_pred, labels):
    """Confusion matrix + accuracy, balanced accuracy, per-class P/R/F1, macro-F1, MCC."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = len(labels)
    cm = np.zeros((k, k), dtype=int)
    li = {lab: n for n, lab in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        cm[li[t], li[p]] += 1
    n = cm.sum()
    per_class, recalls, f1s = {}, [], []
    for lab, j in li.items():
        tp = cm[j, j]; fp = cm[:, j].sum() - tp; fn = cm[j, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[str(lab)] = {'precision': _f(prec), 'recall': _f(rec), 'f1': _f(f1), 'support': int(cm[j, :].sum())}
        if cm[j, :].sum():
            recalls.append(rec)
        f1s.append(f1)
    # Multiclass Matthews correlation coefficient (Gorodkin 2004)
    t_k, p_k, c = cm.sum(axis=1), cm.sum(axis=0), np.trace(cm)
    denom = math.sqrt(max(n ** 2 - (p_k ** 2).sum(), 0) * max(n ** 2 - (t_k ** 2).sum(), 0))
    mcc = (c * n - (t_k * p_k).sum()) / denom if denom else 0.0
    return {
        'labels': [str(x) for x in labels], 'confusion_matrix': cm.tolist(), 'n': int(n),
        'accuracy': _f(c / n if n else np.nan), 'balanced_accuracy': _f(np.mean(recalls) if recalls else np.nan),
        'macro_f1': _f(np.mean(f1s)), 'mcc': _f(mcc), 'per_class': per_class,
    }


def pesaran_timmermann(pred_up, actual_up):
    """
    Pesaran & Timmermann (1992) test of directional predictability.
    H0: predictions and outcomes are independent. Returns (statistic, one-sided p-value).
    """
    x, y = np.asarray(pred_up, bool), np.asarray(actual_up, bool)
    n = len(x)
    if n < 10:
        return None, None
    p_hat = np.mean(x == y)
    py, px = y.mean(), x.mean()
    p_star = py * px + (1 - py) * (1 - px)
    var_p = p_star * (1 - p_star) / n
    var_star = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
                + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    if var_p - var_star <= 0:
        return None, None
    s = (p_hat - p_star) / math.sqrt(var_p - var_star)
    return _f(s), _f(1 - stats.norm.cdf(s), 6)


def directional_report(score, fwd_ret, bull_thr, bear_thr, flat_band, horizon):
    """
    Evaluate a score that is turned into BULL / BEAR / NEUTRAL calls.

    * hit_rate: % of non-neutral calls whose sign matches the realised return
    * coverage: % of bars where the model makes a call at all
    * 3-class matrix: truth is UP / DOWN / FLAT where FLAT = |return| < flat_band (vol-scaled)
    * significance is computed on NON-OVERLAPPING samples (every `horizon`-th bar),
      because overlapping h-bar returns are autocorrelated and would inflate t-stats.
    """
    df = pd.DataFrame({'s': score, 'r': fwd_ret, 'band': flat_band}).dropna()
    call = np.where(df.s >= bull_thr, 'BULL', np.where(df.s <= bear_thr, 'BEAR', 'NEUTRAL'))
    truth = np.where(df.r > df.band, 'UP', np.where(df.r < -df.band, 'DOWN', 'FLAT'))
    pred3 = np.select([call == 'BULL', call == 'BEAR'], ['UP', 'DOWN'], 'FLAT')

    active = call != 'NEUTRAL'
    pred_up = call[active] == 'BULL'
    act_up = df.r.to_numpy()[active] > 0
    hits = pred_up == act_up

    sub = np.zeros(len(df), bool); sub[::max(1, horizon)] = True
    a_sub = active & sub
    hits_sub = (call[a_sub] == 'BULL') == (df.r.to_numpy()[a_sub] > 0)
    binom_p = stats.binomtest(int(hits_sub.sum()), int(len(hits_sub)), 0.5, alternative='greater').pvalue if len(hits_sub) else None
    pt_s, pt_p = pesaran_timmermann(call[a_sub] == 'BULL', df.r.to_numpy()[a_sub] > 0)

    base_up = float((df.r > 0).mean())
    # Persistence / momentum naive baseline: predict sign of the previous h-bar return
    return {
        'n_bars': int(len(df)), 'coverage': _f(active.mean()), 'n_calls': int(active.sum()),
        'pct_bull_calls': _f(pred_up.mean() if len(pred_up) else np.nan),
        'hit_rate': _f(hits.mean() if len(hits) else np.nan),
        'hit_rate_bull_calls': _f(act_up[pred_up].mean() if pred_up.any() else np.nan),
        'hit_rate_bear_calls': _f((~act_up[~pred_up]).mean() if (~pred_up).any() else np.nan),
        'avg_ret_after_bull_bps': _f(df.r.to_numpy()[active][pred_up].mean() * 1e4 if pred_up.any() else np.nan, 2),
        'avg_ret_after_bear_bps': _f(df.r.to_numpy()[active][~pred_up].mean() * 1e4 if (~pred_up).any() else np.nan, 2),
        'base_rate_up': _f(base_up),
        'nonoverlap_calls': int(a_sub.sum()), 'nonoverlap_hit_rate': _f(hits_sub.mean() if len(hits_sub) else np.nan),
        'binomial_p_vs_50pct': _f(binom_p, 6), 'pesaran_timmermann_stat': pt_s, 'pesaran_timmermann_p': pt_p,
        'three_class': confusion_and_scores(truth, pred3, ['UP', 'FLAT', 'DOWN']),
    }


# ---------------------------------------------------------------- probability
def probability_report(p_up, y_up, n_bins=10):
    """Brier score, Brier skill vs climatology, log-loss, ROC-AUC and a reliability table."""
    df = pd.DataFrame({'p': p_up, 'y': y_up}).dropna()
    p, y = df.p.clip(1e-4, 1 - 1e-4).to_numpy(), df.y.astype(float).to_numpy()
    brier = np.mean((p - y) ** 2)
    clim = y.mean()
    brier_ref = np.mean((clim - y) ** 2)
    logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    # ROC-AUC via Mann-Whitney U
    pos, neg = p[y == 1], p[y == 0]
    auc = stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)) if len(pos) and len(neg) else np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    table = []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            table.append({'bin': f'{bins[b]:.1f}-{bins[b+1]:.1f}', 'n': int(m.sum()),
                          'mean_predicted': _f(p[m].mean()), 'observed_freq': _f(y[m].mean())})
    ece = sum(t['n'] * abs(t['mean_predicted'] - t['observed_freq']) for t in table) / len(p)
    return {'n': int(len(p)), 'brier': _f(brier), 'brier_climatology': _f(brier_ref),
            'brier_skill_score': _f(1 - brier / brier_ref if brier_ref else np.nan),
            'log_loss': _f(logloss), 'roc_auc': _f(auc), 'expected_calibration_error': _f(ece),
            'reliability': table}


def information_coefficient(score, fwd_ret, horizon):
    """Spearman rank IC (all bars) + t-stat on non-overlapping bars."""
    df = pd.DataFrame({'s': score, 'r': fwd_ret}).dropna()
    ic_all = stats.spearmanr(df.s, df.r).statistic
    sub = df.iloc[::max(1, horizon)]
    ic_sub, p_sub = stats.spearmanr(sub.s, sub.r)
    return {'ic_all_bars': _f(ic_all), 'ic_nonoverlap': _f(ic_sub), 'ic_nonoverlap_p': _f(p_sub, 6), 'n_nonoverlap': int(len(sub))}


# ---------------------------------------------------------------- regression / price forecasts
def forecast_accuracy(actual, predicted, naive):
    """
    'Map prediction to actual value' metrics for price forecasts.
    naive = random-walk forecast (price now). Theil's U2 < 1 and MASE < 1 mean the
    model beats the random walk; on raw price levels MAPE is always tiny and misleading
    on its own.
    """
    df = pd.DataFrame({'a': actual, 'p': predicted, 'n': naive}).dropna()
    e, en = df.a - df.p, df.a - df.n
    mae, rmse = e.abs().mean(), math.sqrt((e ** 2).mean())
    mae_n, rmse_n = en.abs().mean(), math.sqrt((en ** 2).mean())
    dir_acc = np.mean(np.sign(df.p - df.n) == np.sign(df.a - df.n))
    return {'n': int(len(df)), 'mae': _f(mae), 'rmse': _f(rmse), 'mape_pct': _f((e.abs() / df.a).mean() * 100),
            'r2_vs_mean': _f(1 - (e ** 2).sum() / ((df.a - df.a.mean()) ** 2).sum()),
            'naive_mae': _f(mae_n), 'naive_rmse': _f(rmse_n),
            'theils_u2': _f(rmse / rmse_n if rmse_n else np.nan), 'mase': _f(mae / mae_n if mae_n else np.nan),
            'directional_accuracy': _f(dir_acc)}


# ---------------------------------------------------------------- trading
def equity_stats(bar_returns, periods_per_year=HOURS_PER_YEAR):
    r = pd.Series(bar_returns).fillna(0)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    sd = r.std()
    return {'total_return_pct': _f((eq.iloc[-1] - 1) * 100, 2) if len(eq) else None,
            'ann_sharpe': _f(r.mean() / sd * math.sqrt(periods_per_year) if sd else np.nan, 3),
            'max_drawdown_pct': _f(dd.min() * 100, 2), 'time_in_market_pct': _f((r != 0).mean() * 100, 1)}


def trade_report(trades):
    if trades is None or trades.empty:
        return {}
    done = trades[trades.outcome.isin(['TARGET1', 'STOP', 'TIMEOUT'])]
    out = {'signals_considered': int(len(trades)),
           'outcome_counts': trades.outcome.value_counts().to_dict(),
           'filled_trades': int(len(done))}
    if done.empty:
        return out
    wins = done.r_multiple > 0
    gp, gl = done.pnl_usd[done.pnl_usd > 0].sum(), -done.pnl_usd[done.pnl_usd < 0].sum()
    out.update({
        'target1_hit_rate': _f((done.outcome == 'TARGET1').mean()),
        'win_rate_net': _f(wins.mean()),
        'claimed_win_prob_avg': _f(done.claimed_win_prob.mean() / 100),
        'avg_r_multiple': _f(done.r_multiple.mean()), 'median_r_multiple': _f(done.r_multiple.median()),
        'expectancy_usd_per_trade': _f(done.pnl_usd.mean(), 2), 'total_pnl_usd': _f(done.pnl_usd.sum(), 2),
        'profit_factor': _f(gp / gl if gl else np.nan, 3),
        'avg_bars_held': _f(done.bars_held.mean(), 1),
        'max_drawdown_usd': _f((done.pnl_usd.cumsum() - done.pnl_usd.cumsum().cummax()).min(), 2),
        'by_setup': {},
    })
    for st, g in done.groupby('setup_type'):
        out['by_setup'][st] = {
            'n': int(len(g)), 'claimed_win_prob': _f(g.claimed_win_prob.mean() / 100),
            'target1_hit_rate': _f((g.outcome == 'TARGET1').mean()), 'win_rate_net': _f((g.r_multiple > 0).mean()),
            'avg_r': _f(g.r_multiple.mean()), 'expectancy_usd': _f(g.pnl_usd.mean(), 2),
            'binomial_p_claim': _f(stats.binomtest(int((g.outcome == 'TARGET1').sum()), len(g),
                                                    g.claimed_win_prob.mean() / 100).pvalue, 6),
        }
    return out


# ---------------------------------------------------------------- bands / levels
def band_coverage(next_close, lower, upper, nominal):
    df = pd.DataFrame({'c': next_close, 'lo': lower, 'hi': upper}).dropna()
    inside = (df.c >= df.lo) & (df.c <= df.hi)
    return {'n': int(len(df)), 'observed_coverage': _f(inside.mean()), 'nominal_coverage': nominal,
            'mean_width': _f((df.hi - df.lo).mean())}
