"""
Machine-learning baselines trained on the SAME point-in-time features the
rule-based engine computes.

Purpose: answer "would a learned model beat the hand-tuned 7-pillar weights?"

Validation: expanding-window walk-forward with purging.
  * The model is re-fit every `retrain_every` bars on all data seen so far.
  * Training rows whose label horizon overlaps the test block are dropped
    (purge gap = horizon bars, Lopez de Prado 2018, ch. 7). Using random
    K-fold CV here would leak the future into the past.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_RAW = ['sent_score', 'mtf_score', 'tf15_score', 'tf1h_score', 'tfD_score', 'rsi_15m', 'rsi_1h',
                'ovx', 'crack', 'dxy_chg', 'es_chg', 'gold_oil', 'brent_spread',
                'pillar_mtf_alignment', 'pillar_order_flow_vwap', 'pillar_crack_margin',
                'pillar_brent_spread', 'pillar_usd_impact', 'pillar_equity_risk_on']


def build_features(sig, cl):
    X = sig[FEATURES_RAW].astype(float).copy()
    atr = sig.atr.replace(0, np.nan)
    X['dist_vwap_atr'] = (sig.price - sig.vwap) / atr
    X['dist_ema20_atr'] = (sig.price - sig.ema20) / atr
    X['ema20_50_atr'] = (sig.ema20 - sig.ema50) / atr
    X['dist_poc_atr'] = (sig.price - sig.poc) / atr
    X['atr_pct'] = sig.atr / sig.price
    X['swept_high'] = sig.swept_high.astype(float)
    X['swept_low'] = sig.swept_low.astype(float)
    close = cl['Close']
    for lag in (1, 4, 24):
        X[f'past_ret_{lag}'] = (close / close.shift(lag) - 1).to_numpy()[sig.index]
    hour = pd.DatetimeIndex(sig.time).tz_convert('America/New_York').hour
    X['hour_sin'], X['hour_cos'] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    return X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


def _models(kind):
    if kind == 'clf':
        return {
            'logistic_regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
            'gradient_boosting': lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                        l2_regularization=1.0, random_state=42),
        }
    return {
        'ridge_regression': lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'gradient_boosting_reg': lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                        l2_regularization=1.0, random_state=42),
    }


def walk_forward(X, y, horizon, kind='clf', min_train=2000, retrain_every=250):
    """Return DataFrame of out-of-sample predictions, one column per model."""
    n = len(X)
    preds = {name: np.full(n, np.nan) for name in _models(kind)}
    Xv, yv = X.to_numpy(), y.to_numpy()
    for start in range(min_train, n, retrain_every):
        end = min(start + retrain_every, n)
        train_end = start - horizon            # purge: labels of these rows are fully known at `start`
        tr = np.arange(0, train_end)
        tr = tr[~np.isnan(yv[tr])]
        if len(tr) < 500:
            continue
        for name, make in _models(kind).items():
            m = make()
            if kind == 'clf':
                m.fit(Xv[tr], yv[tr].astype(int))
                preds[name][start:end] = m.predict_proba(Xv[start:end])[:, 1]
            else:
                m.fit(Xv[tr], yv[tr])
                preds[name][start:end] = m.predict(Xv[start:end])
    return pd.DataFrame(preds, index=X.index)


def feature_importance_logit(X, y):
    """Standardised logistic coefficients on the full sample (for interpretation only, not for scoring)."""
    mask = ~np.isnan(y.to_numpy())
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    m.fit(X.to_numpy()[mask], y.to_numpy()[mask].astype(int))
    coefs = m[-1].coef_[0]
    return dict(sorted(zip(X.columns, map(float, coefs)), key=lambda kv: -abs(kv[1])))
