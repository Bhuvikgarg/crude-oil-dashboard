"""
Online (self-learning) models, evaluated test-then-train ("prequential") on the
full 2.4-year history.

At every hourly bar t:
  1. each learner PREDICTS with its current weights      -> scored against the 4-bar outcome
  2. the outcome of the call made at bar t-4 is now known -> the learner UPDATES its weights
So no prediction is ever made by a model that has seen its own answer.

Learners
  * Hedge   : multiplicative weights over the six sentiment pillars (online_learner.Hedge),
              starting from the engine's hand-tuned weights.
  * SGD-LR  : online logistic regression on the 31 features (sklearn SGDClassifier.partial_fit),
              one gradient step per newly labelled hour.

Hyper-parameters (Hedge eta, SGD learning rate) are chosen on the DEV period only and
then scored on the LOCKBOX (Apr-Nov 2026), the same protocol as evaluation/improve.py.

Run: python -m evaluation.online_learning
"""
import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from online_learner import Hedge, PILLARS, HAND_WEIGHTS, HORIZON
from evaluation import metrics as M
from evaluation.backtest import _cl_frame, attach_outcomes
from evaluation.data_loader import CACHE_DIR, load_history
from evaluation.ml_models import build_features, walk_forward
from evaluation.run_evaluation import REPORT_DIR, flat_band

SPLIT = pd.Timestamp('2026-04-01', tz='UTC')
BURN_IN = 2000           # same minimum history as the walk-forward models


def run_hedge(df, eta, alpha=0.01):
    """Prequential Hedge. Returns scores (-100..100) and the weight path."""
    X = np.clip(df[[f'pillar_{k}' for k in PILLARS]].to_numpy(float) / 100.0, -1, 1)
    y = np.where(df.fwd_ret_4.to_numpy() > 0, 1.0, -1.0)
    known = df.fwd_ret_4.notna().to_numpy()
    bars = df.index.to_numpy()
    h = Hedge(eta, alpha)
    scores, W = np.zeros(len(df)), np.zeros((len(df), len(PILLARS)))
    j = 0                                            # next row whose label is waiting to be learned
    for t in range(len(df)):
        while j < t and bars[j] + HORIZON <= bars[t]:  # outcome of row j is known at bar t
            if known[j]:
                h.update(X[j], y[j])
            j += 1
        scores[t] = h.score(X[t])                    # predict BEFORE learning from bar t
        W[t] = h.w
    return pd.Series(scores, index=df.index), W, h


def run_sgd(df, X, eta0):
    """Prequential online logistic regression. Scaler fitted on the burn-in only (no look-ahead)."""
    y = (df.fwd_ret_4 > 0).astype(int).to_numpy()
    known = df.fwd_ret_4.notna().to_numpy()
    bars = df.index.to_numpy()
    Xv = X.to_numpy(float)
    scaler = StandardScaler().fit(Xv[:BURN_IN])
    Z = scaler.transform(Xv)
    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-3, learning_rate='constant', eta0=eta0, random_state=42)
    ok = known[:BURN_IN - HORIZON]
    clf.partial_fit(Z[:BURN_IN - HORIZON][ok], y[:BURN_IN - HORIZON][ok], classes=[0, 1])
    p = np.full(len(df), np.nan)
    j = BURN_IN - HORIZON
    for t in range(BURN_IN, len(df)):
        while j < t and bars[j] + HORIZON <= bars[t]:
            if known[j]:
                clf.partial_fit(Z[j:j + 1], y[j:j + 1])
            j += 1
        p[t] = clf.predict_proba(Z[t:t + 1])[0, 1]
    return pd.Series(p, index=df.index), clf, scaler


def evaluate(score, df, mask, thr=15):
    r, band = df.fwd_ret_4[mask], flat_band(df[mask], HORIZON)
    d = M.directional_report(score[mask], r, thr, -thr, band, HORIZON)
    pr = M.probability_report(((score[mask] / 100) + 1) / 2, (r > 0).astype(float))
    ic = M.information_coefficient(score[mask], r, HORIZON)
    return {'n_bars': d['n_bars'], 'coverage': d['coverage'], 'hit_rate': d['hit_rate'], 'pct_bull_calls': d['pct_bull_calls'],
            'base_rate_up': d['base_rate_up'], 'nonoverlap_calls': d['nonoverlap_calls'],
            'binomial_p_vs_50pct': d['binomial_p_vs_50pct'], 'pesaran_timmermann_p': d['pesaran_timmermann_p'],
            'roc_auc': pr['roc_auc'], 'ic': ic['ic_all_bars']}


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    hist = load_history()
    cl = _cl_frame(hist['macro_1h'])
    df = attach_outcomes(pd.read_pickle(os.path.join(CACHE_DIR, 'signals.pkl')), cl)
    pos = np.arange(len(df))
    after_burn = pos >= BURN_IN
    dev = (df.time < SPLIT).to_numpy() & after_burn
    lock = (df.time >= SPLIT).to_numpy()
    tried = []

    # ---------------- Hedge: choose eta on DEV
    grid = {}
    for eta in (0.01, 0.05, 0.2, 0.5):
        s, W, _ = run_hedge(df, eta)
        grid[eta] = (s, W)
        tried.append(f'hedge eta={eta}')
    dev_auc = {eta: evaluate(s, df, dev)['roc_auc'] or 0 for eta, (s, _) in grid.items()}
    eta_best = max(dev_auc, key=dev_auc.get)
    hedge_s, hedge_W = grid[eta_best]
    logging.info('Hedge eta chosen on DEV: %s (dev AUC %s)', eta_best, dev_auc)

    # ---------------- Online logistic regression: choose eta0 on DEV
    X = build_features(df, cl)
    sgd = {}
    for eta0 in (0.001, 0.01):
        p, clf, scaler = run_sgd(df, X, eta0)
        sgd[eta0] = (p, clf, scaler)
        tried.append(f'sgd eta0={eta0}')
    sgd_dev_auc = {e: evaluate((v[0] - 0.5) * 200, df, dev, thr=1e-9)['roc_auc'] or 0 for e, v in sgd.items()}
    eta0_best = max(sgd_dev_auc, key=sgd_dev_auc.get)
    sgd_p, sgd_clf, sgd_scaler = sgd[eta0_best]
    sgd_score = (sgd_p - 0.5) * 200
    logging.info('SGD eta0 chosen on DEV: %s (dev AUC %s)', eta0_best, sgd_dev_auc)

    # ---------------- Frozen comparisons on the same rows
    y4 = (df.fwd_ret_4 > 0).astype(float).where(df.fwd_ret_4.notna())
    gb = walk_forward(X, y4, HORIZON, 'clf')['gradient_boosting']
    gb_score = (gb - 0.5) * 200
    always = pd.Series(100.0, index=df.index)

    models = {
        'Frozen engine (hand-tuned weights)': (df.sent_score, 15),
        'Hedge self-learning weights': (hedge_s, 15),
        'Online logistic regression (SGD)': (sgd_score, 1e-9),
        'Walk-forward gradient boosting (retrained every 250 h)': (gb_score, 1e-9),
        'Always "up" baseline': (always, 15),
    }
    res = {'protocol': {'split': str(SPLIT), 'burn_in_rows': BURN_IN, 'horizon_bars': HORIZON,
                        'hedge_eta_grid_dev_auc': {str(k): v for k, v in dev_auc.items()}, 'hedge_eta_chosen': eta_best,
                        'hedge_alpha': 0.01, 'sgd_eta0_grid_dev_auc': {str(k): v for k, v in sgd_dev_auc.items()},
                        'sgd_eta0_chosen': eta0_best, 'variants_tried': tried},
           'models': {}}
    for name, (s, thr) in models.items():
        okm = s.notna().to_numpy()
        res['models'][name] = {'dev': evaluate(s, df, dev & okm, thr), 'lockbox': evaluate(s, df, lock & okm, thr)}

    # ---------------- Weight path (daily samples) and final state for the live dashboard
    hand = (HAND_WEIGHTS / HAND_WEIGHTS.sum()).round(4).tolist()
    path = [{'t': str(df.time.iloc[i])[:16], 'w': [round(v, 4) for v in hedge_W[i]]} for i in range(0, len(df), 24)]
    res['hedge_weights'] = {'pillars': PILLARS, 'hand': hand, 'final': [round(v, 4) for v in hedge_W[-1]], 'path': path}
    coef = dict(zip(X.columns, sgd_clf.coef_[0].round(4).tolist()))
    res['sgd_top_coefficients'] = dict(sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:10])

    # Sensitivity: how much of the learned weighting comes from the fixed share alpha?
    Xp = np.clip(df[[f'pillar_{k}' for k in PILLARS]].to_numpy(float) / 100.0, -1, 1)
    yy = np.where(df.fwd_ret_4.to_numpy() > 0, 1.0, -1.0); okp = df.fwd_ret_4.notna().to_numpy()
    res['pillar_diagnostics'] = {k: {'direction_hit_rate': round(float((np.sign(Xp[okp, i]) == yy[okp])[Xp[okp, i] != 0].mean()), 4),
                                     'mean_abs_vote': round(float(np.abs(Xp[okp, i]).mean()), 3),
                                     'mean_squared_loss': round(float(((Xp[okp, i] - yy[okp]) ** 2 / 4).mean()), 4)}
                                 for i, k in enumerate(PILLARS)}
    res['alpha_sensitivity'] = {}
    for a in (0.01, 0.001, 0.0):
        s_a, W_a, _ = run_hedge(df, eta_best, a)
        res['alpha_sensitivity'][str(a)] = {'final_weights': [round(v, 4) for v in W_a[-1]],
                                            'lockbox_hit': evaluate(s_a, df, lock)['hit_rate'], 'lockbox_auc': evaluate(s_a, df, lock)['roc_auc']}
        tried.append(f'hedge alpha={a} (sensitivity, eta={eta_best})')

    # Live learner starts from the weights learned over all history, then keeps updating hourly
    _, _, h_full = run_hedge(df, eta_best)
    state = {'eta': eta_best, 'alpha': 0.01, 'weights': h_full.w.tolist(), 'updates': h_full.updates,
             'backtest_updates': h_full.updates, 'pending': [], 'last_update': None,
             'trained_until': str(df.time.iloc[-1 - HORIZON])[:16],
             'history': [{'t': p['t'], 'w': p['w']} for p in path[-120:]]}
    os.makedirs(REPORT_DIR, exist_ok=True)
    json.dump(state, open(os.path.join(REPORT_DIR, 'online_state_init.json'), 'w'), indent=1)
    json.dump(res, open(os.path.join(REPORT_DIR, 'online_learning_results.json'), 'w'), indent=2, default=str)
    write_markdown(res, os.path.join(REPORT_DIR, 'ONLINE_LEARNING.md'))
    logging.info('wrote reports/online_learning_results.json, ONLINE_LEARNING.md, online_state_init.json')


def write_markdown(res, path):
    P = lambda x: '—' if x is None else f'{x*100:.1f}%'
    N = lambda x, d=3: '—' if x is None else f'{x:.{d}f}'
    pr = res['protocol']
    L = ['# Self-learning models — test-then-train results', '',
         f"Each hour the model predicts first, is scored on the 4-hour outcome, then learns from the outcome of the call made 4 bars earlier. "
         f"Hedge η = {pr['hedge_eta_chosen']} and SGD learning rate = {pr['sgd_eta0_chosen']} were chosen on DEV (before {pr['split'][:10]}); "
         f"LOCKBOX is Apr–Nov 2026. First {pr['burn_in_rows']} rows are burn-in.", '',
         '| Model | DEV hit | DEV AUC | DEV PT p | LOCK hit | LOCK AUC | LOCK PT p | LOCK coverage |', '|---|---|---|---|---|---|---|---|']
    for name, v in res['models'].items():
        d, l = v['dev'], v['lockbox']
        L.append(f"| {name} | {P(d['hit_rate'])} | {N(d['roc_auc'])} | {N(d['pesaran_timmermann_p'])} | {P(l['hit_rate'])} | {N(l['roc_auc'])} | {N(l['pesaran_timmermann_p'])} | {P(l['coverage'])} |")
    hw = res['hedge_weights']
    L += ['', '## Hedge: hand-tuned vs learned pillar weights (end of history)', '', '| Pillar | Hand-tuned | Learned |', '|---|---|---|']
    for k, a, b in zip(hw['pillars'], hw['hand'], hw['final']):
        L.append(f'| {k} | {a*100:.1f}% | {b*100:.1f}% |')
    L += ['', '## Why the weights moved: pillar diagnostics', '',
          '| Pillar | Direction hit rate | Mean |vote| | Mean squared loss |', '|---|---|---|---|']
    for k, v in res['pillar_diagnostics'].items():
        L.append(f"| {k} | {P(v['direction_hit_rate'])} | {v['mean_abs_vote']} | {v['mean_squared_loss']} |")
    L += ['', 'No pillar beats 50 % on direction. Under squared loss a vote near 0 costs about 0.25, while a bold vote that is right '
          'half the time costs more on average, so Hedge shifts weight to the pillars that are usually near zero (dollar, stocks).', '',
          '## Sensitivity to the fixed share α (η fixed)', '', '| α | ' + ' | '.join(hw['pillars']) + ' | Lockbox hit | Lockbox AUC |',
          '|---|' + '---|' * (len(hw['pillars']) + 2)]
    for a, v in res['alpha_sensitivity'].items():
        L.append(f'| {a} | ' + ' | '.join(f'{w*100:.1f}%' for w in v['final_weights']) + f" | {P(v['lockbox_hit'])} | {N(v['lockbox_auc'])} |")
    L += ['', 'With α = 0.01 the fixed share keeps the weights near equal; with α = 0 almost all weight goes to the dollar and stock pillars. '
          'Accuracy is at chance level in every case.']
    L += ['', '## Online logistic regression: largest coefficients (standardised)', '']
    L += [f'- `{k}`: {v:+.4f}' for k, v in res['sgd_top_coefficients'].items()]
    L += ['', f"Variants tried: {', '.join(pr['variants_tried'])}"]
    open(path, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
