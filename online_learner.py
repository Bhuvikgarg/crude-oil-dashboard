"""
Self-learning pillar weights with the Hedge algorithm (online learning).

Each of the engine's six sentiment pillars is treated as an "expert" that votes on
where WTI will be 4 hourly bars later (its pillar value / 100, in [-1, +1]).
After each outcome becomes known, every expert's weight is multiplied by
exp(-eta * loss), so pillars that were right gain influence and pillars that were
wrong lose it (Freund & Schapire, 1997). A small "fixed share" alpha mixes some
weight back to every pillar so the learner can follow a market whose regime
changes (Herbster & Warmuth, 1998).

Delayed feedback: a 4-hour call can only be scored 4 bars later, so an update
always uses a prediction made at least 4 bars ago. The learner never sees the
outcome of a call before that outcome exists.

Used in two places:
  * evaluation/online_learning.py  replays it hour by hour on 2.4 years of history
  * engine.py                      keeps learning live, every hourly refresh
This module has no project imports, so both can use it without circular imports.
"""
import json
import math
import os
import threading
from datetime import datetime, timezone

import numpy as np

PILLARS = ['mtf_alignment', 'order_flow_vwap', 'crack_margin', 'brent_spread', 'usd_impact', 'equity_risk_on']
PILLAR_NAMES = {
    'mtf_alignment': 'Multi-timeframe EMA alignment',
    'order_flow_vwap': 'Order flow / VWAP distance',
    'crack_margin': 'Refining crack margin',
    'brent_spread': 'Brent–WTI spread',
    'usd_impact': 'US dollar (DXY) pressure',
    'equity_risk_on': 'S&P 500 risk appetite',
}
HAND_WEIGHTS = np.array([25, 20, 15, 15, 10, 10], dtype=float)   # engine.py weights (OVX pillar has no breakdown value)
HORIZON = 4


def experts_from(pillars):
    """Pillar dict (engine breakdown, -100..100) -> expert predictions in [-1, 1]."""
    return np.clip(np.array([float(pillars[k]) for k in PILLARS]) / 100.0, -1.0, 1.0)


class Hedge:
    def __init__(self, eta=0.05, alpha=0.01, weights=None):
        w = np.array(HAND_WEIGHTS if weights is None else weights, dtype=float)
        self.w = w / w.sum()
        self.eta, self.alpha = float(eta), float(alpha)
        self.updates = 0

    def score(self, x):
        """Learned composite in [-100, 100], same scale as the engine's sentiment score."""
        return float(np.dot(self.w, x) * 100.0)

    def update(self, x, y):
        """x: expert predictions made 4 bars ago; y: +1 if price rose since then, else -1."""
        loss = (x - y) ** 2 / 4.0                       # squared loss scaled to [0, 1]
        w = self.w * np.exp(-self.eta * loss)
        w = w / w.sum()
        self.w = (1 - self.alpha) * w + self.alpha / len(w)   # fixed share
        self.updates += 1


class LiveHedge:
    """Hedge learner that persists its state and learns from the live hourly refresh."""

    def __init__(self, state_path, init_path):
        self.state_path, self.lock = state_path, threading.Lock()
        path = state_path if os.path.exists(state_path) else init_path
        with open(path) as f:
            s = json.load(f)
        self.h = Hedge(s['eta'], s['alpha'], s['weights'])
        self.h.updates = s.get('updates', 0)
        self.backtest_updates = s.get('backtest_updates', self.h.updates)
        self.pending = s.get('pending', [])
        self.history = s.get('history', [])
        self.last_update = s.get('last_update')
        self.trained_until = s.get('trained_until')

    @classmethod
    def load(cls, root):
        init = os.path.join(root, 'reports', 'online_state_init.json')
        if not os.path.exists(init):
            return None
        return cls(os.path.join(root, 'state', 'online_state.json'), init)

    def _save(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp = self.state_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'eta': self.h.eta, 'alpha': self.h.alpha, 'weights': self.h.w.tolist(), 'updates': self.h.updates,
                       'backtest_updates': self.backtest_updates, 'pending': self.pending[-200:],
                       'history': self.history[-400:], 'last_update': self.last_update,
                       'trained_until': self.trained_until}, f)
        os.replace(tmp, self.state_path)

    def observe(self, close_series, pillars, now=None):
        """
        close_series: WTI hourly closes indexed by bar CLOSE time (UTC, tz-aware pandas index).
        pillars: the latest engine breakdown dict.
        1) resolves pending calls whose 4-bar outcome is now known and updates the weights,
        2) records the latest bar's call as pending.
        """
        now = now or datetime.now(timezone.utc)
        idx = close_series.index
        with self.lock:
            changed = False
            keep = []
            for p in self.pending:
                t = p['bar_close']
                matches = np.where(idx.strftime('%Y-%m-%d %H:%M') == t)[0]
                if len(matches) and matches[0] + HORIZON < len(idx) and idx[matches[0] + HORIZON].to_pydatetime() <= now:
                    fut = float(close_series.iloc[matches[0] + HORIZON])
                    y = 1.0 if fut > p['price'] else -1.0
                    self.h.update(np.array(p['x']), y)
                    self.last_update = now.strftime('%Y-%m-%d %H:%M UTC')
                    self.trained_until = t
                    self.history.append({'t': t, 'w': [round(v, 4) for v in self.h.w]})
                    changed = True
                elif (now - datetime.strptime(t, '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)).days < 5:
                    keep.append(p)                      # not resolvable yet; drop if older than 5 days
                else:
                    changed = True
            self.pending = keep
            latest = idx[-1].strftime('%Y-%m-%d %H:%M')
            if all(p['bar_close'] != latest for p in self.pending) and (self.trained_until is None or latest > self.trained_until):
                self.pending.append({'bar_close': latest, 'price': float(close_series.iloc[-1]),
                                     'x': [round(v, 4) for v in experts_from(pillars)]})
                changed = True
            if changed:
                self._save()

    def summary(self, pillars):
        x = experts_from(pillars)
        s = self.h.score(x)
        hand = HAND_WEIGHTS / HAND_WEIGHTS.sum()
        return {
            'algorithm': 'Hedge (multiplicative weights) with fixed share',
            'eta': self.h.eta, 'alpha': self.h.alpha,
            'pillars': [{'key': k, 'name': PILLAR_NAMES[k], 'hand_weight': round(float(hand[i]), 4),
                         'learned_weight': round(float(self.h.w[i]), 4), 'vote': round(float(x[i]) * 100, 1)}
                        for i, k in enumerate(PILLARS)],
            'learned_score': round(s, 1),
            'learned_call': 'BULL' if s >= 15 else 'BEAR' if s <= -15 else 'NEUTRAL',
            'updates_total': self.h.updates, 'updates_live': self.h.updates - self.backtest_updates,
            'pending': len(self.pending), 'last_update': self.last_update, 'trained_until': self.trained_until,
            'history': self.history[-60:],
        }
