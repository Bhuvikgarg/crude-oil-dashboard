"""Hedge update rule and the live learner's delayed-feedback logic."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from online_learner import Hedge, LiveHedge, PILLARS


def test_hedge_rewards_the_right_expert():
    h = Hedge(eta=0.5, alpha=0.0, weights=[1, 1, 1, 1, 1, 1])
    x = np.array([1, -1, 0, 0, 0, 0], float)          # expert 0 says up, expert 1 says down
    for _ in range(10):
        h.update(x, 1.0)                                # price went up
    assert h.w[0] > h.w[2] > h.w[1]
    assert abs(h.w.sum() - 1) < 1e-9


def test_fixed_share_keeps_every_weight_alive():
    h = Hedge(eta=5.0, alpha=0.05, weights=[1, 1, 1, 1, 1, 1])
    for _ in range(200):
        h.update(np.array([1, -1, -1, -1, -1, -1], float), 1.0)
    assert h.w.min() >= 0.05 / 6 - 1e-12


def test_live_learner_only_learns_after_outcome_exists(tmp_path):
    init = tmp_path / 'init.json'
    json.dump({'eta': 0.5, 'alpha': 0.0, 'weights': [1] * 6, 'updates': 0, 'history': []}, open(init, 'w'))
    lh = LiveHedge(str(tmp_path / 'state.json'), str(init))
    pillars = {k: 0.0 for k in PILLARS}
    pillars['mtf_alignment'] = 100.0                   # one pillar votes "up"
    idx = pd.date_range('2026-01-05 10:00', periods=3, freq='h', tz='UTC')
    now = datetime(2026, 1, 5, 13, 0, tzinfo=timezone.utc)
    lh.observe(pd.Series([70.0, 70.5, 71.0], index=idx), pillars, now=now)
    assert lh.h.updates == 0 and len(lh.pending) == 1   # call recorded, outcome not known yet
    idx2 = pd.date_range('2026-01-05 10:00', periods=7, freq='h', tz='UTC')
    later = datetime(2026, 1, 5, 17, 0, tzinfo=timezone.utc)
    lh.observe(pd.Series([70, 70.5, 71, 71.5, 72, 72.5, 73.0], index=idx2), pillars, now=later)
    assert lh.h.updates == 1                            # 4 bars later the outcome exists -> one update
    assert lh.h.w[0] == lh.h.w.max()                    # the pillar that called "up" gained weight
