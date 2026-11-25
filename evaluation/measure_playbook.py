"""
Replace the playbook's hard-coded win rates with measured ones.

Reads the walk-forward trade simulation (reports/trades.csv, written by run_evaluation)
and, for each of the five setups, measures:
  * how often Target 1 was hit before the stop (with a 95% Wilson interval),
  * the same rate in the development period and in the lockbox (stability check),
  * average result per contract after costs, profit factor and average R.

Writes reports/playbook_measured.json, which engine.py reads so the dashboard shows the
measured win rate instead of the number that was typed into the code.

    python -m evaluation.measure_playbook
"""
import json
import math
import os
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADES = os.path.join(ROOT, 'reports', 'trades.csv')
OUT = os.path.join(ROOT, 'reports', 'playbook_measured.json')
LOCKBOX_START = '2026-04-01'
FILLED = {'TARGET1', 'STOP', 'TIMEOUT'}


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion (better than p ± 2·SE for small n)."""
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def summarise(g):
    n = len(g)
    k = int((g.outcome == 'TARGET1').sum())
    lo, hi = wilson(k, n)
    wins, losses = g.pnl_usd[g.pnl_usd > 0].sum(), -g.pnl_usd[g.pnl_usd < 0].sum()
    return {
        'n': n, 'target1_hits': k,
        'win_rate': round(k / n, 4) if n else None,
        'ci95': [round(lo, 4), round(hi, 4)] if n else None,
        'expectancy_usd': round(float(g.pnl_usd.mean()), 1) if n else None,
        'profit_factor': round(float(wins / losses), 3) if losses > 0 else None,
        'avg_r': round(float(g.r_multiple.mean()), 3) if n else None,
        'avg_win_usd': round(float(g.pnl_usd[g.outcome == 'TARGET1'].mean()), 0) if k else None,
        'avg_loss_usd': round(float(g.pnl_usd[g.outcome == 'STOP'].mean()), 0) if (g.outcome == 'STOP').any() else None,
        'median_planned_rr': round(float(((g.target1 - g.entry).abs() / (g.entry - g.stop).abs()).median()), 2) if n else None,
    }


def main():
    t = pd.read_csv(TRADES)
    t = t[t.outcome.isin(FILLED)].copy()
    t['time'] = pd.to_datetime(t.time, utc=True)
    lock = t.time >= pd.Timestamp(LOCKBOX_START, tz='UTC')
    setups = {}
    for name, g in t.groupby('setup_type'):
        s = summarise(g)
        s['claimed'] = round(float(g.claimed_win_prob.iloc[0]) / 100, 2)
        s['dev'] = summarise(g[~lock.loc[g.index]])
        s['lockbox'] = summarise(g[lock.loc[g.index]])
        setups[name] = s
    out = {
        'what': 'Target 1 hit before the stop, 1 CL contract, after 2 ticks round-turn cost, non-overlapping trades',
        'source': 'reports/trades.csv (walk-forward replay of engine.py)',
        'sample': [str(t.time.min()), str(t.time.max())],
        'lockbox_start': LOCKBOX_START,
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'overall': summarise(t),
        'setups': setups,
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"{'setup':48s} {'n':>4s} {'claim':>6s} {'measured':>9s} {'95% CI':>14s} {'dev':>6s} {'lock':>6s} {'$/trade':>8s}")
    for name, s in sorted(setups.items(), key=lambda kv: -kv[1]['n']):
        print(f"{name:48s} {s['n']:4d} {s['claimed']:6.0%} {s['win_rate']:9.1%} "
              f"{s['ci95'][0]:6.1%}–{s['ci95'][1]:5.1%} {s['dev']['win_rate']:6.1%} {s['lockbox']['win_rate']:6.1%} {s['expectancy_usd']:8.0f}")
    print('wrote', OUT)


if __name__ == '__main__':
    main()
