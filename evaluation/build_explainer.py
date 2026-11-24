"""Build docs/explainer.html (interactive from-scratch explainer) from the project's real data.

Run: python -m evaluation.build_explainer   (needs reports/ and evaluation/data_cache/signals.pkl)
"""
import json
import os

import pandas as pd

from evaluation.backtest import _cl_frame, attach_outcomes
from evaluation.data_loader import CACHE_DIR, load_history

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILLARS = ['pillar_mtf_alignment', 'pillar_order_flow_vwap', 'pillar_crack_margin',
           'pillar_brent_spread', 'pillar_usd_impact', 'pillar_equity_risk_on']


def main():
    cl = _cl_frame(load_history()['macro_1h'])
    df = attach_outcomes(pd.read_pickle(os.path.join(CACHE_DIR, 'signals.pkl')), cl)
    series = json.load(open(os.path.join(ROOT, 'reports', 'chart_series.json')))[-240:]
    lock = df[df.time >= '2026-04-01'].dropna(subset=['fwd_ret_4'])
    sample = lock.iloc[::6].head(600)            # every 6th hour: (score, went up?) pairs for the learning demo
    seg = lock.iloc[:600]                        # 600 consecutive hours for the Hedge simulator
    ev = json.load(open(os.path.join(ROOT, 'reports', 'evaluation_results.json')))
    cal = json.load(open(os.path.join(ROOT, 'reports', 'engine_calibration.json')))
    data = {
        'price': [{'t': s['time'][:16], 'p': round(s['price'], 2)} for s in series],
        'real': [[round(float(s), 1), int(r > 0)] for s, r in zip(sample.sent_score, sample.fwd_ret_4)],
        'hedge': [{'t': str(t)[:13], 'x': [round(float(v) / 100, 2) for v in row], 'y': 1 if r > 0 else -1}
                  for t, row, r in zip(seg.time, seg[PILLARS].to_numpy(), seg.fwd_ret_4)],
        'rel': ev['signals']['h4']['sentiment_7pillar']['probability']['reliability'],
        'cal': {'coef': cal['coef'], 'intercept': cal['intercept']},
    }
    tpl = open(os.path.join(ROOT, 'evaluation', 'explainer_template.html')).read()
    out = tpl.replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
    open(os.path.join(ROOT, 'docs', 'explainer.html'), 'w').write(out)
    print('wrote docs/explainer.html', len(out) // 1024, 'KB')


if __name__ == '__main__':
    main()
