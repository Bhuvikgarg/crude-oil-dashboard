"""
Walk-forward (point-in-time) replay of the live PetroSync engine.

For every hourly bar i of WTI we rebuild *exactly* the inputs the live
dashboard would have had at the close of that bar, call the unmodified
`engine.compute_snapshot(...)`, and store the signals. Nothing after the
decision time is visible to the engine -> no look-ahead bias.

Time convention: Yahoo stamps a 1h bar with its START time. A bar stamped
10:00 closes at 11:00, so the decision time is t_close = t_start + 1h.
  * 1h basket : rows with start <= t_start
  * 15m bars  : rows with start <  t_close   (real 15m only exists for ~60 days;
                before that the 1h series is used as a documented proxy)
  * Daily     : completed sessions before the current CME trade date, plus a
                synthetic "today" bar whose close = last 1h close (this is what
                Yahoo's live, still-forming daily bar looks like).
"""
import logging
import time
import numpy as np
import pandas as pd

from engine import AdvancedOilEngine

ONE_HOUR = pd.Timedelta(hours=1)


def _cl_frame(macro_1h):
    return pd.DataFrame({k: macro_1h[k]['CL=F'] for k in ['Open', 'High', 'Low', 'Close', 'Volume']}).dropna(subset=['Close'])


def _trade_date(ts_utc):
    """CME Globex session for date D opens 18:00 ET on D-1."""
    return (ts_utc.tz_convert('America/New_York') + pd.Timedelta(hours=6)).date()


def replay(hist, lookback_1h=300, warmup=120, step=1, max_bars=None, force_15m_proxy=False):
    eng = AdvancedOilEngine()
    macro, cl_15m, cl_daily = hist['macro_1h'], hist['cl_15m'], hist['cl_daily']
    cl = _cl_frame(macro)
    daily_dates = np.array([d.date() for d in cl_daily.index.tz_convert('America/New_York')])
    real_15m_from = cl_15m.index[0] + pd.Timedelta(days=3)

    idx_positions = range(warmup, len(cl), step)
    if max_bars:
        idx_positions = list(idx_positions)[-max_bars:]

    rows, t0 = [], time.time()
    for n, i in enumerate(idx_positions):
        t_start = cl.index[i]
        t_close = t_start + ONE_HOUR
        macro_slice = macro.loc[:t_start].iloc[-lookback_1h:]

        if t_close >= real_15m_from and not force_15m_proxy:
            s15 = cl_15m[cl_15m.index < t_close].iloc[-300:]
            tf15_real = True
        else:
            s15 = cl.iloc[max(0, i - 299): i + 1]
            tf15_real = False

        td = _trade_date(t_close)
        prior = cl_daily[daily_dates < td].iloc[-29:][['Close']]
        today = pd.DataFrame({'Close': [float(cl['Close'].iloc[i])]}, index=[t_close])
        sdaily = pd.concat([prior, today])

        try:
            snap = eng.compute_snapshot(macro_slice, s15, sdaily, include_display=False)
        except Exception as e:  # missing intermarket data on a given bar, etc.
            logging.debug('skip %s: %s', t_close, e)
            continue

        ts, w, s, m = snap['trade_setup'], snap['wti'], snap['sentiment'], snap['macro_radar']
        tfs = snap['mtf_confluence']['timeframes']
        rec = {
            'bar_index': i, 'time': t_close, 'price': w['price'], 'tf15_real': tf15_real,
            'sent_score': s['score'], 'sent_bull_pct': s['bullish_pct'], 'sent_badge': s['badge'],
            'sent_cal_p': s.get('calibrated_p_up_4h'),
            'mtf_score': snap['mtf_confluence']['overall_score'],
            'tf15_score': tfs[0]['score'], 'tf1h_score': tfs[1]['score'], 'tfD_score': tfs[2]['score'],
            'rsi_15m': tfs[0]['rsi'], 'rsi_1h': w['rsi'], 'atr': w['atr'],
            'vwap': w['vwap'], 'vwap_u1': w['vwap_u1'], 'vwap_l1': w['vwap_l1'],
            'vwap_u2': w['vwap_u2'], 'vwap_l2': w['vwap_l2'],
            'ema9': w['ema9'], 'ema20': w['ema20'], 'ema50': w['ema50'],
            'poc': snap['levels']['poc'], 'vah': snap['levels']['vah'], 'val': snap['levels']['val'],
            'swept_high': snap['liquidity_sweeps']['swept_high'], 'swept_low': snap['liquidity_sweeps']['swept_low'],
            'ovx': m['ovx_price'], 'crack': m['crack_margin'], 'dxy_chg': m['dxy_change_pct'],
            'es_chg': m['sp500_change_pct'], 'gold_oil': m['gold_oil_ratio'],
            'brent_spread': snap['brent']['spread_to_wti'],
            'setup_type': ts['setup_type'], 'setup_bias': ts['bias'],
            'entry_min': ts['entry_min'], 'entry_max': ts['entry_max'],
            'stop': ts['stop_loss'], 'target1': ts['target1'], 'target2': ts['target2'],
            'claimed_win_prob': ts.get('claimed_win_probability', ts['win_probability']),  # the playbook's original claim
        }
        for k, v in s['breakdown'].items():
            rec[f'pillar_{k}'] = v
        rows.append(rec)
        if n and n % 1000 == 0:
            logging.info('replayed %d bars (%.0fs)', n, time.time() - t0)

    sig = pd.DataFrame(rows).set_index('bar_index')
    return sig, cl


def attach_outcomes(sig, cl, horizons=(1, 4, 24)):
    """Add realised forward returns / price paths (the 'ground truth')."""
    close = cl['Close'].to_numpy()
    out = sig.copy()
    for h in horizons:
        fut = np.full(len(close), np.nan)
        fut[:-h] = close[h:]
        out[f'fwd_close_{h}'] = fut[out.index]
        out[f'fwd_ret_{h}'] = out[f'fwd_close_{h}'] / out['price'] - 1
    nxt = np.full(len(close), np.nan); nxt[:-1] = close[1:]
    out['next_close'] = nxt[out.index]
    return out


def simulate_trades(sig, cl, fill_window=6, max_hold=24, cost_per_rt=0.02, contract_mult=1000, overlapping=False):
    """
    Path-dependent simulation of the TT Playbook setups on 1h bars.

    * LONG  : limit buy at entry_max (fills at open if the bar gaps below it)
    * SHORT : limit sell at entry_min (fills at open if the bar gaps above it)
    * NEUTRAL (range rotation): only the long leg at VAL has a defined stop -> simulated as LONG
    * After fill: stop / target1 checked on each bar's Low/High. If both are touched in
      the same bar we assume the STOP hit first (conservative). Exit at close after max_hold bars.
    * One position at a time (non-overlapping) so trades are not double counted.
      overlapping=True simulates every signal independently (used to build ML labels).
    * cost_per_rt: round-turn friction in price units (0.02 = 2 ticks = $20 per CL contract).
    """
    o, h, l, c = (cl[k].to_numpy() for k in ['Open', 'High', 'Low', 'Close'])
    trades, next_free = [], -1
    for i, r in sig.iterrows():
        if i <= next_free and not overlapping:
            continue
        side = 'SHORT' if r.setup_bias == 'SHORT' else 'LONG'
        entry = r.entry_min if side == 'SHORT' else r.entry_max
        stop, t1 = r.stop, r.target1
        base = {'signal_bar': i, 'time': r.time, 'setup_type': r.setup_type, 'side': side,
                'claimed_win_prob': r.claimed_win_prob, 'entry': entry, 'stop': stop, 'target1': t1}

        geometry_ok = (stop < entry < t1) if side == 'LONG' else (t1 < entry < stop)
        if not geometry_ok:
            trades.append({**base, 'outcome': 'INVALID_GEOMETRY'})
            continue

        fill_bar, fill_px = None, None
        for j in range(i + 1, min(i + 1 + fill_window, len(c))):
            if side == 'LONG' and l[j] <= entry:
                fill_bar, fill_px = j, min(o[j], entry); break
            if side == 'SHORT' and h[j] >= entry:
                fill_bar, fill_px = j, max(o[j], entry); break
        if fill_bar is None:
            trades.append({**base, 'outcome': 'NOT_FILLED'})
            next_free = i + fill_window
            continue

        risk = abs(entry - stop)  # PLANNED risk (1R). Gap fills can make the real loss bigger than 1R.
        outcome, exit_px, exit_bar = 'TIMEOUT', None, None
        for j in range(fill_bar, min(fill_bar + max_hold, len(c))):
            hit_sl = l[j] <= stop if side == 'LONG' else h[j] >= stop
            hit_tp = h[j] >= t1 if side == 'LONG' else l[j] <= t1
            if hit_sl:
                outcome, exit_bar = 'STOP', j
                exit_px = min(o[j], stop) if side == 'LONG' else max(o[j], stop)  # gaps make it worse
                break
            # On the fill bar we cannot know if the high/low came before the fill -> only stops count there
            if hit_tp and j > fill_bar:
                outcome, exit_px, exit_bar = 'TARGET1', t1, j
                break
        if exit_bar is None:
            exit_bar = min(fill_bar + max_hold, len(c)) - 1
            exit_px = c[exit_bar]
        pnl_px = (exit_px - fill_px) if side == 'LONG' else (fill_px - exit_px)
        pnl_px -= cost_per_rt
        trades.append({**base, 'outcome': outcome, 'fill_px': fill_px, 'exit_px': exit_px,
                       'bars_held': exit_bar - fill_bar + 1, 'risk_px': risk,
                       'r_multiple': pnl_px / risk if risk > 0 else np.nan,
                       'pnl_usd': pnl_px * contract_mult})
        next_free = exit_bar
    return pd.DataFrame(trades)
