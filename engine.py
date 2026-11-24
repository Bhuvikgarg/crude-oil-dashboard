import os
import time
import math
import logging
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class AdvancedOilEngine:
    def __init__(self):
        self.cached_data = None
        self.last_fetch_time = 0
        self.min_cache_seconds = 45 # cache for 45 seconds to keep it fresh yet rate-limit safe
        self.calibration = self._load_calibration()
        # Volatility forecast (HAR ridge, fitted on 2.4 years by evaluation/volatility.py)
        try:
            import json as _json
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports', 'vol_model.json')) as f:
                self.vol_model = _json.load(f)
        except (OSError, ValueError):
            self.vol_model = None
        # Self-learning pillar weights (Hedge). Starts from the weights learned on 2.4 years of
        # history (reports/online_state_init.json) and keeps updating at every hourly refresh.
        try:
            from online_learner import LiveHedge
            self.learner = LiveHedge.load(os.path.dirname(os.path.abspath(__file__)))
        except Exception as e:
            logging.warning(f"Online learner unavailable: {e}")
            self.learner = None

    @staticmethod
    def _load_calibration():
        """Platt calibration of the sentiment score, produced by `python -m evaluation.improve`."""
        import json, os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports', 'engine_calibration.json')
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def fetch_market_basket(self):
        """Fetch 1h and 15m data for WTI, Brent, DXY, Gasoline, Heating Oil, Gold, OVX, S&P, XLE"""
        try:
            # 1h basket for macro & session
            macro_tickers = ['CL=F', 'BZ=F', 'DX-Y.NYB', 'RB=F', 'HO=F', 'GC=F', '^OVX', 'ES=F', 'XLE']
            df_1h = yf.download(macro_tickers, period='5d', interval='1h', progress=False)

            # 15m and Daily for WTI multi-timeframe analysis
            cl_ticker = yf.Ticker('CL=F')
            df_15m = cl_ticker.history(period='3d', interval='15m')
            df_daily = cl_ticker.history(period='30d', interval='1d')

            return {
                "macro_1h": df_1h,
                "cl_15m": df_15m,
                "cl_daily": df_daily
            }
        except Exception as e:
            logging.error(f"Error fetching market basket: {e}")
            return None

    def calculate_technical_series(self, cl_df):
        """Compute EMAs, RSI, ATR, VWAP, and VWAP Standard Deviation Bands"""
        df = cl_df.copy().dropna(subset=['Close'])

        # EMAs
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss.replace(0, 0.0001))
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)

        # ATR (14)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean().bfill()

        # Session VWAP & StDev Bands (rolling 24 bars)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
        df['VolPrice'] = typical_price * df['Volume']
        roll_vol = df['Volume'].rolling(24, min_periods=1).sum().replace(0, 1)
        df['VWAP'] = df['VolPrice'].rolling(24, min_periods=1).sum() / roll_vol

        # Rolling standard deviation from VWAP
        df['Variance'] = ((typical_price - df['VWAP']) ** 2) * df['Volume']
        df['VwStDev'] = np.sqrt(df['Variance'].rolling(24, min_periods=1).sum() / roll_vol)
        df['VWAP_Upper1'] = df['VWAP'] + df['VwStDev']
        df['VWAP_Lower1'] = df['VWAP'] - df['VwStDev']
        df['VWAP_Upper2'] = df['VWAP'] + (2 * df['VwStDev'])
        df['VWAP_Lower2'] = df['VWAP'] - (2 * df['VwStDev'])

        return df

    def calculate_timeframe_state(self, df, label):
        """Analyze a specific timeframe (15m, 1h, Daily) for trend, momentum, and signal"""
        if df is None or len(df) < 15:
            return {
                "timeframe": label,
                "bias": "NEUTRAL",
                "score": 50,
                "rsi": 50.0,
                "trend": "Insufficient Data",
                "color": "#94a3b8"
            }

        close = df['Close']
        last = float(close.iloc[-1])
        ema9 = float(close.ewm(span=9).mean().iloc[-1])
        ema21 = float(close.ewm(span=21).mean().iloc[-1])

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = float((100 - (100 / (1 + (gain / loss.replace(0, 0.0001))))).iloc[-1])

        score = 50
        # EMA Structure
        if last > ema9 and ema9 > ema21:
            score += 30
            trend = "Bullish Alignment (Price > 9 EMA > 21 EMA)"
        elif last < ema9 and ema9 < ema21:
            score -= 30
            trend = "Bearish Alignment (Price < 9 EMA < 21 EMA)"
        else:
            trend = "Consolidating / Mixed EMAs"

        # RSI Momentum
        if rsi >= 60:
            score += 20
        elif rsi <= 40:
            score -= 20

        score = max(0, min(100, score))

        if score >= 70:
            bias = "BULLISH"
            color = "#10b981"
        elif score <= 30:
            bias = "BEARISH"
            color = "#ef4444"
        else:
            bias = "NEUTRAL"
            color = "#f59e0b"

        return {
            "timeframe": label,
            "bias": bias,
            "score": score,
            "rsi": round(rsi, 1),
            "trend": trend,
            "color": color
        }

    def calculate_volume_profile(self, cl_df, lookback_bars=24):
        """Calculate Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL)"""
        recent = cl_df.iloc[-lookback_bars:]
        if len(recent) < 5:
            recent = cl_df

        prices = []
        volumes = []
        for _, row in recent.iterrows():
            low = row['Low']
            high = row['High']
            vol = row['Volume'] if row['Volume'] > 0 else 100
            if math.isnan(low) or math.isnan(high) or high <= low:
                prices.append(row['Close'])
                volumes.append(vol)
                continue
            sub_prices = np.linspace(low, high, 8)
            vol_per_slice = vol / 8.0
            for p in sub_prices:
                prices.append(p)
                volumes.append(vol_per_slice)

        if not prices:
            last_p = cl_df['Close'].iloc[-1]
            return round(last_p, 2), round(last_p + 1.0, 2), round(last_p - 1.0, 2)

        df_vp = pd.DataFrame({'price': prices, 'volume': volumes})
        df_vp['bin'] = (df_vp['price'] * 10).round() / 10
        profile = df_vp.groupby('bin')['volume'].sum().sort_index()

        poc = float(profile.idxmax())
        total_vol = profile.sum()
        target_vol = total_vol * 0.70

        poc_idx = profile.index.get_loc(poc)
        up_idx = poc_idx
        down_idx = poc_idx
        accum_vol = profile.iloc[poc_idx]

        while accum_vol < target_vol and (up_idx < len(profile) - 1 or down_idx > 0):
            up_vol = profile.iloc[up_idx + 1] if up_idx < len(profile) - 1 else 0
            down_vol = profile.iloc[down_idx - 1] if down_idx > 0 else 0
            if up_vol >= down_vol and up_idx < len(profile) - 1:
                up_idx += 1
                accum_vol += up_vol
            elif down_idx > 0:
                down_idx -= 1
                accum_vol += down_vol
            else:
                break

        vah = float(profile.index[up_idx])
        val = float(profile.index[down_idx])
        return round(poc, 2), round(vah, 2), round(val, 2)

    def detect_liquidity_sweeps(self, cl_df):
        """Identify if recent bars swept the Overnight High/Low or Prior Session extremes"""
        if len(cl_df) < 24:
            return {"swept_high": False, "swept_low": False, "status": "Normal Trading Inside Session"}

        recent_24 = cl_df.iloc[-24:]
        session_high = recent_24['High'].max()
        session_low = recent_24['Low'].min()
        last_bar = cl_df.iloc[-1]
        last_close = last_bar['Close']
        last_high = last_bar['High']
        last_low = last_bar['Low']

        swept_high = False
        swept_low = False
        details = "No extreme sweep detected."

        # Bearish Sweep: broke above session high then closed back below it
        if last_high >= session_high and last_close < session_high:
            swept_high = True
            details = f"Liquidity Sweep at High (${session_high:.2f}): Sellers absorbed buyers above range."
        elif last_low <= session_low and last_close > session_low:
            swept_low = True
            details = f"Liquidity Sweep at Low (${session_low:.2f}): Buyers absorbed sellers below range."

        return {
            "swept_high": swept_high,
            "swept_low": swept_low,
            "status": details
        }

    def generate_trade_setup(self, last_price, vwap, vwap_u1, vwap_l1, vah, val, poc, atr, mtf_confluence, ovx_val, rsi_1h, rsi_15m):
        """
        Synthesizes all quantitative signals into a concrete institutional trade setup
        with exact Entry, Stop Loss, Target 1, Target 2, Risk/Reward, and Probability.
        """
        # Direction determined by MTF confluence + Volume Area position
        score = mtf_confluence['overall_score']
        
        # Determine setup archetype
        if rsi_15m <= 25 and last_price < val:
            # Counter-trend oversold scalp
            setup_type = "COUNTER-TREND EXHAUSTION BOUNCE (LONG)"
            bias = "LONG"
            action = "Buy the Dip (Mean Reversion)"
            entry_min = round(last_price - (0.2 * atr), 2)
            entry_max = round(last_price + (0.1 * atr), 2)
            entry_str = f"${entry_min:.2f} - ${entry_max:.2f}"
            stop_loss = round(entry_min - (0.6 * atr), 2)
            stop_ticks = round((entry_min - stop_loss) * 100)
            target1 = round(vwap_l1 if vwap_l1 > last_price else last_price + (0.8 * atr), 2)
            target2 = round(vwap, 2)
            tp1_ticks = round((target1 - entry_min) * 100)
            rr_ratio = round(tp1_ticks / max(1, stop_ticks), 2)
            win_prob = 68
            rationale = f"15m RSI is deeply oversold ({rsi_15m:.1f}) stretched below Value Area Low (${val:.2f}). Expect a sharp mean-reversion bounce toward VWAP."
            tt_tip = "Place a Limit Buy on the MD Trader bid once sell volume on the ladder slows down. Keep stops strict."

        elif rsi_15m >= 75 and last_price > vah:
            # Counter-trend overbought fade
            setup_type = "EXHAUSTION BLOW-OFF FADE (SHORT)"
            bias = "SHORT"
            action = "Fade the Highs (Mean Reversion)"
            entry_min = round(last_price - (0.1 * atr), 2)
            entry_max = round(last_price + (0.2 * atr), 2)
            entry_str = f"${entry_min:.2f} - ${entry_max:.2f}"
            stop_loss = round(entry_max + (0.6 * atr), 2)
            stop_ticks = round((stop_loss - entry_max) * 100)
            target1 = round(vwap_u1 if vwap_u1 < last_price else last_price - (0.8 * atr), 2)
            target2 = round(vwap, 2)
            tp1_ticks = round((entry_max - target1) * 100)
            rr_ratio = round(tp1_ticks / max(1, stop_ticks), 2)
            win_prob = 67
            rationale = f"15m RSI is overbought ({rsi_15m:.1f}) piercing above VAH (${vah:.2f}). High probability of exhaustion pullback toward VWAP."
            tt_tip = "Watch for absorption at the Ask on MD Trader, then enter short when buyers fail to advance."

        elif score <= 35:
            # Bearish Trend Following
            setup_type = "BEARISH TREND CONTINUATION (SHORT PULLBACK)"
            bias = "SHORT"
            action = "Sell Rallies into Resistance"
            pullback_target = min(round(vwap, 2), round(last_price + (0.4 * atr), 2))
            entry_min = round(pullback_target - 0.15, 2)
            entry_max = round(pullback_target + 0.15, 2)
            entry_str = f"${entry_min:.2f} - ${entry_max:.2f}"
            stop_loss = round(entry_max + (0.7 * atr), 2)
            stop_ticks = round((stop_loss - entry_max) * 100)
            target1 = round(val if val < last_price else last_price - (0.9 * atr), 2)
            target2 = round(target1 - (1.2 * atr), 2)
            tp1_ticks = round((entry_min - target1) * 100)
            rr_ratio = round(tp1_ticks / max(1, stop_ticks), 2)
            win_prob = 74
            rationale = f"All timeframes (15m, 1h, Daily) are aligned bearish. Do not chase lows; wait for a pullback rally into VWAP (${vwap:.2f}) to enter short."
            tt_tip = "Place resting Sell Limit orders just below VWAP on the red side of the MD Trader ladder."

        elif score >= 65:
            # Bullish Trend Following
            setup_type = "BULLISH TREND CONTINUATION (LONG PULLBACK)"
            bias = "LONG"
            action = "Buy Dips into Support"
            pullback_target = max(round(vwap, 2), round(last_price - (0.4 * atr), 2))
            entry_min = round(pullback_target - 0.15, 2)
            entry_max = round(pullback_target + 0.15, 2)
            entry_str = f"${entry_min:.2f} - ${entry_max:.2f}"
            stop_loss = round(entry_min - (0.7 * atr), 2)
            stop_ticks = round((entry_min - stop_loss) * 100)
            target1 = round(vah if vah > last_price else last_price + (0.9 * atr), 2)
            target2 = round(target1 + (1.2 * atr), 2)
            tp1_ticks = round((target1 - entry_min) * 100)
            rr_ratio = round(tp1_ticks / max(1, stop_ticks), 2)
            win_prob = 73
            rationale = f"Multi-timeframe momentum is strongly bullish above VWAP (${vwap:.2f}). Look for liquidity pullbacks to reload longs."
            tt_tip = "Use a bracket order with an automatic stop loss 7-9 ticks below entry on MD Trader."

        else:
            # Balanced Range Play
            setup_type = "VALUE AREA RANGE ROTATION (FADE EDGES)"
            bias = "NEUTRAL"
            action = "Two-Way Edge Scalping"
            entry_str = f"Buy near ${val:.2f} / Sell near ${vah:.2f}"
            entry_min = entry_max = round(val, 2)  # long leg (the only leg with a defined stop)
            stop_loss = round(val - (0.5 * atr), 2)
            stop_ticks = 40
            target1 = round(poc, 2)
            target2 = round(vah, 2)
            rr_ratio = 1.8
            win_prob = 64
            rationale = f"Price is revolving inside the Value Area (${val:.2f} to ${vah:.2f}). Market has no directional imbalance; fade the edges toward POC (${poc:.2f})."
            tt_tip = "Trade smaller contract size and exit promptly at POC on the MD Trader ladder."

        # Adjust for high volatility (OVX)
        if ovx_val > 45:
            vol_warning = f"High Volatility Regime (OVX {ovx_val:.1f}): Spreads are wider; widen your stop by 5-10 ticks or trade Micro Crude (MCL)."
        else:
            vol_warning = "Standard Volatility Regime: Normal order flow and tight execution."

        return {
            "setup_type": setup_type,
            "bias": bias,
            "action": action,
            "entry_range": entry_str,
            "entry_min": entry_min,
            "entry_max": entry_max,
            "stop_loss": stop_loss,
            "stop_ticks": stop_ticks,
            "target1": target1,
            "target2": target2,
            "tp1_ticks": tp1_ticks if 'tp1_ticks' in locals() else 80,
            "risk_reward": f"1 : {rr_ratio:.1f}",
            "win_probability": win_prob,
            "rationale": rationale,
            "tt_tip": tt_tip,
            "vol_warning": vol_warning
        }

    def compute_advanced_sentiment(self, rsi, price, vwap, ema20, ema50, dxy_change, crack_spread, brent_wti_spread, gold_oil_ratio, ovx_val, es_change, mtf_score):
        """
        Institutional 7-Pillar Quantitative Sentiment Model:
        Outputs composite directional score (-100 to +100) and probability.
        """
        # 1. MTF Alignment Score (-100 to +100) - 25% weight
        mtf_component = (mtf_score - 50) * 2.0

        # 2. VWAP / Order Flow Position (-100 to +100) - 20% weight
        vwap_diff_pct = ((price - vwap) / vwap) * 100
        vwap_component = max(-100, min(100, vwap_diff_pct * 40))

        # 3. Refining Crack Margin (-100 to +100) - 15% weight
        # Neutral crack margin is ~$26/bbl
        crack_component = max(-100, min(100, (crack_spread - 26.0) * 6.0))

        # 4. Brent-WTI International Spread (-100 to +100) - 15% weight
        # Historical median spread ~$5.00
        spread_component = max(-100, min(100, (brent_wti_spread - 5.0) * 15.0))

        # 5. US Dollar Headwind (-100 to +100) - 10% weight
        dxy_component = -max(-100, min(100, dxy_change * 70))

        # 6. Global Risk Appetite via S&P 500 (-100 to +100) - 10% weight
        es_component = max(-100, min(100, es_change * 50))

        # 7. Volatility Drag / OVX (-100 to +100) - 5% weight
        # Very high OVX (>50) indicates panic or extreme distribution
        ovx_drag = -25 if ovx_val > 50 else (15 if ovx_val < 32 else 0)

        composite = (
            (mtf_component * 0.25) +
            (vwap_component * 0.20) +
            (crack_component * 0.15) +
            (spread_component * 0.15) +
            (dxy_component * 0.10) +
            (es_component * 0.10) +
            (ovx_drag * 0.05)
        )
        composite = round(max(-100, min(100, composite)), 1)
        bullish_pct = round((composite + 100) / 2.0, 1)

        if composite >= 50:
            label = "Institutional Bullish Trend"
            color = "#10b981"
            badge = "bull-strong"
            summary = "Dominant buyer conviction confirmed across multi-timeframe EMAs, physical refiner demand, and intermarket flows."
        elif composite >= 15:
            label = "Moderate Bullish Accumulation"
            color = "#34d399"
            badge = "bull-mod"
            summary = "Favorable technical and physical backdrop. Dips are being actively supported above institutional VWAP."
        elif composite <= -50:
            label = "Institutional Bearish Trend"
            color = "#ef4444"
            badge = "bear-strong"
            summary = "Aggressive institutional liquidation underway. Multi-timeframe trend is downward with heavy seller dominance."
        elif composite <= -15:
            label = "Moderate Bearish Distribution"
            color = "#f87171"
            badge = "bear-mod"
            summary = "Macro headwinds and seller resistance capping rallies. Price struggling below key moving averages."
        else:
            label = "Balanced / Two-Way Equilibrium"
            color = "#f59e0b"
            badge = "neutral"
            summary = "No clear directional edge. Market is oscillating in a fair-value auction range."

        # Calibrated probability: maps the composite onto how often price actually rose 4h later
        # in the walk-forward backtest (Platt scaling). None if no calibration file exists.
        calibrated = None
        if self.calibration:
            z = self.calibration['coef'] * composite + self.calibration['intercept']
            calibrated = round(100.0 / (1.0 + math.exp(-z)), 1)

        return {
            "score": composite,
            "bullish_pct": bullish_pct,
            "calibrated_p_up_4h": calibrated,
            "bearish_pct": round(100 - bullish_pct, 1),
            "label": label,
            "color": color,
            "badge": badge,
            "summary": summary,
            "breakdown": {
                "mtf_alignment": round(mtf_component, 1),
                "order_flow_vwap": round(vwap_component, 1),
                "crack_margin": round(crack_component, 1),
                "brent_spread": round(spread_component, 1),
                "usd_impact": round(dxy_component, 1),
                "equity_risk_on": round(es_component, 1)
            }
        }

    def get_eia_countdown(self):
        """Calculate exact countdown to Wednesday 10:30 AM EST (14:30 or 15:30 UTC)"""
        now = datetime.now(timezone.utc)
        days_ahead = (2 - now.weekday()) % 7
        target_date = now.date() + timedelta(days=days_ahead)
        target_time = datetime(target_date.year, target_date.month, target_date.day, 14, 30, tzinfo=timezone.utc)

        if target_time <= now:
            target_time += timedelta(days=7)

        diff = target_time - now
        total_seconds = int(diff.total_seconds())
        hours = total_seconds // 3600
        mins = (total_seconds % 3600) // 60
        return f"{hours}h {mins}m"

    def refresh(self, force=False):
        """Execute full quantitative cycle"""
        now_ts = time.time()
        if not force and self.cached_data and (now_ts - self.last_fetch_time < self.min_cache_seconds):
            return self.cached_data

        data_bundle = self.fetch_market_basket()
        if not data_bundle:
            if self.cached_data:
                return self.cached_data
            raise RuntimeError("Unable to load live market data.")

        data = self.compute_snapshot(data_bundle["macro_1h"], data_bundle["cl_15m"], data_bundle["cl_daily"])
        if self.vol_model is not None:
            try:
                data['volatility_forecast'] = self.forecast_volatility(data_bundle["macro_1h"], data['wti']['price'])
            except Exception as e:
                logging.warning(f"Volatility forecast skipped: {e}")
        if self.learner is not None:
            try:
                close = data_bundle["macro_1h"]['Close']['CL=F'].dropna()
                idx = close.index.tz_convert('UTC') if close.index.tz is not None else close.index.tz_localize('UTC')
                close.index = idx + pd.Timedelta(hours=1)          # key each bar by its CLOSE time
                self.learner.observe(close, data['sentiment']['breakdown'])
                data['online_learning'] = self.learner.summary(data['sentiment']['breakdown'])
            except Exception as e:
                logging.warning(f"Online learner step skipped: {e}")
        self.cached_data = data
        self.last_fetch_time = now_ts
        return data

    def forecast_volatility(self, macro_1h, price):
        """
        Expected size of WTI's move over the next 4 hourly bars (HAR model: log realised
        volatility over the last 1, 4, 24, 96 bars + log OVX + hour of day).
        Direction is not predictable, but the size of moves is (volatility clustering).
        """
        m = self.vol_model
        close = macro_1h['Close']['CL=F'].dropna()
        r = np.log(close).diff().dropna()
        feats = {}
        for k in m['lags']:
            feats[f'log_rv_{k}'] = math.log(math.sqrt(float((r.iloc[-k:] ** 2).mean())) + m['eps'])
        ovx = float(macro_1h['Close']['^OVX'].dropna().iloc[-1])
        feats['log_ovx'] = math.log(max(ovx, 5.0))
        last = close.index[-1]
        last = (last.tz_convert('America/New_York') if last.tzinfo else last.tz_localize('UTC').tz_convert('America/New_York')) + pd.Timedelta(hours=1)
        feats['hour_sin'], feats['hour_cos'] = math.sin(2 * math.pi * last.hour / 24), math.cos(2 * math.pi * last.hour / 24)
        x = np.array([feats[f] for f in m['features']])
        z = (x - np.array(m['scaler_mean'])) / np.array(m['scaler_scale'])
        log_rv = float(np.dot(z, m['coef']) + m['intercept'])
        rv_bar = max(math.exp(log_rv) - m['eps'], 0.0)
        move = price * rv_bar * math.sqrt(m['horizon_bars'])          # 1-sigma move in $ over 4 bars
        cuts = m['percentiles_log_rv_forecast']
        level = sum(log_rv > c for c in cuts)                           # 0..4
        names = ['Very calm', 'Calm', 'Normal', 'Stormy', 'Very stormy']
        return {
            'horizon_hours': m['horizon_bars'], 'expected_move_usd': round(move, 2),
            'expected_move_pct': round(rv_bar * math.sqrt(m['horizon_bars']) * 100, 2),
            'range_low': round(price - move, 2), 'range_high': round(price + move, 2),
            'regime_level': int(level), 'regime': names[level],
            'past_24h_move_pct': round(math.exp(feats['log_rv_24']) * math.sqrt(m['horizon_bars']) * 100, 2),
            'lockbox_r2': m.get('lockbox_r2_walkforward'),
        }

    def compute_snapshot(self, macro_1h, cl_15m, cl_daily, include_display=True):
        """
        Pure, side-effect-free quantitative cycle.

        Takes already-downloaded market data and returns the full signal snapshot.
        Separated from `refresh()` so the exact same code can be replayed bar-by-bar
        on historical data by `evaluation/backtest.py` (walk-forward, no look-ahead).
        `include_display=False` skips chart/history payloads that only the UI needs.
        """

        # Extract WTI 1h
        cl_1h = pd.DataFrame({
            'Open': macro_1h['Open']['CL=F'],
            'High': macro_1h['High']['CL=F'],
            'Low': macro_1h['Low']['CL=F'],
            'Close': macro_1h['Close']['CL=F'],
            'Volume': macro_1h['Volume']['CL=F']
        }).dropna(subset=['Close'])

        # Compute technicals on 1h
        cl_df = self.calculate_technical_series(cl_1h)
        last_row = cl_df.iloc[-1]
        prev_row = cl_df.iloc[-2] if len(cl_df) > 1 else last_row
        row_24h_ago = cl_df.iloc[-25] if len(cl_df) >= 25 else cl_df.iloc[0]

        last_price = float(last_row['Close'])
        prev_price = float(prev_row['Close'])
        price_24h = float(row_24h_ago['Close'])

        move_1h = last_price - prev_price
        move_1h_pct = (move_1h / prev_price) * 100
        move_24h = last_price - price_24h
        move_24h_pct = (move_24h / price_24h) * 100

        day_high = float(cl_df['High'].iloc[-24:].max())
        day_low = float(cl_df['Low'].iloc[-24:].min())

        rsi_1h = float(last_row['RSI'])
        atr = float(last_row['ATR'])
        vwap = float(last_row['VWAP'])
        vwap_u1 = float(last_row['VWAP_Upper1'])
        vwap_l1 = float(last_row['VWAP_Lower1'])
        vwap_u2 = float(last_row['VWAP_Upper2'])
        vwap_l2 = float(last_row['VWAP_Lower2'])
        ema9 = float(last_row['EMA9'])
        ema20 = float(last_row['EMA20'])
        ema50 = float(last_row['EMA50'])

        # Multi-timeframe analysis
        state_15m = self.calculate_timeframe_state(cl_15m, "15-Minute (Scalp)")
        state_1h = self.calculate_timeframe_state(cl_1h, "1-Hour (Session)")
        state_daily = self.calculate_timeframe_state(cl_daily, "Daily (Macro)")

        # Overall MTF Confluence Score (weighted: 15m=25%, 1h=45%, Daily=30%)
        mtf_overall_score = round((state_15m['score'] * 0.25) + (state_1h['score'] * 0.45) + (state_daily['score'] * 0.30), 1)
        if mtf_overall_score >= 65:
            mtf_alignment = "Bullish Confluence Across Timeframes"
            mtf_badge = "bull"
        elif mtf_overall_score <= 35:
            mtf_alignment = "Bearish Confluence Across Timeframes"
            mtf_badge = "bear"
        else:
            mtf_alignment = "Conflicted / Mixed Timeframes"
            mtf_badge = "neutral"

        mtf_confluence = {
            "overall_score": mtf_overall_score,
            "alignment": mtf_alignment,
            "badge": mtf_badge,
            "timeframes": [state_15m, state_1h, state_daily]
        }

        # Intermarket drivers
        brent_close = float(macro_1h['Close']['BZ=F'].dropna().iloc[-1])
        brent_spread = brent_close - last_price

        dxy_close = float(macro_1h['Close']['DX-Y.NYB'].dropna().iloc[-1])
        dxy_prev = float(macro_1h['Close']['DX-Y.NYB'].dropna().iloc[-2])
        dxy_change = ((dxy_close - dxy_prev) / dxy_prev) * 100

        rb_price = float(macro_1h['Close']['RB=F'].dropna().iloc[-1])
        ho_price = float(macro_1h['Close']['HO=F'].dropna().iloc[-1])
        crack_margin = round(((2 * (rb_price * 42.0)) + (1 * (ho_price * 42.0)) - (3 * last_price)) / 3.0, 2)

        gc_close = float(macro_1h['Close']['GC=F'].dropna().iloc[-1])
        gold_oil_ratio = round(gc_close / last_price, 2)

        ovx_close = float(macro_1h['Close']['^OVX'].dropna().iloc[-1])
        es_close = float(macro_1h['Close']['ES=F'].dropna().iloc[-1])
        es_prev = float(macro_1h['Close']['ES=F'].dropna().iloc[-2])
        es_change = ((es_close - es_prev) / es_prev) * 100

        xle_close = float(macro_1h['Close']['XLE'].dropna().iloc[-1])

        # OVX Regime interpretation
        if ovx_close < 30:
            ovx_regime = "Low Volatility (Coiling / Compression)"
            ovx_color = "#38bdf8"
        elif ovx_close <= 45:
            ovx_regime = "Normal Volatility (Predictable Order Flow)"
            ovx_color = "#10b981"
        else:
            ovx_regime = "Elevated / Stress Volatility (Expect Violent Sweeps)"
            ovx_color = "#ef4444"

        # Volume Profile
        poc, vah, val = self.calculate_volume_profile(cl_df, lookback_bars=24)

        # Liquidity Sweep Detection
        sweeps = self.detect_liquidity_sweeps(cl_df)

        # Actionable Trade Setup
        trade_setup = self.generate_trade_setup(
            last_price, vwap, vwap_u1, vwap_l1, vah, val, poc, atr, mtf_confluence, ovx_close, rsi_1h, state_15m['rsi']
        )

        # Advanced Quantitative Sentiment
        sentiment = self.compute_advanced_sentiment(
            rsi_1h, last_price, vwap, ema20, ema50, dxy_change, crack_margin, brent_spread, gold_oil_ratio, ovx_close, es_change, mtf_overall_score
        )

        # Pivot Points
        pivot = (day_high + day_low + last_price) / 3.0
        r1 = (2 * pivot) - day_low
        s1 = (2 * pivot) - day_high
        r2 = pivot + (day_high - day_low)
        s2 = pivot - (day_high - day_low)

        # Chart Series for Chart.js
        chart_candles = []
        recent_bars = cl_df.iloc[-48:] if include_display else cl_df.iloc[0:0]
        for idx, bar in recent_bars.iterrows():
            chart_candles.append({
                "time": (idx.tz_convert("UTC") if idx.tzinfo else idx).strftime("%b %d, %H:%M"),
                "open": round(float(bar['Open']), 2),
                "high": round(float(bar['High']), 2),
                "low": round(float(bar['Low']), 2),
                "close": round(float(bar['Close']), 2),
                "volume": int(bar['Volume']),
                "vwap": round(float(bar['VWAP']), 2),
                "vwap_u1": round(float(bar['VWAP_Upper1']), 2),
                "vwap_l1": round(float(bar['VWAP_Lower1']), 2),
                "ema20": round(float(bar['EMA20']), 2),
                "ema50": round(float(bar['EMA50']), 2)
            })

        # Compile Hourly History Log (24h)
        history_records = []
        hist_sample = cl_df.iloc[-24:] if include_display else cl_df.iloc[0:0]
        for i in range(len(hist_sample)):
            bar = hist_sample.iloc[i]
            p_close = float(bar['Close'])
            p_prev = float(hist_sample.iloc[i-1]['Close']) if i > 0 else p_close
            diff = p_close - p_prev
            diff_pct = (diff / p_prev) * 100 if p_prev > 0 else 0
            b_rsi = float(bar['RSI'])
            b_vwap = float(bar['VWAP'])

            if p_close > b_vwap and b_rsi > 52:
                h_reg = "Bullish Imbalance"
                h_badge = "bull"
            elif p_close < b_vwap and b_rsi < 48:
                h_reg = "Bearish Imbalance"
                h_badge = "bear"
            else:
                h_reg = "Balanced Rotation"
                h_badge = "neutral"

            history_records.append({
                "timestamp": (hist_sample.index[i].tz_convert("UTC") if hist_sample.index[i].tzinfo else hist_sample.index[i]).strftime("%H:%M UTC"),
                "price": round(p_close, 2),
                "change_1h": round(diff, 2),
                "change_1h_pct": round(diff_pct, 2),
                "rsi": round(b_rsi, 1),
                "vwap": round(b_vwap, 2),
                "regime": h_reg,
                "badge": h_badge
            })

        history_records = history_records[::-1]

        data = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "wti": {
                "symbol": "CL",
                "name": "WTI Light Sweet Crude Oil",
                "price": round(last_price, 2),
                "change_1h": round(move_1h, 2),
                "change_1h_pct": round(move_1h_pct, 2),
                "change_24h": round(move_24h, 2),
                "change_24h_pct": round(move_24h_pct, 2),
                "day_high": round(day_high, 2),
                "day_low": round(day_low, 2),
                "rsi": round(rsi_1h, 1),
                "atr": round(atr, 2),
                "vwap": round(vwap, 2),
                "vwap_u1": round(vwap_u1, 2),
                "vwap_l1": round(vwap_l1, 2),
                "vwap_u2": round(vwap_u2, 2),
                "vwap_l2": round(vwap_l2, 2),
                "ema9": round(ema9, 2),
                "ema20": round(ema20, 2),
                "ema50": round(ema50, 2)
            },
            "brent": {
                "symbol": "BZ",
                "name": "Brent Crude Oil",
                "price": round(brent_close, 2),
                "spread_to_wti": round(brent_spread, 2),
                "spread_status": "Wide Seaborne Premium (Export Bullish)" if brent_spread > 5.0 else "Normal Benchmark Parity"
            },
            "macro_radar": {
                "ovx_price": round(ovx_close, 2),
                "ovx_regime": ovx_regime,
                "ovx_color": ovx_color,
                "dxy_price": round(dxy_close, 2),
                "dxy_change_pct": round(dxy_change, 2),
                "crack_margin": round(crack_margin, 2),
                "gold_oil_ratio": gold_oil_ratio,
                "sp500_price": round(es_close, 1),
                "sp500_change_pct": round(es_change, 2),
                "xle_price": round(xle_close, 2),
                "eia_countdown": self.get_eia_countdown()
            },
            "mtf_confluence": mtf_confluence,
            "trade_setup": trade_setup,
            "liquidity_sweeps": sweeps,
            "sentiment": sentiment,
            "levels": {
                "r2": round(r2, 2),
                "r1": round(r1, 2),
                "vwap_u2": round(vwap_u2, 2),
                "vah": round(vah, 2),
                "vwap_u1": round(vwap_u1, 2),
                "poc": round(poc, 2),
                "vwap": round(vwap, 2),
                "vwap_l1": round(vwap_l1, 2),
                "val": round(val, 2),
                "vwap_l2": round(vwap_l2, 2),
                "s1": round(s1, 2),
                "s2": round(s2, 2)
            },
            "chart_candles": chart_candles,
            "history": history_records
        }

        return data

# Singleton instance
engine = AdvancedOilEngine()
