# PetroSync Pro — Technical Documentation

> Crude-oil (WTI `CL=F` / Brent `BZ=F`) direction & regime dashboard built with
> Python · Flask · pandas · NumPy · yfinance · Chart.js, with an added
> walk-forward evaluation suite and machine-learning baselines (scikit-learn).

---

## 1. What the project does (one paragraph)

Every hour the server downloads market data from Yahoo Finance for crude oil and
eight related markets, computes ~20 technical / intermarket indicators, combines
them into (a) a **7-pillar sentiment score** from −100 (bearish) to +100
(bullish), (b) a **multi-timeframe (MTF) confluence score** 0–100, and (c) a
concrete **trade setup** (entry, stop, targets). A single-page dashboard shows
everything. The new `evaluation/` package replays this exact logic on ~2.4
years of history and measures how often it was right.

## 2. Honest classification: is this "Machine Learning"?

| Component | Type | Learned from data? |
|---|---|---|
| `engine.py` indicators (EMA, RSI, ATR, VWAP, Volume Profile, pivots) | Classical **technical analysis** / descriptive statistics | No — fixed formulas |
| 7-pillar sentiment | **Rule-based expert system / linear scoring model** with hand-chosen weights (25/20/15/15/10/10/5 %) and hand-chosen neutral points ($26 crack, $5 Brent-WTI) | No |
| MTF confluence | Rule-based voting (EMA alignment + RSI thresholds), weighted 25/45/30 % | No |
| Trade setup + "win probability" | Decision tree written by hand; probabilities are **hard-coded constants** (64–74 %) | No |
| `evaluation/ml_models.py` (new) | **Supervised ML**: Logistic Regression, Ridge Regression, Histogram Gradient Boosting (classifier & regressor) | **Yes** — walk-forward training |

So the original dashboard is a **quantitative, rule-based model**, not ML.
That is not a weakness to hide — it is the correct baseline. The evaluation
suite now gives the project a genuine ML component and a fair, out-of-sample
comparison: *does learning the weights beat hand-tuning them?*

Good wording for a report: *"A rule-based quantitative signal engine, evaluated
with a walk-forward backtest and benchmarked against supervised machine-learning
models (logistic regression and gradient-boosted trees)."*

## 3. Architecture

```
            ┌────────────────────────────── Yahoo Finance (yfinance) ──────────────────────────────┐
            │ 1h: CL BZ DXY RB HO GC ^OVX ES XLE (5 days)   15m: CL (3 days)   1d: CL (30 days)        │
            └──────────────────────────────────────────────┬────────────────────────────────────────┘
                                                           │ fetch_market_basket()
                                                           ▼
 engine.py  AdvancedOilEngine.refresh()  ──►  compute_snapshot(macro_1h, cl_15m, cl_daily)   (pure function)
                │ 45 s cache                      ├─ calculate_technical_series  (EMA, RSI, ATR, VWAP ±σ)
                │                                 ├─ calculate_timeframe_state ×3 (15m / 1h / daily)
                │                                 ├─ calculate_volume_profile    (POC, VAH, VAL)
                │                                 ├─ detect_liquidity_sweeps
                │                                 ├─ generate_trade_setup
                │                                 ├─ compute_advanced_sentiment  (7 pillars)
                │                                 └─ pivots, chart series, 24h history
                ▼
 app.py (Flask)  /  /api/data  /api/refresh  /api/health  /api/metrics   + hourly background thread
                ▼
 templates/index.html + static/dashboard.js (Chart.js)  ── polls /api/data, renders the terminal UI

 evaluation/ (offline)
   data_loader.py   → 730 days of 1h, 60 days of 15m, 5 y daily, cached to evaluation/data_cache/
   backtest.py      → replays compute_snapshot() bar-by-bar with NO look-ahead, simulates trades
   metrics.py       → accuracy, F1, MCC, Brier, AUC, IC, Pesaran–Timmermann, Theil U, Sharpe …
   ml_models.py     → purged walk-forward Logistic / Ridge / Gradient Boosting
   run_evaluation.py→ writes reports/*.json, *.md, *.csv
```

`compute_snapshot` was split out of `refresh()` during this documentation
pass so that the **identical** code path runs live and in the backtest (a
replay of the latest bar reproduces the live sentiment score to ±0.1).

## 4. Data

| Ticker | Meaning | Used for |
|---|---|---|
| `CL=F` | NYMEX WTI front-month future | Price, all technicals, trade setups |
| `BZ=F` | ICE Brent future | Brent–WTI spread pillar |
| `DX-Y.NYB` | US Dollar Index | USD pillar (oil is priced in USD → inverse relation) |
| `RB=F`, `HO=F` | RBOB gasoline, ULSD heating oil ($/gal) | 3:2:1 crack spread pillar |
| `GC=F` | Gold | Gold/Oil ratio (displayed only — **not** used in the score) |
| `^OVX` | CBOE Crude Oil Volatility Index | Volatility regime & pillar |
| `ES=F` | E-mini S&P 500 | Risk-appetite pillar |
| `XLE` | Energy-sector ETF | Displayed only |

Yahoo limits: 1h bars ≈ 730 days back, 15m bars ≈ 60 days. Continuous
front-month futures have **roll gaps** (price jumps on contract expiry) — a
known limitation for any indicator that spans a roll date.

## 5. Every indicator, with its formula

Notation: C = close, H = high, L = low, V = volume, t = current bar.

| Indicator | Formula in code | Notes / theory |
|---|---|---|
| **EMA(n)** | `C.ewm(span=n, adjust=False)`; α = 2/(n+1); EMAₜ = α·Cₜ + (1−α)·EMAₜ₋₁ | 9, 20, 50 on 1h; 9/21 in MTF |
| **RSI(14)** | RS = SMA₁₄(gains) / SMA₁₄(losses); RSI = 100 − 100/(1+RS) | ⚠ Uses **simple** moving average ("Cutler's RSI"), not Wilder's smoothing (α = 1/14). Values differ from TradingView. |
| **ATR(14)** | TR = max(H−L, \|H−Cₜ₋₁\|, \|L−Cₜ₋₁\|); ATR = SMA₁₄(TR) | Wilder (1978) used smoothed, not simple, average |
| **VWAP** | Σ(TP·V)/ΣV over a **rolling 24 bars**, TP = (H+L+C)/3 | Not a true session VWAP (no reset at session open) |
| **VWAP σ bands** | σ = √(Σ V·(TP−VWAP)² / ΣV); bands = VWAP ± 1σ, ± 2σ | Volume-weighted dispersion |
| **Volume profile** | Each bar's volume split evenly over 8 price points between L and H, binned at $0.10. POC = max-volume bin; Value Area expands from POC until 70 % of volume is covered | Market Profile concept (Steidlmayer) |
| **Liquidity sweep** | Last bar's H ≥ 24-bar high but C < that high (bearish), or mirror for lows | ⚠ The 24-bar window *includes* the last bar, so H ≥ max is satisfied whenever the last bar makes the high |
| **Pivot points** | P = (H₂₄+L₂₄+C)/3; R1 = 2P−L; S1 = 2P−H; R2 = P+(H−L); S2 = P−(H−L) | Classic floor-trader pivots on the last 24 h |
| **3:2:1 crack** | (2·RB·42 + 1·HO·42 − 3·CL) / 3  [$/bbl] | 42 gal per barrel; refiner margin proxy |
| **Brent–WTI** | BZ − CL | Seaborne vs landlocked crude premium |
| **Gold/Oil** | GC / CL | Display only |

### 5.1 Multi-timeframe state (per timeframe)
```
score = 50
+30 if C > EMA9 > EMA21      −30 if C < EMA9 < EMA21
+20 if RSI ≥ 60              −20 if RSI ≤ 40
bias = BULLISH (≥70) | BEARISH (≤30) | NEUTRAL
MTF = 0.25·score15m + 0.45·score1h + 0.30·scoreDaily
```

### 5.2 7-Pillar sentiment (linear scoring model)
| # | Pillar | Raw component (clipped to ±100) | Weight |
|---|---|---|---|
| 1 | MTF alignment | (MTF − 50) × 2 | 25 % |
| 2 | VWAP position | % distance from VWAP × 40 | 20 % |
| 3 | Crack margin | (crack − 26) × 6 | 15 % |
| 4 | Brent–WTI | (spread − 5) × 15 | 15 % |
| 5 | USD | −(ΔDXY % 1h) × 70 | 10 % |
| 6 | Equities | (ΔES % 1h) × 50 | 10 % |
| 7 | OVX | −25 if OVX > 50, +15 if < 32, else 0 | 5 % |

Composite = Σ wᵢ·componentᵢ ∈ [−100, 100]; `bullish_pct` = (composite + 100)/2.
Labels: ≥ 50 strong bull, ≥ 15 moderate bull, ≤ −15 moderate bear, ≤ −50 strong bear.

This is mathematically a **linear model** `y = wᵀx` — the same functional form
as logistic regression, except the weights were chosen by intuition instead of
being fitted. That is exactly why logistic regression is the natural ML
benchmark.

### 5.3 Trade-setup decision tree
```
if RSI15m ≤ 25 and price < VAL   → Exhaustion bounce LONG     (claimed 68 %)
elif RSI15m ≥ 75 and price > VAH → Blow-off fade SHORT        (claimed 67 %)
elif MTF ≤ 35                    → Bearish continuation SHORT (claimed 74 %)
elif MTF ≥ 65                    → Bullish continuation LONG  (claimed 73 %)
else                             → Value-area rotation        (claimed 64 %)
```
Entry/stop/targets are expressed as multiples of ATR around VWAP / value-area edges.

## 6. API

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard |
| `GET /api/data[?force=true]` | Latest snapshot (cached 45 s) |
| `GET/POST /api/refresh` | Force re-download |
| `GET /api/health` | Liveness |
| `GET /api/metrics` | Latest backtest summary from `reports/evaluation_results.json` (new) |

## 7. Code-review findings (things a teacher may ask about)

| # | Finding | Impact | Status |
|---|---|---|---|
| 1 | Win probabilities are constants, not estimates | Misleading UI | Measured in backtest; see `reports/EVALUATION_RESULTS.md` |
| 2 | Timestamps were labelled "UTC" but were exchange time (US/Eastern) | Wrong clock by 4–5 h | **Fixed** (converted to UTC) |
| 3 | RSI/ATR use SMA, not Wilder smoothing | Values differ from standard platforms | Documented |
| 4 | "Session VWAP" is a rolling 24-bar VWAP | No session reset | Documented |
| 5 | `rsi`, `ema20`, `ema50`, `gold_oil_ratio` are passed to `compute_advanced_sentiment` but unused | Dead parameters | Documented |
| 6 | EIA countdown hard-codes 14:30 UTC | Off by 1 h during US standard time (should be 15:30 UTC), ignores holiday delays | Documented |
| 7 | Neutral range setup has no `tp1_ticks`; defaults to 80 | Cosmetic | Documented |
| 8 | Neutral-point constants ($26 crack, $5 spread) are fixed while fundamentals drift | Pillars can be saturated at ±100 for months | Visible in evaluation; candidate for rolling z-scores |
| 9 | Brent and WTI front months expire on different dates | Spread jumps at rolls | Documented |
| 10 | `start.sh` hard-codes `/home/bhuvik/anaconda3/bin/python` | Not portable | Documented |

## 8. How to run

```bash
pip install -r requirements.txt
python app.py                                  # dashboard on http://localhost:8080
pip install -r requirements-eval.txt
python -m evaluation.run_evaluation --refresh  # full backtest (~4 min) → reports/
```

See `docs/EVALUATION_METHODOLOGY.md` for how accuracy is measured and
`docs/REFERENCES.md` for the literature.
