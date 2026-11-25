# PetroSync Pro // Institutional Crude Oil Direction & Regime Terminal

An advanced institutional quantitative analytics terminal for Crude Oil (`CL` WTI & `BZ` Brent) traders on Trading Technologies (TT).

## Advanced Features & Quant Models

### 1. Multi-Timeframe Confluence Matrix (MTF Heatmap)
- Simultaneously computes EMA alignment (9, 20, 50), 14-period RSI, and structural momentum across **15-Minute (Scalp)**, **1-Hour (Session Trend)**, and **Daily (Macro Structure)** timeframes.
- Outputs an overall weighted confluence score (0–100%) and alerts to when all three timeframes confirm directional continuation.

### 2. CBOE Crude Oil Volatility Index (`^OVX`) Regime Engine
- Real-time tracking of the Crude Oil "VIX" (`^OVX`).
- Classifies volatility into:
  - **Low Volatility (< 30)**: Coiling / Compression — breakout watch.
  - **Normal Volatility (30 - 45)**: Standard two-way order flow and predictable technical reactions.
  - **Elevated Volatility (> 45)**: Extreme liquidation or panic expansion — triggers automated wider stops and stop-sweep alerts.

### 3. Actionable Quantitative Trade Setup Generator (TT Playbook)
- Directly outputs:
  - **Setup Archetype** (e.g. *Bullish Trend Continuation*, *Bearish Pullback Short*, *Exhaustion Bounce Long*, *Range Rotation Fade*)
  - **Exact Entry Trigger Zone** (e.g. `$89.53 - $89.75`)
  - **Stop Loss** in price and exact ticks (e.g. `$89.08` / -45 ticks)
  - **Target 1 & Target 2** with tick count
  - **Risk/Reward Ratio** (e.g. `1 : 2.2`)
  - **Estimated Win Probability %** based on confluence
  - **TT MD Trader Execution Instruction** for one-click bracket orders.

### 4. 7-Pillar Institutional Intermarket Radar
- **3:2:1 Crack Spread** (`RB=F` Gasoline + `HO=F` Heating Oil vs `CL=F`): US refiner physical buying appetite.
- **Brent-WTI Spread** (`BZ - CL`): Global waterborne export premium.
- **US Dollar Index** (`DXY`): Currency headwind/tailwind.
- **Gold/Oil Ratio** (`GC / CL`): Inflation vs industrial demand macro ratio.
- **S&P 500 Risk Appetite** (`ES=F`): Global cyclical equity demand.
- **Energy Equities** (`XLE`): Smart-money institutional sector positioning.
- **EIA Inventory Countdown**: Exact timer to Wednesday 10:30 AM EST inventory releases.

### 5. Order Flow & Statistical Envelopes
- Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).
- Session Volume-Weighted Average Price (VWAP).
- VWAP Standard Deviation Bands (±1σ and ±2σ statistical volatility envelopes).
- Session Liquidity Sweep & Trap Detector (Overnight High / Low stop run detection).

### 6. Backtest Reality Check (new)
- Shows the live setup's **claimed** win probability next to its **realised** Target-1 rate from the walk-forward backtest, with a verdict (Claim holds / Overstated / Negative edge).
- Shows overall evidence: sentiment hit rate, Brier skill of "bullish %", playbook profit factor and $/trade, best ML AUC, and ML price-forecast Theil U.
- Reads `reports/evaluation_results.json` through `/api/metrics`. If no evaluation has been run, the card says how to generate one.

### 7. Measured win rates (replaces the typed-in claims)
- The playbook's win rates (64–74 %) were numbers typed into `engine.py`. The dashboard now shows each setup's **measured** Target-1 rate from the walk-forward replay, with a 95 % range, the development vs lockbox rate, and the average win, loss and $/trade.
- Produced by `python -m evaluation.measure_playbook` → `reports/playbook_measured.json`. The old claim stays visible (crossed out) and is still what the backtest tests.
- Reward : risk is now measured from the same entry price the backtest fills at. It used to take the reward and the risk from opposite ends of the entry zone, which made it look about twice as good as it was.

## Documentation & Accuracy Evaluation

| Document | Contents |
|---|---|
| [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md) | Architecture, every formula, rule-based vs ML classification, code-review findings |
| [`docs/EVALUATION_METHODOLOGY.md`](docs/EVALUATION_METHODOLOGY.md) | How accuracy is measured (walk-forward replay, baselines, statistical tests) + headline results |
| [`reports/EVALUATION_RESULTS.md`](reports/EVALUATION_RESULTS.md) | Auto-generated backtest tables (13,436 hourly decisions, Jul 2024 → Nov 2026) |
| [`reports/predictions_vs_actual.csv`](reports/predictions_vs_actual.csv) | Every hourly call mapped to the realised price |
| [`reports/IMPROVEMENT_EXPERIMENTS.md`](reports/IMPROVEMENT_EXPERIMENTS.md) | 38 improvement variants, tuned on Jul 2024–Mar 2026, tested once on Apr–Nov 2026 |
| [`reports/VOLATILITY.md`](reports/VOLATILITY.md) | Volatility forecasting (HAR model): the size of the next move is predictable even though its direction is not |
| [`reports/ONLINE_LEARNING.md`](reports/ONLINE_LEARNING.md) | Self-learning models (Hedge pillar weights, online logistic regression), evaluated test-then-train |
| [`docs/study_guide.html`](docs/study_guide.html) | Presentation and viva study guide (open in a browser) |
| [`docs/FORWARD_TEST.md`](docs/FORWARD_TEST.md) | Live, pre-registered 91-day forward test logged hourly by GitHub Actions to the `forward-test-log` branch |
| [`docs/REFERENCES.md`](docs/REFERENCES.md) | 41 research papers / books behind the indicators, ML models and metrics |
| [`docs/learning_report.html`](docs/learning_report.html) | Interactive learning report (open in a browser) |

**Headline:** replayed point-in-time on 2.4 years of data, the directional signals are statistically
indistinguishable from a coin flip (hit rate 49.6–51.6 %, all p > 0.3), and the playbook's claimed
64–74 % win probabilities realise at 20–56 %. Gradient-boosting and logistic-regression baselines
trained walk-forward on the same features do not beat a random walk either. See the methodology for why
this is the expected — and still valuable — result.

```bash
pip install -r requirements-eval.txt
python -m evaluation.run_evaluation --refresh   # ~4 min, writes reports/
python -m evaluation.online_learning            # self-learning models (~1 min)
python -m evaluation.volatility                 # volatility forecasting (~1 min)
python -m evaluation.improve                    # lockbox experiment: can accuracy be improved? (~3 min)
python -m evaluation.measure_playbook           # measured win rate per setup (seconds)
python -m pytest tests                          # metric unit tests
```

## Running the Terminal
The server is currently running live at:
```
http://localhost:8080
```

To run or restart manually:
```bash
pip install -r requirements.txt
./start.sh            # or: python app.py
```
