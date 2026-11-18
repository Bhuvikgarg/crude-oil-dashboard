# How PetroSync's Accuracy Is Measured

> **Run it:** `pip install -r requirements-eval.txt && python -m evaluation.run_evaluation --refresh`
> **Results:** `reports/EVALUATION_RESULTS.md` (tables) · `reports/evaluation_results.json` (all numbers) ·
> `reports/predictions_vs_actual.csv` (every hourly call next to what actually happened) · `reports/trades.csv`
> **Tests:** `python -m pytest tests` (metrics cross-checked against scikit-learn)

---

## 1. The core problem: the engine does not predict a single number

PetroSync outputs **several kinds of prediction**, and each needs a different kind of "accuracy":

| Engine output | What it implicitly claims | Ground truth we compare to | Metric family |
|---|---|---|---|
| Sentiment label (bull / neutral / bear) | Price will go up / sideways / down | Sign & size of the realised return 1h, 4h, 24h later | **Classification**: hit rate, confusion matrix, balanced accuracy, MCC, Pesaran–Timmermann |
| `bullish_pct` (0–100 %) | Probability that price rises | Did it rise? (0/1) | **Probabilistic**: Brier score, Brier skill, log-loss, ROC-AUC, calibration (ECE) |
| Sentiment score (−100…+100) | Stronger score → bigger move | Realised return | **Ranking**: Spearman Information Coefficient |
| Trade setup (entry, stop, TP1, "win %") | TP1 is hit before the stop X % of the time | Simulated path on real high/low bars | **Trading**: realised win rate vs claimed, expectancy, R-multiple, profit factor, drawdown |
| VWAP ±σ bands, Value Area | Price stays inside ~68 % / 95 % / 70 % of the time | Next close | **Coverage**: observed vs nominal |
| Liquidity sweep flag | A reversal follows | 4h return sign | Event study vs base rate |
| ML price forecast (new) | Price in h hours = X | Actual price in h hours | **Regression**: MAE, RMSE, MAPE, **Theil U2, MASE vs random walk** |

"Mapping the prediction to the actual value" is done literally in
`reports/predictions_vs_actual.csv`: one row per hour with the call, the price
4 h / 24 h later, and a YES/NO for whether the call was right.

## 2. Walk-forward, point-in-time replay (no look-ahead)

```
time ─────────────────────────────────────────────────────────────────────►
      [ data the engine is allowed to see ]│ decision at bar close t │[ future: used ONLY for scoring ]
                                            ▲
                         engine.compute_snapshot(slice up to t)  →  signal(t)
                         compare signal(t) with price(t+h)
```

* `evaluation/backtest.py` calls the **unmodified live function** `compute_snapshot` once per hourly bar (13,436 times).
* Yahoo stamps bars with their *start* time; the decision is made at the bar's *close* (start + 1 h).
* 15-minute data only exists for the last ~60 days. Before that the 1h series is a **documented proxy**; results are reported separately for both windows (Section 9 of the results).
* The daily timeframe uses only *completed* sessions plus a synthetic "today" bar equal to the last price — exactly what Yahoo's live, still-forming daily candle looks like.
* **Validation of the replay:** the replayed sentiment on the latest bar was −24.9 vs −24.8 live.

**Why this matters:** using `df['Close'].shift(-1)` by mistake, or computing daily indicators with the *full* day's close, silently leaks the future and produces fake 60–70 % accuracies. Bailey et al. (2014) and López de Prado (2018) document how common this is.

## 3. Choices that keep the numbers honest

| Choice | Why |
|---|---|
| **Baselines** (always-long, persistence, random coin) | Oil drifted up in 51–52 % of windows. A model that "always says up" scores 51 % with zero skill. You must beat the baseline, not 50 %. |
| **Sanity checks** (an oracle that cheats → must score 100 %; a coin → must score ~50 %) | Proves the scoring code itself is correct. |
| **Non-overlapping samples for p-values** | Overlapping 24h returns sampled every hour share 23 hours of path → strongly autocorrelated → naive t-stats are inflated ~√24×. We test on every h-th bar only (alternative: Newey–West). |
| **Pesaran–Timmermann test** | Tests direction-forecast skill *given* the base rates; catches models that only learned the drift. |
| **3-class truth with a volatility band** | A +0.01 % move is noise, not "up". FLAT = \|return\| < 0.25 · ATR · √h. |
| **Balanced accuracy & MCC** | Robust to class imbalance (Chicco & Jurman 2020). |
| **Brier Skill Score vs climatology** | A probability forecast must beat "always say the historical up-frequency". Negative skill = worse than knowing nothing. |
| **Conservative trade simulation** | If stop and target are touched in the same 1h bar → assume stop first; on the fill bar only stops count; gap fills use the worse open price; 2-tick round-turn cost ($20/contract); one position at a time. |
| **Purged walk-forward for ML** | Model refit every 250 bars on past data only, with a gap of h bars so no training label overlaps the test period. Random K-fold would leak the future. |
| **Same window comparison** | Rule-based and ML models are compared on the *identical* out-of-sample period. |
| **Theil U2 / MASE for price forecasts** | MAPE of ~0.4 % on a $70 asset *looks* excellent but is the same as predicting "no change". U2 < 1 is the real bar. |

## 4. Headline results (Jul 2024 → Nov 2026, 13,436 hourly decisions)

| Question | Answer | Evidence |
|---|---|---|
| Does the 7-pillar sentiment predict direction? | **No measurable skill.** Hit rate 50.0 % (1h), 49.6 % (4h), 51.6 % (24h); all p > 0.3; MCC ≈ 0 | Results §1 |
| Is "bullish %" a valid probability? | **No.** Brier skill −0.08 (worse than climatology); observed up-frequency is ~50 % in every bin from 27 % to 75 % predicted | Results §2, reliability table in JSON |
| Are the playbook win probabilities true? | **No.** Claimed 64–74 %, realised TP1 hit rate 20–56 %, overall 45.8 % vs 71.2 % claimed (p < 0.0001). Net −$46 per trade, profit factor 0.84 | Results §3 |
| Do VWAP bands behave like ±1σ/±2σ? | Roughly, but too narrow: 60 % / 89 % vs 68 % / 95 % nominal (fat tails) | Results §5 |
| Does the Value Area predict where price will be? | Only 50 % of next-hour closes and 24 % of 24h-later closes fall in it — it describes the *past* 24 h, not the future | Results §5 |
| Liquidity-sweep flag? | Reversal rate ≈ base rate (p > 0.2) — no edge | Results §6 |
| Does ML beat the hand-tuned weights? | **Marginally, not convincingly.** Gradient boosting AUC 0.52 vs rule-based 0.49; logistic regression's apparent 1h significance disappears under Pesaran–Timmermann (p = 0.08) because it mostly predicts "up" | Results §8 |
| Can ML forecast the price level? | **No better than "no change"**: Theil U2 = 1.004–1.32 | Results §8 |
| Any regime where it works? | OVX < 30: 57.5 % non-overlap hit rate, p = 0.04 — but with 3 regimes tested, Bonferroni-adjusted p ≈ 0.12, and only 965 bars. A hypothesis to test forward, not a finding | Results §7 |

**The interesting contradiction:** the sentiment long/short equity curve returned +17 % vs +2 % buy-and-hold despite a 50 % hit rate. That can happen when a few large moves are caught. But an annualised Sharpe of 0.37 over ~2.4 years has t ≈ 0.37·√2.4 ≈ 0.57: **not statistically different from luck**. Report it, but don't sell it.

## 5. How to explain this result positively (it *is* a good project outcome)

1. It matches the literature: Alquist, Kilian & Vigfusson (2013) find the no-change forecast is very hard to beat for oil; Marshall et al. (2008) find that technical-rule profits in commodity futures vanish after data-snooping control.
2. The project's contribution is the **evaluation framework**: point-in-time replay, baselines, sanity checks, proper significance tests and purged ML validation. Most student and retail "trading AI" projects report in-sample accuracy with look-ahead bias.
3. The framework found specific, fixable problems (hard-coded probabilities, a saturating crack-spread pillar, a negative-expectancy range setup), which is exactly what evaluation is for.

## 6. Improvement roadmap (each can be re-measured with the same harness)

| Idea | Expected effect | How to test |
|---|---|---|
| Replace hard-coded win % with the **empirical** rates from `trades.csv` (or a calibrated logistic model) | Honest UI | Brier/ECE of the new probabilities |
| Replace fixed neutral points ($26 crack, $5 spread) with **rolling z-scores** | Pillars stop saturating at ±100 | IC of each pillar before/after |
| Disable or redesign the Value-Area rotation setup (19.7 % TP1 rate) | Removes the worst setup | Profit factor |
| Fit pillar weights with **regularised logistic regression** inside the walk-forward loop | Data-driven weights | Same-window AUC vs current |
| Use **triple-barrier labels** (López de Prado 2018) as the ML target instead of a fixed horizon | Labels match how trades are actually closed | Trade expectancy |
| Add **Diebold–Mariano** and White's Reality Check | Formal model comparison and data-snooping control | p-values |
| Forward test: log live signals daily for 3 months | Truly out-of-sample | Same metrics |

## 7. Improvement experiment (lockbox protocol)

Run with `python -m evaluation.improve` → `reports/IMPROVEMENT_EXPERIMENTS.md`.

**Protocol.** All choices (feature sets, model, confidence threshold, stop/target geometry,
meta-label threshold) were made on the DEV period (Jul 2024 → Mar 2026). Each finalist was then
scored once on the LOCKBOX (Apr → Nov 2026). Predictions are walk-forward in both periods. All 38
variants are listed in the report so the multiple-testing risk is visible.

| Idea | DEV | LOCKBOX | Verdict |
|---|---|---|---|
| Learned weights (logistic / gradient boosting on engine features) | AUC 0.50–0.52 | AUC 0.51–0.52 | No robust gain |
| Rolling z-scored pillars + time-series momentum | AUC 0.49–0.50 | AUC 0.52–0.53 | Fails DEV, so treated as noise |
| Best DEV model, only its 30 % most confident calls (4h) | 54.4 % hit, PT p 0.02 | 54.7 % hit, PT p 0.004, coverage 10 % | **Promising, unproven.** Needs a forward test |
| Same idea at 24h | 52.6 %, PT p 0.66 | 49.6 % | Fails |
| Platt calibration of "bullish %" | Brier skill −0.074 → −0.001, ECE 0.115 → 0.007 | −0.097 → −0.001, ECE 0.120 → 0.010 | **Reliable fix. Deployed.** |
| Drop the value-area rotation setup | PF 0.735 → 0.761 | PF 1.017 → 1.132 | Consistent improvement |
| Wider stop (1.5 ATR, 1.5 R target), chosen on DEV | PF 0.735 → 0.852 | PF 1.017 → 1.083 | Consistent improvement |
| Meta-labeling (ML filter on engine setups), label AUC 0.59 / 0.60 | PF 0.735 → 0.809 | PF 1.017 → 1.078 | Consistent improvement |
| Geometry + meta-labeling combined | −$65 → −$6 per trade, PF 0.98 | +$7 → +$98 per trade, PF 1.13 (p = 0.21) | Best candidate, **not statistically profitable** |

**What this means.**
1. *Win rate is not the goal.* The geometry grid shows win rates from 23 % to 66 % with profit
   factor moving the other way. A close target raises the win rate and loses money.
2. *The lockbox months suited the strategy.* The unchanged engine went from PF 0.74 (DEV) to 1.02
   (LOCKBOX). Every variant is therefore judged against the baseline in the same period.
3. *Calibration is the only change that is both reliable and honest*, so it is the only one deployed.
   The dashboard now shows `CALIBRATED P(UP, NEXT 4H)`, which stays between about 50 % and 52.5 %.
   That narrow range is the honest answer.
4. The confidence-filtered model and the combined playbook are **candidates for a forward test**:
   log their live calls for 3 months, then score them with the same harness.

## 8. Limitations

* Yahoo continuous futures include **roll gaps**; no back-adjustment is applied.
* 1h bars hide the intra-bar order of high and low, hence the conservative same-bar rule.
* Volume on continuous contracts is per-contract and drops near expiry.
* ~2.4 years is one macro regime (post-2022 normalisation); results may differ in 2008- or 2020-style markets.
* The trade simulator ignores queue position for limit orders (fills on touch), which flatters entries slightly.
