# Forward-test pre-registration

Frozen 2026-11-18 08:05:26 UTC. **Test window: 2026-11-18 09:00 → 2027-02-17 09:00 UTC (91 days).**
These files must not change during the test. The SHA-256 fingerprints below let anyone check that.

| Artifact | SHA-256 |
|---|---|
| `models/direction_gb_h4.joblib` | `e7276438e19914f14ef8379072373c362c3ba7a105c70d535fc93b4e237ba058` |
| `models/meta_label_gb.joblib` | `8681d8a2b7e1caa66be15dba50eaeb03c0563840aab4d4b81e1f1657612110f7` |
| `reports/engine_calibration.json` | `6e7293d1680b34cfe61efda52e043d0da677ed9f2a2d09f4db06e4b257699f8e` |

Models were trained on 2024-07-09 → 2026-11-18 (13,432 hourly rows).
Settings come from the lockbox experiment (`reports/IMPROVEMENT_EXPERIMENTS.md`) and are not tuned again.

## What is tested

| Track | Rule | Success criterion |
|---|---|---|
| A. Engine baseline | Live sentiment call (≥ +15 bull, ≤ −15 bear) and the engine's own playbook | Reference only |
| B. Calibrated probability | Engine `calibrated_p_up_4h` | PASS if Brier skill vs frozen climatology >= -0.005 (the probability is at least as good as knowing the base rate) |
| C. Confident ML direction | Gradient boosting P(up in 4h); call only if \|p − 0.5\| is in the top 30 % of the model's previous 500 hourly confidences (secondary: fixed threshold 0.1402) | Primary = rolling rule. CONFIRMED if hit rate > 50% AND Pesaran-Timmermann p < 0.05 on non-overlapping calls; REJECTED if hit rate <= 50%; otherwise INCONCLUSIVE. C_fixed is reported with the same criteria as a secondary result. |
| D. Improved playbook | Engine setups with stop 1.5×ATR and target 1.5R, taken only if meta-model p(win) ≥ 0.5 | BETTER if profit factor > engine-playbook profit factor AND mean $/trade > engine mean $/trade; PROFITABLE only if additionally mean $/trade > 0 with one-sided t-test p < 0.05 |

Frozen climatology for Brier skill: P(up in 4h) = 0.5129.

## Amendment made before the test started

The lockbox experiment used a fixed confidence threshold (0.1402). The final model trained on all data is less confident (its 70th-percentile confidence is ~0.065 live vs 0.140 in DEV), so the fixed threshold would call ~5% of hours. The primary rule was therefore changed, before any forward data was logged, to the top 30% relative to the model's own previous 500 hourly confidences. Walk-forward backtest of this rule: DEV 55.2% hit (PT p 0.008, coverage 23.6%), LOCKBOX 54.2% (PT p 0.34, coverage 29.4%). Because its lockbox result was seen while designing the fix, the forward test is its first clean evaluation.

## How evidence is recorded
- A GitHub Actions job runs hourly, records every completed hourly bar **before its outcome exists**, and commits the log to the `forward-test-log` branch. The commit timestamps are the proof.
- Bars recorded more than 75 minutes after they closed (scheduler delays, outages) are marked `backfilled`. They are still computed point-in-time, and the report shows results with and without them.

## Statistical power (read before judging the result)
About 1,495 hourly bars are expected. With 24–30 % coverage, track C will make roughly
358–448 calls, or 89–112 non-overlapping 4-hour calls.
Detecting a true 54.5 % hit rate with 80 % power needs about **764 non-overlapping calls**.
So 91 days can **reject** the candidate (if it lands near 50 %) or show it is **consistent** with the backtest,
but it is unlikely to **prove** it. Plan to extend the test if the result is promising.
