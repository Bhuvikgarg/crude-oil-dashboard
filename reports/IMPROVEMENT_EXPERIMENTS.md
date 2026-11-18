# Can the accuracy be improved? — Lockbox experiment

_DEV 2024-07-09 → 2026-03-31 (9,757 bars) · LOCKBOX 2026-04-01 → 2026-11-18 (3,679 bars) · 38 variants tried. All choices made on DEV; LOCKBOX scored once. All model predictions are walk-forward (trained on the past only)._

> Read lockbox numbers *relative to the baseline row in the same period*. The lockbox months happened to suit the strategy, so even the unchanged engine looks better there.

## Direction, 4-hour horizon

| Variant | DEV hit | DEV % up | DEV PT p | DEV AUC | LOCK hit | LOCK % up | LOCK base | LOCK PT p | LOCK AUC | LOCK coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| A_engine_sentiment | 49.0% | 39.1% | 0.811 | 0.492 | 50.9% | 75.9% | 50.5% | 0.457 | 0.500 | 71.3% |
| B_engine_features__logistic_regression | 50.2% | 64.5% | 0.210 | 0.504 | 51.6% | 54.2% | 50.5% | 0.013 | 0.512 | 100.0% |
| B_engine_features__gradient_boosting | 51.5% | 58.1% | 0.033 | 0.520 | 51.0% | 70.8% | 50.5% | 0.094 | 0.515 | 100.0% |
| C_zscore_plus_momentum__logistic_regression | 49.4% | 54.9% | 0.385 | 0.498 | 51.9% | 49.9% | 50.5% | 0.016 | 0.527 | 100.0% |
| C_zscore_plus_momentum__gradient_boosting | 49.4% | 55.8% | 0.458 | 0.493 | 52.0% | 54.3% | 50.5% | 0.011 | 0.518 | 100.0% |
| D_all_features__logistic_regression | 49.8% | 60.8% | 0.250 | 0.495 | 50.9% | 50.9% | 50.5% | 0.069 | 0.508 | 100.0% |
| D_all_features__gradient_boosting | 49.9% | 57.7% | 0.677 | 0.505 | 51.1% | 70.3% | 50.5% | 0.053 | 0.505 | 100.0% |
| E_best_dev_model_top30pct_confident (= B_engine_features__gradient_boosting) | 54.4% | 66.6% | 0.021 | — | 54.7% | 79.4% | 50.5% | 0.004 | — | 10.0% |

## Direction, 24-hour horizon

| Variant | DEV hit | DEV % up | DEV PT p | DEV AUC | LOCK hit | LOCK % up | LOCK base | LOCK PT p | LOCK AUC | LOCK coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| A_engine_sentiment | 49.5% | 39.1% | 0.698 | 0.503 | 56.4% | 75.9% | 52.9% | 0.308 | 0.541 | 71.3% |
| B_engine_features__logistic_regression | 51.7% | 74.9% | 0.385 | 0.484 | 49.1% | 55.9% | 52.9% | 0.963 | 0.497 | 100.0% |
| B_engine_features__gradient_boosting | 50.8% | 52.8% | 0.214 | 0.513 | 53.0% | 64.9% | 52.9% | 0.241 | 0.529 | 100.0% |
| C_zscore_plus_momentum__logistic_regression | 49.5% | 51.0% | 0.374 | 0.490 | 52.3% | 57.1% | 52.9% | 0.853 | 0.521 | 100.0% |
| C_zscore_plus_momentum__gradient_boosting | 46.3% | 45.4% | 0.579 | 0.451 | 48.9% | 44.8% | 52.9% | 0.691 | 0.479 | 100.0% |
| D_all_features__logistic_regression | 47.3% | 62.0% | 0.885 | 0.456 | 51.0% | 50.9% | 52.9% | 0.959 | 0.502 | 100.0% |
| D_all_features__gradient_boosting | 49.5% | 50.2% | 0.185 | 0.490 | 51.4% | 59.9% | 52.9% | 0.461 | 0.505 | 100.0% |
| E_best_dev_model_top30pct_confident (= B_engine_features__gradient_boosting) | 52.6% | 62.8% | 0.661 | — | 49.6% | 58.8% | 52.9% | 0.706 | — | 10.4% |

## Probability calibration of "bullish %"

| Horizon | Method | DEV Brier skill | DEV ECE | LOCK Brier skill | LOCK ECE |
|---|---|---|---|---|---|
| 4h | engine_bullish_pct | -0.0744 | 0.115 | -0.0968 | 0.119 |
| 4h | platt_calibrated_sentiment | -0.0011 | 0.007 | -0.0010 | 0.010 |
| 4h | past_base_rate_only | -0.0012 | 0.006 | -0.0006 | 0.008 |
| 24h | engine_bullish_pct | -0.0722 | 0.105 | -0.0468 | 0.084 |
| 24h | platt_calibrated_sentiment | -0.0085 | 0.067 | -0.0004 | 0.008 |
| 24h | past_base_rate_only | -0.0070 | 0.026 | -0.0025 | 0.016 |

## Trade playbook variants

| Variant | DEV n | DEV win | DEV PF | DEV $/trade | LOCK n | LOCK win | LOCK PF | LOCK $/trade | LOCK p($>0) |
|---|---|---|---|---|---|---|---|---|---|
| T0_engine_as_is | 1551 | 46.0% | 0.735 | -65.23 | 563 | 45.6% | 1.017 | 6.59 | 0.4377 |
| T1_drop_range_rotation | 1605 | 51.1% | 0.761 | -60.5 | 580 | 50.7% | 1.132 | 53.53 | 0.1096 |
| T2_only_trend_continuation | 1416 | 54.7% | 0.744 | -66.38 | 495 | 51.9% | 1.044 | 18.12 | 0.3392 |
| T3_ovx_below_30_only | 188 | 43.6% | 0.607 | -65.8 | 0 | — | — | None | — |
| T4 stop0.7ATR_target0.5R | 2355 | 37.1% | 0.455 | -80.42 | 726 | 46.4% | 0.548 | -125.71 | 1.0000 |
| T4 stop0.7ATR_target1.0R | 2115 | 30.6% | 0.602 | -67.17 | 660 | 37.7% | 0.726 | -90.8 | 0.9998 |
| T4 stop0.7ATR_target1.5R | 1949 | 26.8% | 0.680 | -59.65 | 638 | 34.5% | 0.894 | -38.53 | 0.8842 |
| T4 stop0.7ATR_target2.0R | 1844 | 23.2% | 0.732 | -53.59 | 570 | 29.3% | 0.900 | -39.42 | 0.8506 |
| T4 stop1.0ATR_target0.5R | 1960 | 45.4% | 0.591 | -79.45 | 644 | 55.6% | 0.784 | -71.54 | 0.9953 |
| T4 stop1.0ATR_target1.0R | 1672 | 35.8% | 0.692 | -74.23 | 561 | 46.4% | 1.006 | 2.35 | 0.4773 |
| T4 stop1.0ATR_target1.5R | 1482 | 30.7% | 0.796 | -53.84 | 487 | 37.8% | 1.023 | 11.12 | 0.4158 |
| T4 stop1.0ATR_target2.0R | 1316 | 26.0% | 0.816 | -52.9 | 433 | 31.4% | 1.021 | 11.85 | 0.4281 |
| T4 stop1.5ATR_target0.5R | 1525 | 54.9% | 0.734 | -70.84 | 500 | 66.2% | 1.110 | 45.57 | 0.1534 |
| T4 stop1.5ATR_target1.0R | 1170 | 42.9% | 0.841 | -55.49 | 380 | 51.3% | 1.175 | 103.63 | 0.0783 |
| T4 stop1.5ATR_target1.5R **(chosen on DEV)** | 959 | 35.0% | 0.852 | -59.11 | 317 | 41.3% | 1.083 | 60.6 | 0.2669 |
| T4 stop1.5ATR_target2.0R | 836 | 32.1% | 0.832 | -70.91 | 286 | 35.7% | 1.043 | 34.16 | 0.3792 |
| T5 meta-labeling (threshold 0.5) | 739 | 56.7% | 0.809 | -46.43 | 322 | 54.0% | 1.078 | 31.06 | 0.2818 |
| T6_geometry_plus_meta | 499 | 39.7% | 0.984 | -5.8 | 212 | 44.8% | 1.133 | 98.28 | 0.2069 |

Meta-label model AUC (does ML predict whether a setup hits TP1 before the stop?): DEV 0.592, LOCK 0.605. Part of this is mechanical: targets that are close relative to ATR are hit more often.

## Every variant tried

`direction h4: B_engine_features__logistic_regression`, `direction h4: B_engine_features__gradient_boosting`, `direction h4: C_zscore_plus_momentum__logistic_regression`, `direction h4: C_zscore_plus_momentum__gradient_boosting`, `direction h4: D_all_features__logistic_regression`, `direction h4: D_all_features__gradient_boosting`, `direction h4: E_best(B_engine_features__gradient_boosting)_top30pct_confidence`, `direction h24: B_engine_features__logistic_regression`, `direction h24: B_engine_features__gradient_boosting`, `direction h24: C_zscore_plus_momentum__logistic_regression`, `direction h24: C_zscore_plus_momentum__gradient_boosting`, `direction h24: D_all_features__logistic_regression`, `direction h24: D_all_features__gradient_boosting`, `direction h24: E_best(B_engine_features__gradient_boosting)_top30pct_confidence`, `calibration h4: platt`, `calibration h4: past base rate`, `calibration h24: platt`, `calibration h24: past base rate`, `trades: T1 drop range`, `trades: T2 trend only`, `trades: T3 OVX<30`, `trades: geometry stop0.7ATR_target0.5R`, `trades: geometry stop0.7ATR_target1.0R`, `trades: geometry stop0.7ATR_target1.5R`, `trades: geometry stop0.7ATR_target2.0R`, `trades: geometry stop1.0ATR_target0.5R`, `trades: geometry stop1.0ATR_target1.0R`, `trades: geometry stop1.0ATR_target1.5R`, `trades: geometry stop1.0ATR_target2.0R`, `trades: geometry stop1.5ATR_target0.5R`, `trades: geometry stop1.5ATR_target1.0R`, `trades: geometry stop1.5ATR_target1.5R`, `trades: geometry stop1.5ATR_target2.0R`, `trades: meta-label thr 0.45`, `trades: meta-label thr 0.5`, `trades: meta-label thr 0.55`, `trades: meta-label thr 0.6`, `trades: T6 geometry + meta`
