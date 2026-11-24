# Self-learning models — test-then-train results

Each hour the model predicts first, is scored on the 4-hour outcome, then learns from the outcome of the call made 4 bars earlier. Hedge η = 0.01 and SGD learning rate = 0.001 were chosen on DEV (before 2026-04-01); LOCKBOX is Apr–Nov 2026. First 2000 rows are burn-in.

| Model | DEV hit | DEV AUC | DEV PT p | LOCK hit | LOCK AUC | LOCK PT p | LOCK coverage |
|---|---|---|---|---|---|---|---|
| Frozen engine (hand-tuned weights) | 48.2% | 0.486 | 0.883 | 50.9% | 0.501 | 0.457 | 71.3% |
| Hedge self-learning weights | 48.3% | 0.488 | 0.560 | 50.9% | 0.502 | 0.458 | 63.9% |
| Online logistic regression (SGD) | 50.4% | 0.501 | 0.294 | 51.5% | 0.508 | 0.224 | 100.0% |
| Walk-forward gradient boosting (retrained every 250 h) | 51.5% | 0.520 | 0.033 | 51.0% | 0.514 | 0.094 | 100.0% |
| Always "up" baseline | 51.7% | 0.500 | — | 50.5% | 0.500 | — | 100.0% |

## Hedge: hand-tuned vs learned pillar weights (end of history)

| Pillar | Hand-tuned | Learned |
|---|---|---|
| mtf_alignment | 26.3% | 16.7% |
| order_flow_vwap | 21.1% | 16.8% |
| crack_margin | 15.8% | 13.6% |
| brent_spread | 15.8% | 17.1% |
| usd_impact | 10.5% | 17.9% |
| equity_risk_on | 10.5% | 17.9% |

## Online logistic regression: largest coefficients (standardised)

- `gold_oil`: +0.4101
- `tfD_score`: -0.2767
- `ema20_50_atr`: -0.2497
- `past_ret_24`: -0.1561
- `pillar_crack_margin`: -0.1459
- `sent_score`: -0.1105
- `crack`: -0.0971
- `mtf_score`: -0.0931
- `pillar_mtf_alignment`: -0.0931
- `atr_pct`: -0.0921

Variants tried: hedge eta=0.01, hedge eta=0.05, hedge eta=0.2, hedge eta=0.5, sgd eta0=0.001, sgd eta0=0.01
