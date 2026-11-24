# Volatility forecasting, and does volatility help direction?

Return autocorrelation (lag 1): -0.0082 · |return| autocorrelation: lag 1 0.2177, lag 4 0.1958, lag 24 0.1563. Direction is nearly random; the size of moves clusters.

Target: log realised volatility over the next h hourly bars. Walk-forward, DEV before 2026-04-01, LOCKBOX Apr–Nov 2026.

## Volatility, next 4 bars

| Model | DEV R² | LOCK R² | LOCK Spearman | LOCK stormy-vs-calm accuracy | LOCK stormy AUC |
|---|---|---|---|---|---|
| Naive (last 24 bars) | 0.058 | -0.302 | 0.203 | 51.0% | 0.522 |
| HAR ridge | 0.451 | 0.219 | 0.511 | 68.7% | 0.697 |
| Gradient boosting, all features | 0.406 | 0.215 | 0.499 | 67.8% | 0.681 |

## Volatility, next 24 bars

| Model | DEV R² | LOCK R² | LOCK Spearman | LOCK stormy-vs-calm accuracy | LOCK stormy AUC |
|---|---|---|---|---|---|
| Naive (last 24 bars) | 0.325 | -0.246 | 0.382 | 61.4% | 0.569 |
| HAR ridge | 0.513 | 0.273 | 0.515 | 66.5% | 0.578 |
| Gradient boosting, all features | 0.190 | 0.107 | 0.433 | 55.7% | 0.529 |

## Is the forecast range honest? (HAR, share of real moves inside the range)

| Horizon | DEV inside ±1σ | DEV inside ±2σ | LOCK inside ±1σ | LOCK inside ±2σ |
|---|---|---|---|---|
| 4 bars | 63.2% | 89.5% | 60.7% | 87.7% |
| 24 bars | 70.3% | 94.1% | 68.3% | 92.4% |

A normal distribution would put 68 % / 95 % inside. Oil has fat tails, so the real shares are lower; the dashboard states the measured share.

## Does volatility help predict direction? (4h)

| Model | DEV hit | DEV AUC | LOCK hit | LOCK AUC | LOCK PT p |
|---|---|---|---|---|---|
| gb_31_features | 51.5% | 0.520 | 51.0% | 0.515 | 0.094 |
| gb_31_plus_volatility | 51.0% | 0.513 | 51.4% | 0.506 | 0.251 |

| Baseline direction model in… | DEV hit | DEV AUC | LOCK hit | LOCK AUC | LOCK calls |
|---|---|---|---|---|---|
| predicted calm hours | 52.4% | 0.522 | 48.7% | 0.503 | 1945 |
| predicted stormy hours | 50.9% | 0.519 | 53.5% | 0.527 | 1730 |
