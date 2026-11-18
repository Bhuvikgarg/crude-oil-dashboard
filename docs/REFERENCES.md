# References — literature behind every part of PetroSync

Grouped by the part of the project they justify. ✅ = bibliographic details
checked against an index (RePEc/IDEAS, EconPapers, PMC, publisher) while writing
this file. 📘 = standard textbook / foundational source.
**Read at least the abstract of anything you cite**, and be ready to say in
one sentence why it is relevant (provided below each entry).

---

## A. Technical indicators used in `engine.py`

1. 📘 **Wilder, J. W. (1978).** *New Concepts in Technical Trading Systems.* Trend Research.
   → Original definitions of **RSI** and **ATR** (the engine uses a simple-average variant).
2. 📘 **Murphy, J. J. (1999).** *Technical Analysis of the Financial Markets.* New York Institute of Finance.
   → Standard reference for moving-average crossovers, EMA alignment, pivot points, multi-timeframe analysis.
3. ✅ **Berkowitz, S. A., Logue, D. E., & Noser, E. A. (1988).** The total cost of transactions on the NYSE. *Journal of Finance*, 43(1), 97–112. doi:10.1111/j.1540-6261.1988.tb02591.x
   → Introduced the **VWAP** as an execution benchmark — the origin of the VWAP concept used for the order-flow pillar.
4. 📘 **Steidlmayer, J. P., & Koy, K. (1986).** *Markets and Market Logic.* Porcupine Press.
   → **Market Profile**: Point of Control, Value Area (70 %), the basis of `calculate_volume_profile`.
5. 📘 **Whaley, R. E. (2000).** The investor fear gauge. *Journal of Portfolio Management*, 26(3), 12–17.
   → Theory behind implied-volatility indices (VIX); **OVX** applies the same CBOE methodology to USO options.

## B. Does technical analysis work? (justifies *why we must test it*)

6. **Brock, W., Lakonishok, J., & LeBaron, B. (1992).** Simple technical trading rules and the stochastic properties of stock returns. *Journal of Finance*, 47(5), 1731–1764.
   → Classic evidence that moving-average rules had predictive power (on the DJIA, 1897–1986); introduced bootstrap testing of trading rules.
7. **Lo, A. W., Mamaysky, H., & Wang, J. (2000).** Foundations of technical analysis: Computational algorithms, statistical inference, and empirical implementation. *Journal of Finance*, 55(4), 1705–1765.
   → Shows how to turn chart patterns into testable algorithms — the philosophy of this project's backtest.
8. **Park, C.-H., & Irwin, S. H. (2007).** What do we know about the profitability of technical analysis? *Journal of Economic Surveys*, 21(4), 786–826.
   → Survey of 95 modern studies: profits often exist but shrink after costs and data-snooping control.
9. ✅ **Marshall, B. R., Cahan, R. H., & Cahan, J. M. (2008).** Can commodity futures be profitably traded with quantitative market timing strategies? *Journal of Banking & Finance*, 32(9), 1810–1819.
   → Tests ~7,800 technical rules on 15 commodity futures; after data-snooping adjustment, reliable profits disappear for 14 of 15. **Directly relevant to crude oil.**
10. ✅ **Szakmary, A. C., Shen, Q., & Sharma, S. C. (2010).** Trend-following trading strategies in commodity futures: A re-examination. *Journal of Banking & Finance*, 34(2), 409–426.
    → Moving-average trend rules earn positive net returns in 22 of 28 commodity markets (monthly data) — supports the MTF/EMA trend pillar, but at much lower frequency than 1h.
11. **Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012).** Time series momentum. *Journal of Financial Economics*, 104(2), 228–250.
    → Past 12-month return predicts future returns across 58 futures incl. crude — the academic basis for "trend continuation" setups.

## C. Crude-oil fundamentals behind the intermarket pillars

12. **Kilian, L. (2009).** Not all oil price shocks are alike: Disentangling demand and supply shocks in the crude oil market. *American Economic Review*, 99(3), 1053–1069.
    → Separates supply, global-demand and precautionary-demand shocks — why equities (risk appetite) and oil co-move.
13. **Kilian, L., & Park, C. (2009).** The impact of oil price shocks on the U.S. stock market. *International Economic Review*, 50(4), 1267–1287.
    → Justifies the **S&P 500 pillar**: the sign of the oil–equity correlation depends on the type of shock.
14. **Akram, Q. F. (2009).** Commodity prices, interest rates and the dollar. *Energy Economics*, 31(6), 838–851.
    → A weaker USD raises commodity prices — basis of the **DXY pillar** (negative weight).
15. ✅ **Haigh, M. S., & Holt, M. T. (2002).** Crack spread hedging: accounting for time-varying volatility spillovers in the energy futures markets. *Journal of Applied Econometrics*, 17(3), 269–289.
    → Crude, heating oil and gasoline futures as a system; theoretical grounding for the **3:2:1 crack spread pillar**.
16. ✅ **Büyükşahin, B., Lee, T. K., Moser, J. T., & Robe, M. A. (2013).** Physical markets, paper markets and the WTI–Brent spread. *The Energy Journal*, 34(3), 129–152. doi:10.5547/01956574.34.3.7
    → Explains what drives the **Brent–WTI spread** (Cushing inventories, logistics, financial positions).
17. **Hamilton, J. D. (2009).** Causes and consequences of the oil shock of 2007–08. *Brookings Papers on Economic Activity*, Spring, 215–261.
    → Demand growth, low price elasticity, and the role of speculation in oil price spikes.

## D. Forecasting oil prices (what accuracy is realistic?)

18. **Alquist, R., Kilian, L., & Vigfusson, R. J. (2013).** Forecasting the price of oil. In *Handbook of Economic Forecasting*, Vol. 2A, 427–507. Elsevier.
    → Key finding: the **no-change (random-walk) forecast is very hard to beat**. This is why our report compares every model to the random walk (Theil U2, MASE).
19. **Baumeister, C., & Kilian, L. (2012).** Real-time forecasts of the real price of oil. *Journal of Business & Economic Statistics*, 30(2), 326–336.
    → Why forecasts must be evaluated with data *available at the time* — the same principle as our point-in-time replay.
20. ✅ **Yu, L., Wang, S., & Lai, K. K. (2008).** Forecasting crude oil price with an EMD-based neural network ensemble learning paradigm. *Energy Economics*, 30(5), 2623–2635. doi:10.1016/j.eneco.2008.05.003
    → Widely cited ML paper for WTI/Brent forecasting; reports **directional accuracy** alongside RMSE — the same pair of metrics used here.

## E. Machine learning methods used in `evaluation/ml_models.py`

21. **Breiman, L. (2001).** Random forests. *Machine Learning*, 45(1), 5–32. → Tree ensembles.
22. **Friedman, J. H. (2001).** Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. → The algorithm behind `HistGradientBoosting*`.
23. 📘 **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning* (2nd ed.). Springer. → Logistic & ridge regression, regularisation, cross-validation.
24. **Gu, S., Kelly, B., & Xiu, D. (2020).** Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
    → Benchmark study comparing linear models, trees and neural nets for return prediction; shows small but real gains, mostly from non-linear interactions.
25. 📘 **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley.
    → **Purged walk-forward cross-validation** (ch. 7) and the triple-barrier labelling idea (ch. 3) — both used in our evaluation.
26. **Pedregosa, F., et al. (2011).** Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. → Software citation.

## F. Evaluation metrics and statistical tests used in `evaluation/metrics.py`

27. ✅ **Pesaran, M. H., & Timmermann, A. (1992).** A simple nonparametric test of predictive performance. *Journal of Business & Economic Statistics*, 10(4), 461–465 (some indexes list pp. 561–565).
    → The **directional accuracy test** we report: is the hit rate better than chance *given* the base rates?
28. **Diebold, F. X., & Mariano, R. S. (1995).** Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. → Formal test for comparing two forecasts' losses (suggested extension).
29. **Brier, G. W. (1950).** Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3. → **Brier score** for the "bullish %" probability.
30. **Niculescu-Mizil, A., & Caruana, R. (2005).** Predicting good probabilities with supervised learning. *Proc. ICML*, 625–632. → Reliability diagrams / calibration.
31. ✅ **Chicco, D., & Jurman, G. (2020).** The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21, 6. doi:10.1186/s12864-019-6413-7
    → Why we report **MCC** and balanced accuracy, not only accuracy.
32. **Hyndman, R. J., & Koehler, A. B. (2006).** Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688. → **MASE**; why MAPE on price levels is misleading.
33. 📘 **Theil, H. (1966).** *Applied Economic Forecasting.* North-Holland. → **Theil's U** statistic vs. the no-change forecast.
34. **Bergmeir, C., & Benítez, J. M. (2012).** On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192–213. → Why ordinary K-fold CV is wrong for time series.
35. **Sharpe, W. F. (1994).** The Sharpe ratio. *Journal of Portfolio Management*, 21(1), 49–58.
36. 📘 **Grinold, R. C., & Kahn, R. N. (2000).** *Active Portfolio Management* (2nd ed.). McGraw-Hill. → **Information Coefficient** (rank correlation of signal vs. future return).
37. **Newey, W. K., & West, K. D. (1987).** A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708. → Why overlapping returns need correction — we use non-overlapping subsamples instead.

## G. Backtest pitfalls (cite these when explaining your methodology choices)

38. ✅ **Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014).** Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance. *Notices of the AMS*, 61(5), 458–471. doi:10.1090/noti1105
39. **Sullivan, R., Timmermann, A., & White, H. (1999).** Data-snooping, technical trading rule performance, and the bootstrap. *Journal of Finance*, 54(5), 1647–1691.
40. **White, H. (2000).** A reality check for data snooping. *Econometrica*, 68(5), 1097–1126.
41. **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** …and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68. → Argues t > 3 should be the bar for "new" factors.

---

### Suggested "core 8" if you can only cite a few
Wilder (1978) · Brock et al. (1992) · Marshall et al. (2008) · Alquist et al. (2013) ·
Pesaran & Timmermann (1992) · Brier (1950) · López de Prado (2018) · Bailey et al. (2014).

### Verification links (indexes used to check entries marked ✅)
- Pesaran & Timmermann: https://ideas.repec.org/a/bes/jnlbes/v10y1992i4p561-65.html
- Haigh & Holt: https://ideas.repec.org/a/jae/japmet/v17y2002i3p269-289.html
- Büyükşahin et al.: https://ideas.repec.org/a/sae/enejou/v34y2013i3p129-152.html
- Yu, Wang & Lai: https://ideas.repec.org/a/eee/eneeco/v30y2008i5p2623-2635.html
- Szakmary et al.: https://ideas.repec.org/a/eee/jbfina/v34y2010i2p409-426.html
- Marshall et al.: https://ideas.repec.org/a/eee/jbfina/v32y2008i9p1810-1819.html
- Chicco & Jurman: https://pmc.ncbi.nlm.nih.gov/articles/PMC6941312
- Bailey et al.: https://nova.newcastle.edu.au/vital/access/manager/Repository/uon:20685
- Berkowitz et al.: https://doi.org/10.1111/j.1540-6261.1988.tb02591.x
