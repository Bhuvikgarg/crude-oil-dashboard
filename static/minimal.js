/* PetroSync minimal dashboard.
 * Every element has an (i) button. Its explanation says what the number is, how it is
 * calculated (with the live values), where it comes from and what it feeds, and whether
 * the backtest says it works. The pipeline strip shows the whole system in one line. */

let D = null;   // live snapshot from /api/data
let M = null;   // walk-forward evaluation from /api/metrics

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const usd = (v, d = 2) => (v == null || isNaN(v)) ? '--' : `$${Number(v).toFixed(d)}`;
const pct = (v, d = 1) => (v == null || isNaN(v)) ? '--' : `${(v * 100).toFixed(d)}%`;
const sgn = (v, d = 1) => { if (v == null || isNaN(v)) return '--'; const a = Math.abs(v).toFixed(d); return Number(a) === 0 ? a : `${v > 0 ? '+' : '−'}${a}`; };
const safe = fn => { try { return fn(); } catch (e) { return '<span class="muted">waiting for live data…</span>'; } };

/* ------------------------------------------------------------------ */
/* The pipeline                                                        */
/* ------------------------------------------------------------------ */
const STAGES = [
  { key: 'data', name: 'Market data' },
  { key: 'indicators', name: 'Indicators' },
  { key: 'pillars', name: '7 pillars' },
  { key: 'score', name: 'Signal score' },
  { key: 'calibrate', name: 'Calibration' },
  { key: 'learn', name: 'Self-learning' },
  { key: 'vol', name: 'Volatility' },
  { key: 'plan', name: 'Trade plan' },
  { key: 'evaluate', name: 'Evaluation' },
];
const stageIndex = k => STAGES.findIndex(s => s.key === k);

const PILLARS = [
  { key: 'mtf_alignment', name: 'Trend alignment', w: 25 },
  { key: 'order_flow_vwap', name: 'Price vs VWAP', w: 20 },
  { key: 'crack_margin', name: 'Refining margin', w: 15 },
  { key: 'brent_spread', name: 'Brent–WTI spread', w: 15 },
  { key: 'usd_impact', name: 'US dollar', w: 10 },
  { key: 'equity_risk_on', name: 'Stock market mood', w: 10 },
  { key: 'ovx_drag', name: 'Fear gauge (OVX)', w: 5 },
];
const ovxVote = ovx => ovx > 50 ? -25 : (ovx < 32 ? 15 : 0);
const pillarVote = k => k === 'ovx_drag' ? ovxVote(D.macro_radar.ovx_price) : D.sentiment.breakdown[k];
const learned = k => (D.online_learning && D.online_learning.pillars || []).find(p => p.key === k);

/* ------------------------------------------------------------------ */
/* Explanations                                                         */
/* Each entry: stage, title, what, how (fn), live (fn), from, to,       */
/* check (fn -> {tone, html}), why                                      */
/* ------------------------------------------------------------------ */
const INFO = {
  /* ---------- stage overviews ---------- */
  data: {
    stage: 'data', title: 'Market data: where everything starts',
    what: 'Every hour the server downloads fresh prices from Yahoo Finance. Everything else on this page is calculated from these numbers, so if the data is wrong, everything downstream is wrong.',
    how: () => `<ul>
      <li><b>Hourly bars</b> for 9 markets: WTI crude (CL), Brent crude (BZ), US dollar index (DXY), gasoline (RB), heating oil (HO), gold (GC), the oil volatility index (OVX), S&amp;P 500 futures (ES) and the energy stocks ETF (XLE).</li>
      <li><b>15-minute</b> and <b>daily</b> bars for WTI, for the short- and long-term trend.</li>
      <li>Cleaning: timestamps are converted to UTC, gaps are dropped per market, and only data that existed at that moment is used, so there is no peeking at the future.</li></ul>`,
    to: ['indicators', 'vol', 'price'],
    why: 'The backtest replays exactly this pipeline on 2.4 years of the same data (13,436 hourly decisions), so the live page and the evaluation use identical code.',
  },
  indicators: {
    stage: 'indicators', title: 'Indicators: turning prices into features',
    what: 'Raw prices are hard to compare. Indicators turn them into features with a clear meaning: is the trend up, is the price stretched, where did most trading happen?',
    how: () => `<ul>
      <li><b>EMAs</b> (9, 20, 50): smoothed averages that show the trend.</li>
      <li><b>RSI</b> (14): momentum on a 0–100 scale; above 70 is stretched up, below 30 stretched down.</li>
      <li><b>VWAP</b> and its bands: the average price paid, weighted by volume.</li>
      <li><b>Volume profile</b>: the price where most contracts traded (POC) and the 70% value area (VAL–VAH).</li>
      <li><b>ATR</b>: the typical size of one bar, used for stops.</li>
      <li><b>Intermarket</b>: Brent–WTI spread, refining (crack) margin, dollar and stock moves, gold/oil ratio.</li></ul>`,
    from: ['data'], to: ['pillars', 'plan', 'context'],
    why: 'In the ML part of the project the same indicators became the 31 features of the logistic regression and gradient boosting models.',
  },
  pillars: {
    stage: 'pillars', title: 'The 7 pillars: seven opinions about direction',
    what: 'Each pillar looks at one kind of evidence and votes from −100 (very bearish) to +100 (very bullish). Mixing technical, physical and macro evidence was meant to make the signal more robust than any single indicator.',
    how: () => `<span class="formula">vote = clip(scale × (indicator − neutral), −100, +100)</span>
      <ul>${PILLARS.map(p => `<li><b>${p.name}</b>: weight ${p.w}%</li>`).join('')}</ul>
      <p>The weights were set by hand in the original project. Click any pillar row for its exact formula.</p>`,
    live: () => `Votes now: ${PILLARS.map(p => `${p.name} <b>${sgn(pillarVote(p.key), 0)}</b>`).join(' · ')}`,
    from: ['indicators'], to: ['score', 'hedge'],
    check: () => ({ tone: 'bad', html: 'In the backtest <b>no single pillar beat 50%</b> at calling the 4-hour direction (all between 49.2% and 50.4%). That is why the score built from them is not predictive either, and why the self-learning model was added to test the weights.' }),
  },
  score: {
    stage: 'score', title: 'Signal score: the engine’s overall opinion',
    what: 'One number from −100 to +100 that combines all seven pillar votes. Positive means the engine leans bullish, negative bearish, and near zero means no clear opinion.',
    how: () => `<span class="formula">score = 0.25·trend + 0.20·VWAP + 0.15·crack
      + 0.15·Brent–WTI + 0.10·dollar + 0.10·stocks + 0.05·OVX</span>
      <p>Labels: ≥ +50 strong bullish · ≥ +15 moderate bullish · between −15 and +15 balanced · ≤ −15 moderate bearish · ≤ −50 strong bearish.</p>`,
    live: () => `Score now <b>${sgn(D.sentiment.score)}</b> → “${esc(D.sentiment.label)}”.`,
    from: ['pillars'], to: ['prob', 'plan'],
    check: () => {
      const s = M && M.signals && M.signals.h4 && M.signals.h4.sentiment_7pillar;
      return { tone: 'bad', html: `Over ${M ? M.meta.n_signals.toLocaleString() : '13,436'} past hours the score called the 4-hour direction right <b>${s ? pct(s.hit_rate) : '49.6%'}</b> of the time, which is no better than a coin flip. Treat it as a description of the market, not a forecast.` };
    },
  },
  calibrate: { alias: 'prob' },
  learn: { alias: 'hedge' },
  vol: {
    stage: 'vol', title: 'Volatility forecast: how far, not which way',
    what: 'A forecast of how much WTI is likely to move in the next 4 hours, up <i>or</i> down. This is the part of the project that predicts something reliably.',
    how: () => `<p><b>Model:</b> HAR (Corsi, 2009), a ridge regression trained walk-forward on 2.4 years of hourly data.</p>
      <span class="formula">log(vol next 4h) = b0
  + b1·log(vol last 1h) + b2·log(vol last 4h)
  + b3·log(vol last 24h) + b4·log(vol last 96h)
  + b5·log(OVX) + b6·sin(hour) + b7·cos(hour)</span>
      <p><b>Why it works:</b> big moves cluster. The autocorrelation of the <i>size</i> of hourly moves is <b>0.22</b>, while for the <i>direction</i> it is <b>−0.008</b> (zero). Using 1, 4, 24 and 96 hours captures traders who react on different time scales.</p>
      <p>Expected move = price × forecast volatility per hour × √4.</p>`,
    live: () => { const v = D.volatility_forecast; return `Price ${usd(D.wti.price)} × ${v.expected_move_pct}% → <b>±${usd(v.expected_move_usd)}</b>, range <b>${usd(v.range_low)}–${usd(v.range_high)}</b>. The last 24 hours ran at ${v.past_24h_move_pct}% per 4 hours, so the model expects ${v.expected_move_pct < v.past_24h_move_pct ? 'calmer' : 'livelier'} trading.`; },
    from: ['data'], to: ['plan', 'regime'],
    check: () => ({ tone: 'good', html: 'On the lockbox (Apr–Nov 2026, never used for training): R² <b>0.22</b>, calm vs stormy called right <b>68.7%</b> (AUC <b>0.70</b>). It matched gradient boosting with 38 features, so the simpler model was kept. A naive “same as the last 24 hours” guess scored R² −0.30.' }),
    why: 'This is why the project is still useful even though direction cannot be predicted: knowing the likely size of a move tells you how wide to set a stop and how big a position to take.',
  },
  plan: {
    stage: 'plan', title: 'Trade plan: turning the signals into an action',
    what: 'The engine picks one of five setups and gives exact entry, stop and target prices. This is the part a trader would act on, so it is also the part tested hardest.',
    how: () => `<p>Rules are checked in this order, and the first match wins:</p>
      <ul>
      <li>15-min RSI ≤ 25 and price below the value area → <b>exhaustion bounce (long)</b>: measured 37.8% (old claim 68%)</li>
      <li>15-min RSI ≥ 75 and price above the value area → <b>blow-off fade (short)</b>: measured 39.7% (old claim 67%)</li>
      <li>Trend score ≤ 35 → <b>bearish trend continuation</b>: measured 49.4% (old claim 74%)</li>
      <li>Trend score ≥ 65 → <b>bullish trend continuation</b>: measured 56.0% (old claim 73%)</li>
      <li>Otherwise → <b>value area range rotation</b>: measured 19.7% (old claim 64%)</li></ul>
      <p>Stops and targets are placed with ATR, VWAP and the volume profile.</p>`,
    live: () => `Trend score now ${D.mtf_confluence.overall_score}, 1h RSI ${D.wti.rsi} → <b>${esc(D.trade_setup.action)}</b>.`,
    from: ['indicators', 'score', 'vol'], to: ['evaluate', 'claimed', 'backtestHit'],
    check: () => {
      const T = M && M.trades;
      return { tone: 'bad', html: T ? `In ${T.filled_trades.toLocaleString()} simulated trades Target 1 was hit <b>${pct(T.target1_hit_rate)}</b> of the time against ${pct(T.claimed_win_prob_avg, 0)} claimed. Profit factor <b>${T.profit_factor.toFixed(2)}</b> (needs to be above 1.0), average <b>${sgn(T.expectancy_usd_per_trade, 0)} $</b> per contract after costs. The old win rates were typed into the code and overstated; the dashboard now shows the measured ones.` : 'Backtest not loaded.' };
    },
  },
  evaluate: {
    stage: 'evaluate', title: 'Evaluation: checking every claim against reality',
    what: 'The core of the project. The whole engine was replayed hour by hour on 2.4 years of history, using only data available at each moment, and every prediction was compared with what the price actually did.',
    how: () => `<ul>
      <li><b>Point-in-time replay</b>: 13,436 hourly decisions, with no future data leaking in.</li>
      <li><b>Walk-forward ML</b>: models retrain every 250 hours on the past only, with a gap equal to the forecast horizon.</li>
      <li><b>Lockbox</b>: Apr–Nov 2026 was locked away and tested once at the end.</li>
      <li><b>Baselines</b>: coin flip, “always up” and an oracle as a sanity check.</li>
      <li><b>Metrics</b>: hit rate with a significance test, AUC, Brier skill, profit factor, Theil U2, range coverage.</li>
      <li><b>Forward test</b>: models frozen with a SHA-256 fingerprint and scored on truly new data every hour.</li></ul>`,
    from: ['plan', 'score', 'prob', 'vol', 'hedge'],
    check: () => ({ tone: 'mid', html: 'Overall result: direction is <b>not</b> predictable at the hourly scale (best AUC about 0.52). Volatility <b>is</b> (AUC 0.70). Calibration fixed misleading probabilities. Measuring this honestly is what the project contributes.' }),
  },

  /* ---------- price & chart ---------- */
  price: {
    stage: 'data', title: 'WTI price',
    what: 'The latest price of the front-month WTI crude oil future (ticker CL) on the NYMEX exchange, in US dollars per barrel. It is the main US oil benchmark.',
    how: () => '<p>The close of the most recent hourly bar. The 1h and 24h changes compare it with the close 1 and 24 bars earlier.</p>',
    live: () => `${usd(D.wti.price)} · 1h ${sgn(D.wti.change_1h_pct, 2)}% · 24h ${sgn(D.wti.change_24h_pct, 2)}%`,
    from: ['data'], to: ['indicators', 'vol', 'chart'],
    why: 'The target of every prediction is this price 1, 4 or 24 hours later. In the evaluation each forecast is lined up with the price that actually came.',
  },
  dayrange: {
    stage: 'indicators', title: 'Day range',
    what: 'The lowest and highest price of the current trading day. The dot shows where the price sits now.',
    how: () => '<span class="formula">position = (price − low) / (high − low)</span>',
    live: () => `Low ${usd(D.wti.day_low)}, high ${usd(D.wti.day_high)}; price is <b>${Math.round((D.wti.price - D.wti.day_low) / Math.max(0.01, D.wti.day_high - D.wti.day_low) * 100)}%</b> of the way up.`,
    from: ['price'], to: ['context'],
  },
  chart: {
    stage: 'indicators', title: 'The 48-hour chart',
    what: 'Hourly closing prices with two reference lines: the 20-hour EMA (the short-term trend) and VWAP (the volume-weighted average price, a “fair value” many traders watch).',
    how: () => `<span class="formula">EMA(t) = α·price(t) + (1 − α)·EMA(t−1), α = 2/(20+1)
VWAP = Σ(typical price × volume) / Σ(volume)</span>`,
    live: () => `Price ${usd(D.wti.price)}, EMA 20 ${usd(D.wti.ema20)}, VWAP ${usd(D.wti.vwap)}: price is ${D.wti.price > D.wti.vwap ? 'above' : 'below'} fair value and ${D.wti.price > D.wti.ema20 ? 'above' : 'below'} its short trend.`,
    from: ['data'], to: ['order_flow_vwap', 'mtf_alignment'],
  },

  /* ---------- outputs ---------- */
  prob: {
    stage: 'calibrate', title: 'Calibrated probability (ML)',
    what: 'The honest chance that WTI is higher in 4 hours, given today’s score. It is learned from history instead of assumed.',
    how: () => `<p>The raw engine turns its score straight into a percentage: <span class="f">(score + 100) / 2</span>. A score of +40 claims 70%. To check this, a <b>logistic regression (Platt scaling)</b> was fitted on 13,432 past hours: score → did the price actually rise 4 hours later?</p>
      <span class="formula">P(up) = 1 / (1 + e^−(a·score + b))
a = −0.00076, b = 0.052</span>
      <p>The slope is almost zero, and slightly negative: high scores were <i>not</i> followed by more rises. So the honest answer is always close to 50%.</p>`,
    live: () => `Score ${sgn(D.sentiment.score)} → raw claim <b>${D.sentiment.bullish_pct}%</b> → calibrated <b>${D.sentiment.calibrated_p_up_4h}%</b>.`,
    from: ['score'], to: ['evaluate'],
    check: () => ({ tone: 'good', html: 'Brier skill (0 = as good as always saying 50%): raw score <b>−0.079</b> (worse than guessing) → calibrated <b>−0.001</b> on the lockbox. Calibration does not add accuracy, but it stops the dashboard from promising confidence it does not have. It is deployed live.' }),
  },
  regime: {
    stage: 'vol', title: 'Volatility regime',
    what: 'Where today’s forecast sits compared with all past forecasts: very calm, calm, normal, stormy or very stormy. Each band holds one fifth of history.',
    how: () => '<p>The forecast is compared with the 20th, 40th, 60th and 80th percentiles of past forecasts. Five bands, so each is equally common.</p>',
    live: () => `Now: <b>${esc(D.volatility_forecast.regime)}</b> (band ${D.volatility_forecast.regime_level + 1} of 5).`,
    from: ['vol'], to: ['plan'],
    check: () => ({ tone: 'mid', html: 'Does a stormy regime make direction easier to call? Tested: no. Stormy hours scored 53.5% on the lockbox but only 50.9% in development, so that difference is noise.' }),
  },

  /* ---------- self learning ---------- */
  hedge: {
    stage: 'learn', title: 'Self-learning weights (online ML)',
    what: 'The hand-set pillar weights were a guess. This model treats each pillar as an “expert” and adjusts its weight every hour, depending on how good its recent calls were.',
    how: () => `<p><b>Algorithm:</b> Hedge (multiplicative weights, Freund &amp; Schapire 1997) with fixed share (Herbster &amp; Warmuth 1998).</p>
      <span class="formula">loss_i = (vote_i − outcome)² / 4
w_i ← w_i · e^(−η · loss_i),  η = 0.01
normalise, then mix in α = 1% of the average weight</span>
      <ul><li>It learns from the outcome of the call made 4 hours ago, once that outcome is known.</li>
      <li>It always predicts first and learns after (test-then-train), so it never sees the answer early.</li>
      <li>Fixed share stops any weight reaching zero, so the model can adapt if the market changes.</li></ul>`,
    live: () => { const L = D.online_learning; return `Learned score <b>${sgn(L.learned_score)}</b> (${L.learned_call}). ${L.updates_total.toLocaleString()} updates so far, ${L.updates_live} live on this server. Last update ${esc(L.last_update)}.`; },
    from: ['pillars'], to: ['evaluate'],
    check: () => ({ tone: 'mid', html: 'Lockbox: hand weights 50.9% hit, AUC 0.501 → learned weights 50.9%, AUC 0.502. No accuracy gain, but it <b>showed which weights were unjustified</b>: weight moved from the trend and VWAP pillars toward the dollar and stocks, because their small votes lose less when no pillar beats 50%.' }),
  },

  /* ---------- trade plan parts ---------- */
  entry: {
    stage: 'plan', title: 'Entry', what: 'The price zone where the setup says to enter.',
    how: () => '<ul><li>Trend setups: a pullback toward VWAP (or 0.4 × ATR), ± $0.15.</li><li>Exhaustion setups: around the current price, a fraction of ATR either side.</li><li>Range setup: buy near the value area low, sell near the value area high.</li></ul>',
    live: () => esc(D.trade_setup.entry_range), from: ['indicators'], to: ['stop', 'targets'],
  },
  stop: {
    stage: 'plan', title: 'Stop loss', what: 'The price at which the trade is wrong and should be closed to cap the loss.',
    how: () => '<p>Placed 0.5–0.7 × ATR beyond the entry, so a normal-sized wiggle does not trigger it. 1 tick = $0.01 = $10 per CL contract.</p>',
    live: () => `${usd(D.trade_setup.stop_loss)} · ${D.trade_setup.stop_ticks} ticks risk (≈ $${D.trade_setup.stop_ticks * 10} per contract). ATR now ${usd(D.wti.atr)}.`,
    from: ['entry', 'indicators'], to: ['rr', 'evaluate'],
    check: () => ({ tone: 'mid', html: `Compare the stop distance with the volatility forecast: ±${usd(D.volatility_forecast.expected_move_usd)} is a normal 4-hour move right now. A stop tighter than that is likely to be hit by noise.` }),
  },
  targets: {
    stage: 'plan', title: 'Targets', what: 'Where to take profit. Target 1 is the main exit, Target 2 the stretch.',
    how: () => '<p>Usually the edge of the value area, the POC or a VWAP band; otherwise 0.8–0.9 × ATR away. Target 2 adds 1.2 × ATR more.</p>',
    live: () => `T1 ${usd(D.trade_setup.target1)} · T2 ${usd(D.trade_setup.target2)}`, from: ['entry'], to: ['rr', 'backtestHit'],
  },
  rr: {
    stage: 'plan', title: 'Reward : risk',
    what: 'How many dollars the plan aims to make for every dollar it risks.',
    how: () => '<span class="formula">R:R = (Target 1 − entry) / (entry − stop)</span><p>Break-even win rate = 1 / (1 + R:R), before costs. At 1 : 0.6 you must win 63% of trades just to break even; at 1 : 1.5, 40%.</p><p>Reward and risk are measured from the same entry price the backtest fills at. An earlier version measured them from opposite ends of the entry zone, which made R:R look about twice as good as it was.</p>',
    live: () => esc(D.trade_setup.risk_reward), from: ['stop', 'targets'], to: ['evaluate'],
  },
  claimed: {
    stage: 'plan', title: 'Old typed-in claim (replaced)',
    what: 'The win rate the original playbook stated for this setup. It was a fixed number typed into the code (64–74%) and never measured.',
    how: () => '<p>It is kept on screen, crossed out, so the difference is visible. The backtest still tests the claim, so the comparison stays honest. Everywhere else the dashboard now uses the <b>measured</b> win rate.</p>',
    live: () => `Typed-in claim ${D.trade_setup.claimed_win_probability ?? D.trade_setup.win_probability}% → measured ${D.trade_setup.win_probability}%.`,
    from: ['plan'], to: ['backtestHit'],
  },
  backtestHit: {
    stage: 'evaluate', title: 'Measured win rate',
    what: 'How often this exact setup actually reached Target 1 before its stop, measured on 2.4 years of history. It replaces the typed-in claim.',
    how: () => `<p>Every historical signal was simulated as a real trade: it fills only if price reaches the entry, then it is a win if Target 1 comes before the stop. 2 ticks of costs, one trade at a time.</p>
      <span class="formula">win rate = Target 1 hits / filled trades
95% range: Wilson score interval</span>
      <p>The grey band is the 95% range: with a few hundred trades the true rate could be a few points either side. It is measured separately on the development period and the lockbox to check that it is stable.</p>
      <p>Made with <code>python -m evaluation.measure_playbook</code>, which writes <code>reports/playbook_measured.json</code>.</p>`,
    live: () => { const m = D.trade_setup.measured; return m ? `${m.n} trades: <b>${D.trade_setup.win_probability}%</b> (95% range ${pct(m.ci95[0])}–${pct(m.ci95[1])}); development ${pct(m.dev)}, lockbox ${pct(m.lockbox)}; profit factor ${m.profit_factor}; average ${sgn(m.expectancy_usd, 0)} $ per contract.` : 'No measured history for this setup.'; },
    from: ['claimed', 'plan'], to: ['evaluate'],
    check: () => ({ tone: 'mid', html: 'All five setups fall well short of their old claims (by 17 to 44 points). Win rate alone is not the whole story: the blow-off fade wins only 40% but makes +$90 per trade, because its wins are bigger than its losses.' }),
  },

  /* ---------- context ---------- */
  context: {
    stage: 'indicators', title: 'Market context',
    what: 'The supporting indicators the pillars and the trade plan are built from. Click any one for what it means and where it is used.',
    from: ['data'], to: ['pillars', 'plan'],
  },
  ovx: {
    stage: 'indicators', title: 'OVX: the oil fear gauge',
    what: 'The CBOE crude oil volatility index: how much movement options traders expect over the next 30 days, in % per year. It is the market’s own volatility forecast.',
    how: () => '<p>Below 30 low · 30–45 normal · above 45 stressed. Used three ways: a small pillar (−25 if above 50, +15 if below 32), a warning on the trade plan above 45, and one input of the volatility model.</p>',
    live: () => `${D.macro_radar.ovx_price} → ${esc(D.macro_radar.ovx_regime)}`, from: ['data'], to: ['ovx_drag', 'vol', 'plan'],
  },
  brent: {
    stage: 'indicators', title: 'Brent–WTI spread',
    what: 'The price gap between Brent (the global benchmark) and WTI (the US one). A wide gap makes US oil cheap to export, which supports WTI demand.',
    how: () => '<span class="formula">spread = Brent − WTI   (typical ≈ $5)</span>',
    live: () => `Brent ${usd(D.brent.price)} − WTI ${usd(D.wti.price)} = <b>${usd(D.brent.spread_to_wti)}</b>`, from: ['data'], to: ['brent_spread'],
  },
  crack: {
    stage: 'indicators', title: 'Crack margin (3-2-1)',
    what: 'The profit a refinery makes turning 3 barrels of crude into 2 of gasoline and 1 of heating oil. High margins mean refiners want to buy crude.',
    how: () => '<span class="formula">crack = (2·gasoline·42 + 1·heating oil·42 − 3·WTI) / 3</span><p>×42 converts $/gallon to $/barrel. Around $26 is treated as neutral.</p>',
    live: () => `<b>${usd(D.macro_radar.crack_margin)}</b> per barrel`, from: ['data'], to: ['crack_margin'],
  },
  dxy: {
    stage: 'indicators', title: 'US dollar index (DXY)',
    what: 'The dollar against a basket of major currencies. Oil is priced in dollars, so a stronger dollar usually weighs on oil.',
    live: () => `${D.macro_radar.dxy_price} · last hour ${sgn(D.macro_radar.dxy_change_pct, 2)}%`, from: ['data'], to: ['usd_impact'],
  },
  spx: {
    stage: 'indicators', title: 'S&P 500 futures',
    what: 'US stocks as a proxy for risk appetite and economic optimism. When investors feel confident they tend to buy oil too.',
    live: () => `${D.macro_radar.sp500_price.toLocaleString()} · last hour ${sgn(D.macro_radar.sp500_change_pct, 2)}%`, from: ['data'], to: ['equity_risk_on'],
  },
  goldoil: {
    stage: 'indicators', title: 'Gold / oil ratio',
    what: 'How many barrels one ounce of gold buys. A high ratio means oil is cheap compared with gold, often in times of fear about growth.',
    how: () => '<span class="formula">ratio = gold price / WTI price</span><p>Shown for context; it is not one of the scored pillars.</p>',
    live: () => `${D.macro_radar.gold_oil_ratio} barrels per ounce`, from: ['data'],
  },
  vwapInfo: {
    stage: 'indicators', title: 'VWAP and its bands',
    what: 'Volume-weighted average price: the average price paid by everyone today, giving more weight to busy periods. The bands sit one and two standard deviations away.',
    how: () => '<span class="formula">VWAP = Σ(typical price × volume) / Σ volume</span>',
    live: () => `VWAP ${usd(D.levels.vwap)} · bands ${usd(D.levels.vwap_l1)} – ${usd(D.levels.vwap_u1)}`, from: ['data'], to: ['order_flow_vwap', 'plan'],
  },
  profile: {
    stage: 'indicators', title: 'Volume profile: POC and value area',
    what: 'Where trading actually happened over the last 24 hours. The POC (point of control) is the busiest price; the value area holds 70% of the volume.',
    how: () => '<p>Volume is grouped into price buckets. Starting from the busiest bucket, neighbouring buckets are added until 70% of volume is covered: that is VAL to VAH.</p>',
    live: () => `POC ${usd(D.levels.poc)} · value area ${usd(D.levels.val)}–${usd(D.levels.vah)}`, from: ['data'], to: ['plan'],
  },
  atr: {
    stage: 'indicators', title: 'ATR: average true range',
    what: 'The typical size of one hourly bar, including gaps. It is a backward-looking measure of volatility.',
    how: () => '<span class="formula">TR = max(high − low, |high − prev close|, |low − prev close|)\nATR = 14-bar average of TR</span>',
    live: () => usd(D.wti.atr), from: ['data'], to: ['stop', 'targets'],
    check: () => ({ tone: 'mid', html: 'ATR only looks back. The volatility model looks forward and uses several windows plus OVX, which is why it predicts the next move size much better than “same as recently”.' }),
  },
  rsi: {
    stage: 'indicators', title: 'RSI (14)',
    what: 'Relative strength index: compares the size of recent up moves with recent down moves, on a 0–100 scale.',
    how: () => '<span class="formula">RSI = 100 − 100 / (1 + avg gain / avg loss)</span><p>≥ 60 adds +20 to a timeframe’s trend score, ≤ 40 subtracts 20. Extremes on the 15-minute chart trigger the exhaustion setups.</p>',
    live: () => `1h RSI ${D.wti.rsi}`, from: ['data'], to: ['mtf_alignment', 'plan'],
  },
  eia: {
    stage: 'data', title: 'EIA inventory report',
    what: 'The weekly US crude stock report (Wednesdays, 10:30 New York time). It often causes the biggest move of the week.',
    live: () => `Next report in ${esc(D.macro_radar.eia_countdown)}`, to: ['vol'],
  },
  sweep: {
    stage: 'indicators', title: 'Liquidity sweep',
    what: 'A quick push beyond a recent high or low that then reverses, typically when stop orders get triggered.',
    live: () => esc(D.liquidity_sweeps.status), from: ['data'], to: ['evaluate'],
    check: () => ({ tone: 'mid', html: 'Sweep signals were tested separately in the backtest; see docs/EVALUATION_METHODOLOGY.md for the numbers.' }),
  },
  tf: {
    stage: 'indicators', title: 'Timeframes: 15-minute, 1-hour, daily',
    what: 'The same trend check run on three time scales. When all three agree, the trend is considered strong.',
    how: () => `<span class="formula">start at 50
+30 if price > EMA9 > EMA21   (−30 if below)
+20 if RSI ≥ 60               (−20 if ≤ 40)
overall = 0.25·15m + 0.45·1h + 0.30·daily</span><p>≥ 70 bullish, ≤ 30 bearish.</p>`,
    live: () => `${D.mtf_confluence.timeframes.map(t => `${esc(t.timeframe.split(' ')[0])} ${t.score}`).join(' · ')} → overall <b>${D.mtf_confluence.overall_score}</b> (${esc(D.mtf_confluence.alignment)})`,
    from: ['data'], to: ['mtf_alignment', 'plan'],
  },

  /* ---------- evaluation stats ---------- */
  hit: {
    stage: 'evaluate', title: 'Direction hit rate',
    what: 'How often the engine’s up or down call over the next 4 hours was right, measured on every hour of the 2.4-year replay.',
    how: () => '<span class="formula">hit rate = correct calls / all calls</span><p>The comparison is 50% (a coin flip) and the share of hours that went up (51.3%). A binomial test and the Pesaran–Timmermann test check whether any gap is more than luck.</p>',
    from: ['score'], check: () => ({ tone: 'bad', html: 'Not better than chance.' }),
  },
  brier: {
    stage: 'evaluate', title: 'Brier skill score',
    what: 'Measures how good the probabilities are, not just the up or down call. 0 means as good as always saying 50%; above 0 is better, below 0 is worse.',
    how: () => '<span class="formula">Brier = average (p − outcome)²\nskill = 1 − Brier / Brier(always 50%)</span>',
    from: ['prob'], check: () => ({ tone: 'good', html: 'Raw −0.079 → calibrated −0.001: calibration removed the overconfidence.' }),
  },
  auc: {
    stage: 'evaluate', title: 'ROC-AUC of the ML models',
    what: 'The probability that the model ranks a random “up” hour above a random “down” hour. 0.5 is random, 1.0 is perfect.',
    how: () => '<p>Logistic regression (C = 0.1) and gradient boosting (depth 3, 200 trees) on 31 features, trained walk-forward and retrained every 250 hours.</p>',
    from: ['indicators'], check: () => ({ tone: 'bad', html: 'Best direction AUC ≈ 0.52: barely above random, and it did not hold up on the lockbox.' }),
  },
  pf: {
    stage: 'evaluate', title: 'Profit factor',
    what: 'Total money won divided by total money lost in the simulated trades. Above 1 the strategy makes money.',
    how: () => '<span class="formula">PF = Σ winning trades / |Σ losing trades|</span>',
    from: ['plan'], check: () => ({ tone: 'bad', html: 'Below 1, so the trade plan as written lost money after costs.' }),
  },
  volstat: { alias: 'vol' },
};

// Pillar entries
const PILLAR_INFO = {
  mtf_alignment: { what: 'Do the 15-minute, 1-hour and daily trends point the same way?', f: 'vote = (overall trend score − 50) × 2', src: 'tf' },
  order_flow_vwap: { what: 'Is the price above or below the average price paid today? Above suggests buyers are in control.', f: 'vote = clip(40 × (price − VWAP) / VWAP × 100)', src: 'vwapInfo' },
  crack_margin: { what: 'Are refineries making good money? High margins mean they buy more crude.', f: 'vote = clip(6 × (crack − $26))', src: 'crack' },
  brent_spread: { what: 'Is US oil cheap compared with global oil? That supports exports and WTI demand.', f: 'vote = clip(15 × (spread − $5))', src: 'brent' },
  usd_impact: { what: 'Is the dollar getting stronger? That usually pushes oil down, so the sign is flipped.', f: 'vote = −clip(70 × DXY % change)', src: 'dxy' },
  equity_risk_on: { what: 'Are stocks rising? Risk appetite tends to lift oil too.', f: 'vote = clip(50 × S&P % change)', src: 'spx' },
  ovx_drag: { what: 'Is the market panicking? Very high fear is treated as bearish, very low fear as mildly bullish.', f: 'vote = −25 if OVX > 50, +15 if OVX < 32, else 0', src: 'ovx' },
};
PILLARS.forEach(p => {
  const pi = PILLAR_INFO[p.key];
  INFO[p.key] = {
    stage: 'pillars', title: `Pillar: ${p.name}`,
    what: pi.what,
    how: () => `<span class="formula">${pi.f}
clip keeps it between −100 and +100</span><p>Hand weight in the score: <b>${p.w}%</b>.</p>`,
    live: () => {
      const v = pillarVote(p.key), L = learned(p.key);
      return `Vote now <b>${sgn(v, 1)}</b>, adding ${sgn(v * p.w / 100, 1)} points to the score.` +
        (L ? ` Learned weight <b>${(L.learned_weight * 100).toFixed(1)}%</b> vs hand ${(L.hand_weight * 100).toFixed(1)}% (out of the 6 learnable pillars).` : ' This pillar is not part of the self-learning model because the engine does not report its vote separately.');
    },
    from: [pi.src], to: ['score', 'hedge'],
    check: () => {
      const hr = { mtf_alignment: 49.7, order_flow_vwap: 49.2, crack_margin: 49.5, brent_spread: 50.4, usd_impact: 49.3, equity_risk_on: 49.7 }[p.key];
      return hr ? { tone: 'bad', html: `On its own this pillar called the 4-hour direction right <b>${hr}%</b> of the time in the backtest.` } : { tone: 'mid', html: 'Not tested separately.' };
    },
  };
});

const CONTEXT = [
  ['ovx', 'Oil fear gauge (OVX)', () => D.macro_radar.ovx_price.toFixed(1), () => D.macro_radar.ovx_regime.split(' (')[0]],
  ['brent', 'Brent – WTI', () => usd(D.brent.spread_to_wti), () => `Brent ${usd(D.brent.price)}`],
  ['crack', 'Refining margin', () => usd(D.macro_radar.crack_margin), () => 'per barrel, 3-2-1'],
  ['dxy', 'US dollar (DXY)', () => D.macro_radar.dxy_price.toFixed(2), () => `${sgn(D.macro_radar.dxy_change_pct, 2)}% 1h`],
  ['spx', 'S&P 500 futures', () => Math.round(D.macro_radar.sp500_price).toLocaleString(), () => `${sgn(D.macro_radar.sp500_change_pct, 2)}% 1h`],
  ['goldoil', 'Gold / oil', () => `${D.macro_radar.gold_oil_ratio}`, () => 'barrels per ounce'],
  ['vwapInfo', 'VWAP', () => usd(D.levels.vwap), () => `bands ${usd(D.levels.vwap_l1)}–${usd(D.levels.vwap_u1)}`],
  ['profile', 'Value area · POC', () => usd(D.levels.poc), () => `${usd(D.levels.val)}–${usd(D.levels.vah)}`],
  ['atr', 'ATR (hourly)', () => usd(D.wti.atr), () => 'typical bar size'],
  ['rsi', 'RSI (1h)', () => D.wti.rsi.toFixed(0), () => D.wti.rsi >= 70 ? 'stretched up' : D.wti.rsi <= 30 ? 'stretched down' : 'neutral zone'],
  ['eia', 'EIA report in', () => D.macro_radar.eia_countdown, () => 'Wednesdays 10:30 NY'],
  ['sweep', 'Liquidity sweep', () => D.liquidity_sweeps.swept_high ? 'High swept' : D.liquidity_sweeps.swept_low ? 'Low swept' : 'None', () => 'stop runs'],
];

/* ------------------------------------------------------------------ */
/* Rendering                                                            */
/* ------------------------------------------------------------------ */
function renderPipeline() {
  $('pipeline').innerHTML = STAGES.map((s, i) =>
    `<div class="step">${i ? '<span class="arrow">→</span>' : ''}<button data-info="${s.key}" data-step="${s.key}"><span class="n">${i + 1}</span>${s.name}</button></div>`).join('');
}

function setupStats() {
  if (!M || !M.trades || !D) return null;
  return M.trades.by_setup[D.trade_setup.setup_type] || null;
}

function renderSpark() {
  const c = D.chart_candles || [];
  const svg = $('spark');
  if (c.length < 2) { svg.innerHTML = ''; return; }
  const vals = c.flatMap(k => [k.close, k.ema20, k.vwap]).filter(v => v != null);
  const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.08 || 1;
  const W = 600, H = 180;
  const x = i => i / (c.length - 1) * W;
  const y = v => H - (v - lo + pad) / (hi - lo + 2 * pad) * H;
  const path = key => c.map((k, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(k[key]).toFixed(1)}`).join('');
  const area = `${path('close')}L${W},${H}L0,${H}Z`;
  const last = c[c.length - 1];
  svg.innerHTML = `
    <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="var(--ink)" stop-opacity=".08"/><stop offset="1" stop-color="var(--ink)" stop-opacity="0"/></linearGradient></defs>
    <path d="${area}" fill="url(#g)"/>
    <path d="${path('vwap')}" fill="none" stroke="var(--warn)" stroke-width="1.5" stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/>
    <path d="${path('ema20')}" fill="none" stroke="var(--accent)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>
    <path d="${path('close')}" fill="none" stroke="var(--ink)" stroke-width="2" vector-effect="non-scaling-stroke"/>
    <circle cx="${x(c.length - 1)}" cy="${y(last.close)}" r="3.5" fill="var(--ink)"/>`;
}

function chg(el, label, v) {
  el.textContent = `${label} ${sgn(v, 2)}%`;
  el.className = 'chg ' + (v > 0 ? 'up' : v < 0 ? 'down' : '');
}

function render() {
  if (!D) return;
  const w = D.wti, s = D.sentiment, t = D.trade_setup, v = D.volatility_forecast;

  // hero
  $('price').textContent = usd(w.price);
  chg($('chg1h'), '1h', w.change_1h_pct);
  chg($('chg24h'), '24h', w.change_24h_pct);
  $('dayLow').textContent = usd(w.day_low);
  $('dayHigh').textContent = usd(w.day_high);
  const pos = Math.max(0, Math.min(1, (w.price - w.day_low) / Math.max(0.01, w.day_high - w.day_low)));
  $('rangeDot').style.left = `${pos * 100}%`;
  renderSpark();

  // score
  $('score').textContent = sgn(s.score);
  $('score').className = 'big ' + (s.score >= 15 ? 'bull' : s.score <= -15 ? 'bear' : '');
  $('scoreLabel').textContent = s.label;
  const f = $('scoreFill');
  f.style.left = s.score >= 0 ? '50%' : `${50 + s.score / 2}%`;
  f.style.width = `${Math.abs(s.score) / 2}%`;
  f.style.background = s.score >= 0 ? 'var(--bull)' : 'var(--bear)';

  // probability
  const cal = s.calibrated_p_up_4h;
  $('prob').textContent = cal == null ? '--' : `${cal.toFixed(1)}%`;
  $('probSub').textContent = cal == null ? 'Calibration file missing' : Math.abs(cal - 50) < 3 ? 'Essentially a coin flip, and that is the honest answer' : (cal > 50 ? 'Slight lean up' : 'Slight lean down');
  $('rawPct').textContent = `${s.bullish_pct}%`;
  $('calPct').textContent = cal == null ? '--' : `${cal.toFixed(1)}%`;

  // volatility
  if (v) {
    $('volMove').textContent = `±${usd(v.expected_move_usd)}`;
    $('volRange').textContent = `Likely range ${usd(v.range_low)} – ${usd(v.range_high)} · ${v.expected_move_pct}%`;
    $('regimeDots').innerHTML = [0, 1, 2, 3, 4].map(i => `<i class="${i <= v.regime_level ? 'on' : ''}"></i>`).join('');
    $('regimeName').textContent = v.regime;
  } else {
    $('volMove').textContent = '--';
    $('volRange').textContent = 'Volatility model not loaded';
  }

  // pillars
  $('pillarRows').innerHTML = PILLARS.map(p => {
    const vote = pillarVote(p.key) ?? 0, L = learned(p.key);
    const width = Math.min(100, Math.abs(vote)) / 2, left = vote >= 0 ? 50 : 50 - width;
    let lw = '<span class="muted">—</span>';
    if (L) {
      const d = (L.learned_weight - L.hand_weight) * 100;
      lw = `${(L.learned_weight * 100).toFixed(1)}%<span class="wdiff ${d > 0.5 ? 'bull' : d < -0.5 ? 'bear' : 'muted'}">${d > 0.5 ? '▲' : d < -0.5 ? '▼' : ''}</span>`;
    }
    return `<div class="prow">
      <span class="pname">${p.name}<button class="i" data-info="${p.key}" aria-label="About ${p.name}">i</button></span>
      <span class="vcell"><span class="vbar"><span class="mid"></span><span class="fill" style="left:${left}%;width:${width}%;background:${vote >= 0 ? 'var(--bull)' : 'var(--bear)'}"></span></span><span class="vval num">${sgn(vote, 0)}</span></span>
      <span class="num">${p.w}%</span>
      <span class="num">${lw}</span></div>`;
  }).join('');
  renderLearn();

  // trade plan
  $('planAction').textContent = t.action;
  $('planType').textContent = t.setup_type;
  $('planBias').textContent = t.bias;
  $('planBias').className = 'tag ' + (t.bias === 'LONG' ? 'bull' : t.bias === 'SHORT' ? 'bear' : '');
  $('pEntry').textContent = t.entry_min === t.entry_max ? usd(t.entry_min) : `${usd(t.entry_min)}–${t.entry_max.toFixed(2)}`;
  $('pStop').textContent = usd(t.stop_loss);
  $('pT1').textContent = usd(t.target1);
  $('pT2').textContent = usd(t.target2);
  $('pRR').textContent = t.risk_reward;
  $('planWhy').textContent = t.rationale;
  const claimed = t.claimed_win_probability ?? t.win_probability;
  $('claimBar').style.width = `${claimed}%`;
  $('claimVal').textContent = `${claimed}%`;
  renderReality();

  // context
  $('contextList').innerHTML = CONTEXT.map(([k, name, val, sub]) => `<div class="kv-item">
      <span class="k">${name}<button class="i" data-info="${k}" aria-label="About ${name}">i</button></span>
      <span class="v">${safe(val)}<small>${safe(sub)}</small></span></div>`).join('');
  $('tfRow').innerHTML = D.mtf_confluence.timeframes.map(tf => `<div>
      <span style="display:flex;align-items:center;gap:5px">${esc(tf.timeframe)}<button class="i" data-info="tf" aria-label="About timeframes">i</button></span>
      <b class="${tf.bias === 'BULLISH' ? 'bull' : tf.bias === 'BEARISH' ? 'bear' : ''}">${tf.bias} · ${tf.score}</b></div>`).join('');

  $('updated').textContent = D.timestamp;
}

function renderReality() {
  const t = D && D.trade_setup, m = t && t.measured, bar = $('realBar');
  if (!m) { bar.style.width = '0'; $('realCi').style.width = '0'; $('realVal').textContent = 'n/a'; $('planNote').textContent = 'No measured history for this setup yet.'; return; }
  const p = t.win_probability;
  bar.style.width = `${p}%`;
  bar.className = 'bar-fill ' + (m.expectancy_usd > 0 ? 'good' : 'mid');
  $('realCi').style.left = `${m.ci95[0] * 100}%`;
  $('realCi').style.width = `${(m.ci95[1] - m.ci95[0]) * 100}%`;
  $('realVal').textContent = `${p.toFixed(1)}%`;
  const be = (() => { const r = parseFloat(String(t.risk_reward).split(':')[1]); return r ? 100 / (1 + r) : null; })();
  $('planNote').innerHTML = `Measured on <b>${m.n}</b> past trades: likely between ${pct(m.ci95[0], 0)} and ${pct(m.ci95[1], 0)} (95%). `
    + `Development ${pct(m.dev, 0)}, lockbox ${pct(m.lockbox, 0)}, so it is stable. `
    + (m.avg_win_usd != null ? `Wins average +$${Math.round(m.avg_win_usd)}, losses −$${Math.abs(Math.round(m.avg_loss_usd))}, so the average trade made ` : 'Average ')
    + `<b class="${m.expectancy_usd > 0 ? 'bull' : 'bear'}">${sgn(m.expectancy_usd, 0)} $</b> per contract after costs.`
    + (be ? ` This plan's reward : risk (${esc(t.risk_reward)}) needs about ${be.toFixed(0)}% wins to break even.` : '');
}

function renderEval() {
  if (!M) return;
  const s4 = M.signals.h4.sentiment_7pillar, T = M.trades;
  const auc = Math.max(M.ml.h4.logistic_regression.probability.roc_auc, M.ml.h4.gradient_boosting.probability.roc_auc);
  $('evalMeta').textContent = `${M.meta.n_signals.toLocaleString()} hourly decisions · ${M.meta.sample_start.slice(0, 10)} → ${M.meta.sample_end.slice(0, 10)}`;
  const stats = [
    ['hit', 'Direction hit rate', pct(s4.hit_rate), 'coin flip = 50%', ''],
    ['brier', 'Probability skill', '−0.001', `raw ${s4.probability.brier_skill_score.toFixed(3)} → calibrated`, ''],
    ['auc', 'Best ML direction AUC', auc.toFixed(3), 'random = 0.500', ''],
    ['pf', 'Profit factor', T.profit_factor.toFixed(2), `${T.filled_trades.toLocaleString()} trades · needs > 1`, ''],
    ['volstat', 'Volatility calm vs stormy', '68.7%', 'AUC 0.70 · lockbox', 'hl'],
  ];
  $('evalStats').innerHTML = stats.map(([k, name, v, sub, cls]) => `<div class="stat ${cls}">
      <span class="k">${name}<button class="i" data-info="${k}" aria-label="About ${name}">i</button></span>
      <span class="v">${v}</span><span class="s">${sub}</span></div>`).join('');
}

/* ------------------------------------------------------------------ */
/* Self-learning card + hover tips                                      */
/* ------------------------------------------------------------------ */
const WCOLORS = ['#2563eb', '#db2777', '#ea580c', '#16a34a', '#7c3aed', '#0891b2'];
const LEARN_PILLARS = PILLARS.filter(p => p.key !== 'ovx_drag');

function renderLearn() {
  const L = D.online_learning, card = $('learnCard');
  if (!L) { card.hidden = true; return; }
  card.hidden = false;
  $('lScore').textContent = sgn(L.learned_score);
  $('lCall').textContent = L.learned_call;
  $('lCall').className = 'tag ' + (L.learned_call === 'BULL' ? 'bull' : L.learned_call === 'BEAR' ? 'bear' : '');
  $('lFixed').textContent = sgn(D.sentiment.score);
  $('lUpdates').textContent = L.updates_total.toLocaleString();
  $('lUpdatesSub').textContent = `${L.updates_live} live on this server · ${L.pending} waiting for their answer`;

  const lv = L.live || { n: 0 };
  if (lv.n) {
    $('lBoard').innerHTML = `${(lv.learned_rate * 100).toFixed(0)}% <span class="muted" style="font-size:14px;font-weight:500">vs ${(lv.engine_rate * 100).toFixed(0)}%</span>`;
    $('lBoardSub').textContent = `learned vs fixed · ${lv.n} live call${lv.n === 1 ? '' : 's'} so far`;
  } else {
    $('lBoard').textContent = 'Starting';
    $('lBoardSub').textContent = 'first live calls are resolved 4 h after start';
  }

  // weight history
  const H = (L.history || []).filter(h => h.w && h.w.length === 6);
  const svg = $('wSvg');
  if (H.length > 1) {
    const all = H.flatMap(h => h.w), lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.15 || 0.01;
    const x = i => i / (H.length - 1) * 600, y = v => 140 - (v - lo + pad) / (hi - lo + 2 * pad) * 140;
    document.querySelector('.wchart .label').textContent = `How the weights moved · last 60 updates · axis ${((lo - pad) * 100).toFixed(1)}% to ${((hi + pad) * 100).toFixed(1)}% (zoomed in)`;
    svg.innerHTML = WCOLORS.map((c, k) =>
      `<path d="${H.map((h, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(h.w[k]).toFixed(1)}`).join('')}" fill="none" stroke="${c}" stroke-width="2" vector-effect="non-scaling-stroke"/>`).join('');
    const last = H[H.length - 1].w;
    $('wLegend').innerHTML = LEARN_PILLARS.map((p, k) => `<span><i style="background:${WCOLORS[k]}"></i>${p.name} ${(last[k] * 100).toFixed(1)}%</span>`).join('') +
      `<span class="muted">${esc(H[0].t.slice(0, 10))} → ${esc(H[H.length - 1].t.slice(0, 10))}</span>`;
  }

  // when will it improve
  const prog = lv.progress || 0;
  $('lProg').style.width = `${Math.max(prog * 100, 0.8)}%`;
  $('lEta').textContent = `${lv.nonoverlap ?? 0} of ${lv.needed_nonoverlap ?? 764} independent calls · about ${lv.days_remaining_if_always_on ?? 186} days left if always on`;
  $('lImproveText').innerHTML = 'It will not get smarter just by waiting: after 13,432 updates its weights have settled, and in the backtest it matched the fixed weights (50.9% vs 50.9%). '
    + 'What time gives is <b>proof</b>: about 764 independent 4-hour calls are needed before a real edge can be told apart from luck.';
}

const TIPS = {
  cycle: () => `<h5>How it learns, every hour</h5><ol>
    <li><b>Predict:</b> weighted average of the 6 pillar votes.</li>
    <li><b>Wait:</b> the answer is known 4 hours later.</li>
    <li><b>Score:</b> how wrong was each pillar?</li>
    <li><b>Re-weight:</b> wrong pillars lose a little trust.</li></ol>
    <span class="tnote">It always predicts before it sees the answer (test-then-train), so it can never cheat.</span>`,
  c1: () => { const L = D.online_learning; return `<h5>1 · Predict</h5>Each pillar votes from −1 (down) to +1 (up). The model multiplies each vote by its current weight and adds them up.
    <span class="tf">${L.pillars.map(p => `${(p.vote / 100).toFixed(2).padStart(5)} × ${(p.learned_weight * 100).toFixed(1)}%`).join('\n')}
= ${sgn(L.learned_score)}  →  ${L.learned_call}</span>Above 0 means it expects the price to rise.`; },
  c2: () => `<h5>2 · Wait 4 hours</h5>The call is saved with today's price. Only when 4 more hourly bars have closed does the model know if it was right.
    <span class="tnote">Waiting right now: ${D.online_learning.pending} call(s).</span>`,
  c3: () => `<h5>3 · Score each pillar</h5>Outcome = +1 if the price rose, −1 if it fell.
    <span class="tf">loss = (vote − outcome)² / 4</span>Example: a pillar voted +0.7 and the price fell → (0.7 + 1)² / 4 = <b>0.72</b> (bad). A vote of 0 always costs 0.25.`,
  c4: () => `<h5>4 · Re-weight</h5><span class="tf">w ← w × e^(−0.01 × loss)
then rescale to 100%
then mix in 1% equal weight</span>Loss 0.72 → weight × 0.9928, a 0.7% cut. Small steps on purpose, so one bad hour can't wreck it. The 1% mix (“fixed share”) keeps every pillar alive in case it becomes useful again.`,
  lscore: () => `<h5>Same votes, different weights</h5>Fixed weights (typed by hand, never change): <b>${sgn(D.sentiment.score)}</b><br>Learned weights (updated every hour): <b>${sgn(D.online_learning.learned_score)}</b>
    <span class="tnote">The trade plan still uses the fixed score. The learned one runs beside it as a challenger, so the two can be compared fairly.</span>`,
  lupdates: () => { const L = D.online_learning; return `<h5>One update = one hourly call whose answer is known</h5><b>13,432</b> came from replaying 2.4 years of history (Jul 2024 → Nov 2026), predicting first and learning after, hour by hour.<br><b>${L.updates_live}</b> more happened live on this server.
    <span class="tnote">Last update: ${esc(L.last_update || 'not yet on this server')}</span>`; },
  lboard: () => { const lv = D.online_learning.live || {}; return `<h5>Live scoreboard</h5>Hit rate on calls made <b>after</b> this server started, each predicted before its answer existed.
    <span class="tf">learned weights: ${lv.n ? `${lv.learned_hits}/${lv.n}` : '–'}
fixed weights:   ${lv.n ? `${lv.engine_hits}/${lv.n}` : '–'}</span>Backtest on the lockbox (Apr–Nov 2026): 50.9% vs 50.9%.
    <span class="tnote">With only ${lv.n || 'a few'} calls this is mostly luck: with 20 calls, anything from 30% to 70% is normal for a coin flip.</span>`; },
  lweights: () => `<h5>Proof that it learned</h5>Each line is one pillar's weight. They started from the hand weights (25 · 20 · 15 · 15 · 10 · 10) and drifted:
    <ul><li>US dollar and stocks <b>up</b></li><li>Trend and VWAP <b>down</b></li></ul>Why: no pillar beats 50%, and the dollar and stock pillars usually vote near 0, so their mistakes cost less. It learned to be <b>less confident</b>.
    <span class="tnote">Nearly flat lines = the weights have settled.</span>`,
  limprove: () => { const lv = D.online_learning.live || {}; return `<h5>Honest answer: maybe never, and that is a result</h5>
    <ul><li><b>Accuracy:</b> it improves only if some pillar starts predicting better than 50%. Then its weight will grow automatically.</li>
    <li><b>Proof:</b> needs ~764 independent calls (one per 4 hours) to tell 54.5% from a coin flip with 80% confidence. Running 24/7 that is about <b>6 months</b>; now ${lv.nonoverlap ?? 0} done.</li>
    <li>It only counts while the dashboard server is running. On a laptop that's opened now and then, it takes much longer.</li></ul>
    <span class="tnote">More data cannot make a coin flip predictable. It can only tell us, with confidence, whether it is one.</span>`; },
};

(function tips() {
  const tip = $('tip');
  let current = null;
  const show = el => {
    const fn = TIPS[el.dataset.tip];
    if (!fn || !D) return;
    try { tip.innerHTML = fn(); } catch (_) { return; }
    tip.hidden = false;
    current = el;
    const r = el.getBoundingClientRect(), tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = Math.min(Math.max(16, r.left + r.width / 2 - tw / 2), innerWidth - tw - 16);
    let top = r.bottom + 10;
    if (top + th > innerHeight - 12) top = Math.max(12, r.top - th - 10);
    tip.style.left = `${left}px`; tip.style.top = `${top}px`;
  };
  const hide = () => { tip.hidden = true; current = null; };
  document.addEventListener('mouseover', e => { const el = e.target.closest('[data-tip]'); if (el && el !== current) show(el); if (!el && current) hide(); });
  document.addEventListener('focusin', e => { const el = e.target.closest('[data-tip]'); if (el) show(el); });
  document.addEventListener('focusout', hide);
  document.addEventListener('click', e => { const el = e.target.closest('[data-tip]'); if (el && matchMedia('(hover: none)').matches) { el === current ? hide() : show(el); } });
  addEventListener('scroll', hide, { passive: true });
})();

/* ------------------------------------------------------------------ */
/* Info drawer                                                          */
/* ------------------------------------------------------------------ */
const titleOf = k => { const e = resolve(k); return e ? e.title.replace(/^Pillar: /, '').split(':')[0] : k; };
function resolve(k) { let e = INFO[k]; while (e && e.alias) e = INFO[e.alias]; return e; }

let currentKey = null;
function openInfo(key) {
  const e = resolve(key);
  if (!e) return;
  currentKey = key;
  const si = stageIndex(e.stage), st = STAGES[si];
  $('dStage').textContent = `Step ${si + 1} of ${STAGES.length} · ${st.name}`;
  $('dTitle').textContent = e.title;
  $('dMap').innerHTML = STAGES.map((s, i) =>
    `<button class="${i === si ? 'on' : i < si ? 'past' : ''}" data-info="${s.key}" title="${i + 1}. ${s.name}" aria-label="${s.name}"></button>`).join('');
  const label = document.querySelector('.d-map-label') || Object.assign(document.createElement('div'), { className: 'd-map-label' });
  label.innerHTML = `<span>Data</span><span>you are here: <b>${st.name}</b></span><span>Evaluation</span>`;
  $('dMap').after(label);

  let h = `<h4>What it is</h4><p>${e.what}</p>`;
  if (e.how) h += `<h4>How it is calculated</h4>${safe(e.how)}`;
  if (e.live && D) h += `<h4>Right now</h4><div class="nowbox">${safe(e.live)}</div>`;
  if ((e.from && e.from.length) || (e.to && e.to.length)) {
    h += '<h4>How it connects</h4><div class="links">';
    if (e.from && e.from.length) h += `<span class="dir">Gets its input from</span>${e.from.map(k => `<button data-info="${k}">← ${titleOf(k)}</button>`).join('')}`;
    if (e.to && e.to.length) h += `<span class="dir">Feeds into</span>${e.to.map(k => `<button data-info="${k}">${titleOf(k)} →</button>`).join('')}`;
    h += '</div>';
  }
  if (e.check) { const c = (() => { try { return e.check(); } catch (_) { return null; } })(); if (c) h += `<h4>Does it work?</h4><div class="check ${c.tone}">${c.html}</div>`; }
  if (e.why) h += `<h4>Why it matters for the whole</h4><p>${e.why}</p>`;
  $('dBody').innerHTML = h;

  const prev = STAGES[si - 1], next = STAGES[si + 1];
  const isStage = STAGES.some(s => s.key === key || (INFO[s.key].alias && INFO[s.key].alias === key));
  $('dNav').innerHTML =
    (isStage ? (prev ? `<button class="ghost" data-info="${prev.key}">← ${prev.name}</button>` : '<span></span>')
      : `<button class="ghost" data-info="${st.key}">← ${st.name} overview</button>`) +
    (next ? `<button class="ghost" data-info="${next.key}">Next: ${next.name} →</button>` : '<span></span>');

  // highlight where this lives on the page
  document.querySelectorAll('.lit').forEach(n => n.classList.remove('lit'));
  document.querySelectorAll('.step button').forEach(b => b.classList.toggle('on', b.dataset.step === e.stage));
  document.querySelectorAll(`[data-stage="${e.stage}"]`).forEach(n => n.classList.add('lit'));

  $('drawer').classList.add('open');
  $('drawer').setAttribute('aria-hidden', 'false');
  $('scrim').hidden = false;
  $('drawer').scrollTop = 0;
}
function closeInfo() {
  $('drawer').classList.remove('open');
  $('drawer').setAttribute('aria-hidden', 'true');
  $('scrim').hidden = true;
  document.querySelectorAll('.lit').forEach(n => n.classList.remove('lit'));
  document.querySelectorAll('.step button.on').forEach(b => b.classList.remove('on'));
  currentKey = null;
}

document.addEventListener('click', ev => {
  const b = ev.target.closest('[data-info]');
  if (b) { ev.preventDefault(); openInfo(b.dataset.info); }
});
$('scrim').addEventListener('click', closeInfo);
$('dClose').addEventListener('click', closeInfo);
document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape') closeInfo();
  if (!currentKey) return;
  const e = resolve(currentKey), si = stageIndex(e.stage);
  if (ev.key === 'ArrowRight' && STAGES[si + 1]) openInfo(STAGES[si + 1].key);
  if (ev.key === 'ArrowLeft' && STAGES[si - 1]) openInfo(STAGES[si - 1].key);
});
$('tourBtn').addEventListener('click', () => openInfo('data'));

/* theme */
(function theme() {
  let saved = null;
  try { saved = localStorage.getItem('ps-theme'); } catch (_) { }
  if (saved) document.documentElement.dataset.theme = saved;
  $('themeBtn').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    const next = dark ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('ps-theme', next); } catch (_) { }
  });
})();

/* ------------------------------------------------------------------ */
/* Data loading                                                         */
/* ------------------------------------------------------------------ */
function status(kind, text) {
  $('status').className = 'status ' + kind;
  $('statusText').textContent = text;
}

async function load(force = false) {
  $('refreshIcon').classList.add('spin');
  try {
    const r = await fetch(force ? '/api/refresh' : '/api/data', { method: force ? 'POST' : 'GET' });
    const j = await r.json();
    if (!j.success) throw new Error(j.error || 'failed');
    D = j.data;
    render();
    status('live', `Live · ${D.timestamp.slice(11, 16)} UTC`);
    if (currentKey) openInfo(currentKey);
  } catch (e) {
    status('err', D ? 'Offline · showing last data' : 'Could not load market data');
  } finally {
    $('refreshIcon').classList.remove('spin');
  }
}

async function loadMetrics() {
  try {
    const r = await fetch('/api/metrics');
    const j = await r.json();
    if (j.success) { M = j.data; renderEval(); renderReality(); }
  } catch (_) { }
}

renderPipeline();
$('refreshBtn').addEventListener('click', () => load(true));
loadMetrics();
load();
setInterval(load, 5 * 60 * 1000);
