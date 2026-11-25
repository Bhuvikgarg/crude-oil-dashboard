/* /ml — the PetroSync pipeline as a live flowchart.
 * Hover (or tap) a box: its whole path lights up (everything it depends on and everything that
 * depends on it) and the panel explains the backend code behind it, with live values.
 * "Run a live request" triggers a real refresh and replays it through the boxes using the
 * server's own stage timings. "Replay the training run" walks the offline evaluation loop. */

let D = null, M = null, playing = false;
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const sgn = (v, d = 1) => { if (v == null || isNaN(v)) return '--'; const a = Math.abs(v).toFixed(d); return Number(a) === 0 ? a : `${v > 0 ? '+' : '−'}${a}`; };
const usd = (v, d = 2) => v == null ? '--' : `$${Number(v).toFixed(d)}`;
const pct = (v, d = 1) => v == null ? '--' : `${(v * 100).toFixed(d)}%`;
const ms = v => v == null ? '' : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(v < 10 ? 1 : 0)} ms`;
const T = () => (D && D.pipeline && D.pipeline.timings_ms) || {};
const safe = fn => { try { return fn(); } catch (_) { return ''; } };

/* ------------------------------------------------------------------ */
/* Layout                                                               */
/* ------------------------------------------------------------------ */
const C = { c1: 95, c2: 265, c3: 445, c4: 650, c5: 840, c6: 1050 };
const W = { c1: 150, c2: 150, c3: 166, c4: 172, c5: 150, c6: 170 };
const H = 62;

const NODES = [
  // live band
  { id: 'yahoo', k: 'data', x: C.c1, y: 260, w: W.c1, t1: 'Yahoo Finance', t2: 'yfinance · 9 markets', live: () => `${D.pipeline.bars.hourly} hourly bars` },
  { id: 'fetch', k: 'data', x: C.c2, y: 260, w: W.c2, t1: 'Download & clean', t2: 'fetch_market_basket()', live: () => ms(T().fetch_ms) },
  { id: 'feat', k: 'group', x: C.c3, y: 261, w: 190, h: 338, t1: 'Feature engineering' },
  { id: 'tech', k: 'rule', x: C.c3, y: 160, w: W.c3, t1: 'Indicators', t2: 'EMA · RSI · VWAP · ATR', live: () => `RSI ${D.wti.rsi.toFixed(0)} · ATR ${usd(D.wti.atr)}` },
  { id: 'mtf', k: 'rule', x: C.c3, y: 235, w: W.c3, t1: 'Timeframes', t2: '15m · 1h · daily trend', live: () => `trend score ${D.mtf_confluence.overall_score}` },
  { id: 'prof', k: 'rule', x: C.c3, y: 310, w: W.c3, t1: 'Volume profile', t2: 'POC · value area', live: () => `POC ${usd(D.levels.poc)}` },
  { id: 'inter', k: 'rule', x: C.c3, y: 385, w: W.c3, t1: 'Intermarket', t2: 'crack · Brent · DXY · S&P', live: () => `crack ${usd(D.macro_radar.crack_margin, 1)}` },
  { id: 'pillars', k: 'rule', x: C.c4, y: 150, w: W.c4, t1: '7 pillars → score', t2: 'weighted vote, −100…+100', live: () => `score ${sgn(D.sentiment.score)}` },
  { id: 'platt', k: 'ml', x: C.c4, y: 240, w: W.c4, t1: 'Platt calibration', t2: 'P(price up in 4 h)', live: () => `${D.sentiment.calibrated_p_up_4h}% up` },
  { id: 'hedge', k: 'ml', x: C.c4, y: 330, w: W.c4, t1: 'Hedge self-learning', t2: 're-weights pillars hourly', live: () => `${sgn(D.online_learning.learned_score)} ${D.online_learning.learned_call}` },
  { id: 'har', k: 'ml', x: C.c4, y: 420, w: W.c4, t1: 'HAR volatility', t2: 'expected 4-hour move', live: () => `±${usd(D.volatility_forecast.expected_move_usd)} · ${D.volatility_forecast.regime}` },
  { id: 'plan', k: 'rule', x: C.c5, y: 150, w: W.c5, t1: 'Trade plan', t2: 'setup · entry · stop', live: () => `${D.trade_setup.bias} · ${D.trade_setup.win_probability}% won` },
  { id: 'api', k: 'serve', x: C.c5, y: 300, w: W.c5, t1: 'Flask /api/data', t2: 'cache 45 s · hourly', live: () => `cache age ${D.pipeline.cache_age_s ?? 0} s` },
  { id: 'mapi', k: 'serve', x: C.c5, y: 420, w: W.c5, t1: '/api/metrics', t2: 'backtest results', live: () => M ? `${M.meta.n_signals.toLocaleString()} decisions` : '' },
  { id: 'ui_dash', k: 'ui', x: C.c6, y: 200, w: W.c6, t1: 'Dashboard  /', t2: 'minimal view + info', live: () => 'every number explained' },
  { id: 'ui_ml', k: 'ui', x: C.c6, y: 300, w: W.c6, t1: 'Pipeline  /ml', t2: 'this page', live: () => 'you are here' },
  { id: 'ui_classic', k: 'ui', x: C.c6, y: 400, w: W.c6, t1: 'Classic  /classic', t2: 'original dashboard', live: () => 'kept as backup' },
  // offline band
  { id: 'hist', k: 'data', x: C.c1, y: 625, w: W.c1, t1: '2.4 years history', t2: 'data_loader.py · cached', live: () => 'Jul 2024 → Nov 2026' },
  { id: 'replay', k: 'rule', x: C.c2, y: 625, w: W.c2, t1: 'Replay engine', t2: 'backtest.replay()', live: () => M ? `${M.meta.n_signals.toLocaleString()} decisions` : '13,436 decisions' },
  { id: 'wf', k: 'ml', x: C.c3, y: 580, w: W.c3, t1: 'Walk-forward ML', t2: 'logistic reg · boosting', live: () => M ? `best AUC ${bestAuc().toFixed(3)}` : 'AUC 0.519' },
  { id: 'trades', k: 'rule', x: C.c3, y: 670, w: W.c3, t1: 'Trade simulation', t2: 'simulate_trades()', live: () => M ? `${M.trades.filled_trades.toLocaleString()} trades` : '2,114 trades' },
  { id: 'metrics', k: 'rule', x: C.c4, y: 580, w: W.c4, t1: 'Metrics & tests', t2: 'AUC · Brier · PT test', live: () => M ? `hit rate ${pct(M.signals.h4.sentiment_7pillar.hit_rate)}` : 'hit rate 49.6%' },
  { id: 'fit', k: 'ml', x: C.c4, y: 670, w: W.c4, t1: 'Fit live models', t2: 'Platt · HAR · Hedge', live: () => '4 model files' },
  { id: 'reports', k: 'data', x: C.c5, y: 625, w: W.c5, t1: 'reports/', t2: 'results + model files', live: () => 'JSON + CSV' },
  { id: 'forward', k: 'serve', x: C.c6, y: 625, w: W.c6, t1: 'Forward test', t2: 'GitHub Actions, hourly', live: () => 'models frozen, SHA-256' },
];
const NODE = Object.fromEntries(NODES.map(n => [n.id, n]));
const L = n => n.x - n.w / 2, R = n => n.x + n.w / 2, TOP = n => n.y - (n.h || H) / 2, BOT = n => n.y + (n.h || H) / 2;

// Drawn edges: [from, to, path, options]
const EDGES = [
  ['yahoo', 'fetch', 'M170 260H190'],
  ['fetch', 'feat', 'M340 260H350'],
  ['feat', 'pillars', 'M540 150H564'],
  ['feat', 'plan', 'M445 92V72H840V119', { label: ['setup rules use ATR, VWAP, RSI, trend, value area', 470, 66] }],
  ['pillars', 'platt', 'M650 181V209'],
  ['pillars', 'hedge', 'M564 168H552V330H564'],
  ['fetch', 'har', 'M265 291V470H552V420H564', { label: ['raw hourly returns + OVX', 290, 462] }],
  ['pillars', 'api', 'M736 150H750V300H765'],
  ['platt', 'api', 'M736 240H750V300H765'],
  ['hedge', 'api', 'M736 330H750V300H765'],
  ['har', 'api', 'M736 420H750V300H765'],
  ['plan', 'api', 'M840 181V269'],
  ['api', 'ui_dash', 'M915 300H940V200H965'],
  ['api', 'ui_ml', 'M915 300H965'],
  ['api', 'ui_classic', 'M915 300H940V400H965'],
  ['mapi', 'ui_dash', 'M915 420H948V212H965'],
  ['mapi', 'ui_classic', 'M915 420H948V408H965'],
  ['hist', 'replay', 'M170 625H190'],
  ['replay', 'wf', 'M340 625H350V580H362'],
  ['replay', 'trades', 'M340 625H350V670H362'],
  ['wf', 'metrics', 'M528 580H564'],
  ['trades', 'metrics', 'M528 670H546V594H564'],
  ['trades', 'fit', 'M528 670H564'],
  ['metrics', 'reports', 'M736 580H750V615H765'],
  ['fit', 'reports', 'M736 670H750V635H765'],
  ['reports', 'forward', 'M915 625H965'],
  ['reports', 'mapi', 'M860 594V451', { label: ['results', 866, 520] }],
  ['reports', 'har', 'M815 594V500H660V451', { dashed: true, label: ['model files, loaded at start ↑', 668, 486] }],
];
// Logical links used for lighting (not drawn): what really feeds what in the code
const LOGICAL = [
  ['feat', 'tech'], ['feat', 'mtf'], ['feat', 'prof'], ['feat', 'inter'],
  ['tech', 'pillars'], ['mtf', 'pillars'], ['inter', 'pillars'],
  ['tech', 'plan'], ['mtf', 'plan'], ['prof', 'plan'],
  ['fetch', 'hedge'],                                   // learns from the realised price 4 h later
  ['reports', 'platt'], ['reports', 'hedge'], ['reports', 'plan'],   // model files / measured win rates
  ['replay', 'fit'], ['hist', 'fit'],
];

/* ------------------------------------------------------------------ */
/* Backend explanations                                                 */
/* ------------------------------------------------------------------ */
const KIND = { data: 'Data', rule: 'Rule logic', ml: 'ML model', serve: 'Serving', ui: 'Screen', group: 'Stage' };

const INFO = {
  yahoo: {
    code: 'yfinance (free Yahoo Finance API)',
    what: 'The only data source. Prices for 9 markets, downloaded fresh on every refresh. No API key; the hostnames must be reachable from the server.',
    io: ['Internet request', 'OHLCV bars: open, high, low, close, volume'],
    live: () => `${D.pipeline.bars.hourly} hourly bars × ${D.pipeline.bars.tickers} markets, ${D.pipeline.bars.m15} 15-minute bars, ${D.pipeline.bars.daily} daily bars.`,
  },
  fetch: {
    code: 'engine.py · PetroSyncEngine.fetch_market_basket()\nengine.py · refresh()',
    what: 'Downloads 5 days of hourly bars for CL, BZ, DXY, RB, HO, GC, OVX, ES and XLE in one call, plus 3 days of 15-minute and 30 days of daily WTI bars. Timestamps are converted to UTC and empty bars are dropped per market.',
    io: ['Yahoo Finance', 'Three pandas DataFrames'],
    live: () => `Took <b>${ms(T().fetch_ms)}</b> on the last refresh, by far the slowest stage (it waits for the network). Refreshed every hour by a background thread in app.py, or on demand.`,
  },
  feat: {
    code: 'engine.py · compute_snapshot()',
    what: 'Turns raw prices into meaningful numbers. Pure pandas/numpy, no learning. The same function runs live and in the backtest, so the evaluation tests exactly this code.',
    io: ['Price DataFrames', 'Indicators, trend states, levels, intermarket values'],
    live: () => `The whole snapshot (features + pillars + trade plan) took <b>${ms(T().snapshot_ms)}</b>.`,
  },
  tech: {
    code: 'engine.py · calculate_technical_series()',
    what: 'EMA 9/20/50 (trend), RSI 14 (momentum), VWAP with ±1σ/±2σ bands (fair value), ATR 14 (typical bar size, used for stops).',
    io: ['WTI hourly bars', 'Indicator columns'],
    live: () => `Price ${usd(D.wti.price)}, EMA20 ${usd(D.wti.ema20)}, VWAP ${usd(D.wti.vwap)}, RSI ${D.wti.rsi}, ATR ${usd(D.wti.atr)}.`,
  },
  mtf: {
    code: 'engine.py · calculate_timeframe_state()  × 3',
    what: 'The same trend check on 15-minute, 1-hour and daily bars: start at 50, ±30 for EMA alignment, ±20 for RSI. Combined 25% / 45% / 30%.',
    io: ['15m, 1h, daily bars', 'Trend score 0–100'],
    live: () => `${D.mtf_confluence.timeframes.map(t => `${esc(t.timeframe.split(' ')[0])} ${t.score}`).join(' · ')} → <b>${D.mtf_confluence.overall_score}</b>.`,
  },
  prof: {
    code: 'engine.py · calculate_volume_profile()',
    what: 'Groups the last 24 hours of volume into price buckets: POC is the busiest price, and the value area (VAL–VAH) holds 70% of the volume.',
    io: ['Last 24 hourly bars', 'POC, VAL, VAH'],
    live: () => `POC ${usd(D.levels.poc)}, value area ${usd(D.levels.val)}–${usd(D.levels.vah)}.`,
  },
  inter: {
    code: 'engine.py · compute_snapshot()',
    what: '3-2-1 crack margin from gasoline and heating oil, the Brent–WTI spread, hourly % change of the dollar index and S&P futures, the gold/oil ratio and the OVX level.',
    io: ['RB, HO, BZ, DXY, ES, GC, OVX', 'Intermarket values'],
    live: () => `Crack ${usd(D.macro_radar.crack_margin)}, Brent–WTI ${usd(D.brent.spread_to_wti)}, OVX ${D.macro_radar.ovx_price}.`,
  },
  pillars: {
    code: 'engine.py · compute_advanced_sentiment()',
    what: 'Seven pillars each vote from −100 to +100; the score is their weighted sum with hand-set weights 25/20/15/15/10/10/5. This is rule logic, not ML; the ML parts below test and correct it.',
    io: ['Indicators, trend score, intermarket', 'Score −100…+100 and 6 pillar votes'],
    live: () => `Score <b>${sgn(D.sentiment.score)}</b> (“${esc(D.sentiment.label)}”). Votes: ${Object.entries(D.sentiment.breakdown).map(([k, v]) => `${k.split('_')[0]} ${sgn(v, 0)}`).join(', ')}.`,
    check: 'Backtest: the score called the 4-hour direction right 49.6% of the time over 13,436 hours.',
  },
  platt: {
    code: 'engine.py · compute_advanced_sentiment()\nloads reports/engine_calibration.json\nmade by evaluation/improve.py',
    what: 'Turns the score into an honest probability. A logistic regression was fitted offline on 13,432 past hours (score → did the price rise 4 h later?). Live it is one line of maths.',
    ml: 'P(up) = 1 / (1 + e^−(a·score + b))\na = −0.00076, b = 0.052',
    io: ['Score', 'P(price higher in 4 h)'],
    live: () => `Raw claim ${D.sentiment.bullish_pct}% → calibrated <b>${D.sentiment.calibrated_p_up_4h}%</b>.`,
    check: 'Brier skill −0.079 → −0.001 on the lockbox: it removed the overconfidence.',
  },
  hedge: {
    code: 'online_learner.py · LiveHedge.observe() / summary()\nstate: state/online_state.json\nstart: reports/online_state_init.json',
    what: 'Online learning. Each pillar is an expert; every hour it predicts first, then, once the price 4 h later is known, multiplies each pillar’s weight by e^(−η·loss). The state is saved to disk, so it keeps learning across restarts.',
    ml: 'loss = (vote − outcome)² / 4\nw ← w · e^(−0.01 · loss), normalise\nfixed share α = 0.01',
    io: ['Pillar votes + realised price', 'Learned weights + learned score'],
    live: () => { const l = D.online_learning, lv = l.live || {}; return `Learned score <b>${sgn(l.learned_score)}</b> (${l.learned_call}). ${l.updates_total.toLocaleString()} updates, ${l.updates_live} live; ${l.pending} waiting. Live scoreboard: ${lv.n ? `${lv.learned_hits}/${lv.n} vs ${lv.engine_hits}/${lv.n}` : 'starting'}. Step took ${ms(T().learning_ms)}.`; },
    check: 'Lockbox: 50.9% hit, same as the hand weights. It showed which hand weights the data does not support.',
  },
  har: {
    code: 'engine.py · forecast_volatility()\nloads reports/vol_model.json\nmade by evaluation/volatility.py',
    what: 'The model that works. HAR (Corsi 2009): a ridge regression of future log volatility on past volatility over 1, 4, 24 and 96 hours, plus log OVX and hour of day. Inputs are rebuilt from the raw bars exactly as in training.',
    ml: 'log vol(next 4h) = b0 + b1·log vol(1h) + b2·log vol(4h)\n  + b3·log vol(24h) + b4·log vol(96h)\n  + b5·log OVX + b6·sin(hour) + b7·cos(hour)\nridge α = 10, standardised inputs',
    io: ['Raw hourly returns + OVX', 'Expected move ±$, range, regime'],
    live: () => `Expected 4-hour move <b>±${usd(D.volatility_forecast.expected_move_usd)}</b> (${D.volatility_forecast.expected_move_pct}%), range ${usd(D.volatility_forecast.range_low)}–${usd(D.volatility_forecast.range_high)}, regime ${D.volatility_forecast.regime}. Took ${ms(T().volatility_ms)}.`,
    check: 'Lockbox: R² 0.22, calm vs stormy right 68.7%, AUC 0.70.',
  },
  plan: {
    code: 'engine.py · generate_trade_setup()\nmeasured win rates: reports/playbook_measured.json\nmade by evaluation/measure_playbook.py',
    what: 'Picks one of five setups by rules (RSI extremes, trend score, value area) and places entry, stop and targets with ATR, VWAP and the volume profile. The win rate shown is now measured on 2,114 past trades, not typed in.',
    io: ['Indicators, trend, value area', 'Setup, entry, stop, targets, measured win rate'],
    live: () => { const t = D.trade_setup; return `<b>${esc(t.action)}</b> (${t.bias}). Entry ${t.entry_min === t.entry_max ? usd(t.entry_min) : `${usd(t.entry_min)}–${usd(t.entry_max)}`}, stop ${usd(t.stop_loss)}, target ${usd(t.target1)}, ${esc(t.risk_reward)}. Measured win rate ${t.win_probability}% (old claim ${t.claimed_win_probability}%).`; },
    check: 'Profit factor 0.84 over 2,114 simulated trades: the rules as written lose money after costs.',
  },
  api: {
    code: 'app.py · /api/data → engine.refresh()\nbackground_hourly_updater() thread',
    what: 'Flask web server. A background thread refreshes everything every hour; requests in between get the cached snapshot (45 s cache) so Yahoo is not hammered. /api/refresh forces a new run.',
    io: ['Snapshot from the engine', 'JSON to the browser'],
    live: () => `Computed ${esc(D.pipeline.computed_utc)}; served from cache ${D.pipeline.cache_age_s ?? 0} s old. Stage timings: fetch ${ms(T().fetch_ms)}, snapshot ${ms(T().snapshot_ms)}, volatility ${ms(T().volatility_ms)}, learning ${ms(T().learning_ms)}.`,
  },
  mapi: {
    code: 'app.py · /api/metrics\nserves reports/evaluation_results.json',
    what: 'Serves the offline evaluation results to the dashboards, for the “Does it actually work?” section and the reality check.',
    io: ['reports/evaluation_results.json', 'JSON to the browser'],
    live: () => M ? `${M.meta.n_signals.toLocaleString()} hourly decisions, ${M.meta.sample_start.slice(0, 10)} → ${M.meta.sample_end.slice(0, 10)}, generated ${esc(M.meta.generated_utc)}.` : 'Not loaded.',
  },
  ui_dash: {
    code: 'templates/minimal.html · static/minimal.js',
    what: 'The main screen. Fetches /api/data every 5 minutes and /api/metrics once. Every number has an (i) explanation and the self-learning card explains itself on hover.',
    io: ['/api/data, /api/metrics', 'What you see'],
    live: () => 'Open it at <a href="/">/</a>.',
  },
  ui_ml: {
    code: 'templates/ml.html · static/ml.js',
    what: 'This page. The same live JSON drawn as the pipeline, with the server’s own stage timings.',
    io: ['/api/data, /api/metrics, /api/refresh', 'This flowchart'],
    live: () => 'You are here.',
  },
  ui_classic: {
    code: 'templates/index.html · static/dashboard.js',
    what: 'The original dashboard, kept unchanged as a fallback, plus the backtest reality-check card.',
    io: ['/api/data, /api/metrics', 'Classic screen'],
    live: () => 'Open it at <a href="/classic">/classic</a>.',
  },
  hist: {
    code: 'evaluation/data_loader.py',
    what: 'Downloads and caches 2.4 years of the same 9 markets so the backtest is repeatable. The cache folder is git-ignored.',
    io: ['Yahoo Finance (once)', 'Cached history'],
    live: () => 'July 2024 → November 2026.',
  },
  replay: {
    code: 'evaluation/backtest.py · replay()',
    what: 'Runs the real compute_snapshot() for every past hour, giving it only the data that existed at that moment. No future data can leak in. Then attaches what the price actually did 1, 4 and 24 hours later.',
    io: ['Cached history', '13,436 point-in-time decisions + outcomes'],
    live: () => M ? `${M.meta.n_signals.toLocaleString()} decisions, ${M.meta.sample_start.slice(0, 10)} → ${M.meta.sample_end.slice(0, 10)}.` : '',
  },
  wf: {
    code: 'evaluation/ml_models.py · build_features(), walk_forward()',
    what: 'Can ML predict direction better than the rules? 31 features, logistic regression (C = 0.1) and gradient boosting (depth 3, 200 trees). Trained only on the past: start with 2,000 hours, retrain every 250, with a gap equal to the horizon.',
    ml: 'walk-forward: train [0 … t−gap] → predict [t … t+250]\nrepeat to the end; lockbox Apr–Nov 2026 untouched',
    io: ['Replay decisions + outcomes', 'Out-of-sample predictions'],
    live: () => M ? `4-hour AUC: logistic ${M.ml.h4.logistic_regression.probability.roc_auc.toFixed(3)}, boosting ${M.ml.h4.gradient_boosting.probability.roc_auc.toFixed(3)}.` : '',
    check: 'Barely above random (0.5). Direction is not predictable at the hourly scale.',
  },
  trades: {
    code: 'evaluation/backtest.py · simulate_trades()',
    what: 'Every trade plan is simulated bar by bar: fill only if price reaches the entry, then Target 1 or the stop, whichever comes first (stop wins ties). 2 ticks of cost, one trade at a time.',
    io: ['Replay decisions', '2,114 trades with results'],
    live: () => M ? `${M.trades.filled_trades.toLocaleString()} trades, Target 1 hit ${pct(M.trades.target1_hit_rate)}, profit factor ${M.trades.profit_factor.toFixed(2)}.` : '',
  },
  metrics: {
    code: 'evaluation/metrics.py · run_evaluation.py',
    what: 'Scores everything against baselines: hit rate with binomial and Pesaran–Timmermann tests, ROC-AUC, MCC, Brier skill, calibration error, Theil U2, MASE, profit factor.',
    io: ['Predictions, trades', 'evaluation_results.json, EVALUATION_RESULTS.md'],
    live: () => M ? `Direction hit rate ${pct(M.signals.h4.sentiment_7pillar.hit_rate)}; Brier skill ${M.signals.h4.sentiment_7pillar.probability.brier_skill_score.toFixed(3)}.` : '',
  },
  fit: {
    code: 'evaluation/improve.py · volatility.py · online_learning.py · measure_playbook.py',
    what: 'Fits the models that run live and writes them as small JSON files: Platt coefficients, HAR coefficients and scaler, the Hedge starting weights after 13,432 updates, and the measured win rate of each setup.',
    io: ['Replay + trades', 'engine_calibration.json, vol_model.json, online_state_init.json, playbook_measured.json'],
    live: () => 'Parameters chosen on Jul 2024–Mar 2026, tested once on the lockbox.',
  },
  reports: {
    code: 'reports/',
    what: 'The hand-off between the offline and the live halves. The server only reads these files at start: no training happens in the live loop except the Hedge update.',
    io: ['Offline scripts', 'Live models, /api/metrics, the report files'],
    live: () => 'evaluation_results.json, engine_calibration.json, vol_model.json, online_state_init.json, playbook_measured.json, trades.csv, predictions_vs_actual.csv.',
  },
  forward: {
    code: '.github/workflows/forward-test.yml\nevaluation/forward_test.py\nforward_test/manifest.json',
    what: 'Pre-registered test on truly new data. Models were frozen on 18 Nov 2026 with SHA-256 fingerprints; every hour GitHub Actions records the prediction before the outcome exists and commits it to the forward-test-log branch.',
    io: ['Frozen models + new hourly bars', 'Timestamped prediction log'],
    live: () => 'Runs once the branch is merged into main (GitHub only schedules jobs on the default branch). About 764 independent calls are needed to prove a 54.5% edge.',
  },
};

/* ------------------------------------------------------------------ */
/* Drawing                                                              */
/* ------------------------------------------------------------------ */
function el(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

function draw() {
  const svg = $('flow');
  svg.innerHTML = '';
  const defs = el('defs');
  const mk = el('marker', { id: 'arr', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' });
  mk.append(el('path', { d: 'M0 0L10 5L0 10z', fill: 'var(--ink-3)' }));
  defs.append(mk);
  svg.append(defs);

  svg.append(el('rect', { class: 'band', x: 8, y: 34, width: 1164, height: 456, rx: 14 }));
  svg.append(el('text', { class: 'band-label', x: 24, y: 56 }, 'Live · every hour'));
  svg.append(el('rect', { class: 'band', x: 8, y: 512, width: 1164, height: 240, rx: 14 }));
  svg.append(el('text', { class: 'band-label', x: 24, y: 534 }, 'Offline · training and evaluation'));
  svg.append(el('text', { class: 'band-note', x: 24, y: 744 }, 'Run once with python -m evaluation.…  Its files are what the live half loads.'));

  const eg = el('g');
  EDGES.forEach(([a, b, d, o = {}], i) => {
    const p = el('path', { d, class: 'edge' + (o.dashed ? ' dashed' : ''), 'marker-end': 'url(#arr)', 'data-a': a, 'data-b': b });
    eg.append(p);
    if (o.label) eg.append(el('text', { class: 'edge-label', x: o.label[1], y: o.label[2], 'data-a': a, 'data-b': b }, o.label[0]));
  });

  const ng = el('g');
  // group first so it sits behind its children
  NODES.slice().sort((a, b) => (a.k === 'group' ? -1 : 0) - (b.k === 'group' ? -1 : 0)).forEach(n => {
    const h = n.h || H;
    const g = el('g', { class: `node k-${n.k}`, 'data-id': n.id, tabindex: 0, role: 'button', 'aria-label': n.t1 });
    g.append(el('rect', { class: 'box', x: L(n), y: TOP(n), width: n.w, height: h, rx: 10 }));
    if (n.k === 'group') {
      g.append(el('text', { class: 't1', x: n.x, y: TOP(n) + 20, 'text-anchor': 'middle' }, n.t1));
    } else {
      g.append(el('text', { class: 't1', x: n.x, y: TOP(n) + 20, 'text-anchor': 'middle' }, n.t1));
      g.append(el('text', { class: 't2', x: n.x, y: TOP(n) + 36, 'text-anchor': 'middle' }, n.t2));
      g.append(el('text', { class: 'live', x: n.x, y: TOP(n) + 53, 'text-anchor': 'middle', 'data-live': n.id }, ''));
      if (n.k === 'ml') g.append(el('text', { class: 'badge', x: R(n) - 8, y: TOP(n) + 13, 'text-anchor': 'end' }, 'ML'));
    }
    ng.append(g);
  });
  svg.append(eg, ng);
}

function updateLive() {
  if (!D) return;
  NODES.forEach(n => {
    const t = document.querySelector(`[data-live="${n.id}"]`);
    if (t && n.live) t.textContent = safe(n.live);
  });
}

/* ------------------------------------------------------------------ */
/* Lighting the path                                                    */
/* ------------------------------------------------------------------ */
const LINKS = [...EDGES.map(e => [e[0], e[1]]), ...LOGICAL];
const up = {}, down = {};
LINKS.forEach(([a, b]) => { (down[a] ||= []).push(b); (up[b] ||= []).push(a); });
function walk(start, graph) {
  const seen = new Set(), stack = [start];
  while (stack.length) { const x = stack.pop(); (graph[x] || []).forEach(y => { if (!seen.has(y)) { seen.add(y); stack.push(y); } }); }
  return seen;
}
function light(id) {
  const svg = $('flow');
  const on = new Set([id, ...walk(id, up), ...walk(id, down)]);
  svg.classList.add('focus');
  svg.querySelectorAll('.node').forEach(g => { g.classList.toggle('lit', on.has(g.dataset.id)); g.classList.toggle('me', g.dataset.id === id); });
  svg.querySelectorAll('[data-a]').forEach(p => p.classList.toggle('lit', on.has(p.dataset.a) && on.has(p.dataset.b)));
}
function unlight() {
  const svg = $('flow');
  svg.classList.remove('focus');
  svg.querySelectorAll('.lit, .me').forEach(e => e.classList.remove('lit', 'me'));
}

/* ------------------------------------------------------------------ */
/* Panel                                                                */
/* ------------------------------------------------------------------ */
let pinned = null;
function show(id) {
  const n = NODE[id], i = INFO[id];
  if (!n || !i) return;
  $('pEmpty').hidden = true;
  const body = $('pBody');
  body.hidden = false;
  const ups = [...new Set(LINKS.filter(l => l[1] === id).map(l => l[0]))].filter(x => NODE[x]);
  const downs = [...new Set(LINKS.filter(l => l[0] === id).map(l => l[1]))].filter(x => NODE[x]);
  body.innerHTML = `
    <span class="p-kind k-${n.k}">${KIND[n.k]}</span>
    <h3>${esc(n.t1.replace(/\s+/g, ' '))}</h3>
    <p>${i.what}</p>
    <div class="p-sec"><h4>Where in the code</h4><code class="p-code">${esc(i.code)}</code></div>
    ${i.ml ? `<div class="p-sec"><h4>The maths</h4><code class="p-code">${esc(i.ml)}</code></div>` : ''}
    <div class="p-sec"><h4>In → out</h4><div class="p-io"><div><b>Takes</b>${esc(i.io[0])}</div><div><b>Gives</b>${esc(i.io[1])}</div></div></div>
    ${D && i.live ? `<div class="p-sec"><h4>Right now</h4><div class="p-live">${safe(i.live) || 'waiting for data'}</div></div>` : ''}
    ${i.check ? `<div class="p-sec"><h4>Does it work?</h4><p>${i.check}</p></div>` : ''}
    ${ups.length || downs.length ? `<div class="p-sec"><h4>Connected to</h4><div class="p-chips">
      ${ups.map(x => `<button data-go="${x}">← ${esc(NODE[x].t1.replace(/\s+/g, ' '))}</button>`).join('')}
      ${downs.map(x => `<button data-go="${x}">${esc(NODE[x].t1.replace(/\s+/g, ' '))} →</button>`).join('')}</div></div>` : ''}`;
}

function focusNode(id, pin = false) {
  if (playing) return;
  if (pin) { pinned = id; try { history.replaceState(null, '', `#${id}`); } catch (_) { } }
  light(id); show(id);
}

function bind() {
  const svg = $('flow');
  svg.addEventListener('mouseover', e => { const g = e.target.closest('.node'); if (g) focusNode(g.dataset.id); });
  svg.addEventListener('mouseleave', () => { if (playing) return; if (pinned) focusNode(pinned); else { unlight(); $('pBody').hidden = true; $('pEmpty').hidden = false; } });
  svg.addEventListener('click', e => { const g = e.target.closest('.node'); if (g) focusNode(g.dataset.id, true); });
  svg.addEventListener('keydown', e => { const g = e.target.closest('.node'); if (g && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); focusNode(g.dataset.id, true); } });
  svg.addEventListener('focusin', e => { const g = e.target.closest('.node'); if (g) focusNode(g.dataset.id); });
  $('pBody').addEventListener('click', e => { const b = e.target.closest('[data-go]'); if (b) focusNode(b.dataset.go, true); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && !playing) { pinned = null; unlight(); $('pBody').hidden = true; $('pEmpty').hidden = false; } });
}

/* ------------------------------------------------------------------ */
/* Replays                                                              */
/* ------------------------------------------------------------------ */
const wait = t => new Promise(r => setTimeout(r, t));
const pace = t => Math.round(380 + Math.min(900, Math.log10((t || 1) + 1) * 260));

function setRun(ids, cls) {
  const svg = $('flow');
  svg.classList.add('focus');
  ids.forEach(id => { const g = svg.querySelector(`.node[data-id="${id}"]`); if (g) g.classList.add('lit', cls); });
  svg.querySelectorAll('[data-a]').forEach(p => {
    const a = svg.querySelector(`.node[data-id="${p.dataset.a}"]`), b = svg.querySelector(`.node[data-id="${p.dataset.b}"]`);
    p.classList.toggle('lit', !!(a && b && a.classList.contains('lit') && b.classList.contains('lit')));
  });
}

async function playLive() {
  if (playing) return;
  let fresh = null;
  const steps = [
    { id: 'yahoo', label: 'Request market data', delay: 300, note: 'POST /api/refresh asks the server for a brand-new run.',
      before: async li => { try { const r = await fetch('/api/refresh', { method: 'POST' }); const j = await r.json(); if (j.success) { fresh = j.data; D = fresh; updateLive(); } } catch (_) { } li.innerHTML += fresh ? '' : '<br><span class="bear">Refresh failed; replaying the last run.</span>'; } },
  ];
  // Timings are only known after the refresh, so build the rest lazily
  const rest = () => {
    const t = T(), b = D.pipeline.bars;
    return [
      { id: 'fetch', label: 'Download & clean', ms: t.fetch_ms, note: `${b.hourly} hourly bars × ${b.tickers} markets, ${b.m15} 15-min, ${b.daily} daily.` },
      { id: ['feat', 'tech', 'mtf', 'prof', 'inter'], label: 'Feature engineering', ms: t.snapshot_ms, note: `RSI ${D.wti.rsi}, trend score ${D.mtf_confluence.overall_score}, POC ${usd(D.levels.poc)}, crack ${usd(D.macro_radar.crack_margin)}.` },
      { id: 'pillars', label: '7 pillars → score', delay: 650, note: `Score ${sgn(D.sentiment.score)}: ${esc(D.sentiment.label)}.` },
      { id: 'platt', label: 'Platt calibration (ML)', delay: 650, note: `Raw ${D.sentiment.bullish_pct}% → honest ${D.sentiment.calibrated_p_up_4h}%.` },
      { id: 'hedge', label: 'Hedge self-learning (ML)', ms: t.learning_ms, note: `Learned score ${sgn(D.online_learning.learned_score)}; ${D.online_learning.pending} call(s) waiting for their 4-hour answer.` },
      { id: 'har', label: 'HAR volatility (ML)', ms: t.volatility_ms, note: `Expected move ±${usd(D.volatility_forecast.expected_move_usd)}, ${D.volatility_forecast.regime}.` },
      { id: 'plan', label: 'Trade plan', delay: 650, note: `${esc(D.trade_setup.action)}; measured win rate ${D.trade_setup.win_probability}%.` },
      { id: 'api', label: 'Serve JSON', delay: 550, note: `Cached for 45 s; next automatic run in under an hour.` },
      { id: ['ui_dash', 'ui_ml', 'ui_classic'], label: 'Screens update', delay: 700, note: `Total server time ${ms((t.fetch_ms || 0) + (t.snapshot_ms || 0) + (t.volatility_ms || 0) + (t.learning_ms || 0))}.` },
    ];
  };
  // The first step performs the real refresh; the timed steps are added once its timings are known
  const first = steps[0], before = first.before;
  first.before = async li => { await before(li); steps.push(...rest()); };
  await replayDynamic(steps, 'A live request, step by step');
}

async function replayDynamic(steps, title) {
  // like replay(), but tolerates steps being appended while it runs
  if (playing) return;
  playing = true; pinned = null;
  $('playLive').disabled = $('playTrain').disabled = true;
  unlight();
  $('pEmpty').hidden = true; $('pBody').hidden = false;
  $('pBody').innerHTML = `<span class="p-kind k-ml">Replay</span><h3>${title}</h3><p>Each box lights up as its stage runs, paced by the server's real timings.</p>`;
  const log = $('pLog'); log.hidden = false; log.innerHTML = '';
  for (let i = 0; i < steps.length; i++) {
    const s = steps[i];
    if (!s.label) continue;
    const ids = [].concat(s.id);
    const li = document.createElement('li'); li.className = 'cur';
    li.innerHTML = `<b>${esc(s.label)}</b> ${s.ms != null ? `<span class="ms">${ms(s.ms)}</span>` : ''}<br>${s.note || ''}`;
    log.querySelectorAll('.cur').forEach(x => x.classList.remove('cur'));
    log.append(li);
    setRun(ids, 'run');
    if (s.before) await s.before(li);
    await wait(s.delay ?? pace(s.ms));
    ids.forEach(id => { const g = $('flow').querySelector(`.node[data-id="${id}"]`); if (g) { g.classList.remove('run'); g.classList.add('done'); } });
  }
  log.querySelectorAll('.cur').forEach(x => x.classList.remove('cur'));
  await wait(1500);
  $('flow').querySelectorAll('.done').forEach(g => g.classList.remove('done'));
  unlight();
  playing = false;
  $('playLive').disabled = $('playTrain').disabled = false;
}

function playTrain() {
  const m = M;
  replayDynamic([
    { id: 'hist', label: 'Load 2.4 years of history', delay: 700, note: '9 markets, hourly, Jul 2024 → Nov 2026 (cached).' },
    { id: 'replay', label: 'Replay the engine hour by hour', delay: 1000, note: `${m ? m.meta.n_signals.toLocaleString() : '13,436'} point-in-time decisions, each with the real outcome 1, 4 and 24 h later.` },
    { id: 'wf', label: 'Walk-forward ML (direction)', delay: 1000, note: m ? `Logistic AUC ${m.ml.h4.logistic_regression.probability.roc_auc.toFixed(3)}, boosting ${m.ml.h4.gradient_boosting.probability.roc_auc.toFixed(3)}: barely above random.` : '' },
    { id: 'trades', label: 'Simulate every trade', delay: 900, note: m ? `${m.trades.filled_trades.toLocaleString()} trades, profit factor ${m.trades.profit_factor.toFixed(2)}.` : '' },
    { id: 'metrics', label: 'Score against baselines', delay: 900, note: m ? `Hit rate ${pct(m.signals.h4.sentiment_7pillar.hit_rate)} vs coin flip 50%; Brier skill ${m.signals.h4.sentiment_7pillar.probability.brier_skill_score.toFixed(3)}.` : '' },
    { id: 'fit', label: 'Fit the live models', delay: 1000, note: 'Platt a = −0.00076; HAR lockbox R² 0.22; Hedge after 13,432 updates; measured win rates.' },
    { id: 'reports', label: 'Write reports/ and model files', delay: 800, note: 'JSON files the server loads at start.' },
    { id: ['platt', 'hedge', 'har', 'plan', 'mapi'], label: 'Live server loads them', delay: 1100, note: 'Calibration, volatility model, Hedge state, measured win rates, evaluation results.' },
    { id: 'forward', label: 'Forward test on new data', delay: 900, note: 'Frozen models, logged hourly by GitHub Actions.' },
  ], 'The training run, step by step');
}

/* ------------------------------------------------------------------ */
/* Data + theme                                                         */
/* ------------------------------------------------------------------ */
function bestAuc() { return Math.max(M.ml.h4.logistic_regression.probability.roc_auc, M.ml.h4.gradient_boosting.probability.roc_auc); }

async function load() {
  try {
    const r = await fetch('/api/data'); const j = await r.json();
    if (!j.success) throw new Error();
    D = j.data; updateLive();
    $('status').className = 'status live'; $('statusText').textContent = `Live · ${D.timestamp.slice(11, 16)} UTC`;
    if (pinned && !playing) show(pinned);
  } catch (_) { $('status').className = 'status err'; $('statusText').textContent = 'Could not load data'; }
}
async function loadMetrics() {
  try { const r = await fetch('/api/metrics'); const j = await r.json(); if (j.success) { M = j.data; updateLive(); } } catch (_) { }
}

(function theme() {
  let saved = null;
  try { saved = localStorage.getItem('ps-theme'); } catch (_) { }
  if (saved) document.documentElement.dataset.theme = saved;
  $('themeBtn').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    const next = dark ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('ps-theme', next); } catch (_) { }
  });
})();

draw(); bind();
$('playLive').addEventListener('click', playLive);
$('playTrain').addEventListener('click', playTrain);
Promise.all([load(), loadMetrics()]).then(() => {
  const h = location.hash.slice(1);
  if (NODE[h]) focusNode(h, true);
});
setInterval(load, 5 * 60 * 1000);
