// PetroSync Pro Terminal Frontend Engine
let chartInstance = null;
let currentData = null;
let audioEnabled = true;
let countdownTimer = null;
let secondsRemaining = 3600;
let refreshIntervalSeconds = 3600;

// Audio Chime using Web Audio API
function playChime(type = 'update') {
    if (!audioEnabled) return;
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);

        const now = audioCtx.currentTime;
        if (type === 'signal_change') {
            // High double ping for trade setup / signal shift
            osc.frequency.setValueAtTime(659.25, now); // E5
            osc.frequency.setValueAtTime(987.77, now + 0.12); // B5
            gain.gain.setValueAtTime(0.18, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
            osc.start(now);
            osc.stop(now + 0.5);
        } else {
            // Subtle sync chime
            osc.frequency.setValueAtTime(523.25, now); // C5
            gain.gain.setValueAtTime(0.08, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
            osc.start(now);
            osc.stop(now + 0.3);
        }
    } catch (e) {
        console.warn("Audio Context error:", e);
    }
}

// Format numbers with sign
function formatSigned(num, decimals = 2, prefix = "") {
    if (num > 0) return `+${prefix}${num.toFixed(decimals)}`;
    if (num < 0) return `-${prefix}${Math.abs(num).toFixed(decimals)}`;
    return `${prefix}0.00`;
}

// Calculate seconds until next top of the hour
function getSecondsUntilNextHour() {
    const now = new Date();
    const minutes = now.getUTCMinutes();
    const seconds = now.getUTCSeconds();
    return (59 - minutes) * 60 + (60 - seconds);
}

// Start Countdown Clock
function initCountdown() {
    if (countdownTimer) clearInterval(countdownTimer);

    if (refreshIntervalSeconds === 3600) {
        secondsRemaining = getSecondsUntilNextHour();
    } else if (refreshIntervalSeconds > 0) {
        secondsRemaining = refreshIntervalSeconds;
    } else {
        document.getElementById('nextHourlyCountdown').textContent = "MANUAL";
        return;
    }

    countdownTimer = setInterval(() => {
        if (refreshIntervalSeconds === 0) return;

        secondsRemaining--;
        if (secondsRemaining <= 0) {
            fetchMarketData(true);
            if (refreshIntervalSeconds === 3600) {
                secondsRemaining = getSecondsUntilNextHour();
            } else {
                secondsRemaining = refreshIntervalSeconds;
            }
        }

        const mins = Math.floor(secondsRemaining / 60);
        const secs = secondsRemaining % 60;
        document.getElementById('nextHourlyCountdown').textContent = 
            `${mins}m ${secs < 10 ? '0' : ''}${secs}s`;
    }, 1000);
}

// Fetch Market Data from API
async function fetchMarketData(force = false) {
    const spinner = document.getElementById('refreshSpinner');
    const btn = document.getElementById('manualRefreshBtn');
    spinner.classList.add('spinning');
    btn.disabled = true;

    try {
        const url = force ? '/api/refresh' : '/api/data';
        const res = await fetch(url);
        const json = await res.json();

        if (json.success && json.data) {
            const previousSignal = currentData ? currentData.trade_setup.setup_type : null;
            currentData = json.data;
            updateUI(currentData);

            if (previousSignal && previousSignal !== currentData.trade_setup.setup_type) {
                playChime('signal_change');
            } else if (force) {
                playChime('update');
            }
        }
    } catch (err) {
        console.error("Fetch error:", err);
    } finally {
        spinner.classList.remove('spinning');
        btn.disabled = false;
    }
}

// Update the full UI dashboard
function updateUI(d) {
    // 0. Top Nav
    document.getElementById('lastSyncTime').textContent = d.timestamp.split(' ')[1] + ' UTC';
    const mtf = d.mtf_confluence;
    const topConf = document.getElementById('topConfluenceVal');
    topConf.textContent = `${mtf.overall_score}% ${mtf.badge.toUpperCase()}`;
    topConf.style.color = mtf.badge === 'bull' ? 'var(--bull-green)' : (mtf.badge === 'bear' ? 'var(--bear-red)' : 'var(--warn-amber)');

    // 1. WTI Card
    const wti = d.wti;
    document.getElementById('wtiPrice').textContent = `$${wti.price.toFixed(2)}`;
    
    const wti1h = document.getElementById('wti1hBadge');
    wti1h.textContent = `1h: ${formatSigned(wti.change_1h, 2, '$')} (${formatSigned(wti.change_1h_pct, 2)}%)`;
    wti1h.className = `change-badge ${wti.change_1h >= 0 ? 'positive' : 'negative'}`;

    const wti24h = document.getElementById('wti24hBadge');
    wti24h.textContent = `24h: ${formatSigned(wti.change_24h, 2, '$')} (${formatSigned(wti.change_24h_pct, 2)}%)`;
    wti24h.className = `change-badge-sub ${wti.change_24h >= 0 ? 'positive' : 'negative'}`;

    document.getElementById('wtiDayLow').textContent = `L: $${wti.day_low.toFixed(2)}`;
    document.getElementById('wtiDayHigh').textContent = `H: $${wti.day_high.toFixed(2)}`;

    // Range track
    const rangeSpan = wti.day_high - wti.day_low;
    let fillPct = 50;
    if (rangeSpan > 0) {
        fillPct = Math.max(5, Math.min(95, ((wti.price - wti.day_low) / rangeSpan) * 100));
    }
    document.getElementById('wtiRangeFill').style.width = `${fillPct}%`;

    // 2. OVX Card
    const macro = d.macro_radar;
    document.getElementById('ovxPrice').textContent = macro.ovx_price.toFixed(2);
    const ovxBadge = document.getElementById('ovxRegimeBadge');
    ovxBadge.textContent = macro.ovx_price > 45 ? "HIGH VOL" : (macro.ovx_price < 30 ? "LOW VOL" : "NORMAL");
    ovxBadge.style.color = macro.ovx_color;
    ovxBadge.style.background = macro.ovx_color + '25';
    document.getElementById('ovxStatusText').textContent = macro.ovx_regime;
    document.getElementById('ovxStatusText').style.color = macro.ovx_color;

    // 3. Brent Card
    const brent = d.brent;
    document.getElementById('brentSpread').textContent = `+$${brent.spread_to_wti.toFixed(2)}/bbl`;
    document.getElementById('brentRawPrice').textContent = `BZ: $${brent.price.toFixed(2)}`;
    document.getElementById('brentStatus').textContent = brent.spread_status;

    // 4. Crack Margin Card
    document.getElementById('crackPrice').textContent = `$${macro.crack_margin.toFixed(2)}`;
    document.getElementById('crackStatus').textContent = macro.crack_margin >= 25 ? "High (Refiners Buying)" : "Weak";
    document.getElementById('crackStatus').style.color = macro.crack_margin >= 25 ? 'var(--bull-green)' : 'var(--warn-amber)';

    // 5. Macro Headwind (DXY & Gold/Oil)
    document.getElementById('dxyPrice').textContent = macro.dxy_price.toFixed(2);
    const dxyBadge = document.getElementById('dxyChangeBadge');
    dxyBadge.textContent = `${formatSigned(macro.dxy_change_pct, 2)}%`;
    dxyBadge.className = `change-badge ${macro.dxy_change_pct > 0 ? 'negative' : 'positive'}`;
    document.getElementById('goldOilRatio').textContent = `${macro.gold_oil_ratio}:1`;

    // 6. EIA Countdown
    document.getElementById('eiaCountdown').textContent = macro.eia_countdown;

    // 7. QUANTITATIVE TRADE SETUP CARD
    const ts = d.trade_setup;
    document.getElementById('winProbVal').textContent = `${ts.win_probability}%`;
    renderRealityCheck(ts.setup_type);
    document.getElementById('setupTypeLabel').textContent = ts.setup_type;
    document.getElementById('setupActionTitle').textContent = ts.action.toUpperCase();

    const biasBadge = document.getElementById('setupBiasBadge');
    biasBadge.textContent = ts.bias;
    if (ts.bias === 'LONG') {
        biasBadge.style.background = 'rgba(16, 185, 129, 0.2)';
        biasBadge.style.color = 'var(--bull-green)';
        biasBadge.style.border = '1px solid rgba(16, 185, 129, 0.4)';
    } else if (ts.bias === 'SHORT') {
        biasBadge.style.background = 'rgba(244, 63, 94, 0.2)';
        biasBadge.style.color = 'var(--bear-red)';
        biasBadge.style.border = '1px solid rgba(244, 63, 94, 0.4)';
    } else {
        biasBadge.style.background = 'rgba(245, 158, 11, 0.2)';
        biasBadge.style.color = 'var(--warn-amber)';
        biasBadge.style.border = '1px solid rgba(245, 158, 11, 0.4)';
    }

    document.getElementById('paramEntry').textContent = ts.entry_range;
    document.getElementById('paramStop').textContent = `$${ts.stop_loss.toFixed(2)}`;
    document.getElementById('paramStopTicks').textContent = `-${ts.stop_ticks} ticks risk`;
    document.getElementById('paramTarget1').textContent = `$${ts.target1.toFixed(2)}`;
    document.getElementById('paramTp1Ticks').textContent = `+${ts.tp1_ticks} ticks`;
    document.getElementById('paramTarget2').textContent = `$${ts.target2.toFixed(2)}`;
    document.getElementById('paramRR').textContent = ts.risk_reward;

    document.getElementById('setupRationaleText').textContent = ts.rationale;
    document.getElementById('setupTtTip').textContent = ts.tt_tip;
    document.getElementById('volAlertText').textContent = ts.vol_warning;

    // 8. MULTI-TIMEFRAME CONFLUENCE MATRIX
    document.getElementById('mtfScoreBadge').textContent = `SCORE: ${mtf.overall_score}/100`;
    
    // Fill MTF columns (15m, 1h, Daily)
    const tfs = mtf.timeframes;
    if (tfs && tfs.length >= 3) {
        // 15m
        document.getElementById('mtfBadge15m').textContent = tfs[0].bias;
        document.getElementById('mtfBadge15m').style.color = tfs[0].color;
        document.getElementById('mtfBadge15m').style.background = tfs[0].color + '20';
        document.getElementById('mtfRsi15m').textContent = tfs[0].rsi;
        document.getElementById('mtfTrend15m').textContent = tfs[0].trend;

        // 1h
        document.getElementById('mtfBadge1h').textContent = tfs[1].bias;
        document.getElementById('mtfBadge1h').style.color = tfs[1].color;
        document.getElementById('mtfBadge1h').style.background = tfs[1].color + '20';
        document.getElementById('mtfRsi1h').textContent = tfs[1].rsi;
        document.getElementById('mtfTrend1h').textContent = tfs[1].trend;

        // Daily
        document.getElementById('mtfBadgeDaily').textContent = tfs[2].bias;
        document.getElementById('mtfBadgeDaily').style.color = tfs[2].color;
        document.getElementById('mtfBadgeDaily').style.background = tfs[2].color + '20';
        document.getElementById('mtfRsiDaily').textContent = tfs[2].rsi;
        document.getElementById('mtfTrendDaily').textContent = tfs[2].trend;
    }

    // Liquidity Sweeps
    const sw = d.liquidity_sweeps;
    document.getElementById('sweepDetails').textContent = sw.status;

    // Benchmarks
    document.getElementById('benchVwap').textContent = `$${wti.vwap.toFixed(2)}`;
    document.getElementById('benchPoc').textContent = `$${d.levels.poc.toFixed(2)}`;
    document.getElementById('benchVwapU1').textContent = `$${wti.vwap_u1.toFixed(2)}`;
    document.getElementById('benchVwapL1').textContent = `$${wti.vwap_l1.toFixed(2)}`;

    // 9. SENTIMENT GAUGE & ATTRIBUTION
    const sent = d.sentiment;
    document.getElementById('sentimentScoreBadge').textContent = `SCORE: ${sent.score > 0 ? '+' : ''}${sent.score}`;
    document.getElementById('sentimentLabel').textContent = sent.label.toUpperCase();
    document.getElementById('sentimentLabel').style.color = sent.color;
    document.getElementById('sentimentDesc').textContent = sent.summary;

    document.getElementById('probBullText').textContent = `BULL: ${sent.bullish_pct}%`;
    document.getElementById('probBearText').textContent = `BEAR: ${sent.bearish_pct}%`;
    document.getElementById('splitFillBull').style.width = `${sent.bullish_pct}%`;

    // Rotate Needle (-80 to +80 deg)
    const angle = (sent.score / 100) * 80;
    document.getElementById('gaugeNeedle').style.transform = `rotate(${angle}deg)`;

    // Factor bars
    setFactorBar('barMtf', 'valMtf', sent.breakdown.mtf_alignment);
    setFactorBar('barFlow', 'valFlow', sent.breakdown.order_flow_vwap);
    setFactorBar('barCrack', 'valCrack', sent.breakdown.crack_margin);
    setFactorBar('barSpread', 'valSpread', sent.breakdown.brent_spread);
    setFactorBar('barDxy', 'valDxy', sent.breakdown.usd_impact);
    setFactorBar('barRisk', 'valRisk', sent.breakdown.equity_risk_on);

    // 10. Levels Table
    renderLevelsTable(d.levels, wti.price);

    // 11. History Table
    renderHistoryTable(d.history);

    // 12. Chart
    renderChart(d.chart_candles);
}

function setFactorBar(barId, valId, score) {
    const bar = document.getElementById(barId);
    const val = document.getElementById(valId);
    const pct = Math.max(5, Math.min(95, (score + 100) / 2));
    bar.style.width = `${pct}%`;
    val.textContent = `${score > 0 ? '+' : ''}${score}`;

    if (score >= 15) {
        bar.style.backgroundColor = 'var(--bull-green)';
    } else if (score <= -15) {
        bar.style.backgroundColor = 'var(--bear-red)';
    } else {
        bar.style.backgroundColor = 'var(--warn-amber)';
    }
}

// Render Reference Levels Table
function renderLevelsTable(levels, currentPrice) {
    const tbody = document.getElementById('levelsTableBody');
    const rows = [
        { desc: "Resistance 2 (Exhaustion Target)", code: "R2", price: levels.r2, type: "res", rule: "Major target for profitable longs; sharp fade potential" },
        { desc: "Resistance 1 (Day Pivot High)", code: "R1", price: levels.r1, type: "res", rule: "First barrier for bull continuation vs rejection" },
        { desc: "VWAP Upper +2σ Band", code: "+2σ", price: levels.vwap_u2, type: "vwap", rule: "Extreme 2-standard-deviation overbought boundary" },
        { desc: "Value Area High (70% Volume Roof)", code: "VAH", price: levels.vah, type: "vah", rule: "Must hold on pullbacks to sustain bullish regime" },
        { desc: "VWAP Upper +1σ Band", code: "+1σ", price: levels.vwap_u1, type: "vwap", rule: "First statistical expansion band; potential pullback zone" },
        { desc: "Point of Control (Volume Magnet)", code: "POC", price: levels.poc, type: "poc", rule: "Highest liquidity node of session; expect consolidation" },
        { desc: "Session VWAP (Institutional Baseline)", code: "VWAP", price: levels.vwap, type: "poc", rule: "Primary directional pivot: Long above, Short below" },
        { desc: "VWAP Lower -1σ Band", code: "-1σ", price: levels.vwap_l1, type: "vwap", rule: "First statistical discount band; test of seller power" },
        { desc: "Value Area Low (70% Volume Floor)", code: "VAL", price: levels.val, type: "val", rule: "Must be defended by buyers; breakdown triggers dump" },
        { desc: "VWAP Lower -2σ Band", code: "-2σ", price: levels.vwap_l2, type: "vwap", rule: "Extreme 2-standard-deviation oversold capitulation" },
        { desc: "Support 1 (Day Pivot Low)", code: "S1", price: levels.s1, type: "sup", rule: "First bounce zone for mean-reversion scalpers" },
        { desc: "Support 2 (Deep Liquidity Basin)", code: "S2", price: levels.s2, type: "sup", rule: "Capitulation floor; high reward long with tight stops" }
    ];

    tbody.innerHTML = rows.map(r => {
        const diff = currentPrice - r.price;
        const diffTicks = Math.round(diff * 100);
        const distStr = diff >= 0 ? `+$${diff.toFixed(2)} (+${diffTicks}t)` : `-$${Math.abs(diff).toFixed(2)} (${diffTicks}t)`;
        const isClosest = Math.abs(diff) <= 0.35;

        return `
            <tr class="${isClosest ? 'current-level-row' : ''}">
                <td>${r.desc}</td>
                <td><span class="level-code code-${r.type}">${r.code}</span></td>
                <td class="level-price">$${r.price.toFixed(2)}</td>
                <td class="level-dist" style="color: ${diff >= 0 ? 'var(--bull-green)' : 'var(--bear-red)'}">${distStr}</td>
                <td style="font-size: 0.7rem; color: var(--text-muted);">${r.rule}</td>
            </tr>
        `;
    }).join('');
}

// Render 24-Hour History Table
function renderHistoryTable(history) {
    const tbody = document.getElementById('historyTableBody');
    if (!history || history.length === 0) return;

    tbody.innerHTML = history.map(h => {
        const changeClass = h.change_1h >= 0 ? 'positive' : 'negative';
        let badgeColor = 'rgba(59, 130, 246, 0.15); color: #60a5fa;';
        if (h.badge === 'bull') badgeColor = 'rgba(16, 185, 129, 0.15); color: #34d399;';
        if (h.badge === 'bear') badgeColor = 'rgba(244, 63, 94, 0.15); color: #fb7185;';

        return `
            <tr>
                <td style="font-family: var(--font-mono); font-weight: 600;">${h.timestamp}</td>
                <td style="font-family: var(--font-mono); font-weight: 700; color: #ffffff;">$${h.price.toFixed(2)}</td>
                <td><span class="change-badge ${changeClass}">${formatSigned(h.change_1h, 2, '$')} (${formatSigned(h.change_1h_pct, 2)}%)</span></td>
                <td style="font-family: var(--font-mono);">${h.rsi.toFixed(1)}</td>
                <td style="font-family: var(--font-mono); color: var(--accent-cyan);">$${h.vwap.toFixed(2)}</td>
                <td><span class="level-code" style="background: ${badgeColor}">${h.regime}</span></td>
            </tr>
        `;
    }).join('');
}

// Render Multi-Series Interactive Chart
function renderChart(candles) {
    if (!candles || candles.length === 0) return;

    const ctx = document.getElementById('crudeChart').getContext('2d');
    const labels = candles.map(c => c.time);
    const closePrices = candles.map(c => c.close);
    const vwapValues = candles.map(c => c.vwap);
    const vwapUpper = candles.map(c => c.vwap_u1);
    const vwapLower = candles.map(c => c.vwap_l1);
    const ema20Values = candles.map(c => c.ema20);
    const ema50Values = candles.map(c => c.ema50);
    const volumes = candles.map(c => c.volume);

    if (chartInstance) {
        chartInstance.data.labels = labels;
        chartInstance.data.datasets[0].data = closePrices;
        chartInstance.data.datasets[1].data = vwapValues;
        chartInstance.data.datasets[2].data = vwapUpper;
        chartInstance.data.datasets[3].data = vwapLower;
        chartInstance.data.datasets[4].data = ema20Values;
        chartInstance.data.datasets[5].data = ema50Values;
        chartInstance.data.datasets[6].data = volumes;
        chartInstance.update();
        return;
    }

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'WTI Close ($)',
                    data: closePrices,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.08)',
                    borderWidth: 2.2,
                    fill: false,
                    tension: 0.15,
                    pointRadius: 1,
                    pointHoverRadius: 5,
                    yAxisID: 'y'
                },
                {
                    label: 'Session VWAP',
                    data: vwapValues,
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    borderDash: [5, 4],
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'VWAP +1σ Band',
                    data: vwapUpper,
                    borderColor: 'rgba(16, 185, 129, 0.65)',
                    borderWidth: 1.2,
                    borderDash: [2, 2],
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'VWAP -1σ Band',
                    data: vwapLower,
                    borderColor: 'rgba(244, 63, 94, 0.65)',
                    borderWidth: 1.2,
                    borderDash: [2, 2],
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: '20 EMA',
                    data: ema20Values,
                    borderColor: '#10b981',
                    borderWidth: 1.3,
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: '50 EMA',
                    data: ema50Values,
                    borderColor: '#a855f7',
                    borderWidth: 1.3,
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'Volume',
                    data: volumes,
                    type: 'bar',
                    backgroundColor: 'rgba(148, 163, 184, 0.18)',
                    yAxisID: 'yVolume'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { family: "'Plus Jakarta Sans', sans-serif", size: 11 },
                        usePointStyle: true,
                        pointStyle: 'line'
                    }
                },
                tooltip: {
                    backgroundColor: '#0c1220',
                    titleColor: '#f8fafc',
                    bodyColor: '#cbd5e1',
                    borderColor: '#1a2538',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: true
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(51, 65, 85, 0.22)' },
                    ticks: {
                        color: '#64748b',
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 12
                    }
                },
                y: {
                    position: 'right',
                    grid: { color: 'rgba(51, 65, 85, 0.3)' },
                    ticks: {
                        color: '#94a3b8',
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                        callback: val => `$${val.toFixed(2)}`
                    }
                },
                yVolume: {
                    position: 'left',
                    grid: { display: false },
                    ticks: { display: false },
                    min: 0,
                    max: Math.max(...volumes) * 4
                }
            }
        }
    });

    // Wire chart toggle buttons
    document.getElementById('toggleVwap').addEventListener('click', function() {
        this.classList.toggle('active');
        chartInstance.setDatasetVisibility(1, this.classList.contains('active'));
        chartInstance.update();
    });
    document.getElementById('toggleBands').addEventListener('click', function() {
        this.classList.toggle('active');
        const show = this.classList.contains('active');
        chartInstance.setDatasetVisibility(2, show);
        chartInstance.setDatasetVisibility(3, show);
        chartInstance.update();
    });
    document.getElementById('toggleEma20').addEventListener('click', function() {
        this.classList.toggle('active');
        chartInstance.setDatasetVisibility(4, this.classList.contains('active'));
        chartInstance.update();
    });
    document.getElementById('toggleEma50').addEventListener('click', function() {
        this.classList.toggle('active');
        chartInstance.setDatasetVisibility(5, this.classList.contains('active'));
        chartInstance.update();
    });
}


// ===================== BACKTEST REALITY CHECK =====================
// Loads the offline walk-forward evaluation (/api/metrics) once and compares the
// live setup's hard-coded win probability with how that setup actually performed.
let backtestMetrics = null;
let backtestLoaded = false;

async function loadBacktestMetrics() {
    try {
        const res = await fetch('/api/metrics');
        const json = await res.json();
        backtestMetrics = json.success ? json.data : null;
    } catch (e) {
        console.warn('Backtest metrics unavailable', e);
        backtestMetrics = null;
    }
    backtestLoaded = true;
    renderRealityCheck(currentData ? currentData.trade_setup.setup_type : null);
}

const rcPct = (x, d = 1) => (x === null || x === undefined) ? '--' : `${(x * 100).toFixed(d)}%`;
const rcTone = (gapPts, exp) => (gapPts <= -15 || exp < 0) ? 'bad' : (gapPts <= -5 ? 'warn' : 'good');
const rcTitle = s => s.replace(/\s*\(.*\)/, '').toLowerCase().replace(/(^|\s)\S/g, m => m.toUpperCase());

function setTone(el, tone) {
    el.classList.remove('text-bull', 'text-bear', 'text-warning');
    el.classList.add(tone === 'good' ? 'text-bull' : tone === 'bad' ? 'text-bear' : 'text-warning');
}

function renderRealityCheck(setupType) {
    const empty = document.getElementById('realityEmpty');
    const body = document.getElementById('realityBody');
    const meta = document.getElementById('realityMeta');
    const chipBt = document.getElementById('winProbBacktest');
    if (!backtestLoaded) return;
    if (!backtestMetrics || !backtestMetrics.trades || !backtestMetrics.trades.by_setup) {
        empty.hidden = false; body.hidden = true;
        meta.textContent = 'No evaluation found';
        chipBt.textContent = 'n/a';
        return;
    }
    empty.hidden = true; body.hidden = false;
    const m = backtestMetrics, T = m.trades, bySetup = T.by_setup;
    meta.textContent = `Walk-forward replay · ${m.meta.sample_start.slice(0, 10)} → ${m.meta.sample_end.slice(0, 10)} · ${m.meta.n_signals.toLocaleString()} hourly decisions · generated ${m.meta.generated_utc}`;

    // --- current setup ---
    const g = setupType ? bySetup[setupType] : null;
    chipBt.classList.remove('rc-bad', 'rc-warn', 'rc-good', 'text-muted-strong');
    if (g) {
        const gap = (g.target1_hit_rate - g.claimed_win_prob) * 100;
        const tone = rcTone(gap, g.expectancy_usd);
        chipBt.textContent = rcPct(g.target1_hit_rate);
        chipBt.classList.add('rc-' + tone);
        document.getElementById('realitySetupName').textContent = rcTitle(setupType);
        document.getElementById('realityClaimBar').style.width = `${g.claimed_win_prob * 100}%`;
        document.getElementById('realityClaimVal').textContent = rcPct(g.claimed_win_prob, 0);
        const realBar = document.getElementById('realityRealBar');
        realBar.style.width = `${g.target1_hit_rate * 100}%`;
        realBar.style.background = tone === 'good' ? 'var(--bull-green)' : tone === 'warn' ? 'var(--warn-amber)' : 'var(--bear-red)';
        document.getElementById('realityRealVal').textContent = rcPct(g.target1_hit_rate);
        const verdict = document.getElementById('realityVerdict');
        verdict.className = 'reality-verdict ' + tone;
        verdict.textContent = tone === 'good' ? 'CLAIM HOLDS' : (g.expectancy_usd < 0 ? 'NEGATIVE EDGE' : 'OVERSTATED');
        const sign = g.expectancy_usd >= 0 ? '+' : '−';
        document.getElementById('realityCurrentNote').textContent =
            `In ${g.n} simulated trades this setup reached Target 1 ${rcPct(g.target1_hit_rate)} of the time, ` +
            `${Math.abs(gap).toFixed(1)} points ${gap < 0 ? 'below' : 'above'} the ${rcPct(g.claimed_win_prob, 0)} the playbook claims. ` +
            `Average result ${sign}$${Math.abs(g.expectancy_usd).toFixed(0)} per contract after costs. ` +
            (g.binomial_p_claim !== null && g.binomial_p_claim < 0.05 ? 'The gap is statistically significant (p < 0.05).' : 'The gap is not statistically significant.');
    } else {
        chipBt.textContent = 'n/a';
        chipBt.classList.add('text-muted-strong');
        document.getElementById('realitySetupName').textContent = setupType ? rcTitle(setupType) : 'Waiting for live setup…';
        document.getElementById('realityVerdict').className = 'reality-verdict';
        document.getElementById('realityVerdict').textContent = 'NO HISTORY';
        document.getElementById('realityCurrentNote').textContent = 'This setup did not trigger a filled trade in the backtest sample.';
    }

    // --- global stats ---
    const s4 = m.signals && m.signals.h4 ? m.signals.h4.sentiment_7pillar : null;
    if (s4) {
        const hit = document.getElementById('rcHit');
        hit.textContent = rcPct(s4.hit_rate);
        setTone(hit, s4.binomial_p_vs_50pct < 0.05 && s4.hit_rate > 0.5 ? 'good' : 'bad');
        document.getElementById('rcHitSub').textContent = `coin flip = 50% · p = ${s4.binomial_p_vs_50pct.toFixed(2)}`;
        const bs = s4.probability.brier_skill_score, b = document.getElementById('rcBrier');
        b.textContent = (bs >= 0 ? '+' : '') + bs.toFixed(3);
        setTone(b, bs > 0 ? 'good' : 'bad');
    }
    const pf = document.getElementById('rcPf');
    pf.textContent = T.profit_factor.toFixed(2);
    setTone(pf, T.profit_factor > 1.1 ? 'good' : T.profit_factor >= 1 ? 'warn' : 'bad');
    document.getElementById('rcPfSub').textContent = `needs > 1.0 · ${T.filled_trades.toLocaleString()} trades`;
    const ex = document.getElementById('rcExp');
    ex.textContent = `${T.expectancy_usd_per_trade >= 0 ? '+' : '−'}$${Math.abs(T.expectancy_usd_per_trade).toFixed(0)}`;
    setTone(ex, T.expectancy_usd_per_trade > 0 ? 'good' : 'bad');
    document.getElementById('rcExpSub').textContent = `1 CL contract, after costs · win ${rcPct(T.target1_hit_rate)} vs ${rcPct(T.claimed_win_prob_avg, 0)} claimed`;
    if (m.ml && m.ml.h4) {
        const auc = Math.max(m.ml.h4.logistic_regression.probability.roc_auc, m.ml.h4.gradient_boosting.probability.roc_auc);
        const a = document.getElementById('rcAuc');
        a.textContent = auc.toFixed(3);
        setTone(a, auc >= 0.55 ? 'good' : auc >= 0.52 ? 'warn' : 'bad');
        const theil = Math.min(...Object.values(m.ml.h4.price_forecast).map(v => v.theils_u2));
        const t = document.getElementById('rcTheil');
        t.textContent = theil.toFixed(3);
        setTone(t, theil < 0.98 ? 'good' : theil <= 1.0 ? 'warn' : 'bad');
    }

    // --- per-setup table ---
    const rows = Object.entries(bySetup).sort((a, b) => b[1].n - a[1].n).map(([name, r]) => {
        const gap = (r.target1_hit_rate - r.claimed_win_prob) * 100;
        const tone = rcTone(gap, r.expectancy_usd);
        const cls = tone === 'good' ? 'text-bull' : tone === 'warn' ? 'text-warning' : 'text-bear';
        return `<tr class="${name === setupType ? 'is-current' : ''}">
            <td>${rcTitle(name)}${name === setupType ? ' <span class="text-accent">● live</span>' : ''}</td>
            <td>${r.n}</td>
            <td>${rcPct(r.claimed_win_prob, 0)}</td>
            <td class="${cls}">${rcPct(r.target1_hit_rate)}</td>
            <td class="reality-gap ${cls}">${gap >= 0 ? '+' : ''}${gap.toFixed(1)} pts</td>
            <td class="${r.expectancy_usd >= 0 ? 'text-bull' : 'text-bear'}">${r.expectancy_usd >= 0 ? '+' : '−'}$${Math.abs(r.expectancy_usd).toFixed(0)}</td>
        </tr>`;
    });
    document.getElementById('realityTableBody').innerHTML = rows.join('');
}

// Event Listeners on DOM Load
document.addEventListener('DOMContentLoaded', () => {
    // Initial fetch
    fetchMarketData(false);
    loadBacktestMetrics();

    // Manual refresh button
    document.getElementById('manualRefreshBtn').addEventListener('click', () => {
        fetchMarketData(true);
        if (refreshIntervalSeconds === 3600) {
            secondsRemaining = getSecondsUntilNextHour();
        } else if (refreshIntervalSeconds > 0) {
            secondsRemaining = refreshIntervalSeconds;
        }
    });

    // Interval selector
    document.getElementById('refreshInterval').addEventListener('change', (e) => {
        refreshIntervalSeconds = parseInt(e.target.value);
        initCountdown();
    });

    // Audio chime toggle
    document.getElementById('audioToggleBtn').addEventListener('click', () => {
        audioEnabled = !audioEnabled;
        const icon = document.getElementById('audioIcon');
        icon.textContent = audioEnabled ? '🔔' : '🔕';
        document.getElementById('audioToggleBtn').style.opacity = audioEnabled ? '1' : '0.5';
    });

    // Start timer
    initCountdown();
});
