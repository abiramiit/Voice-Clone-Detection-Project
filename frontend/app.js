/* =================================================================
   VOICE FORENSICS LAB — app.js
   Canvas bug fix: ResizeObserver on container, never size canvas
   from JS style properties. All canvases use position:absolute inset:0.
   ================================================================= */
'use strict';

/* ─── DOM shortcuts ───────────────────────────────────────────── */
const $ = id => document.getElementById(id);

/* ─── Color constants ─────────────────────────────────────────── */
const C = {
    BON: '#22e3a0', SPO: '#ff3d6e', UNC: '#ffb020', INS: '#64748b',
    CYAN: '#22d3ee', VIO: '#8b5cf6', DARK: '#070a12'
};
function verdictColor(v) {
    return v === 'BONAFIDE' ? C.BON : v === 'SPOOF' ? C.SPO : v === 'UNCERTAIN' ? C.UNC : C.INS;
}
function pillClass(v) {
    return v === 'BONAFIDE' ? 'pill pill-bon' : v === 'SPOOF' ? 'pill pill-spo' :
        v === 'UNCERTAIN' ? 'pill pill-unc' : 'pill pill-ins';
}

/* ─── State ───────────────────────────────────────────────────── */
let audioCtx, processor, source, analyser, gainNode, ws, stream;
let isStreaming = false, isReconnecting = false;
let reconnectDelay = 1000;
let currentMode = 'orb';   // orb | wf | spec | scope
let animId = null;
let resizeRafs = {};        // container id -> raf id
let idlePhase = 0;
let sessionStart = null;
let timerInterval = null;

// Data arrays
const TL_MAX = 120;
let tlRaw = [], tlAgg = [], tlVerdicts = [];
const HIST_MAX = 100;
let histRows = [];
const SPARK_MAX = 40;
let spkInfData = [], spkPreData = [], spkTotData = [], spkStabData = [], spkErrData = [];

// Spectrogram accumulator
let spectroCol = 0;

// Waterfall accumulator
let wfRows = [];
const WF_MAX_ROWS = 80;

// Window ring buffer (last 60 windows for orb ring-3)
const RING_MAX = 60;
let ringWindows = [];

// Scope phosphor trail
let scopeTrail = [];

// RMS peak hold
let rmsPeakVal = 0;
let rmsPeakTimer = null;

// Errors
let errWS = 0, errModel = 0, errUI = 0;

// Flip counter
let lastRawPred = null, flipCount = 0;

// Window count
let winCount = 0;

/* ─── roundRect polyfill (Chrome < 99) ───────────────────────── */
if (!CanvasRenderingContext2D.prototype.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
        const rad = Array.isArray(r) ? r[0] : (r || 0);
        this.moveTo(x + rad, y); this.lineTo(x + w - rad, y);
        this.quadraticCurveTo(x + w, y, x + w, y + rad); this.lineTo(x + w, y + h);
        this.quadraticCurveTo(x + w, y + h, x + w - rad, y + h); this.lineTo(x + rad, y + h);
        this.quadraticCurveTo(x, y + h, x, y + h - rad); this.lineTo(x, y + rad);
        this.quadraticCurveTo(x, y, x + rad, y); this.closePath();
    };
}

/* =================================================================
   CANVAS INIT & RESIZE — THE BUG FIX
   Size from the CONTAINER (not the canvas), via ResizeObserver.
   Never set canvas.style.width/height.
   ================================================================= */
const observers = {};

function initCanvas(containerId) {
    const container = $(containerId);
    if (!container) return;
    const canvas = container.querySelector('canvas');
    if (!canvas) return;

    function doResize() {
        const dpr = window.devicePixelRatio || 1;
        const r = container.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) return;
        canvas.width = Math.round(r.width * dpr);
        canvas.height = Math.round(r.height * dpr);
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        // Re-draw idle state on resize
        if (!isStreaming && (containerId === 'orbWrap')) drawOrbIdle();
    }

    // Debounce resize with rAF so we never resize inside draw loop
    const ro = new ResizeObserver(() => {
        if (resizeRafs[containerId]) cancelAnimationFrame(resizeRafs[containerId]);
        resizeRafs[containerId] = requestAnimationFrame(() => {
            doResize();
            delete resizeRafs[containerId];
        });
    });
    ro.observe(container);
    observers[containerId] = ro;
    doResize(); // initial sizing
}

// Helpers — read CSS dimensions from the container, not canvas.style
function cW(canvas) { return canvas.width / (window.devicePixelRatio || 1); }
function cH(canvas) { return canvas.height / (window.devicePixelRatio || 1); }

function ctxOf(canvasId) { return $(canvasId)?.getContext('2d'); }

/* ─── Safe DOM setter ─────────────────────────────────────────── */
function setEl(id, val, asWidth) {
    try {
        const el = $(id); if (!el) { console.error('[UI ERROR] Missing #' + id); return; }
        if (asWidth) el.style.width = val; else el.textContent = val;
    } catch (e) { console.error('[UI ERROR] setEl:', e); errUI++; updateErrors(); }
}

/* =================================================================
   AMBIENT GLOW — update CSS var on the .lab element
   ================================================================= */
function setAmbient(verdict) {
    const lab = document.querySelector('.lab');
    if (!lab) return;
    const col = verdictColor(verdict);
    // rgba with low opacity for ambient bg tint, higher for border
    lab.style.setProperty('--ambient', hexAlpha(col, .04));
    lab.style.setProperty('--ambient-border', hexAlpha(col, .20));
}
function hexAlpha(hex, a) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${a})`;
}

/* =================================================================
   SESSION TIMER
   ================================================================= */
function startTimer() {
    sessionStart = Date.now();
    timerInterval = setInterval(() => {
        if (!sessionStart) return;
        const s = Math.floor((Date.now() - sessionStart) / 1000);
        const h = String(Math.floor(s / 3600)).padStart(2, '0');
        const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
        const ss = String(s % 60).padStart(2, '0');
        setEl('sessionTimer', `${h}:${m}:${ss}`);
    }, 1000);
}
function stopTimer() {
    clearInterval(timerInterval);
    timerInterval = null;
    sessionStart = null;
    setEl('sessionTimer', '00:00:00');
}

/* =================================================================
   CONNECTION STATUS
   ================================================================= */
function setConn(label, dotClass) {
    const d = $('connDot'); const t = $('connText');
    if (d) d.className = 'dot ' + dotClass;
    if (t) t.textContent = label;
}

/* =================================================================
   IDLE ORB ANIMATION  (breathing ring)
   ================================================================= */
function drawOrbIdle() {
    const canvas = $('orbCanvas'); if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = cW(canvas), h = cH(canvas);
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;
    const baseR = Math.min(w, h) * 0.22;
    const breathe = Math.sin(idlePhase) * 0.06;
    const r = baseR * (1 + breathe);

    // Outer glow
    const grad = ctx.createRadialGradient(cx, cy, r * .6, cx, cy, r * 1.5);
    grad.addColorStop(0, hexAlpha(C.CYAN, .10));
    grad.addColorStop(1, 'transparent');
    ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(cx, cy, r * 1.5, 0, Math.PI * 2); ctx.fill();

    // Ring
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = hexAlpha(C.CYAN, .4 + breathe * .4);
    ctx.lineWidth = 2; ctx.stroke();

    // Inner dot
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fillStyle = hexAlpha(C.CYAN, .6); ctx.fill();

    idlePhase += 0.022;
    if (!isStreaming) animId = requestAnimationFrame(drawOrbIdle);
}

/* =================================================================
   VOICE ORB  (live mode)
   Ring 1: polar waveform (time domain)
   Ring 2: radial spectrum (frequency, log-spaced)
   Ring 3: window ring (last 60 backend windows)
   Center: orb core + verdict text
   ================================================================= */
let orbTooltipData = null;
let orbMouseX = 0, orbMouseY = 0;

function drawOrb() {
    if (!isStreaming || !analyser || currentMode !== 'orb') { return; }
    const canvas = $('orbCanvas'); if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = cW(canvas), h = cH(canvas);
    const cx = w * .5, cy = h * .5;
    const baseR = Math.min(w, h) * .18;

    // Get audio data
    const fftSize = analyser.fftSize;
    const tdBuf = new Float32Array(fftSize);
    analyser.getFloatTimeDomainData(tdBuf);
    const fdBuf = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(fdBuf);

    // Compute RMS from time-domain
    let rmsLive = 0;
    for (let i = 0; i < tdBuf.length; i++) rmsLive += tdBuf[i] * tdBuf[i];
    rmsLive = Math.sqrt(rmsLive / tdBuf.length);

    ctx.clearRect(0, 0, w, h);

    // --- ORB CORE ---
    const verdict = tlVerdicts.length ? tlVerdicts[tlVerdicts.length - 1] : null;
    const orbColor = verdict ? verdictColor(verdict) : C.CYAN;
    const orbR = baseR * (1 + Math.min(rmsLive * 2.5, 0.5));

    // Outer ambient glow
    const og = ctx.createRadialGradient(cx, cy, orbR * .5, cx, cy, orbR * 2.5);
    og.addColorStop(0, hexAlpha(orbColor, .18));
    og.addColorStop(1, 'transparent');
    ctx.fillStyle = og; ctx.beginPath(); ctx.arc(cx, cy, orbR * 2.5, 0, Math.PI * 2); ctx.fill();

    // Orb fill
    const og2 = ctx.createRadialGradient(cx - orbR * .25, cy - orbR * .25, 0, cx, cy, orbR);
    og2.addColorStop(0, hexAlpha(orbColor, .65));
    og2.addColorStop(1, hexAlpha(orbColor, .18));
    ctx.fillStyle = og2; ctx.beginPath(); ctx.arc(cx, cy, orbR, 0, Math.PI * 2); ctx.fill();

    // --- RING 1: POLAR WAVEFORM (time domain) ---
    const r1 = baseR * 1.45;
    ctx.beginPath();
    for (let i = 0; i < fftSize; i++) {
        const angle = (i / fftSize) * Math.PI * 2 - Math.PI / 2;
        const amp = tdBuf[i] * baseR * 0.4;
        const rx = cx + Math.cos(angle) * (r1 + amp);
        const ry = cy + Math.sin(angle) * (r1 + amp);
        i === 0 ? ctx.moveTo(rx, ry) : ctx.lineTo(rx, ry);
    }
    ctx.closePath();
    ctx.strokeStyle = hexAlpha(orbColor, .55);
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // --- RING 2: RADIAL SPECTRUM (frequency, log-spaced outward) ---
    const r2inner = baseR * 1.75, r2outer = baseR * 2.1;
    const freqBins = analyser.frequencyBinCount;
    const specCount = 80;
    for (let i = 0; i < specCount; i++) {
        const logIdx = Math.round(Math.pow(i / specCount, 1.8) * (freqBins - 1));
        const val = fdBuf[logIdx] / 255;
        const angle = (i / specCount) * Math.PI * 2 - Math.PI / 2;
        const rOuter = r2inner + val * (r2outer - r2inner) * 1.4;
        const peakR = r2inner + val * (r2outer - r2inner) * 1.6;

        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(angle) * r2inner, cy + Math.sin(angle) * r2inner);
        ctx.lineTo(cx + Math.cos(angle) * rOuter, cy + Math.sin(angle) * rOuter);
        // gradient per bar
        const sg = ctx.createLinearGradient(
            cx + Math.cos(angle) * r2inner, cy + Math.sin(angle) * r2inner,
            cx + Math.cos(angle) * rOuter, cy + Math.sin(angle) * rOuter
        );
        sg.addColorStop(0, hexAlpha(C.CYAN, .3));
        sg.addColorStop(1, hexAlpha(C.VIO, .7 + val * .3));
        ctx.strokeStyle = sg; ctx.lineWidth = 2.2; ctx.stroke();

        // peak cap
        ctx.beginPath();
        ctx.arc(cx + Math.cos(angle) * peakR, cy + Math.sin(angle) * peakR, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = hexAlpha('#ffffff', .5); ctx.fill();
    }

    // --- RING 3: WINDOW RING (last 60 backend windows) ---
    const r3 = baseR * 2.5;
    const ringThick = 14;
    const n = ringWindows.length;
    orbTooltipData = []; // reset for hover detection
    for (let i = 0; i < n; i++) {
        const win = ringWindows[i];
        // newest at top, clockwise
        const startAngle = -Math.PI / 2 + (i / RING_MAX) * Math.PI * 2;
        const endAngle = -Math.PI / 2 + ((i + 1) / RING_MAX) * Math.PI * 2;
        // height = raw spoof probability (as fraction of ringThick)
        const probH = win.rawProb * (ringThick + 8);
        const innerR = r3;
        const outerR = r3 + probH;
        ctx.beginPath();
        ctx.arc(cx, cy, outerR, startAngle, endAngle);
        ctx.arc(cx, cy, innerR, endAngle, startAngle, true);
        ctx.closePath();
        ctx.fillStyle = hexAlpha(verdictColor(win.rawPred), .65);
        ctx.fill();

        orbTooltipData.push({ i, startAngle, endAngle, innerR, outerR, win, cx, cy });
    }

    // Thin aggregated probability arc above ring
    if (tlAgg.length) {
        const aggPct = tlAgg[tlAgg.length - 1];
        const arcEnd = -Math.PI / 2 + aggPct * Math.PI * 2;
        ctx.beginPath();
        ctx.arc(cx, cy, r3 + ringThick + 5, -Math.PI / 2, arcEnd);
        ctx.strokeStyle = hexAlpha(C.CYAN, .8);
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // --- CENTER TEXT ---
    if (verdict) {
        const aggPct = tlAgg.length ? (tlAgg[tlAgg.length - 1] * 100).toFixed(1) + '%' : '—';
        ctx.fillStyle = hexAlpha(verdictColor(verdict), .9);
        ctx.font = `bold ${Math.max(12, Math.round(orbR * .55))}px var(--f-mono, monospace)`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(verdict, cx, cy - orbR * .22);
        ctx.font = `${Math.max(9, Math.round(orbR * .38))}px var(--f-mono, monospace)`;
        ctx.fillStyle = hexAlpha('#ffffff', .6);
        ctx.fillText(aggPct, cx, cy + orbR * .25);
    }
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; // reset

    // --- ORB HOVER TOOLTIP ---
    drawOrbTooltip(ctx, w, h);

    animId = requestAnimationFrame(drawOrb);
}

function drawOrbTooltip(ctx, w, h) {
    if (!orbTooltipData) return;
    const mx = orbMouseX, my = orbMouseY;
    for (const seg of orbTooltipData) {
        const dx = mx - seg.cx, dy = my - seg.cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < seg.innerR || dist > seg.outerR) continue;
        let angle = Math.atan2(dy, dx);
        if (angle < -Math.PI / 2) angle += Math.PI * 2;
        const sa = seg.startAngle < -Math.PI / 2 ? seg.startAngle + Math.PI * 2 : seg.startAngle;
        const ea = seg.endAngle < -Math.PI / 2 ? seg.endAngle + Math.PI * 2 : seg.endAngle;
        const normalizedAngle = angle < -Math.PI / 2 ? angle + Math.PI * 2 : angle;
        if (normalizedAngle < sa || normalizedAngle > ea) continue;
        // Draw tooltip
        const tw = 180, th = 70, tx = Math.min(mx + 12, w - tw - 5), ty = Math.min(my - 10, h - th - 5);
        ctx.fillStyle = 'rgba(5,8,18,.95)';
        ctx.strokeStyle = hexAlpha(C.CYAN, .35);
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.roundRect(tx, ty, tw, th, 6); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#e8edf5'; ctx.font = '11px monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        const w2 = seg.win;
        ctx.fillText(`Window #${winCount - RING_MAX + (seg.i) + 1}`, tx + 8, ty + 8);
        ctx.fillStyle = hexAlpha(verdictColor(w2.rawPred), .9);
        ctx.fillText(`Raw: ${w2.rawPred}  ${(w2.rawProb * 100).toFixed(1)}%`, tx + 8, ty + 22);
        ctx.fillStyle = hexAlpha(verdictColor(w2.aggPred), .9);
        ctx.fillText(`Agg: ${w2.aggPred}  ${(w2.aggProb * 100).toFixed(1)}%`, tx + 8, ty + 36);
        ctx.fillStyle = hexAlpha('#ffffff', .4);
        ctx.fillText(`RMS: ${w2.rms.toFixed(4)}`, tx + 8, ty + 50);
        ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
        return;
    }
}

// Track mouse for orb tooltip
const orbCanvas = $('orbCanvas');
if (orbCanvas) {
    orbCanvas.addEventListener('mousemove', e => {
        const rect = orbCanvas.getBoundingClientRect();
        // Convert to logical coordinates (not DPR-scaled)
        orbMouseX = (e.clientX - rect.left);
        orbMouseY = (e.clientY - rect.top);
    });
    orbCanvas.addEventListener('mouseleave', () => { orbMouseX = -999; orbMouseY = -999; });
}

/* =================================================================
   WATERFALL (joy-plot / pseudo-3D)
   ================================================================= */
function drawWaterfall() {
    if (!isStreaming || !analyser || currentMode !== 'wf') return;
    const canvas = $('vizCanvas'); if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = cW(canvas), h = cH(canvas);

    // Collect new frequency row
    const fdBuf = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(fdBuf);
    const cols = 80;
    const row = [];
    for (let i = 0; i < cols; i++) {
        const logIdx = Math.round(Math.pow(i / cols, 1.6) * (analyser.frequencyBinCount - 1));
        row.push(fdBuf[logIdx] / 255);
    }
    wfRows.push(row);
    if (wfRows.length > WF_MAX_ROWS) wfRows.shift();

    ctx.clearRect(0, 0, w, h);

    const rowH = h / WF_MAX_ROWS;
    const perspective = 0.6;
    const maxRidgeHeight = h * 0.4;

    for (let ri = 0; ri < wfRows.length; ri++) {
        const r = wfRows[ri];
        const yBase = h - ri * rowH * perspective;
        const alpha = 0.15 + 0.7 * (ri / wfRows.length);
        ctx.beginPath();
        ctx.moveTo(0, yBase);
        for (let ci = 0; ci < cols; ci++) {
            const x = (ci / cols) * w;
            const energy = r[ci];
            const ry = yBase - energy * maxRidgeHeight * (0.3 + 0.7 * ((ri + 1) / wfRows.length));
            ci === 0 ? ctx.moveTo(x, ry) : ctx.lineTo(x, ry);
        }
        ctx.lineTo(w, yBase);
        ctx.closePath();
        // magma colormap approximation
        const t = ri / wfRows.length;
        const rr = Math.round(20 + t * 235); const gr = Math.round(10 + t * 50); const bb = Math.round(60 + t * 80);
        ctx.fillStyle = `rgba(${rr},${gr},${bb},${alpha * .7})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(${rr},${gr},${bb},${Math.min(alpha, 0.6)})`;
        ctx.lineWidth = 1; ctx.stroke();
    }
    animId = requestAnimationFrame(drawWaterfall);
}

/* =================================================================
   SPECTROGRAM (flat, log Hz axis)
   ================================================================= */
function drawSpectrogram() {
    if (!isStreaming || !analyser || currentMode !== 'spec') return;
    const canvas = $('vizCanvas'); if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = cW(canvas), h = cH(canvas);
    const dpr = window.devicePixelRatio || 1;
    const pW = Math.round(w * dpr), pH = Math.round(h * dpr);

    const fdBuf = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(fdBuf);

    // Scroll left by 1 pixel
    if (spectroCol >= w) {
        const imgD = ctx.getImageData(dpr, 0, pW - dpr, pH);
        ctx.putImageData(imgD, 0, 0);
        spectroCol = w - 1;
    }

    // Draw column
    for (let y = 0; y < h; y++) {
        const freqIdx = Math.round((1 - y / h) * (analyser.frequencyBinCount - 1));
        const val = fdBuf[freqIdx] / 255;
        // Viridis-inspired colormap
        const R = Math.round(68 + val * 187);
        const G = Math.round(1 + val * 210);
        const B = Math.round(84 + (1 - val) * 80);
        ctx.fillStyle = `rgb(${R},${G},${B})`;
        ctx.fillRect(spectroCol, y, 1, 1);
    }
    spectroCol++;
    animId = requestAnimationFrame(drawSpectrogram);
}

/* =================================================================
   SCOPE (oscilloscope with phosphor persistence)
   ================================================================= */
function drawScope() {
    if (!isStreaming || !analyser || currentMode !== 'scope') return;
    const canvas = $('vizCanvas'); if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = cW(canvas), h = cH(canvas), mid = h / 2;

    const tdBuf = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(tdBuf);

    // Phosphor: fade existing content (persistence)
    ctx.fillStyle = 'rgba(7,10,18,.22)';
    ctx.fillRect(0, 0, w, h);

    // Build new points
    const pts = [];
    const slice = w / tdBuf.length;
    for (let i = 0; i < tdBuf.length; i++) {
        pts.push({ x: i * slice, y: mid + tdBuf[i] * mid * .85 });
    }

    // Draw with glow
    for (let pass = 0; pass < 2; pass++) {
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        if (pass === 0) {
            ctx.shadowColor = C.CYAN; ctx.shadowBlur = 12;
            ctx.strokeStyle = hexAlpha(C.CYAN, .35);
            ctx.lineWidth = 4;
        } else {
            ctx.shadowBlur = 0;
            const verdict = tlVerdicts.length ? tlVerdicts[tlVerdicts.length - 1] : null;
            ctx.strokeStyle = hexAlpha(verdict ? verdictColor(verdict) : C.CYAN, .9);
            ctx.lineWidth = 1.5;
        }
        ctx.stroke();
    }
    ctx.shadowBlur = 0;

    // Center line
    ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid);
    ctx.strokeStyle = hexAlpha(C.CYAN, .06); ctx.lineWidth = 1; ctx.stroke();

    animId = requestAnimationFrame(drawScope);
}

/* ─── Start appropriate visualizer ─────────────────────────────── */
function startViz() {
    if (animId) cancelAnimationFrame(animId);
    spectroCol = 0; wfRows = [];
    const hint = $('orbHint');

    if (currentMode === 'orb') {
        if ($('vizWrap')) $('vizWrap').style.display = 'none';
        if ($('orbWrap')) $('orbWrap').style.display = 'block';
        if (hint) hint.classList.add('hidden');
        drawOrb();
    } else {
        if ($('orbWrap')) $('orbWrap').style.display = 'block';
        if ($('vizWrap')) $('vizWrap').style.display = 'block';
        if (hint) hint.classList.add('hidden');
        initCanvas('vizWrap'); // ensure properly sized
        const vizCanvas = $('vizCanvas');
        if (vizCanvas) {
            const ctx = vizCanvas.getContext('2d');
            ctx.clearRect(0, 0, cW(vizCanvas), cH(vizCanvas));
        }
        if (currentMode === 'wf') { drawOrb(); drawWaterfall(); }
        else if (currentMode === 'spec') { drawOrb(); drawSpectrogram(); }
        else if (currentMode === 'scope') { drawOrb(); drawScope(); }
    }
}

/* ─── Tab switching ─────────────────────────────────────────────── */
document.querySelectorAll('.viz-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.viz-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentMode = tab.dataset.mode;
        if (animId) cancelAnimationFrame(animId);
        spectroCol = 0; wfRows = [];
        const vizCanvas = $('vizCanvas');
        if (vizCanvas) {
            const ctx = vizCanvas.getContext('2d');
            ctx.clearRect(0, 0, cW(vizCanvas), cH(vizCanvas));
        }
        if (currentMode !== 'orb') $('vizWrap').style.display = 'block';
        else $('vizWrap').style.display = 'none';
        if (isStreaming) startViz();
    });
});

/* =================================================================
   GAUGE (semicircular canvas, ResizeObserver-driven)
   ================================================================= */
function drawGauge(aggProb, rawProb) {
    const canvas = $('gaugeCanvas'); if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = cW(canvas), h = cH(canvas);
    const cx = w / 2, cy = h - 8;
    const r = Math.min(w * .45, h * .75);

    ctx.clearRect(0, 0, w, h);

    const startA = Math.PI, endA = 2 * Math.PI;

    // Zone arcs
    const zones = [
        { from: 0, to: .3, col: C.BON + '28' },
        { from: .3, to: .7, col: C.UNC + '22' },
        { from: .7, to: 1, col: C.SPO + '28' },
    ];
    ctx.lineWidth = 12; ctx.lineCap = 'round';
    for (const z of zones) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, startA + z.from * Math.PI, startA + z.to * Math.PI);
        ctx.strokeStyle = z.col; ctx.stroke();
    }

    // Raw marker
    if (rawProb != null) {
        const ang = startA + rawProb * Math.PI;
        ctx.beginPath();
        ctx.arc(cx + Math.cos(ang) * r, cy + Math.sin(ang) * r, 5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,.55)'; ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();
    }

    // Aggregated needle
    if (aggProb != null) {
        const ang = startA + aggProb * Math.PI;
        const col = aggProb > .7 ? C.SPO : aggProb < .3 ? C.BON : C.UNC;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(ang) * (r - 8), cy + Math.sin(ang) * (r - 8));
        ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.lineCap = 'round'; ctx.stroke();
        // center dot
        ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#fff'; ctx.fill();
    }

    // Labels
    ctx.fillStyle = '#4a5878'; ctx.font = '10px system-ui'; ctx.textBaseline = 'middle';
    ctx.textAlign = 'left'; ctx.fillText('0%', cx - r - 6, cy + 12);
    ctx.textAlign = 'right'; ctx.fillText('100%', cx + r + 6, cy + 12);
    ctx.textAlign = 'center'; ctx.fillText('50%', cx, cy - r - 4);
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
}

/* =================================================================
   FORENSIC TIMELINE (120 windows, hover crosshair)
   ================================================================= */
function drawTimeline() {
    const canvas = $('tlCanvas'); if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = cW(canvas), h = cH(canvas);
    const pad = { l: 36, r: 10, t: 10, b: 22 };
    const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;

    ctx.clearRect(0, 0, w, h);

    // Zone shading
    const z = [
        { y0: 0, y1: .3, col: C.BON + '07' }, { y0: .3, y1: .7, col: C.UNC + '05' }, { y0: .7, y1: 1, col: C.SPO + '07' }
    ];
    for (const zi of z) {
        ctx.fillStyle = zi.col;
        ctx.fillRect(pad.l, pad.t + (1 - zi.y1) * ch, cw, (zi.y1 - zi.y0) * ch);
    }

    // Y grid
    ctx.fillStyle = '#4a5878'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
    for (let v = 0; v <= 1; v += .25) {
        const y = pad.t + (1 - v) * ch;
        ctx.fillText(Math.round(v * 100) + '%', pad.l - 4, y + 3);
        ctx.strokeStyle = 'rgba(80,120,180,.07)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + cw, y); ctx.stroke();
    }

    if (!tlRaw.length) return;

    const stepX = cw / Math.max(TL_MAX - 1, 1);

    // Lines
    function drawLine(arr, col, lw, alpha) {
        ctx.beginPath(); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.globalAlpha = alpha;
        for (let i = 0; i < arr.length; i++) {
            const x = pad.l + i * stepX, y = pad.t + (1 - arr[i]) * ch;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke(); ctx.globalAlpha = 1;
    }
    drawLine(tlRaw, '#8899b8', 1.2, .45);
    drawLine(tlAgg, C.CYAN, 2.5, 1);
    ctx.shadowColor = C.CYAN; ctx.shadowBlur = 6;
    drawLine(tlAgg, C.CYAN, 2.5, 1);
    ctx.shadowBlur = 0;

    // Verdict dots on X
    for (let i = 0; i < tlVerdicts.length; i++) {
        const x = pad.l + i * stepX;
        ctx.beginPath(); ctx.arc(x, pad.t + ch + 8, 3, 0, Math.PI * 2);
        ctx.fillStyle = verdictColor(tlVerdicts[i]); ctx.fill();
    }
    ctx.textAlign = 'left';
}

// Crosshair tooltip
(() => {
    const canvas = $('tlCanvas'), tip = $('tlTip');
    if (!canvas || !tip) return;
    canvas.addEventListener('mousemove', e => {
        if (!tlRaw.length) { tip.classList.remove('show'); return; }
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const pad = { l: 36, r: 10 };
        const cw = rect.width - pad.l - pad.r;
        const stepX = cw / Math.max(TL_MAX - 1, 1);
        const idx = Math.round((mx - pad.l) / stepX);
        if (idx < 0 || idx >= tlRaw.length) { tip.classList.remove('show'); return; }
        const startWin = winCount - tlRaw.length + 1;
        tip.innerHTML = `<b>Win #${startWin + idx}</b><br>`
            + `Raw: ${(tlRaw[idx] * 100).toFixed(1)}% (${tlRaw[idx] > .5 ? 'SPOOF' : 'BONAFIDE'})<br>`
            + `Agg: ${(tlAgg[idx] * 100).toFixed(1)}% → ${tlVerdicts[idx]}`;
        tip.style.left = (mx + 14) + 'px'; tip.style.top = '4px';
        tip.classList.add('show');
    });
    canvas.addEventListener('mouseleave', () => tip.classList.remove('show'));
})();

/* =================================================================
   SPARKLINES (tiny line charts in metrics cards)
   ================================================================= */
function drawSparkline(canvasId, data, col) {
    const canvas = $(canvasId); if (!canvas || !data.length) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height;
    const mn = Math.min(...data), mx2 = Math.max(...data);
    const range = mx2 - mn || 1;
    ctx.clearRect(0, 0, w, h);
    ctx.beginPath();
    data.forEach((v, i) => {
        const x = (i / (data.length - 1 || 1)) * w;
        const y = h - (v - mn) / range * (h * .8) - (h * .1);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = col || C.CYAN; ctx.lineWidth = 1.5; ctx.globalAlpha = .6; ctx.stroke();
    ctx.globalAlpha = 1;
}

function updateSparklines() {
    drawSparkline('spkInf', spkInfData, C.CYAN);
    drawSparkline('spkPre', spkPreData, C.VIO);
    drawSparkline('spkTot', spkTotData, C.CYAN);
    drawSparkline('spkStab', spkStabData, C.BON);
    drawSparkline('spkWin', spkWin ? [] : [], C.CYAN);
    drawSparkline('spkErr', spkErrData, C.SPO);
}

/* =================================================================
   UPDATE ERRORS
   ================================================================= */
function updateErrors() {
    setEl('mErr', `${errWS}/${errModel}/${errUI}`);
    const el = $('mErr'); if (el) el.className = (errWS + errModel + errUI > 0) ? 'metric-val err' : 'metric-val';
    spkErrData.push(errWS + errModel + errUI);
    if (spkErrData.length > SPARK_MAX) spkErrData.shift();
}

/* =================================================================
   SPECTRAL CLIENT-SIDE ESTIMATES
   ================================================================= */
function updateSignalEstimates() {
    if (!analyser) return;
    const sr = audioCtx ? audioCtx.sampleRate : 48000;
    const fd = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(fd);
    const binHz = sr / (analyser.fftSize);

    // Dominant frequency
    let maxVal = 0, maxBin = 0;
    for (let i = 0; i < fd.length; i++) if (fd[i] > maxVal) { maxVal = fd[i]; maxBin = i; }
    setEl('domFreq', (maxBin * binHz).toFixed(0) + ' Hz');

    // Spectral centroid
    let sumWt = 0, sumTotal = 0;
    for (let i = 0; i < fd.length; i++) { sumWt += fd[i] * (i * binHz); sumTotal += fd[i]; }
    const centroid = sumTotal > 0 ? (sumWt / sumTotal) : 0;
    setEl('specCentroid', centroid.toFixed(0) + ' Hz');

    // Noise floor (mean of bottom 10 bins)
    const noiseSlice = fd.slice(1, 11);
    const nf = noiseSlice.reduce((a, b) => a + b, 0) / noiseSlice.length;
    setEl('noiseFloor', '-' + (255 - nf).toFixed(0) + ' dB (est.)');
}

/* =================================================================
   HANDLE BACKEND RESULT
   ================================================================= */
function handleResult(d) {
    try {
        const rawPred = d.raw_prediction;
        const aggPred = d.agg_prediction;
        const rawProb = d.raw_spoof_prob;
        const aggProb = d.agg_spoof_prob;
        const stability = d.stability;

        // Parse RMS
        let rmsVal = 0;
        if (d.audio_status) {
            const m = d.audio_status.match(/RMS=([\d.]+)/);
            if (m) rmsVal = parseFloat(m[1]);
        }

        winCount++;

        // --- Ambient glow ---
        setAmbient(aggPred || '');

        // --- Verdict panel ---
        const vEl = $('verdictMain');
        if (vEl) {
            vEl.textContent = aggPred || '—';
            vEl.className = 'verdict-main' + (
                aggPred === 'BONAFIDE' ? ' v-bon' : aggPred === 'SPOOF' ? ' v-spo' :
                    aggPred === 'UNCERTAIN' ? ' v-unc' : aggPred === 'INSUFFICIENT_AUDIO' ? ' v-ins' : ''
            );
        }
        const iconEl = $('verdictIcon');
        if (iconEl) iconEl.textContent =
            aggPred === 'BONAFIDE' ? '✅' : aggPred === 'SPOOF' ? '🚨' :
                aggPred === 'UNCERTAIN' ? '⚠️' : aggPred === 'INSUFFICIENT_AUDIO' ? '🔇' : '🔈';
        const badgeEl = $('verdictBadge');
        if (badgeEl) {
            if (aggPred === 'BONAFIDE') { badgeEl.textContent = 'Genuine Human Voice'; badgeEl.className = 'verdict-badge vb-bon'; }
            else if (aggPred === 'SPOOF') { badgeEl.textContent = 'AI-Generated / Spoofed'; badgeEl.className = 'verdict-badge vb-spo'; }
            else if (aggPred === 'UNCERTAIN') { badgeEl.textContent = 'Unstable Probabilities'; badgeEl.className = 'verdict-badge vb-unc'; }
            else badgeEl.className = 'verdict-badge vb-hidden';
        }
        setEl('verdictRaw', `Raw: ${rawPred || '—'}`);

        // --- Gauge ---
        const skipProb = rawPred === 'INSUFFICIENT_AUDIO';
        drawGauge(skipProb ? null : aggProb, skipProb ? null : rawProb);
        setEl('gaugeAgg', skipProb ? '—' : (aggProb * 100).toFixed(1) + '%');
        setEl('gaugeRaw', skipProb ? '—' : (rawProb * 100).toFixed(1) + '%');

        // --- RMS meter ---
        const rmsPct = Math.min(rmsVal / .5 * 100, 100);
        setEl('rmsFill', rmsPct + '%', true);
        if (rmsVal > rmsPeakVal) rmsPeakVal = rmsVal;
        setEl('rmsPeak', Math.min(rmsPeakVal / .5 * 100, 100) + '%', true);
        setEl('rmsValue', 'RMS=' + rmsVal.toFixed(4));
        const rsEl = $('rmsStatus');
        if (rsEl) {
            if (rmsVal < .01) { rsEl.textContent = 'Too Quiet'; rsEl.className = 'rms-status quiet'; }
            else if (rmsVal > .45) { rsEl.textContent = 'Clipping'; rsEl.className = 'rms-status clipping'; }
            else { rsEl.textContent = 'Good'; rsEl.className = 'rms-status good'; }
        }
        clearTimeout(rmsPeakTimer);
        rmsPeakTimer = setTimeout(() => { rmsPeakVal = Math.max(rmsPeakVal * .85, rmsVal); }, 900);

        // --- Flip counter ---
        if (rawPred && rawPred !== 'INSUFFICIENT_AUDIO') {
            if (lastRawPred && lastRawPred !== rawPred) flipCount++;
            lastRawPred = rawPred;
            setEl('flipCount', flipCount);
        }

        // --- Metrics ---
        setEl('mWin', winCount);
        if (d.latency_inference_ms != null) {
            const v = parseFloat(d.latency_inference_ms);
            setEl('mInf', v.toFixed(0) + ' ms');
            spkInfData.push(v); if (spkInfData.length > SPARK_MAX) spkInfData.shift();
        }
        if (d.latency_pre_ms != null) {
            const v = parseFloat(d.latency_pre_ms);
            setEl('mPre', v.toFixed(0) + ' ms');
            spkPreData.push(v); if (spkPreData.length > SPARK_MAX) spkPreData.shift();
        }
        if (d.latency_total_ms != null) {
            const v = parseFloat(d.latency_total_ms);
            setEl('mTot', v.toFixed(0) + ' ms');
            spkTotData.push(v); if (spkTotData.length > SPARK_MAX) spkTotData.shift();
        }
        if (stability != null) {
            const v = parseFloat(stability);
            setEl('mStab', (v * 100).toFixed(1) + '%');
            spkStabData.push(v); if (spkStabData.length > SPARK_MAX) spkStabData.shift();
        }
        updateSparklines();

        // --- Timeline + Ring (exclude INSUFFICIENT_AUDIO) ---
        if (rawPred !== 'INSUFFICIENT_AUDIO') {
            tlRaw.push(rawProb || 0); tlAgg.push(aggProb || 0); tlVerdicts.push(aggPred);
            if (tlRaw.length > TL_MAX) { tlRaw.shift(); tlAgg.shift(); tlVerdicts.shift(); }
            drawTimeline();

            ringWindows.push({ rawPred, rawProb: rawProb || 0, aggPred, aggProb: aggProb || 0, rms: rmsVal });
            if (ringWindows.length > RING_MAX) ringWindows.shift();
        }

        // --- Session log ---
        if (rawPred !== 'INSUFFICIENT_AUDIO') {
            const now = new Date();
            const ts = now.toTimeString().slice(0, 8);
            const row = {
                n: winCount, ts, rawPred, rawProb, aggPred, aggProb, rms: rmsVal,
                lat: d.latency_total_ms != null ? parseFloat(d.latency_total_ms).toFixed(0) : '—',
                stab: stability != null ? (parseFloat(stability) * 100).toFixed(1) + '%' : '—'
            };
            histRows.unshift(row);
            if (histRows.length > HIST_MAX) histRows.pop();
            insertLogRow(row);
        }

        // Client-side signal estimates every 3 frames
        if (winCount % 3 === 0) updateSignalEstimates();

    } catch (e) {
        console.error('[UI ERROR] handleResult:', e);
        errUI++; updateErrors();
    }
}

/* ─── Session log row insert ────────────────────────────────────── */
function insertLogRow(r) {
    const body = $('logBody'); if (!body) return;
    const tr = document.createElement('tr');
    const rp = (v, p) => v != null ? (v * 100).toFixed(1) + '%' : '—';
    tr.innerHTML =
        `<td>${r.n}</td><td>${r.ts}</td>` +
        `<td><span class="${pillClass(r.rawPred)}">${r.rawPred}</span></td>` +
        `<td>${rp(r.rawProb)}</td>` +
        `<td><span class="${pillClass(r.aggPred)}">${r.aggPred}</span></td>` +
        `<td>${rp(r.aggProb)}</td>` +
        `<td>${r.rms.toFixed(4)}</td><td>${r.lat} ms</td><td>${r.stab}</td>`;
    body.insertBefore(tr, body.firstChild);
    while (body.children.length > HIST_MAX) body.removeChild(body.lastChild);
}

/* =================================================================
   EXPORT + SNAPSHOT
   ================================================================= */
function exportCSV() {
    if (!histRows.length) return;
    const hdr = '#,Time,Raw Pred,Raw%,Agg Pred,Agg%,RMS,Latency ms,Stability\n';
    const rows = histRows.map(r =>
        `${r.n},${r.ts},${r.rawPred},${r.rawProb != null ? (r.rawProb * 100).toFixed(1) : ''},` +
        `${r.aggPred},${r.aggProb != null ? (r.aggProb * 100).toFixed(1) : ''},${r.rms.toFixed(4)},${r.lat},${r.stab}`
    ).join('\n');
    dlBlob(hdr + rows, 'text/csv', 'vfl_session.csv');
}
function exportJSON() {
    if (!histRows.length) return;
    dlBlob(JSON.stringify({ session: { windows: histRows.length, flips: flipCount }, data: histRows }, null, 2),
        'application/json', 'vfl_session.json');
}
function snapshot() {
    const canvas = $('orbCanvas'); if (!canvas) return;
    canvas.toBlob(blob => { if (!blob) return; dlBlob(blob, 'image/png', 'vfl_orb.png'); }, 'image/png');
}
function dlBlob(data, type, name) {
    const blob = data instanceof Blob ? data : new Blob([data], { type });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

if ($('btnClear')) $('btnClear').onclick = () => { histRows = []; const b = $('logBody'); if (b) b.innerHTML = ''; };
if ($('btnCSV')) $('btnCSV').onclick = exportCSV;
if ($('btnJSON')) $('btnJSON').onclick = exportJSON;
if ($('btnSnapshot')) $('btnSnapshot').onclick = snapshot;

/* =================================================================
   WEBSOCKET
   ================================================================= */
async function connectWS(sampleRate) {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/stream?sample_rate=${sampleRate}`;
    return new Promise(resolve => {
        ws = new WebSocket(url);
        ws.binaryType = 'arraybuffer';
        ws.onopen = () => {
            setConn(isReconnecting ? 'Reconnected' : 'Connected', 'dot-green');
            isReconnecting = false; reconnectDelay = 1000;
            resolve(true);
        };
        ws.onmessage = msg => {
            try {
                let d;
                try { d = JSON.parse(msg.data); }
                catch (e) { console.error('[WS ERROR] Bad JSON:', e); errWS++; updateErrors(); return; }
                if (d.error) { console.error('[MODEL ERROR]', d.error); errModel++; updateErrors(); return; }
                handleResult(d);
            } catch (e) {
                console.error('[WS ERROR] Message processing:', e); errWS++; updateErrors();
            }
        };
        ws.onerror = err => { console.error('[WS ERROR] Socket:', err); errWS++; updateErrors(); resolve(false); };
        ws.onclose = () => {
            setConn('Reconnecting…', 'dot-amber');
            if (isStreaming) {
                isReconnecting = true;
                setTimeout(() => connectWS(sampleRate), reconnectDelay);
                reconnectDelay = Math.min(reconnectDelay * 1.5, 8000);
            } else { setConn('Disconnected', 'dot-gray'); }
        };
    });
}

/* =================================================================
   START MICROPHONE
   ================================================================= */
const startBtn = $('startBtn'), stopBtn = $('stopBtn');
if (startBtn) startBtn.onclick = startMic;

async function startMic() {
    try {
        const micErrEl = $('micError');
        if (micErrEl) { micErrEl.classList.remove('show'); micErrEl.textContent = ''; }
        setConn('Requesting Mic…', 'dot-amber');

        stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

        startBtn.disabled = true;
        if (stopBtn) stopBtn.disabled = false;
        setEl('chipMic', 'Mic: On');

        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const sr = audioCtx.sampleRate;
        setEl('chipSR', (sr / 1000).toFixed(1) + ' kHz');

        const ok = await connectWS(sr);
        if (!ok) throw new Error('WebSocket failed');

        // AnalyserNode — does NOT alter audio
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048; analyser.smoothingTimeConstant = .8;

        source = audioCtx.createMediaStreamSource(stream);
        source.connect(analyser);

        // ScriptProcessor for sending chunks to backend
        const CHUNK_SEC = 1.0;
        const samplesPerChunk = Math.floor(sr * CHUNK_SEC);
        let buf = new Float32Array(samplesPerChunk), bufIdx = 0;

        processor = audioCtx.createScriptProcessor(4096, 1, 1);
        gainNode = audioCtx.createGain(); gainNode.gain.value = 0;

        processor.onaudioprocess = e => {
            const inp = e.inputBuffer.getChannelData(0);
            for (let i = 0; i < inp.length; i++) {
                buf[bufIdx++] = inp[i];
                if (bufIdx >= samplesPerChunk) {
                    if (ws && ws.readyState === WebSocket.OPEN) ws.send(buf.buffer.slice(0));
                    bufIdx = 0;
                }
            }
        };
        source.connect(processor);
        processor.connect(gainNode);
        gainNode.connect(audioCtx.destination);

        isStreaming = true;
        const lb = $('liveBadge'); if (lb) lb.classList.add('on');
        startTimer();
        startViz();

    } catch (err) {
        console.error('[UI ERROR] Mic init:', err);
        errUI++; updateErrors();
        const micErrEl = $('micError');
        if (micErrEl) {
            micErrEl.textContent = err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError'
                ? 'Microphone permission denied. Allow access in browser settings and try again.'
                : 'Microphone error: ' + err.message;
            micErrEl.classList.add('show');
        }
        setConn('Error', 'dot-red');
        setEl('chipMic', 'Mic: Error');
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
    }
}

/* =================================================================
   STOP
   ================================================================= */
function stopMic() {
    isStreaming = false;
    if (animId) { cancelAnimationFrame(animId); animId = null; }
    [processor, gainNode, source, analyser].forEach(n => { try { n && n.disconnect(); } catch (_) { } });
    try { audioCtx && audioCtx.close(); } catch (_) { }
    stream && stream.getTracks().forEach(t => t.stop());
    if (ws) { const _ws = ws; ws = null; try { _ws.close(); } catch (_) { } }

    startBtn && (startBtn.disabled = false);
    stopBtn && (stopBtn.disabled = true);
    setEl('chipMic', 'Mic: Off');
    const lb = $('liveBadge'); if (lb) lb.classList.remove('on');
    setConn('Disconnected', 'dot-gray');
    stopTimer();

    // Cap arrays
    while (tlRaw.length > TL_MAX) { tlRaw.shift(); tlAgg.shift(); tlVerdicts.shift(); }
    while (ringWindows.length > RING_MAX) ringWindows.shift();

    rmsPeakVal = 0;
    spectroCol = 0; wfRows = [];

    // Return to idle orb
    const hint = $('orbHint'); if (hint) hint.classList.remove('hidden');
    idlePhase = 0; drawOrbIdle();
}
if (stopBtn) stopBtn.onclick = stopMic;

/* =================================================================
   VISIBILITY + PAUSE
   ================================================================= */
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        if (animId) { cancelAnimationFrame(animId); animId = null; }
    } else if (isStreaming) startViz();
    else drawOrbIdle();
});

/* =================================================================
   KEYBOARD SHORTCUTS
   ================================================================= */
document.addEventListener('keydown', e => {
    if (e.target !== document.body && e.target.tagName !== 'CANVAS') return;
    if (e.code === 'Space') {
        e.preventDefault();
        isStreaming ? stopMic() : startBtn && !startBtn.disabled && startMic();
    }
    const modeMap = { '1': 'orb', '2': 'wf', '3': 'spec', '4': 'scope' };
    if (modeMap[e.key]) {
        const tab = document.querySelector(`.viz-tab[data-mode="${modeMap[e.key]}"]`);
        if (tab) tab.click();
    }
});

/* =================================================================
   BOOT — init everything on load
   ================================================================= */
// Init all canvas containers
['orbWrap', 'vizWrap', 'gaugeWrap', 'tlWrap'].forEach(initCanvas);

// Draw initial gauge (blank)
drawGauge(null, null);
drawTimeline();

// Start idle orb
drawOrbIdle();

// Log resize observer sanity
console.log('[VFL] Canvas containers initialized with ResizeObserver. No feedback loop possible.');
