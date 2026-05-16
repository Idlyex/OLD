/**
 * Standalone Puppeteer card renderer — exact 1:1 POLYx TradeCard.jsx replica.
 * Usage: node render_card.js <trade_data.json> <output.png>
 *   or:  node render_card.js '<json_string>' <output.png>
 * 
 * Reads trade data, generates HTML with identical CSS to POLYx TradeCard,
 * screenshots via Puppeteer, outputs PNG.
 */
const puppeteer = require('D:\\AF\\POLYx\\cards\\node_modules\\puppeteer');
const fs = require('fs');
const path = require('path');

// ── Parse input ─────────────────────────────────────────
const arg1 = process.argv[2];
const outputPath = process.argv[3] || path.join(__dirname, 'output.png');
let trade;
try {
  if (arg1.startsWith('{')) {
    trade = JSON.parse(arg1);
  } else {
    trade = JSON.parse(fs.readFileSync(arg1, 'utf-8'));
  }
} catch (e) {
  console.error('Failed to parse trade data:', e.message);
  process.exit(1);
}

// ── Themes — exact POLYx ────────────────────────────────
const THEMES = {
  emerald:  { accent: '#00D084', glow: 'rgba(0,208,132,',   bg1: '#0A1210', bg2: '#0C1814', bg3: '#0E1E1A' },
  cyan:     { accent: '#00E5FF', glow: 'rgba(0,229,255,',    bg1: '#0A1318', bg2: '#0C181E', bg3: '#0E1D25' },
  teal:     { accent: '#2DD4BF', glow: 'rgba(45,212,191,',   bg1: '#0A1314', bg2: '#0C181A', bg3: '#0E1E20' },
  gold:     { accent: '#FFB84D', glow: 'rgba(255,184,77,',   bg1: '#12100A', bg2: '#18140C', bg3: '#1E1A0E' },
  purple:   { accent: '#A78BFA', glow: 'rgba(167,139,250,',  bg1: '#0D0A18', bg2: '#110E1E', bg3: '#151225' },
  fuchsia:  { accent: '#E879F9', glow: 'rgba(232,121,249,',  bg1: '#120A14', bg2: '#180C1A', bg3: '#1E0E20' },
  rose:     { accent: '#FF4D6A', glow: 'rgba(255,77,106,',   bg1: '#120A0C', bg2: '#180C10', bg3: '#1E0E14' },
  blue:     { accent: '#60A5FA', glow: 'rgba(96,165,250,',   bg1: '#0A0E18', bg2: '#0C121E', bg3: '#0E1625' },
  indigo:   { accent: '#818CF8', glow: 'rgba(129,140,248,',  bg1: '#0B0C18', bg2: '#0E101E', bg3: '#111425' },
};

function pickTheme(pnlPct, won) {
  if (!won) return 'rose';
  if (pnlPct >= 200) return 'fuchsia';
  if (pnlPct >= 100) return 'purple';
  if (pnlPct >= 50)  return 'gold';
  if (pnlPct >= 25)  return 'teal';
  if (pnlPct >= 10)  return 'cyan';
  return 'emerald';
}

// ── Helpers ──────────────────────────────────────────────
function fmtDollar(v) {
  const abs = Math.abs(v);
  if (abs >= 1000) return `${v >= 0 ? '' : '-'}$${(abs / 1000).toFixed(1)}k`;
  return `${v >= 0 ? '' : '-'}$${abs.toFixed(2)}`;
}
function fmtDate(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return d.toLocaleDateString('en-US', { month: '2-digit', day: '2-digit', year: 'numeric' });
}
function getDuration(secs) {
  if (!secs) return '';
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}
function getMarketDuration(slug) {
  if (!slug) return '';
  if (slug.includes('5m')) return '5min';
  if (slug.includes('15m')) return '15min';
  if (slug.includes('60m') || slug.includes('1h')) return '1h';
  return '';
}

// ── MiniChart SVG ────────────────────────────────────────
function buildMiniChartSVG(data, color, width, height) {
  if (!data || data.length < 3) return '';
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const areaPoints = `0,${height} ${points} ${width},${height}`;
  const zeroY = max > 0 && min < 0
    ? height - ((0 - min) / range) * (height - 4) - 2
    : height * 0.5;
  const gid = 'cg' + Math.random().toString(36).slice(2, 6);
  return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="filter:drop-shadow(0 0 8px ${color}40);display:block">
    <defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${color}" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="${color}" stop-opacity="0.01"/>
    </linearGradient></defs>
    <line x1="0" y1="${zeroY}" x2="${width}" y2="${zeroY}" stroke="rgba(255,255,255,0.025)" stroke-width="1" stroke-dasharray="4,6"/>
    <polygon points="${areaPoints}" fill="url(#${gid})"/>
    <polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;
}

// ── Generate synthetic PnL chart data ────────────────────
function generateChartData(trade) {
  const pnlPct = trade.pnl_pct || trade.pnlPct || 0;
  const n = 30;
  // Simple seeded random
  let seed = 0;
  const slug = trade.slug || '';
  for (let i = 0; i < slug.length; i++) seed = ((seed << 5) - seed + slug.charCodeAt(i)) | 0;
  function rand() { seed = (seed * 16807 + 0) % 2147483647; return (seed & 0x7fffffff) / 0x7fffffff; }
  function gauss() { return Math.sqrt(-2*Math.log(rand()+0.001))*Math.cos(2*Math.PI*rand()); }
  const data = [0];
  for (let i = 1; i < n; i++) {
    const t = i / (n - 1);
    data.push(pnlPct * t + gauss() * (Math.abs(pnlPct) * 0.12 + 1.5));
  }
  data[n - 1] = pnlPct;
  return data;
}

// ── Build HTML ───────────────────────────────────────────
function buildHTML(trade) {
  const won = !!trade.won;
  const pnlPct = trade.pnl_pct || trade.pnlPct || 0;
  const pnlUsd = trade.pnl_usd || trade.pnl || 0;
  const direction = trade.direction || 'UP';
  const isUp = direction === 'UP';
  const shares = trade.shares || 0;
  const slug = trade.slug || '';
  const confidence = trade.confidence || 0;
  const entryPrice = trade.entry_price || trade.entryPrice || 0;
  const exitPrice = trade.exit_price || trade.exitPrice || 0;
  const solEntry = trade.sol_at_entry || trade.entryBinancePrice || 0;
  const solExit = trade.sol_at_exit || trade.exitBinancePrice || solEntry;
  const ptb = trade.ptb || trade.priceToBeat || solEntry;
  const solDelta = solExit - solEntry;
  const holdSecs = trade.hold_time_s || 0;
  const exitTime = trade.exit_time || '';
  const allProbs = trade.all_model_probs || {};
  const dryRun = !!trade.dry_run;
  const marketDur = getMarketDuration(slug);

  const themeKey = trade.theme || pickTheme(pnlPct, won);
  const t = THEMES[themeKey] || THEMES.emerald;
  const accent = t.accent;
  const glow = t.glow;

  // Chart
  const chartData = generateChartData(trade);
  const chartSVG = buildMiniChartSVG(chartData, accent, 440, 56);

  // Model bars HTML
  let modelBarsHTML = '';
  const models = Object.entries(allProbs).sort((a,b) => a[0].localeCompare(b[0]));
  if (models.length > 0) {
    modelBarsHTML = `
    <div style="border-top:1px solid ${glow}0.08);margin-top:12px;padding-top:10px">
      <div style="font-size:8px;color:rgba(255,255,255,0.2);text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:8px">Model Agreement</div>
      ${models.map(([name, pUp]) => {
        const dp = Math.max(pUp, 1 - pUp);
        const d = pUp > 0.5 ? 'UP' : 'DN';
        const agreed = d === direction;
        const barPct = (dp * 100).toFixed(0);
        return `<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
          <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(255,255,255,${agreed?0.6:0.3});width:80px">${name}</span>
          <div style="flex:1;height:6px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden">
            <div style="width:${barPct}%;height:100%;background:${agreed ? accent : glow+'0.15)'};border-radius:3px"></div>
          </div>
          <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:${agreed?accent:'rgba(255,255,255,0.3)'};width:55px;text-align:right">${barPct}% ${d}</span>
        </div>`;
      }).join('')}
    </div>`;
  }

  // Confidence badge
  const confHTML = confidence > 0 ? `
    <div style="display:flex;align-items:center;gap:6px;margin-top:4px">
      <span style="font-size:8px;color:rgba(255,255,255,0.2);text-transform:uppercase;letter-spacing:0.1em">Conf</span>
      <span style="font-size:13px;font-weight:700;color:${accent};font-family:'JetBrains Mono',monospace">${(confidence*100).toFixed(0)}%</span>
    </div>` : '';

  return `<!DOCTYPE html><html><head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  *{margin:0;padding:0;box-sizing:border-box;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
  body{background:transparent;font-family:'Inter',system-ui,sans-serif}
</style>
</head><body>
<div id="trade-card" style="
  width:480px;
  border-radius:22px;
  overflow:hidden;
  position:relative;
  background:linear-gradient(155deg, ${t.bg1} 0%, ${t.bg2} 40%, ${t.bg3} 70%, ${t.bg1} 100%);
  border:1px solid ${glow}0.12);
  box-shadow:0 20px 60px rgba(0,0,0,0.7), 0 0 80px ${glow}0.06), inset 0 1px 0 rgba(255,255,255,0.04);
">
  <!-- Top glow -->
  <div style="position:absolute;top:0;left:50%;transform:translateX(-50%);width:70%;height:120px;pointer-events:none;
    background:radial-gradient(ellipse at 50% 0%, ${glow}0.08) 0%, transparent 70%)"></div>
  
  <!-- Grid -->
  <div style="position:absolute;inset:0;pointer-events:none;opacity:0.015;
    background-image:linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px);
    background-size:40px 40px"></div>

  <div style="position:relative;z-index:10;padding:24px 28px 24px 28px">
    
    <!-- Header -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px">
      <div style="display:flex;align-items:center;gap:12px">
        <!-- SOL icon box -->
        <div style="width:36px;height:36px;border-radius:12px;display:flex;align-items:center;justify-content:center;
          background:${glow}0.08);border:1px solid ${glow}0.15)">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="${accent}">
            <path d="M17.28 13.4a.47.47 0 0 0-.34-.14H4.58a.24.24 0 0 0-.17.41l2.74 2.74c.09.09.21.14.34.14h12.36a.24.24 0 0 0 .17-.41l-2.74-2.74zM7.15 10.74c.09.09.21.14.34.14h12.36a.24.24 0 0 0 .17-.41l-2.74-2.74a.47.47 0 0 0-.34-.14H4.58a.24.24 0 0 0-.17.41l2.74 2.74zM19.85 4.33a.24.24 0 0 0-.17-.07H7.32c-.13 0-.25.05-.34.14L4.24 7.14a.24.24 0 0 0 .17.41h12.36c.13 0 .25-.05.34-.14l2.74-2.74a.24.24 0 0 0 0-.34z"/>
          </svg>
        </div>
        <div style="display:flex;flex-direction:column">
          <span style="font-size:14px;font-weight:700;color:rgba(255,255,255,0.9);letter-spacing:-0.02em">DESTROYER</span>
          <span style="font-size:10px;color:rgba(255,255,255,0.3);font-family:'JetBrains Mono',monospace">${exitTime}</span>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        ${marketDur ? `<span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:rgba(255,255,255,0.25);padding:2px 8px;border-radius:4px;background:rgba(255,255,255,0.03)">${marketDur}</span>` : ''}
        <span style="font-size:11px;font-weight:700;color:rgba(255,255,255,0.35);font-family:'JetBrains Mono',monospace">${shares.toFixed(1)}x</span>
        <div style="display:flex;align-items:center;gap:6px;padding:6px 12px;border-radius:8px;
          background:${glow}0.06);border:1px solid ${glow}0.20)">
          <span style="font-size:9px;color:${accent}">${isUp ? '&#9650;' : '&#9660;'}</span>
          <span style="font-size:11px;font-weight:700;color:${accent}">SOL ${direction}</span>
        </div>
      </div>
    </div>

    <!-- ROI Hero -->
    <div style="margin-bottom:8px">
      <div style="font-size:10px;color:rgba(255,255,255,0.25);text-transform:uppercase;letter-spacing:0.2em;margin-bottom:8px;font-weight:600">ROI</div>
      <div style="display:flex;align-items:baseline;gap:12px">
        <span style="font-size:20px;color:${accent};opacity:0.8">${won ? '&#8593;' : '&#8595;'}</span>
        <span style="font-size:52px;font-weight:900;letter-spacing:-0.04em;line-height:1;color:${accent};
          text-shadow:0 0 40px ${glow}0.3), 0 0 80px ${glow}0.15)">
          ${Math.abs(pnlPct).toFixed(pnlPct >= 100 ? 0 : 1)}%
        </span>
        <div style="display:flex;flex-direction:column;margin-left:8px">
          <span style="font-size:15px;font-weight:700;color:rgba(255,255,255,0.6)">${fmtDollar(pnlUsd)}</span>
          <span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:rgba(255,255,255,0.2)">${getDuration(holdSecs)}</span>
        </div>
        <div style="margin-left:auto">
          <span style="font-size:10px;font-weight:900;text-transform:uppercase;letter-spacing:0.05em;padding:4px 10px;border-radius:6px;
            background:${won ? glow+'0.10)' : 'rgba(255,77,106,0.10)'};
            color:${won ? accent : '#FF4D6A'};
            border:1px solid ${won ? glow+'0.20)' : 'rgba(255,77,106,0.20)'}">
            ${won ? 'WIN' : 'LOSS'}
          </span>
        </div>
      </div>
    </div>

    <!-- Chart -->
    ${chartSVG ? `<div style="margin:20px -4px">${chartSVG}</div>` : ''}

    <!-- SOL Price row -->
    <div style="display:flex;align-items:stretch;padding:16px;margin-bottom:16px;border-radius:12px;margin-left:-4px;margin-right:-4px;
      background:linear-gradient(135deg, ${glow}0.03) 0%, rgba(255,255,255,0.01) 100%);
      border:1px solid ${glow}0.06)">
      <div style="flex:1;text-align:center">
        <div style="font-size:9px;color:rgba(255,255,255,0.25);text-transform:uppercase;letter-spacing:0.15em;margin-bottom:6px;font-weight:600">Entry SOL</div>
        <div style="font-size:16px;font-family:'JetBrains Mono',monospace;font-weight:700;color:rgba(255,255,255,0.8);letter-spacing:-0.02em">$${solEntry.toFixed(2)}</div>
      </div>
      <div style="width:1px;align-self:stretch;margin:0 8px;background:linear-gradient(180deg, transparent, ${glow}0.15), transparent)"></div>
      <div style="flex:1;text-align:center">
        <div style="font-size:9px;color:rgba(255,255,255,0.25);text-transform:uppercase;letter-spacing:0.15em;margin-bottom:6px;font-weight:600">PTB (Open)</div>
        <div style="font-size:16px;font-family:'JetBrains Mono',monospace;font-weight:700;color:rgba(255,255,255,0.5);letter-spacing:-0.02em">$${ptb.toFixed(2)}</div>
      </div>
      <div style="width:1px;align-self:stretch;margin:0 8px;background:linear-gradient(180deg, transparent, ${glow}0.15), transparent)"></div>
      <div style="flex:1;text-align:center">
        <div style="font-size:9px;color:rgba(255,255,255,0.25);text-transform:uppercase;letter-spacing:0.15em;margin-bottom:6px;font-weight:600">Resolve SOL</div>
        <div style="font-size:16px;font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:-0.02em;color:${solDelta >= 0 ? accent : '#FF4D6A'}">$${solExit.toFixed(2)}</div>
      </div>
    </div>

    <!-- Share Entry → Exit + Peak -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;padding:0 4px">
      <div style="display:flex;align-items:center;gap:16px">
        <div>
          <div style="font-size:8px;color:rgba(255,255,255,0.2);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:2px">Share Entry</div>
          <div style="font-size:13px;font-family:'JetBrains Mono',monospace;font-weight:700;color:rgba(255,255,255,0.6)">${(entryPrice).toFixed(2)}¢</div>
        </div>
        <div style="color:rgba(255,255,255,0.1)">→</div>
        <div>
          <div style="font-size:8px;color:rgba(255,255,255,0.2);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:2px">Share Exit</div>
          <div style="font-size:13px;font-family:'JetBrains Mono',monospace;font-weight:700;color:${glow}0.7)">${(exitPrice).toFixed(2)}¢</div>
        </div>
      </div>
      ${confHTML}
    </div>

    <!-- Model agreement -->
    ${modelBarsHTML}

    <!-- Footer -->
    <div style="border-top:1px solid ${glow}0.06);margin-top:12px;padding-top:10px;display:flex;align-items:center;gap:8px">
      <div style="width:7px;height:7px;border-radius:50%;background:${dryRun ? '#60A5FA' : '#FF3C3C'}"></div>
      <span style="font-size:9px;color:rgba(255,255,255,0.3)">${dryRun ? 'DRY RUN' : 'LIVE'}  ·  DESTROYER 2.0</span>
    </div>

  </div>
</div>
</body></html>`;
}

// ── Render ────────────────────────────────────────────────
async function render() {
  const html = buildHTML(trade);
  
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'],
    defaultViewport: { width: 600, height: 900, deviceScaleFactor: 2 },
  });

  const page = await browser.newPage();
  await page.setContent(html, { waitUntil: 'networkidle0', timeout: 10000 });
  
  // Wait for fonts
  await page.evaluateHandle('document.fonts.ready');
  await new Promise(r => setTimeout(r, 300));

  const card = await page.$('#trade-card');
  if (!card) {
    console.error('Card element not found');
    await browser.close();
    process.exit(1);
  }

  await card.screenshot({ path: outputPath, omitBackground: true });
  await browser.close();
  
  const size = fs.statSync(outputPath).size;
  console.log(JSON.stringify({ ok: true, path: outputPath, size }));
}

render().catch(err => {
  console.error(JSON.stringify({ ok: false, error: err.message }));
  process.exit(1);
});
