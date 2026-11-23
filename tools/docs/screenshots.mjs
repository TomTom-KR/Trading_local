#!/usr/bin/env node
// screenshots.mjs - reproducible screenshots of every GUI view of Trading Local for the documentation.
//
// Drives the SYSTEM Microsoft Edge (playwright-core, channel 'msedge', headless; no browser download) against a
// running server and writes PNG files to docs/screenshots/. Read-only: it only loads pages, switches tabs,
// sorts a table, opens the signal drawer and types example values into the order ticket. It never submits a
// form, never presses a button that changes server state, and never touches the database.
//
// Usage (from the project root or from tools/docs; run `npm install` in tools/docs once):
//   node tools/docs/screenshots.mjs [--base=http://127.0.0.1:8000] [--only=overview,signals,...] [--out=docs/screenshots]
//
// Options
//   --base=<url>        server to capture (default http://127.0.0.1:8000). The demo server
//                       (python tools/mock_api.py --port 8765) is detected by its version string ("Demo-Server"):
//                       its files get the suffix "-demo" and, without --only, only the views listed in DEMO_DEFAULT.
//   --only=<ids>        comma-separated view ids (see --list); default: every view except 'depot-trades' (real
//                       server) or DEMO_DEFAULT (demo server)
//   --out=<dir>         output directory (default <project root>/docs/screenshots)
//   --suffix=<text>     file-name suffix; default "-demo" on the demo server, "" otherwise
//   --crypto=<symbol>   crypto instrument for the instrument views (default BTC-EUR, Kraken real-time data)
//   --equity=<symbol>   equity instrument for the 15-minute chart (default SAP.DE)
//   --timeout=<ms>      how long one view may take to render (default 45000)
//   --no-optimize       keep the PNGs as the browser wrote them (default: lossless recompression, see optimizePng)
//   --list              print the view ids and exit
//
// File names: <name>-<theme>-<device>[suffix].png, e.g. overview-dark-desktop.png, signals-light-desktop-demo.png.
// Desktop = 1440x900 CSS px at device scale 1 (long pages as full-page captures); phone = 390x844 CSS px at
// device scale 2 (mobile emulation with touch), so phone PNGs are 780 px wide.
//
// Theme: the app keeps the theme toggle's choice in localStorage under 'tl-theme' ('dark' | 'light'); the script
// sets that key before the app starts AND emulates the matching prefers-color-scheme. It also emulates
// prefers-reduced-motion: reduce, which the app honours (no drawer slide-in or flashes caught mid-animation).
// Time zone Europe/Berlin and locale de-DE are fixed, so the times shown match the user's.
//
// "Rendered" means: the app has booted, the WebSocket is connected (no connection banner), no loading
// placeholder (aria-busy) is left, every price chart shows its OHLC legend, every chart canvas is laid out and
// painted, the web fonts are loaded and no HTTP request has been in flight for 600 ms. Instrument pages of an
// open market additionally wait (at most 75 s) for the first live 'bar' message, so the chart is current (the
// REST call returns closed bars only; the server pushes each closed minute). The crypto views use the 1-minute
// timeframe, the equity view the default 15 minutes. An error state, an empty
// price chart, a console error, an uncaught exception, a failed request or an HTTP status >= 400 fails the view
// loudly: its PNG is deleted, the error is printed and the exit code is 1.
//
// Exit codes: 0 all views captured, 1 at least one view failed (or the server is not reachable), 2 bad arguments.

import { mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';
import { chromium } from 'playwright-core';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');

const DEVICES = {
  desktop: { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, isMobile: false, hasTouch: false },
  phone: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
};
const BOTH = ['dark', 'light'];
const DARK = ['dark'];

/**
 * View catalogue. ``route`` is the hash route, ``full`` a full-page capture (otherwise the viewport),
 * ``element`` an element capture, ``prepare`` runs after the first render (clicks, typing), ``extra``
 * produces additional files from the same page.
 * ``realDefault: false`` leaves a view out of the default set on the real server (still available with --only).
 * @type {Array<{id: string, name: string, route: (o: Options) => string, themes: string[], device: string,
 *   full?: boolean, element?: string, realDefault?: boolean,
 *   prepare?: (page: import('playwright-core').Page, o: Options, ctx: Ctx) => Promise<void>,
 *   extra?: (page: import('playwright-core').Page, o: Options, ctx: Ctx) => Promise<void>}>}
 */
const VIEWS = [
  { id: 'overview', name: 'overview', route: () => '#/', themes: BOTH, device: 'desktop', full: true },
  { id: 'overview-hero', name: 'overview-hero', route: () => '#/', themes: BOTH, device: 'desktop', full: false },
  { id: 'ribbon', name: 'ribbon', route: () => '#/', themes: BOTH, device: 'desktop', element: 'section.ribbon' },
  { id: 'markets', name: 'markets', route: () => '#/maerkte', themes: BOTH, device: 'desktop', full: true },
  {
    id: 'markets-sorted', name: 'markets-sorted', route: () => '#/maerkte', themes: DARK, device: 'desktop', full: false,
    // column-header sort (client side only): 'Veränd.' once = largest change first
    prepare: async (page, o, ctx) => {
      await page.locator('.markets-table th button.th-sort', { hasText: 'Veränd.' }).click();
      await neutralMouse(page);
      await waitForRendered(page, ctx, o.timeout);
      const sorted = await page.locator('.markets-table th[aria-sort="descending"]').count();
      if (sorted !== 1) throw new Error('the markets table did not sort by change');
    },
  },
  {
    id: 'instrument-crypto', name: 'instrument-crypto', route: (o) => `#/markt/${encodeURIComponent(o.crypto)}`,
    themes: BOTH, device: 'desktop', full: true, prepare: expectTimeframe('1m', 'crypto'),
  },
  {
    id: 'instrument-equity', name: 'instrument-equity', route: (o) => `#/markt/${encodeURIComponent(o.equity)}`,
    themes: BOTH, device: 'desktop', full: true, prepare: expectTimeframe('15m', 'equity'),
  },
  { id: 'signals', name: 'signals', route: () => '#/signale', themes: BOTH, device: 'desktop', full: true },
  {
    id: 'signal-drawer', name: 'signal-drawer', route: () => '#/signale', themes: BOTH, device: 'desktop', full: false,
    prepare: openSignalDrawer,
    // the whole drawer in one picture: the viewport is made as tall as the drawer's content
    extra: async (page, o, ctx) => {
      const height = await page.evaluate(() => document.querySelector('.drawer')?.scrollHeight ?? 0);
      if (height <= DEVICES.desktop.viewport.height) return;
      await page.setViewportSize({ width: DEVICES.desktop.viewport.width, height: Math.ceil(height) });
      await waitForRendered(page, ctx, o.timeout);
      await shoot(page, ctx, 'signal-drawer-full', { full: false });
    },
  },
  { id: 'depot', name: 'depot', route: () => '#/depot', themes: BOTH, device: 'desktop', full: true },
  {
    id: 'depot-trades', name: 'depot-trades', route: () => '#/depot', themes: BOTH, device: 'desktop', realDefault: false,
    element: 'section[aria-label="Positionen, Orders und Trades"]',
    prepare: async (page, o, ctx) => {
      await page.getByRole('tab', { name: /^Trades/ }).click();
      await neutralMouse(page);
      await waitForRendered(page, ctx, o.timeout);
    },
  },
  { id: 'news', name: 'news', route: () => '#/news', themes: BOTH, device: 'desktop', full: true },
  { id: 'validation', name: 'validation', route: () => '#/validierung', themes: BOTH, device: 'desktop', full: true },
  { id: 'system', name: 'system', route: () => '#/system', themes: BOTH, device: 'desktop', full: true },
  { id: 'settings', name: 'settings', route: () => '#/einstellungen', themes: BOTH, device: 'desktop', full: true },
  { id: 'overview-phone', name: 'overview', route: () => '#/', themes: DARK, device: 'phone', full: true },
  {
    id: 'instrument-phone', name: 'instrument-crypto', route: (o) => `#/markt/${encodeURIComponent(o.crypto)}`,
    themes: DARK, device: 'phone', full: true, prepare: expectTimeframe('1m', 'crypto'),
  },
  {
    id: 'order-ticket-phone', name: 'order-ticket', route: (o) => `#/markt/${encodeURIComponent(o.crypto)}`,
    themes: DARK, device: 'phone', element: 'section[aria-labelledby="tk-title"]', prepare: fillOrderTicket,
  },
];
const DEMO_DEFAULT = ['overview', 'overview-hero', 'signals', 'signal-drawer', 'depot', 'depot-trades', 'validation'];

/**
 * @typedef {{base: string, out: string, suffix: string, crypto: string, equity: string, timeout: number,
 *   optimize: boolean, demo: boolean}} Options
 * @typedef {{theme: string, device: string, out: string, suffix: string, net: {inflight: number, last: number},
 *   problems: string[], written: Array<{file: string, width: number, height: number}>, note: string[],
 *   liveBars: Set<string>}} Ctx
 */

// ------------------------------------------------------------------------------------------------
// Page state
// ------------------------------------------------------------------------------------------------

/** Runs in the page: {ready} / {ready: false, reason} / {fatal}. */
function pageState() {
  const text = (el) => (el?.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 240);
  const visible = (el) => !!el && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
  const failed = document.querySelector('.state-error, .pchart-error');
  if (failed) return { fatal: `error state in the page: ${text(failed)}` };
  if (document.querySelector('#app .boot')) return { ready: false, reason: 'the app is still booting' };
  const banner = document.querySelector('.conn-banner');
  if (banner) return { ready: false, reason: `WebSocket not connected: ${text(banner)}` };
  const busy = document.querySelector('[aria-busy="true"]');
  if (busy) return { ready: false, reason: `loading placeholder: ${busy.getAttribute('aria-label') || 'skeleton'}` };
  for (const st of document.querySelectorAll('.pchart-state')) {
    const t = text(st);
    if (/noch keine Kursdaten/i.test(t)) return { fatal: `empty price chart: ${t}` };
    return { ready: false, reason: `price chart: ${t}` };
  }
  for (const chart of document.querySelectorAll('.pchart')) {
    if (visible(chart) && !chart.querySelector('.ohlc')) return { ready: false, reason: 'price chart without bars yet' };
  }
  if (document.fonts && document.fonts.status !== 'loaded') return { ready: false, reason: 'web fonts loading' };
  for (const host of document.querySelectorAll('.pchart-host, .lchart')) {
    if (!visible(host)) continue;
    const canvases = [...host.querySelectorAll('canvas')];
    if (!canvases.length) return { ready: false, reason: 'chart without canvas yet' };
    const area = (c) => c.width * c.height;
    const max = Math.max(...canvases.map(area));
    if (max < 100 * 40) return { ready: false, reason: 'chart not laid out yet' };
    const painted = canvases.filter((c) => area(c) === max).some((c) => {
      const ctx = c.getContext('2d');
      if (!ctx) return true; // not a 2d canvas: nothing to inspect
      const { data } = ctx.getImageData(0, 0, c.width, c.height);
      const step = 4 * 7; // every 7th pixel
      let differ = 0;
      for (let i = step; i < data.length; i += step) {
        if (data[i] !== data[0] || data[i + 1] !== data[1] || data[i + 2] !== data[2] || data[i + 3] !== data[3]) differ += 1;
      }
      return differ > (data.length / step) * 0.002;
    });
    if (!painted) return { ready: false, reason: 'chart canvas not painted yet' };
  }
  return { ready: true };
}

/** Wait until the page is rendered (see header) and stable for 600 ms. */
async function waitForRendered(page, ctx, timeout) {
  const deadline = Date.now() + timeout;
  let reason = 'not checked yet';
  let stableSince = 0;
  while (Date.now() < deadline) {
    if (ctx.problems.length) throw new Error(ctx.problems[0]);
    const state = await page.evaluate(pageState);
    if (state.fatal) throw new Error(state.fatal);
    const quiet = ctx.net.inflight === 0 && Date.now() - ctx.net.last >= 600;
    if (state.ready && quiet) {
      stableSince ||= Date.now();
      if (Date.now() - stableSince >= 600) return;
    } else {
      stableSince = 0;
      reason = state.ready ? `${ctx.net.inflight} request(s) in flight` : state.reason;
    }
    await page.waitForTimeout(150);
  }
  throw new Error(`not rendered after ${timeout} ms: ${reason}`);
}

/** Park the mouse where it hovers nothing (top-left corner of the top bar). */
async function neutralMouse(page) {
  await page.mouse.move(2, 2);
}

/**
 * Instrument page: make sure the timeframe ``tf`` is shown; while the venue is open, also wait for the first
 * live 'bar' message of the symbol (the server sends one per closed minute). The REST call that fills the chart
 * returns closed bars only, so without this wait the chart would end one bar early. The crypto views use the
 * 1-minute timeframe: there a live bar is a complete candle. On 5m/15m/1h the app starts the forming candle from
 * the first live minute it receives (the minutes of that bucket before the page load are missing), so a capture
 * shortly after loading would show a candle with the wrong open and volume.
 */
function expectTimeframe(tf, symbolKey) {
  const labels = { '1m': '1 Min', '5m': '5 Min', '15m': '15 Min', '1h': '1 Std', '1d': '1 Tag' };
  return async (page, o, ctx) => {
    const pressed = page.locator('.ins-toolbar .segmented button[aria-pressed="true"]');
    if ((await pressed.textContent())?.trim() !== labels[tf]) {
      await page.locator('.ins-toolbar .segmented button', { hasText: labels[tf] }).click();
      await neutralMouse(page);
      await waitForRendered(page, ctx, o.timeout);
    }
    ctx.note.push(`timeframe ${tf}`);
    const symbol = o[symbolKey];
    const open = await page.evaluate(async (sym) => {
      const res = await fetch('/api/instruments');
      return (await res.json()).find((i) => i.symbol === sym)?.is_open ?? false;
    }, symbol);
    if (!open) return;
    const deadline = Date.now() + 75000;
    while (!ctx.liveBars.has(symbol) && Date.now() < deadline) await page.waitForTimeout(500);
    if (!ctx.liveBars.has(symbol)) {
      ctx.note.push('no live bar within 75 s - chart shows closed bars only');
      return;
    }
    await page.waitForTimeout(500); // the chart applies the bar in the next frame
    await waitForRendered(page, ctx, o.timeout);
    ctx.note.push('live candle included');
  };
}

/** Signals page: tab 'Alle', then the first recommendation that reached its target, else the first active one, else the newest. */
async function openSignalDrawer(page, o, ctx) {
  await page.getByRole('tab', { name: 'Alle', exact: true }).click();
  await waitForRendered(page, ctx, o.timeout);
  const rows = page.locator('table.sigtable tbody tr');
  if (!(await rows.count())) throw new Error('no signal in the list - the signal drawer cannot be shown');
  let row = rows.filter({ hasText: 'Ziel erreicht' }).first();
  if (!(await row.count())) row = rows.filter({ has: page.locator('.chip', { hasText: /^aktiv/ }) }).first();
  if (!(await row.count())) row = rows.first();
  await row.click();
  await page.locator('.drawer').waitFor({ state: 'visible' });
  await neutralMouse(page);
  await waitForRendered(page, ctx, o.timeout);
  ctx.note.push(`signal ${page.url().split('#')[1]}`);
}

/** Instrument page (phone): type an example stop 1 % below and a target 2 % above the price. Never submitted. */
async function fillOrderTicket(page, o, ctx) {
  const price = await page.evaluate(async (symbol) => {
    const res = await fetch('/api/quotes');
    const quotes = await res.json();
    return quotes.find((q) => q.symbol === symbol)?.price ?? null;
  }, o.crypto);
  if (!(price > 0)) throw new Error(`no price for ${o.crypto}`);
  const decimals = price >= 1000 ? 0 : price >= 1 ? 2 : 5;
  const de = (v) => v.toFixed(decimals).replace('.', ',');
  const stop = page.locator('#ins-ot-stop');
  const target = page.locator('#ins-ot-target');
  await stop.fill(de(price * 0.99));
  await target.fill(de(price * 1.02));
  await target.evaluate((el) => el.blur());
  await waitForRendered(page, ctx, o.timeout);
  ctx.note.push(`example stop ${de(price * 0.99)} and target ${de(price * 1.02)} typed, not submitted`);
}

// ------------------------------------------------------------------------------------------------
// Capture
// ------------------------------------------------------------------------------------------------

/** Take one screenshot of the current page; records the written file in ctx. */
async function shoot(page, ctx, name, { full = false, element = null } = {}) {
  if (ctx.problems.length) throw new Error(ctx.problems[0]);
  const file = path.join(ctx.out, `${name}-${ctx.theme}-${ctx.device}${ctx.suffix}.png`);
  if (element) {
    // clip the element out of a full-page capture: there the sticky top bar sits at the top of the document
    // and cannot cover the element (a plain element screenshot scrolls it under the bar)
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(250);
    const box = await page.locator(element).first().evaluate((el) => {
      const r = el.getBoundingClientRect();
      return { x: r.left + window.scrollX, y: r.top + window.scrollY, width: r.width, height: r.height };
    });
    await page.screenshot({ path: file, fullPage: true, clip: box, animations: 'disabled', caret: 'hide' });
  } else {
    await page.screenshot({ path: file, fullPage: full, animations: 'disabled', caret: 'hide' });
  }
  ctx.written.push({ file, ...(await pngSize(file)) });
}

/**
 * Capture one view in one theme: fresh browser context, theme preset, load, prepare, wait, shoot.
 * @returns {Promise<{ok: boolean, files: Array<{file: string, width: number, height: number}>, error?: string, note: string[]}>}
 */
async function captureView(browser, view, theme, o) {
  const dev = DEVICES[view.device];
  const context = await browser.newContext({
    ...dev, colorScheme: theme, reducedMotion: 'reduce', locale: 'de-DE', timezoneId: 'Europe/Berlin',
  });
  await context.addInitScript((t) => {
    try {
      window.localStorage.setItem('tl-theme', t);
    } catch {
      // storage blocked: prefers-color-scheme still selects the theme
    }
  }, theme);
  /** @type {Ctx} */
  const ctx = {
    theme, device: view.device, out: o.out, suffix: o.suffix, net: { inflight: 0, last: Date.now() }, problems: [], written: [], note: [],
    liveBars: new Set(),
  };
  const page = await context.newPage();
  page.setDefaultTimeout(o.timeout);
  page.on('websocket', (socket) => socket.on('framereceived', (frame) => {
    try {
      const msg = JSON.parse(String(frame.payload));
      if (msg?.type === 'bar' && msg.data?.symbol) ctx.liveBars.add(msg.data.symbol);
    } catch {
      // not JSON: not a bar message
    }
  }));
  page.on('console', (msg) => {
    if (msg.type() === 'error') ctx.problems.push(`console error: ${msg.text()} (${msg.location().url}:${msg.location().lineNumber})`);
  });
  page.on('pageerror', (err) => ctx.problems.push(`uncaught exception: ${String(err?.stack || err).split('\n').slice(0, 3).join(' | ')}`));
  page.on('request', () => { ctx.net.inflight += 1; ctx.net.last = Date.now(); });
  const done = () => { ctx.net.inflight = Math.max(0, ctx.net.inflight - 1); ctx.net.last = Date.now(); };
  page.on('requestfinished', done);
  page.on('requestfailed', (req) => {
    done();
    const why = req.failure()?.errorText || '';
    if (!/ERR_ABORTED/.test(why)) ctx.problems.push(`request failed: ${req.method()} ${req.url()} ${why}`);
  });
  page.on('response', (res) => {
    if (res.status() >= 400) ctx.problems.push(`HTTP ${res.status()}: ${res.request().method()} ${res.url()}`);
  });
  try {
    await page.goto(`${o.base}/${view.route(o)}`, { waitUntil: 'load' });
    await waitForRendered(page, ctx, o.timeout);
    if (view.prepare) await view.prepare(page, o, ctx);
    await shoot(page, ctx, view.name, { full: view.full, element: view.element });
    if (view.extra) await view.extra(page, o, ctx);
    if (ctx.problems.length) throw new Error(ctx.problems.join('\n    '));
    return { ok: true, files: ctx.written, note: ctx.note };
  } catch (err) {
    for (const w of ctx.written) await rm(w.file, { force: true });
    const extra = ctx.problems.length > 1 ? `\n    ${ctx.problems.slice(1).join('\n    ')}` : '';
    return { ok: false, files: [], error: `${String(err?.message || err).split('\n')[0]}${extra}`, note: ctx.note };
  } finally {
    await context.close();
  }
}

// ------------------------------------------------------------------------------------------------
// PNG: size and lossless recompression (zlib level 9, best filter per row, alpha dropped when opaque)
// ------------------------------------------------------------------------------------------------
const PNG_SIG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

async function pngSize(file) {
  const head = await readFile(file);
  return { width: head.readUInt32BE(16), height: head.readUInt32BE(20) };
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, 'latin1'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(zlib.crc32(td) >>> 0);
  return Buffer.concat([len, td, crc]);
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/** Re-encode an 8-bit RGB/RGBA, non-interlaced PNG losslessly; returns null when it is not smaller or not supported. */
function optimizePng(buf) {
  if (!buf.subarray(0, 8).equals(PNG_SIG)) return null;
  let pos = 8;
  let ihdr = null;
  const idat = [];
  const ancillary = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString('latin1', pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    if (type === 'IHDR') ihdr = data;
    else if (type === 'IDAT') idat.push(data);
    else if (type !== 'IEND') ancillary.push([type, data]);
    pos += 12 + len;
  }
  if (!ihdr) return null;
  const width = ihdr.readUInt32BE(0);
  const height = ihdr.readUInt32BE(4);
  const [depth, colorType, , , interlace] = ihdr.subarray(8, 13);
  if (depth !== 8 || interlace !== 0 || (colorType !== 2 && colorType !== 6)) return null;
  const bpp = colorType === 6 ? 4 : 3;
  const stride = width * bpp;
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const px = Buffer.alloc(stride * height);
  for (let y = 0; y < height; y += 1) { // unfilter
    const f = raw[y * (stride + 1)];
    const src = y * (stride + 1) + 1;
    const dst = y * stride;
    for (let x = 0; x < stride; x += 1) {
      const a = x >= bpp ? px[dst + x - bpp] : 0;
      const b = y ? px[dst - stride + x] : 0;
      const c = x >= bpp && y ? px[dst - stride + x - bpp] : 0;
      const v = raw[src + x];
      px[dst + x] = (f === 0 ? v : f === 1 ? v + a : f === 2 ? v + b : f === 3 ? v + ((a + b) >> 1) : v + paeth(a, b, c)) & 0xff;
    }
  }
  let outBpp = bpp;
  let image = px;
  if (bpp === 4) { // drop the alpha channel when every pixel is opaque
    let opaque = true;
    for (let i = 3; i < px.length; i += 4) if (px[i] !== 255) { opaque = false; break; }
    if (opaque) {
      outBpp = 3;
      image = Buffer.alloc(width * height * 3);
      for (let i = 0, j = 0; i < px.length; i += 4, j += 3) {
        image[j] = px[i];
        image[j + 1] = px[i + 1];
        image[j + 2] = px[i + 2];
      }
    }
  }
  const s = width * outBpp;
  const filtered = Buffer.alloc((s + 1) * height);
  const cand = [0, 1, 2, 3, 4].map(() => Buffer.alloc(s));
  for (let y = 0; y < height; y += 1) { // per row the filter with the smallest sum of absolute values
    const row = y * s;
    let best = 0;
    let bestSum = Infinity;
    for (let f = 0; f < 5; f += 1) {
      const outRow = cand[f];
      let sum = 0;
      for (let x = 0; x < s; x += 1) {
        const v = image[row + x];
        const a = x >= outBpp ? image[row + x - outBpp] : 0;
        const b = y ? image[row - s + x] : 0;
        const c = x >= outBpp && y ? image[row - s + x - outBpp] : 0;
        const r = (f === 0 ? v : f === 1 ? v - a : f === 2 ? v - b : f === 3 ? v - ((a + b) >> 1) : v - paeth(a, b, c)) & 0xff;
        outRow[x] = r;
        sum += r < 128 ? r : 256 - r;
        if (sum >= bestSum) break;
      }
      if (sum < bestSum) { bestSum = sum; best = f; }
    }
    filtered[y * (s + 1)] = best;
    cand[best].copy(filtered, y * (s + 1) + 1, 0, s);
  }
  const newHdr = Buffer.from(ihdr);
  newHdr[9] = outBpp === 4 ? 6 : 2;
  const out = Buffer.concat([
    PNG_SIG, chunk('IHDR', newHdr), ...ancillary.map(([t, d]) => chunk(t, d)),
    chunk('IDAT', zlib.deflateSync(filtered, { level: 9, memLevel: 9 })), chunk('IEND', Buffer.alloc(0)),
  ]);
  return out.length < buf.length ? out : null;
}

// ------------------------------------------------------------------------------------------------
// Main
// ------------------------------------------------------------------------------------------------
function parseArgs(argv) {
  const flags = {};
  for (const a of argv) {
    if (!a.startsWith('--')) throw new Error(`unexpected argument "${a}" (options look like --name=value)`);
    const eq = a.indexOf('=');
    if (eq === -1) flags[a.slice(2)] = true;
    else flags[a.slice(2, eq)] = a.slice(eq + 1);
  }
  const known = ['base', 'only', 'out', 'suffix', 'crypto', 'equity', 'timeout', 'no-optimize', 'list', 'help'];
  const unknown = Object.keys(flags).filter((k) => !known.includes(k));
  if (unknown.length) throw new Error(`unknown option(s): ${unknown.map((k) => `--${k}`).join(', ')}`);
  return flags;
}

async function main() {
  let flags;
  try {
    flags = parseArgs(process.argv.slice(2));
  } catch (err) {
    console.error(err.message);
    process.exit(2);
  }
  if (flags.help) {
    console.log('usage: node tools/docs/screenshots.mjs [--base=url] [--only=ids] [--out=dir] [--suffix=text] [--crypto=SYM] '
      + '[--equity=SYM] [--timeout=ms] [--no-optimize] [--list]');
    return;
  }
  if (flags.list) {
    for (const v of VIEWS) {
      const kind = v.element ? 'element' : v.full ? 'full page' : 'viewport';
      console.log(`${v.id.padEnd(20)} ${v.device.padEnd(8)} ${v.themes.join('+').padEnd(11)} ${kind.padEnd(10)} ${v.route({ crypto: 'BTC-EUR', equity: 'SAP.DE' })}${v.realDefault === false ? '  (not in the real-server default)' : ''}`);
    }
    console.log(`\ndemo server default: ${DEMO_DEFAULT.join(', ')}`);
    return;
  }
  const base = String(flags.base || 'http://127.0.0.1:8000').replace(/\/+$/, '');
  let status;
  try {
    const res = await fetch(`${base}/api/status`, { signal: AbortSignal.timeout(10000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    status = await res.json();
  } catch (err) {
    console.error(`server not reachable at ${base}: ${err.message}`);
    process.exit(1);
  }
  const demo = /demo/i.test(status.version || '');
  const ids = flags.only ? String(flags.only).split(',').map((s) => s.trim()).filter(Boolean)
    : demo ? DEMO_DEFAULT : VIEWS.filter((v) => v.realDefault !== false).map((v) => v.id);
  const unknown = ids.filter((id) => !VIEWS.some((v) => v.id === id));
  if (unknown.length) {
    console.error(`unknown view id(s): ${unknown.join(', ')}\nknown: ${VIEWS.map((v) => v.id).join(', ')}`);
    process.exit(2);
  }
  /** @type {Options} */
  const o = {
    base,
    out: path.resolve(process.cwd(), String(flags.out || path.join(ROOT, 'docs', 'screenshots'))),
    suffix: flags.suffix !== undefined && flags.suffix !== true ? String(flags.suffix) : demo ? '-demo' : '',
    crypto: String(flags.crypto || 'BTC-EUR'),
    equity: String(flags.equity || 'SAP.DE'),
    timeout: Number(flags.timeout || 45000),
    optimize: !flags['no-optimize'],
    demo,
  };
  await mkdir(o.out, { recursive: true });
  const serverTime = new Date(status.server_time * 1000).toLocaleString('de-DE', { timeZone: 'Europe/Berlin' });
  console.log(`server ${base} - version ${status.version}${demo ? ' (DEMO DATA)' : ''}, server time ${serverTime}`);
  console.log(`output ${o.out}${o.suffix ? `, suffix "${o.suffix}"` : ''}`);

  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  console.log(`browser Microsoft Edge ${browser.version()} (headless)\n`);
  const failures = [];
  const written = [];
  try {
    for (const id of ids) {
      const view = VIEWS.find((v) => v.id === id);
      for (const theme of view.themes) {
        const t0 = Date.now();
        const r = await captureView(browser, view, theme, o);
        const label = `${id} ${theme} ${view.device}`;
        if (!r.ok) {
          failures.push(`${label}: ${r.error}`);
          console.error(`FAIL ${label}: ${r.error}`);
          continue;
        }
        for (const f of r.files) {
          let note = '';
          if (o.optimize) {
            const before = (await stat(f.file)).size;
            const smaller = optimizePng(await readFile(f.file));
            if (smaller) {
              await writeFile(f.file, smaller);
              note = ` (${Math.round(before / 1024)} -> ${Math.round(smaller.length / 1024)} KB)`;
            }
          }
          const size = (await stat(f.file)).size;
          written.push({ ...f, size });
          console.log(`ok   ${path.basename(f.file).padEnd(44)} ${String(f.width).padStart(5)}x${String(f.height).padEnd(6)} ${String(Math.round(size / 1024)).padStart(5)} KB${note}  ${((Date.now() - t0) / 1000).toFixed(1)} s${r.note.length ? `  [${r.note.join('; ')}]` : ''}`);
        }
      }
    }
  } finally {
    await browser.close();
  }
  const total = written.reduce((s, f) => s + f.size, 0);
  console.log(`\n${written.length} file(s), ${(total / 1024 / 1024).toFixed(1)} MB${failures.length ? `, ${failures.length} FAILED` : ''}`);
  if (failures.length) {
    console.error(`\nFailed views:\n  ${failures.join('\n  ')}`);
    process.exit(1);
  }
}

await main();
