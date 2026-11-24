#!/usr/bin/env node
// tutorial.mjs - nimmt das Video-Tutorial "Das Dashboard einrichten" reproduzierbar auf.
//
// Ablauf: (1) Sprechertext aus tutorial-script.mjs mit der Windows-Stimme in WAV-Dateien sprechen, (2) die
// Oberfläche im installierten Microsoft Edge (playwright-core, headless) Schritt für Schritt bedienen und den
// Bildschirm über das DevTools-Protokoll (Page.startScreencast) aufnehmen, (3) mit ffmpeg zu einem MP4 mit
// Bild, Ton und einer SRT-Untertiteldatei zusammensetzen.
//
// SICHERHEIT: Das Skript ändert Einstellungen (es klickt "Speichern") und läuft deshalb nur gegen den Demo-Server
// (python tools/mock_api.py --port 8765), dessen Änderungen nur im Arbeitsspeicher liegen. Gegen eine echte
// Plattform bricht es ab. Alle Daten im Video sind Beispieldaten.
//
// Aufruf (aus dem Projektordner; einmalig `npm install` in tools/docs):
//   node tools/docs/tutorial.mjs --ffmpeg=<pfad\ffmpeg.exe> [--base=http://127.0.0.1:8765] [--out=docs/video/...mp4]
//
// Optionen
//   --base=<url>     Demo-Server (Standard http://127.0.0.1:8765)
//   --out=<datei>    Ziel-MP4 (Standard docs/video/Trading_Local_Dashboard_Tutorial.mp4); daneben liegt die .srt
//   --ffmpeg=<pfad>  ffmpeg (Standard: Umgebungsvariable FFMPEG, sonst "ffmpeg" im PATH)
//   --work=<ordner>  Arbeitsordner für Einzelbilder und Ton (Standard: <temp>/tl-tutorial)
//   --voice=<name>   Windows-Stimme (Standard "Microsoft Hedda Desktop"), --rate=<-10..10> Tempo (Standard -1)
//   --audio-only     nur den Ton erzeugen und die Längen der Abschnitte ausgeben
//   --keep           Arbeitsordner nach dem Zusammenbau behalten
//
// Exit-Codes: 0 fertig, 1 Fehler bei der Aufnahme oder beim Zusammenbau, 2 falsche Aufrufargumente oder Server.

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { CHAPTERS, SEGMENTS } from './tutorial-script.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');

const APP_W = 1600; // Oberfläche
const APP_H = 900;
const CAP_H = 100; // Untertitelstreifen darunter
const VIDEO_H = APP_H + CAP_H;
const FPS = 30;
const LNA_OFF = ['--disable-features=LocalNetworkAccessChecks,BlockInsecurePrivateNetworkRequests,PrivateNetworkAccessSendPreflights'];

// ------------------------------------------------------------------------------------------------
// Aufruf
// ------------------------------------------------------------------------------------------------

function parseArgs(argv) {
  const o = {
    base: 'http://127.0.0.1:8765',
    out: path.join(ROOT, 'docs', 'video', 'Trading_Local_Dashboard_Tutorial.mp4'),
    ffmpeg: process.env.FFMPEG || 'ffmpeg',
    work: path.join(os.tmpdir(), 'tl-tutorial'),
    voice: 'Microsoft Hedda Desktop',
    rate: -1,
    audioOnly: false,
    keep: false,
  };
  for (const arg of argv) {
    const [k, ...rest] = arg.replace(/^--/, '').split('=');
    const v = rest.join('=');
    if (k === 'base') o.base = v.replace(/\/+$/, '');
    else if (k === 'out') o.out = path.resolve(v);
    else if (k === 'ffmpeg') o.ffmpeg = v;
    else if (k === 'work') o.work = path.resolve(v);
    else if (k === 'voice') o.voice = v;
    else if (k === 'rate') o.rate = Number.parseInt(v, 10);
    else if (k === 'audio-only') o.audioOnly = true;
    else if (k === 'keep') o.keep = true;
    else usage(`unbekannte Option: ${arg}`);
  }
  if (!Number.isInteger(o.rate) || o.rate < -10 || o.rate > 10) usage('--rate muss eine ganze Zahl von -10 bis 10 sein');
  return o;
}

function usage(msg) {
  console.error(`${msg}\nAufruf: node tools/docs/tutorial.mjs --ffmpeg=<pfad> [--base=...] [--out=...] [--audio-only] [--keep]`);
  process.exit(2);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function run(cmd, args, { cwd } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd, windowsHide: true });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { err += d; });
    child.on('error', reject);
    child.on('close', (code) => (code === 0 ? resolve(out) : reject(new Error(`${cmd} endete mit Code ${code}\n${err.slice(-1800)}`))));
  });
}

// ------------------------------------------------------------------------------------------------
// Ton
// ------------------------------------------------------------------------------------------------

/** Dauer einer PCM-WAV-Datei in Sekunden. */
function wavSeconds(buf) {
  let pos = 12;
  let byteRate = 0;
  while (pos + 8 <= buf.length) {
    const id = buf.toString('ascii', pos, pos + 4);
    const size = buf.readUInt32LE(pos + 4);
    if (id === 'fmt ') byteRate = buf.readUInt32LE(pos + 16);
    if (id === 'data') {
      if (!byteRate) throw new Error('WAV ohne fmt-Block');
      return Math.min(size, buf.length - pos - 8) / byteRate;
    }
    pos += 8 + size + (size % 2);
  }
  throw new Error('WAV ohne data-Block');
}

/** Spricht alle Abschnitte (nur geänderte neu) und liefert {id: Sekunden}. */
async function synthesize(o) {
  const dir = path.join(o.work, 'audio');
  await mkdir(dir, { recursive: true });
  const todo = [];
  for (const s of SEGMENTS) {
    const text = s.say ?? s.text;
    const hash = createHash('sha1').update(`${o.voice}|${o.rate}|${text}`).digest('hex');
    const stamp = path.join(dir, `${s.id}.sha`);
    const same = await readFile(stamp, 'utf8').then((h) => h === hash).catch(() => false);
    const wavOk = await readFile(path.join(dir, `${s.id}.wav`)).then(() => true).catch(() => false);
    if (!(same && wavOk)) todo.push({ id: s.id, text, hash });
  }
  if (todo.length) {
    const list = path.join(o.work, 'speech.json');
    await writeFile(list, JSON.stringify(todo.map(({ id, text }) => ({ id, text }))), 'utf8');
    console.log(`Sprechertext: ${todo.length} von ${SEGMENTS.length} Abschnitten werden gesprochen ...`);
    await run('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', path.join(HERE, 'tutorial-speak.ps1'),
      '-Json', list, '-OutDir', dir, '-Voice', o.voice, '-Rate', String(o.rate)]);
    for (const t of todo) await writeFile(path.join(dir, `${t.id}.sha`), t.hash, 'utf8');
  }
  const durations = {};
  for (const s of SEGMENTS) durations[s.id] = wavSeconds(await readFile(path.join(dir, `${s.id}.wav`)));
  return durations;
}

// ------------------------------------------------------------------------------------------------
// Bühne: Oberfläche (iframe) oben, Untertitelstreifen unten, Titelkarten und Telefon darüber
// ------------------------------------------------------------------------------------------------

function stageHtml() {
  return `<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Tutorial</title><style>
@font-face{font-family:Archivo;src:url('/fonts/archivo-latin-ext-wdth-wght.woff2') format('woff2-variations');font-weight:100 900;unicode-range:U+0100-02AF,U+1E00-1EFF,U+20A0-20CF,U+2C60-2C7F,U+A720-A7FF}
@font-face{font-family:Archivo;src:url('/fonts/archivo-latin-wdth-wght.woff2') format('woff2-variations');font-weight:100 900;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}
*{box-sizing:border-box}
html,body{margin:0;width:${APP_W}px;height:${VIDEO_H}px;overflow:hidden;background:#080c11;font-family:Archivo,'Segoe UI',system-ui,sans-serif;color:#f3f6fa}
#wrap{position:relative;width:${APP_W}px;height:${APP_H}px;overflow:hidden}
iframe{display:block;border:0;background:#121a24}
#app{width:${APP_W}px;height:${APP_H}px}
#card{position:absolute;inset:0;z-index:20;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;
  background:radial-gradient(circle at 28% 18%,#1d2c3d 0%,#0e151d 62%,#0a1017 100%);opacity:0;pointer-events:none;transition:opacity .7s}
#card svg{width:84px;height:84px;margin-bottom:30px}
#card h1{margin:0 0 18px;font:700 60px/1.1 Archivo,sans-serif;letter-spacing:-.01em}
#card .sub{font:500 26px/1.4 Archivo,sans-serif;color:#a9b4c2;max-width:1000px}
#card .note{margin-top:38px;padding:9px 20px;border:1px solid #2b394a;border-radius:999px;font:500 18px Archivo,sans-serif;color:#7fcfb4;background:rgba(37,155,123,.1)}
#card ul{list-style:none;margin:34px 0 0;padding:0;display:flex;gap:22px;font:600 20px Archivo,sans-serif}
#card li{padding:12px 22px;border-radius:12px;background:#18222e;border:1px solid #2b394a;color:#e8edf3}
#card li b{display:block;font:700 13px Archivo,sans-serif;letter-spacing:.14em;color:#7fcfb4;margin-bottom:4px}
#pal{position:absolute;z-index:15;top:70px;right:24px;width:330px;padding:16px 20px 12px;border-radius:14px;border:1px solid;opacity:0;pointer-events:none;
  transition:opacity .4s,transform .4s;transform:translateY(-8px);box-shadow:0 14px 40px rgba(0,0,0,.35)}
#pal.on{opacity:1;transform:none}
#pal h3{margin:0 0 10px;font:700 15px Archivo,sans-serif;letter-spacing:.1em;text-transform:uppercase}
#pal .row{display:flex;align-items:center;gap:14px;padding:8px 0;font:600 18px Archivo,sans-serif}
#pal .sw{flex:none;width:26px;height:26px;border-radius:50%;box-shadow:0 0 0 3px rgba(128,128,128,.25)}
#pal .hex{margin-left:auto;font:500 14px 'Cascadia Mono',Consolas,monospace;opacity:.7}
#phone{position:absolute;inset:0;z-index:18;background:radial-gradient(circle at 70% 30%,#1d2c3d 0%,#0b1118 70%);opacity:0;pointer-events:none;transition:opacity .6s}
#phone .copy{position:absolute;left:130px;top:300px;width:420px}
#phone h2{margin:0 0 14px;font:700 44px/1.1 Archivo,sans-serif}
#phone p{margin:0;font:500 22px/1.45 Archivo,sans-serif;color:#a9b4c2}
#phone .dev{position:absolute;left:50%;top:26px;transform:translateX(-50%);width:414px;height:848px;padding:12px;border-radius:50px;background:#03060a;
  box-shadow:0 0 0 2px #2b394a,0 30px 70px rgba(0,0,0,.6)}
#phone iframe{width:390px;height:824px;border-radius:38px}
#cap{position:relative;z-index:30;height:${CAP_H}px;display:grid;grid-template-columns:290px 1fr 290px;align-items:center;background:#0a0f15;border-top:1px solid #1f2c3b}
#chap{padding-left:36px;font:700 13px Archivo,sans-serif;letter-spacing:.14em;text-transform:uppercase;color:#7fcfb4;transition:opacity .2s}
#chap span{display:block;margin-top:5px;font:700 17px Archivo,sans-serif;letter-spacing:0;text-transform:none;color:#e8edf3}
#txt{font:600 24px/1.34 Archivo,sans-serif;text-align:center;color:#f3f6fa;transition:opacity .2s}
#brand{justify-self:end;padding-right:36px;font:600 15px Archivo,sans-serif;color:#7c8899}
</style></head><body>
<div id="wrap">
  <iframe id="app" name="app"></iframe>
  <div id="card"></div>
  <div id="pal"></div>
  <div id="phone"><div class="copy"><h2>Telefon-Ansicht</h2><p>390 Pixel breit: Die Bereiche stehen untereinander, Tabellen werden zu Karten.</p></div>
    <div class="dev"><iframe id="phoneframe" name="phone"></iframe></div></div>
</div>
<div id="cap"><div id="chap"></div><div id="txt"></div><div id="brand">Trading Local · Beispieldaten</div></div>
<script>
const $ = (id) => document.getElementById(id);
const BRAND = '<svg viewBox="0 0 18 18"><rect x="0" y="2" width="8" height="3.2" rx="0.6" fill="#56967f"/><rect x="5" y="7.4" width="9" height="3.2" rx="0.6" fill="#56967f"/><rect x="10" y="12.8" width="8" height="3.2" rx="0.6" fill="#56967f"/></svg>';
window.__stage = {
  load() { localStorage.setItem('tl-theme', 'dark'); $('app').src = '/'; },
  caption(chapter, text) {
    $('chap').style.opacity = $('txt').style.opacity = '0';
    setTimeout(() => {
      $('chap').innerHTML = 'Kapitel<span>' + chapter + '</span>';
      $('txt').textContent = text;
      $('chap').style.opacity = $('txt').style.opacity = '1';
    }, 160);
  },
  card(o) {
    const c = $('card');
    c.innerHTML = BRAND + '<h1></h1><div class="sub"></div>' + (o.note ? '<div class="note"></div>' : '') + (o.chips ? '<ul></ul>' : '');
    c.querySelector('h1').textContent = o.title;
    c.querySelector('.sub').textContent = o.sub;
    if (o.note) c.querySelector('.note').textContent = o.note;
    if (o.chips) for (const [k, v] of o.chips) { const li = document.createElement('li'); li.innerHTML = '<b></b><span></span>'; li.firstChild.textContent = k; li.lastChild.textContent = v; c.querySelector('ul').appendChild(li); }
    if (o.instant) { c.style.transition = 'none'; c.style.opacity = '1'; c.offsetHeight; c.style.transition = ''; } else c.style.opacity = '1';
  },
  cardHide() { $('card').style.opacity = '0'; },
  palette(show, v) {
    const p = $('pal');
    if (!show) { p.classList.remove('on'); return; }
    p.style.background = v.bg; p.style.color = v.ink; p.style.borderColor = v.line;
    p.innerHTML = '<h3></h3>' + v.rows.map(() => '<div class="row"><span class="sw"></span><span class="lbl"></span><span class="hex"></span></div>').join('');
    p.querySelector('h3').textContent = v.title; p.querySelector('h3').style.color = v.muted;
    p.querySelectorAll('.row').forEach((r, i) => {
      r.querySelector('.sw').style.background = v.rows[i].color;
      r.querySelector('.lbl').textContent = v.rows[i].label;
      r.querySelector('.hex').textContent = v.rows[i].color.toUpperCase();
    });
    p.classList.add('on');
  },
  phone(show) {
    if (show) $('phoneframe').src = '/#/';
    $('phone').style.opacity = show ? '1' : '0';
  },
};
</script></body></html>`;
}

/** Läuft in jedem Frame der Oberfläche: Mauszeiger mit Klickwelle, Hervorhebungsrahmen, weiches Scrollen. */
function frameOverlay() {
  if (window.__tut) return;
  const T = (window.__tut = {});
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - ((-2 * t + 2) ** 3) / 2);
  T.scrollTo = (y, ms = 1200) => new Promise((resolve) => {
    const root = document.scrollingElement || document.documentElement;
    const from = root.scrollTop;
    const to = Math.max(0, Math.min(y, root.scrollHeight - window.innerHeight));
    if (Math.abs(to - from) < 2 || ms <= 0) { root.scrollTop = to; resolve(); return; }
    const t0 = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - t0) / ms);
      root.scrollTop = from + (to - from) * ease(t);
      if (t < 1) requestAnimationFrame(step); else resolve();
    };
    requestAnimationFrame(step);
  });
  if (window.name !== 'app') return;

  const el = (css, html) => {
    const e = document.createElement('div');
    Object.assign(e.style, css);
    if (html) e.innerHTML = html;
    return e;
  };
  let target = null;
  let pad = 8;
  const cursor = el({ position: 'fixed', left: '0', top: '0', width: '28px', height: '28px', zIndex: '2147483647', pointerEvents: 'none',
    transform: 'translate(-80px,-80px)', filter: 'drop-shadow(0 2px 3px rgba(0,0,0,.55))' },
  '<svg viewBox="0 0 28 28" width="28" height="28"><path d="M5 3l16 9.5-7 1.8-3.4 6.7z" fill="#fff" stroke="#111" stroke-width="1.6" stroke-linejoin="round"/></svg>');
  const ring = el({ position: 'fixed', left: '0', top: '0', width: '0', height: '0', zIndex: '2147483000', pointerEvents: 'none', opacity: '0',
    border: '3px solid #ffb62e', borderRadius: '12px', transition: 'opacity .25s',
    boxShadow: '0 0 0 4px rgba(255,182,46,.28), 0 0 26px rgba(255,182,46,.5)' });
  const mount = () => { document.documentElement.append(ring, cursor); };
  if (document.documentElement) mount(); else document.addEventListener('readystatechange', mount, { once: true });
  window.addEventListener('mousemove', (e) => { cursor.style.transform = `translate(${e.clientX - 5}px,${e.clientY - 3}px)`; }, true);
  window.addEventListener('mousedown', (e) => {
    const wave = el({ position: 'fixed', left: `${e.clientX - 22}px`, top: `${e.clientY - 22}px`, width: '44px', height: '44px', borderRadius: '50%',
      border: '3px solid #ffb62e', zIndex: '2147483646', pointerEvents: 'none' });
    document.documentElement.append(wave);
    wave.animate([{ transform: 'scale(.25)', opacity: 1 }, { transform: 'scale(1.5)', opacity: 0 }], { duration: 520, easing: 'ease-out' }).onfinish = () => wave.remove();
  }, true);
  const tick = () => {
    requestAnimationFrame(tick);
    if (!target) return;
    const e = document.querySelector(target);
    if (!e) { ring.style.opacity = '0'; return; }
    const r = e.getBoundingClientRect();
    Object.assign(ring.style, { left: `${r.left - pad}px`, top: `${r.top - pad}px`, width: `${r.width + 2 * pad}px`, height: `${r.height + 2 * pad}px`, opacity: '1' });
  };
  requestAnimationFrame(tick);
  T.highlight = (selector, p = 8) => {
    target = selector;
    pad = p;
    if (!selector) ring.style.opacity = '0';
  };
}

// ------------------------------------------------------------------------------------------------
// Aufnahme
// ------------------------------------------------------------------------------------------------

async function checkServer(base) {
  let status;
  try {
    const res = await fetch(`${base}/api/status`, { signal: AbortSignal.timeout(8000) });
    status = await res.json();
  } catch (err) {
    console.error(`Server unter ${base} nicht erreichbar (${err.message}). Zuerst starten: python tools/mock_api.py --port 8765`);
    process.exit(2);
  }
  if (!String(status.version ?? '').includes('Demo-Server')) {
    console.error(`${base} ist nicht der Demo-Server. Das Tutorial klickt "Speichern" und darf nur gegen tools/mock_api.py laufen.`);
    process.exit(2);
  }
}

async function record(o, durations) {
  const frameDir = path.join(o.work, 'frames');
  await rm(frameDir, { recursive: true, force: true });
  await mkdir(frameDir, { recursive: true });
  const byId = Object.fromEntries(SEGMENTS.map((s) => [s.id, s]));

  const browser = await chromium.launch({ channel: 'msedge', headless: true, args: LNA_OFF });
  try {
    const context = await browser.newContext({
      viewport: { width: APP_W, height: VIDEO_H }, deviceScaleFactor: 1, locale: 'de-DE', timezoneId: 'Europe/Berlin',
      colorScheme: 'dark', bypassCSP: true,
    });
    await context.addInitScript(frameOverlay);
    const page = await context.newPage();
    const problems = [];
    page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
    await page.route(`${o.base}/__stage`, (route) => route.fulfill({ contentType: 'text/html; charset=utf-8', body: stageHtml() }));
    // the app's CSP forbids embedding (frame-ancestors 'none'); only for the recording the header is dropped
    await page.route((u) => u.href === `${o.base}/`, async (route) => {
      const resp = await route.fetch();
      const headers = { ...resp.headers() };
      delete headers['content-security-policy'];
      await route.fulfill({ response: resp, headers });
    });
    await page.goto(`${o.base}/__stage`);
    await page.evaluate(() => window.__stage.load());
    const app = () => page.frame({ name: 'app' });
    const phone = () => page.frame({ name: 'phone' });

    // ---- Hilfen --------------------------------------------------------------------------------
    const mouse = { x: 820, y: 460 };

    /** Seite (oder Frame) ist fertig gerendert: gestartet, keine Platzhalter, Charts mit Kerzen, Schriften geladen. */
    async function ready(f, timeout = 30000) {
      const end = Date.now() + timeout;
      let last = '';
      while (Date.now() < end) {
        last = await f.evaluate(() => {
          if (document.querySelector('#app .boot')) return 'startet';
          if (document.querySelector('.state-error, .pchart-error')) return 'FEHLER';
          if (document.querySelector('[aria-busy="true"]')) return 'lädt';
          if (document.querySelector('.pchart-state')) return 'Chart lädt';
          for (const c of document.querySelectorAll('.pchart')) if (!c.querySelector('.ohlc')) return 'Chart ohne Kerzen';
          if (document.fonts && document.fonts.status !== 'loaded') return 'Schriften';
          if (document.querySelector('.conn-banner')) return 'Verbindung';
          return 'ok';
        });
        if (last === 'FEHLER') throw new Error('Fehlerzustand in der Oberfläche');
        if (last === 'ok') { await sleep(450); return; }
        await sleep(150);
      }
      throw new Error(`Oberfläche nicht fertig nach ${timeout} ms: ${last}`);
    }

    async function glide(x, y, ms = 650) {
      const x0 = mouse.x;
      const y0 = mouse.y;
      const t0 = Date.now();
      for (;;) {
        const t = Math.min(1, (Date.now() - t0) / ms);
        const e = t < 0.5 ? 2 * t * t : 1 - ((-2 * t + 2) ** 2) / 2;
        await page.mouse.move(x0 + (x - x0) * e, y0 + (y - y0) * e);
        if (t >= 1) break;
        await sleep(12);
      }
      mouse.x = x;
      mouse.y = y;
    }

    /** Scrollt weich, falls das Element nicht bequem im sichtbaren Bereich liegt. */
    async function reveal(loc, f = app()) {
      const b = await loc.boundingBox();
      if (!b) throw new Error('Element nicht sichtbar');
      if (b.y < 80 || b.y + b.height > APP_H - 50) {
        const delta = b.y - 250;
        await f.evaluate((d) => window.__tut.scrollTo((document.scrollingElement || document.documentElement).scrollTop + d, 900), delta);
        await sleep(200);
      }
    }

    async function pointAt(loc, { dx = 0.5, dy = 0.5, ms = 650 } = {}) {
      await reveal(loc);
      const b = await loc.boundingBox();
      await glide(b.x + b.width * dx, b.y + b.height * dy, ms);
      return b;
    }

    async function click(loc, { dx = 0.5, dy = 0.5, after = 450, ms = 650 } = {}) {
      const b = await pointAt(loc, { dx, dy, ms });
      await sleep(140);
      await loc.click({ delay: 70, position: { x: b.width * dx, y: b.height * dy } });
      await sleep(after);
    }

    async function hover(loc, hold = 1500, opts = {}) {
      await pointAt(loc, opts);
      await sleep(hold);
    }

    async function typeInto(loc, text, check) {
      await click(loc, { after: 200 });
      await page.keyboard.press('Control+A');
      await page.keyboard.type(text, { delay: 150 });
      await sleep(200);
      if (check) {
        const got = await loc.evaluate((i) => i.valueAsNumber);
        if (got !== check) { await loc.fill(String(check)); }
      }
    }

    async function highlight(loc, pad = 8) {
      await loc.first().evaluate((e) => {
        document.querySelectorAll('[data-tut-hl]').forEach((x) => x.removeAttribute('data-tut-hl'));
        e.setAttribute('data-tut-hl', '1');
      });
      await app().evaluate((p) => window.__tut.highlight('[data-tut-hl]', p), pad);
    }
    const unhighlight = () => app().evaluate(() => window.__tut.highlight(null));
    const scrollTo = (y, ms = 1200, f = app()) => f.evaluate(([yy, mm]) => window.__tut.scrollTo(yy, mm), [y, ms]);

    const nav = (label) => app().locator('nav.nav a', { hasText: new RegExp(String.raw`^\s*${label}\s*$`) });
    async function goNav(label, after = 700) {
      await click(nav(label), { after: 250 });
      await ready(app());
      await sleep(after);
    }
    const themeButton = () => app().locator('button.icon-btn[aria-label$="Design"]');
    async function paletteCard(title) {
      const v = await app().evaluate(() => {
        const cs = getComputedStyle(document.documentElement);
        const g = (n) => cs.getPropertyValue(n).trim();
        return { bg: g('--surface-2'), ink: g('--ink-1'), muted: g('--ink-3'), line: g('--line'),
          up: g('--up'), down: g('--down'), ai: g('--ai'), me: g('--me') };
      });
      await page.evaluate(([vars, t]) => window.__stage.palette(true, {
        title: t, bg: vars.bg, ink: vars.ink, muted: vars.muted, line: vars.line,
        rows: [
          { label: 'Steigend · Long', color: vars.up }, { label: 'Fallend · Short', color: vars.down },
          { label: 'KI-Depot', color: vars.ai }, { label: 'Mein Depot', color: vars.me },
        ],
      }), [v, title]);
    }

    // ---- Aufnahme starten ----------------------------------------------------------------------
    await ready(app());
    await page.evaluate(() => window.__stage.card({
      title: 'Das Dashboard einrichten', sub: 'Ansichten, Einstellungen und das Design in Hell und Dunkel',
      note: 'Video-Tutorial · alle Daten sind Beispieldaten aus dem Demo-Server', instant: true,
    }));
    await page.evaluate(() => window.__stage.caption('Einleitung', ''));
    const frames = [];
    const writes = [];
    const cdp = await context.newCDPSession(page);
    cdp.on('Page.screencastFrame', (ev) => {
      const file = `${String(frames.length).padStart(6, '0')}.jpg`;
      frames.push({ file, ts: ev.metadata?.timestamp ?? Date.now() / 1000 });
      writes.push(writeFile(path.join(frameDir, file), Buffer.from(ev.data, 'base64')));
      cdp.send('Page.screencastFrameAck', { sessionId: ev.sessionId }).catch(() => {});
    });
    await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 92, maxWidth: APP_W, maxHeight: VIDEO_H, everyNthFrame: 1 });
    await sleep(900);

    const marks = [];
    async function seg(id, work) {
      const s = byId[id];
      if (!s) throw new Error(`Abschnitt ${id} fehlt im Drehbuch`);
      marks.push({ id, ms: Date.now() });
      console.log(`  ${id}`);
      await page.evaluate(([c, t]) => window.__stage.caption(c, t), [CHAPTERS[s.chapter], s.text]);
      const hold = Math.round(durations[id] * 1000) + 500;
      await Promise.all([(async () => { await sleep(350); if (work) await work(); })(), sleep(hold)]);
    }

    // ---- Drehbuch ------------------------------------------------------------------------------
    await seg('intro');
    await page.evaluate(() => window.__stage.cardHide());
    await sleep(900);

    // 1 · Ansichten
    await seg('nav', async () => {
      await highlight(app().locator('nav.nav'), 6);
      for (const label of ['Übersicht', 'Märkte', 'Signale', 'Depot', 'Nachrichten', 'Validierung', 'System']) {
        await hover(nav(label), 420, { ms: 500 });
      }
      await unhighlight();
      await pointAt(nav('Übersicht'), { ms: 500 });
    });

    await seg('overview', async () => {
      await highlight(app().locator('section.ribbon'), 4);
      await sleep(2300);
      await unhighlight();
      await glide(900, 600, 500);
      await scrollTo(470, 1300);
      await sleep(1200);
      await scrollTo(1000, 1500);
      await sleep(1500);
      await scrollTo(0, 1300);
    });

    await seg('lights', async () => {
      const wraps = app().locator('.indicator-wrap');
      for (let i = 0; i < 3; i += 1) await hover(wraps.nth(i).locator('button'), 1900, { ms: 600 });
      await glide(900, 300, 500);
    });

    await seg('markets', async () => {
      await goNav('Märkte');
      await click(app().locator('.markets-table th button.th-sort', { hasText: 'Veränd.' }), { after: 900 });
      await glide(760, 560, 500);
    });

    await seg('filters', async () => {
      await scrollTo(0, 600);
      const search = app().locator('input[type="search"]').first();
      await click(search, { after: 150 });
      await page.keyboard.type('SAP', { delay: 220 });
      await sleep(1100);
      await page.keyboard.press('Control+A');
      await page.keyboard.press('Backspace');
      await sleep(400);
      for (const name of ['Aktiv', 'Handelbar']) {
        const th = app().locator('.markets-table th', { has: app().locator('button.th-sort', { hasText: name }) });
        await reveal(th);
        await highlight(th, 6);
        await hover(th, 1300, { ms: 600 });
      }
      await unhighlight();
    });

    await seg('instrument', async () => {
      const link = app().locator('.markets-table tbody a[href*="SAP.DE"]').first();
      const row = (await link.count()) ? link : app().locator('.markets-table tbody a[href^="#/markt/"]').first();
      await click(row, { after: 300 });
      await ready(app());
      await sleep(600);
      const seg5 = app().locator('.ins-toolbar .segmented button');
      await highlight(app().locator('.ins-toolbar .segmented'), 6);
      await click(seg5.filter({ hasText: /^\s*5 Min\s*$/ }), { after: 1700 });
      await click(seg5.filter({ hasText: /^\s*1 Std\s*$/ }), { after: 1700 });
      await click(seg5.filter({ hasText: /^\s*15 Min\s*$/ }), { after: 800 });
      await unhighlight();
      await scrollTo(520, 1300);
      await sleep(900);
      await scrollTo(0, 1000);
    });

    await seg('signals', async () => {
      await goNav('Signale', 300);
      await click(app().getByRole('tab', { name: 'Abgeschlossen' }), { after: 1100 });
      await click(app().getByRole('tab', { name: 'Alle', exact: true }), { after: 1000 });
      await click(app().locator('table.sigtable tbody tr').first(), { dx: 0.4, after: 600 });
      await app().locator('.drawer').waitFor({ state: 'visible' });
      await sleep(2300);
      await page.keyboard.press('Escape');
      await sleep(700);
    });

    await seg('depot', async () => {
      await goNav('Depot', 300);
      await click(app().getByRole('button', { name: '1M', exact: true }), { after: 1100 });
      await click(app().getByRole('tab', { name: /^Orders/ }), { after: 1300 });
      await click(app().getByRole('tab', { name: /^Trades/ }), { after: 1300 });
    });

    await seg('more', async () => {
      await goNav('Nachrichten', 400);
      await scrollTo(500, 1100);
      await sleep(900);
      await goNav('Validierung', 1700);
      await goNav('System', 500);
      await scrollTo(700, 1400);
      await sleep(900);
      await scrollTo(0, 800);
    });

    // 2 · Einstellungen
    await seg('open-settings', async () => {
      await click(app().locator('a.icon-btn[aria-label="Einstellungen"]'), { after: 300 });
      await ready(app());
      await sleep(500);
    });

    await seg('groups', async () => {
      await scrollTo(99999, 2600);
      await sleep(300);
      await scrollTo(0, 2000);
      await sleep(200);
      await highlight(app().locator('.set-group').first().locator('.set-group-head'), 8);
    });

    await seg('mail-regions', async () => {
      await unhighlight();
      const asia = app().locator('fieldset.set-multi').first().locator('label.check', { hasText: 'Asien' });
      await click(asia, { dx: 0.3, after: 500 });
      await highlight(app().locator('.set-bar'), 4);
      await sleep(1800);
      await unhighlight();
    });

    await seg('signal-group', async () => {
      const head = app().locator('.set-group').nth(1).locator('h2');
      await reveal(head);
      await highlight(app().locator('.set-group').nth(1).locator('.set-group-head'), 8);
      await sleep(1400);
      await unhighlight();
      const setups = app().locator('fieldset.set-multi', { hasText: 'Setups' });
      await click(setups.locator('label.check', { hasText: 'Gegenbewegung' }), { dx: 0.25, after: 900 });
    });

    await seg('score', async () => {
      const score = app().locator('#set-min_score_actionable');
      await reveal(score);
      await highlight(score.locator('xpath=ancestor::div[contains(@class,"set-row")]'), 6);
      await sleep(1200);
      await typeInto(score, '0,65', 0.65);
      await sleep(1400);
      await unhighlight();
    });

    await seg('risk-error', async () => {
      const risk = app().locator('#set-risk_per_trade_pct');
      await reveal(risk);
      await typeInto(risk, '9');
      await sleep(700);
      await highlight(risk.locator('xpath=ancestor::div[contains(@class,"set-row")]'), 6);
      await sleep(1800);
    });

    await seg('risk-fix', async () => {
      const risk = app().locator('#set-risk_per_trade_pct');
      await typeInto(risk, '0,75', 0.75);
      await sleep(1000);
      await unhighlight();
    });

    await seg('benchmarks', async () => {
      const nasdaq = app().locator('fieldset.set-multi', { hasText: 'Vergleichsindizes' }).locator('label.check', { hasText: 'Nasdaq 100' });
      await reveal(nasdaq);
      await highlight(app().locator('.set-group').nth(4), 6);
      await sleep(1500);
      await unhighlight();
      await click(nasdaq, { dx: 0.15, after: 900 });
    });

    await seg('save', async () => {
      const bar = app().locator('.set-bar');
      await highlight(bar, 4);
      await sleep(1700);
      await unhighlight();
      await click(app().locator('.set-bar button[type="submit"]'), { after: 600 });
      await app().locator('.toast').waitFor({ state: 'visible', timeout: 8000 });
      await sleep(1500);
    });

    await seg('fixed', async () => {
      await scrollTo(0, 900);
      await sleep(500);
      await highlight(app().locator('.settings-aside'), 6);
      await sleep(2800);
      await unhighlight();
    });

    // 3 · Design und Farben
    await seg('theme', async () => {
      await goNav('Übersicht', 300);
      await highlight(themeButton(), 8);
      await sleep(1200);
      await click(themeButton(), { after: 300 });
      await unhighlight();
      await sleep(1100);
      await glide(900, 480, 500);
    });

    await seg('colors-light', async () => {
      await paletteCard('Farben · helles Design');
      await sleep(1800);
      await scrollTo(470, 1400);
      await sleep(2200);
    });

    await seg('light-markets', async () => {
      await page.evaluate(() => window.__stage.palette(false));
      await goNav('Märkte', 600);
      const link = app().locator('.markets-table tbody a[href*="SAP.DE"]').first();
      await click((await link.count()) ? link : app().locator('.markets-table tbody a[href^="#/markt/"]').first(), { after: 300 });
      await ready(app());
      await sleep(900);
      await scrollTo(480, 1300);
      await sleep(900);
    });

    await seg('light-depot', async () => {
      await goNav('Depot', 400);
      await highlight(app().locator('h2', { hasText: 'Depotwert im Verlauf' }).locator('xpath=ancestor::section[1]'), 4);
      await sleep(2800);
      await unhighlight();
    });

    await seg('colors-dark', async () => {
      await goNav('Übersicht', 300);
      await click(themeButton(), { after: 1200 });
      await paletteCard('Farben · dunkles Design');
      await glide(900, 480, 500);
      await sleep(2800);
      await page.evaluate(() => window.__stage.palette(false));
    });

    await seg('phone', async () => {
      await page.evaluate(() => window.__stage.phone(true));
      await sleep(700);
      await ready(phone());
      await scrollTo(1500, 3200, phone());
      await sleep(400);
      await phone().evaluate(() => { location.hash = '#/markt/SAP.DE'; });
      await sleep(600);
      await ready(phone());
      await scrollTo(700, 2200, phone());
      await sleep(400);
    });
    await page.evaluate(() => {
      window.__stage.phone(false);
      window.__stage.card({
        title: 'Viel Erfolg mit Trading Local', sub: 'Ansichten über die Kopfleiste, Einstellungen über das Regler-Symbol, Design über den Schalter',
        note: 'Trading Local ist eine Simulation und keine Anlageberatung.',
        chips: [['1 · ANSICHTEN', 'Kopfleiste'], ['2 · EINSTELLUNGEN', 'Regler-Symbol'], ['3 · DESIGN', 'Sonne / Mond']],
      });
    });
    await seg('outro');
    await sleep(1500);

    // ---- Aufnahme beenden ----------------------------------------------------------------------
    const endTs = Date.now() / 1000;
    await cdp.send('Page.stopScreencast');
    await Promise.all(writes);
    if (problems.length) throw new Error(problems.join('\n'));
    if (!frames.length) throw new Error('keine Bilder aufgenommen');
    const t0 = frames[0].ts;
    const timeline = marks.map((m) => ({ id: m.id, start: m.ms / 1000 - t0 }));
    return { frames, t0, endTs, timeline, frameDir };
  } finally {
    await browser.close();
  }
}

// ------------------------------------------------------------------------------------------------
// Zusammenbau
// ------------------------------------------------------------------------------------------------

const srtTime = (s) => {
  const ms = Math.max(0, Math.round(s * 1000));
  const p = (n, w) => String(n).padStart(w, '0');
  return `${p(Math.floor(ms / 3600000), 2)}:${p(Math.floor(ms / 60000) % 60, 2)}:${p(Math.floor(ms / 1000) % 60, 2)},${p(ms % 1000, 3)}`;
};

async function assemble(o, rec, durations) {
  const { frames, t0, endTs, timeline, frameDir } = rec;
  const total = endTs - t0;
  // Einzelbilder mit ihrer Anzeigedauer (concat-Demuxer), danach konstante 30 Bilder pro Sekunde
  const lines = [];
  frames.forEach((f, i) => {
    const next = i + 1 < frames.length ? frames[i + 1].ts : endTs;
    lines.push(`file '${f.file}'`, `duration ${Math.max(0.001, next - f.ts).toFixed(4)}`);
  });
  lines.push(`file '${frames[frames.length - 1].file}'`);
  await writeFile(path.join(frameDir, 'frames.txt'), `${lines.join('\n')}\n`, 'utf8');
  const silent = path.join(o.work, 'video.mp4');
  console.log(`Bild: ${frames.length} Einzelbilder, ${total.toFixed(1)} s ...`);
  await run(o.ffmpeg, ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'frames.txt',
    '-vf', `fps=${FPS},scale=${APP_W}:${VIDEO_H}:flags=lanczos:in_range=pc:out_range=tv:in_color_matrix=bt601:out_color_matrix=bt709,format=yuv420p`,
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
    '-t', total.toFixed(3), silent], { cwd: frameDir });

  // Ton: jeder Abschnitt startet beim Eintrag der Zeitleiste (+0,25 s Vorlauf für den Untertitel)
  const inputs = [];
  const filters = [];
  timeline.forEach((m, i) => {
    inputs.push('-i', path.join(o.work, 'audio', `${m.id}.wav`));
    const ms = Math.max(0, Math.round((m.start + 0.25) * 1000));
    filters.push(`[${i + 1}:a]aresample=44100,adelay=${ms}|${ms}[a${i}]`);
  });
  const mix = `${timeline.map((_, i) => `[a${i}]`).join('')}amix=inputs=${timeline.length}:normalize=0:duration=longest,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=44100,apad[aout]`;
  await mkdir(path.dirname(o.out), { recursive: true });
  console.log('Ton mischen und MP4 schreiben ...');
  await run(o.ffmpeg, ['-y', '-loglevel', 'error', '-i', silent, ...inputs, '-filter_complex', `${filters.join(';')};${mix}`,
    '-map', '0:v', '-map', '[aout]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k', '-ac', '2', '-t', total.toFixed(3),
    '-movflags', '+faststart', o.out]);

  const srt = timeline.map((m, i) => {
    const s = SEGMENTS.find((x) => x.id === m.id);
    const start = m.start + 0.25;
    return `${i + 1}\n${srtTime(start)} --> ${srtTime(start + durations[m.id] + 0.3)}\n${s.text}\n`;
  }).join('\n');
  await writeFile(o.out.replace(/\.mp4$/i, '.de.srt'), srt, 'utf8');
  return total;
}

// ------------------------------------------------------------------------------------------------

async function main() {
  const o = parseArgs(process.argv.slice(2));
  await mkdir(o.work, { recursive: true });
  const durations = await synthesize(o);
  const sum = Object.values(durations).reduce((a, b) => a + b, 0);
  console.log(`Sprechertext: ${SEGMENTS.length} Abschnitte, zusammen ${sum.toFixed(1)} s`);
  if (o.audioOnly) {
    for (const s of SEGMENTS) console.log(`  ${s.id.padEnd(14)} ${durations[s.id].toFixed(1).padStart(5)} s  ${s.text.length} Zeichen`);
    return;
  }
  await run(o.ffmpeg, ['-version']).catch(() => usage(`ffmpeg nicht gefunden (${o.ffmpeg}); Pfad mit --ffmpeg= angeben`));
  await checkServer(o.base);
  console.log('Aufnahme läuft ...');
  const rec = await record(o, durations);
  const total = await assemble(o, rec, durations);
  console.log(`Fertig: ${o.out} (${total.toFixed(1)} s)`);
  if (!o.keep) await rm(path.join(o.work, 'frames'), { recursive: true, force: true });
}

main().catch((err) => {
  console.error(err.stack || err);
  process.exit(1);
});
