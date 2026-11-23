// App frame header: product name, navigation, live status indicators (Daten, KI, Mail) with details on
// hover/focus, WebSocket state while disconnected, local time and theme toggle.

import { html } from '../vendor/preact-htm.mjs';
import { app } from '../lib/data.js';
import {
  fmtAge, fmtDuration, fmtMs, fmtNum, fmtShare, fmtTime, fmtWeekday,
} from '../lib/format.js';
import { mailModeLabel } from '../lib/labels.js';
import { connection, reconnectNow } from '../lib/live.js';
import { useStore, useTick } from '../lib/store.js';
import { themeStore, toggleTheme } from '../lib/theme.js';
import { serverNow } from '../lib/time.js';

const NAV = [
  { path: '/', label: 'Übersicht', routes: ['overview', 'notfound'] },
  { path: '/maerkte', label: 'Märkte', routes: ['markets', 'instrument'] },
  { path: '/signale', label: 'Signale', routes: ['signals'] },
  { path: '/depot', label: 'Depot', routes: ['depot'] },
  { path: '/news', label: 'Nachrichten', routes: ['news'] },
  { path: '/validierung', label: 'Validierung', routes: ['validation'] },
  { path: '/system', label: 'System', routes: ['system'] },
];

/**
 * Traffic-light state of the three subsystems: {tone: 'ok'|'warn'|'error'|'off', word, lines[]}.
 * ``error`` (the last status request failed) turns every light to 'unbekannt': the last known status
 * may be arbitrarily old and must not keep saying 'live'.
 * @param {import('../lib/types.js').SystemStatusOut | null} s
 * @param {number} now
 * @param {string | null} [error]
 */
export function indicatorStates(s, now, error = null) {
  if (!s || error) {
    const lines = error
      ? [error, ...(s ? [`Letzter bekannter Stand: ${fmtTime(s.server_time)}`] : [])]
      : ['Noch kein Status vom Server.'];
    const unknown = { tone: 'off', word: 'unbekannt', lines };
    return { data: unknown, llm: unknown, mail: unknown };
  }
  const d = s.data;
  const dataLines = [
    d.stream_connected ? 'Kursstrom verbunden' : 'Kursstrom getrennt',
    `Letzter Tick: ${d.last_tick_ts ? fmtAge(d.last_tick_ts, now) : 'noch keiner'}`,
    `Letzte REST-Abfrage: ${d.last_rest_ok_ts ? fmtAge(d.last_rest_ok_ts, now) : 'noch keine'}`,
    `${fmtNum(d.symbols_active, 0)} Werte aktiv, ${fmtNum(d.symbols_stale, 0)} ohne frische Daten`,
  ];
  if (d.rate_limited_until && d.rate_limited_until > now) dataLines.push(`Abfragen gedrosselt bis ${fmtTime(d.rate_limited_until)}`);
  if (d.backfill_progress !== null && d.backfill_progress !== undefined) dataLines.push(`Historie wird geladen: ${fmtShare(d.backfill_progress)}`);
  let data;
  if (s.offline) data = { tone: 'off', word: 'aus', lines: ['Offline-Modus: keine Datenabrufe.'] };
  else if (d.rate_limited_until && d.rate_limited_until > now) data = { tone: 'warn', word: 'gedrosselt', lines: dataLines };
  else if (d.stream_connected || (d.last_rest_ok_ts && now - d.last_rest_ok_ts < 300)) {
    data = d.symbols_stale > 0 || !d.stream_connected
      ? { tone: 'warn', word: d.stream_connected ? 'teilweise' : 'nur Abfragen', lines: dataLines }
      : { tone: 'ok', word: 'live', lines: dataLines };
  } else data = { tone: 'error', word: 'gestört', lines: dataLines };

  const l = s.llm;
  const llmLines = [
    `Modell: ${l.model}`,
    l.busy ? `arbeitet, ${fmtNum(l.queue_depth, 0)} in der Warteschlange` : `frei, ${fmtNum(l.queue_depth, 0)} in der Warteschlange`,
    `Letzte Antwortzeit: ${fmtMs(l.last_latency_ms)}`,
    `Letzte Stunde: ${fmtNum(l.calls_last_hour, 0)} Aufrufe, ${fmtNum(l.errors_last_hour, 0)} Fehler`,
  ];
  if (l.last_error) llmLines.push(`Letzter Fehler: ${l.last_error}`);
  let llm;
  if (!l.enabled) llm = { tone: 'off', word: 'aus', lines: ['KI ausgeschaltet.'] };
  else if (!l.available) llm = { tone: 'error', word: 'nicht erreichbar', lines: llmLines };
  else if (l.errors_last_hour > 3) llm = { tone: 'warn', word: 'Fehler', lines: llmLines };
  else llm = { tone: 'ok', word: l.busy ? 'arbeitet' : 'bereit', lines: llmLines };

  const m = s.mail;
  const mailLines = [
    `Modus: ${mailModeLabel(m.mode)}`,
    m.recipient_masked ? `Empfänger: ${m.recipient_masked}` : 'Kein Empfänger eingetragen',
    `Letzte Stunde: ${fmtNum(m.sent_last_hour, 0)} versendet`,
  ];
  if (m.last_error) mailLines.push(`Letzter Fehler: ${m.last_error}`);
  let mail;
  if (m.mode === 'off') mail = { tone: 'off', word: 'aus', lines: mailLines };
  else if (!m.configured) mail = { tone: 'warn', word: 'nicht eingerichtet', lines: mailLines };
  else if (m.last_error) mail = { tone: 'error', word: 'Fehler', lines: mailLines };
  else mail = { tone: 'ok', word: m.mode === 'file' ? 'Datei' : 'aktiv', lines: mailLines };
  return { data, llm, mail };
}

function Indicator({ id, name, state }) {
  return html`<span class="indicator-wrap">
    <button type="button" class="indicator" aria-describedby=${`ind-${id}`}>
      <span class=${`dot dot-${state.tone}`} aria-hidden="true"></span>
      <strong>${name}</strong><span class="state-word">${state.word}</span>
    </button>
    <span class="popover" role="tooltip" id=${`ind-${id}`}>
      <strong>${name}: ${state.word}</strong>
      ${state.lines.map((line) => html`<span>${line}</span>`)}
    </span>
  </span>`;
}

function Connection() {
  const conn = useStore(connection, (s) => s);
  useTick(1);
  if (conn.state === 'open' || (conn.state === 'connecting' && conn.attempts === 0)) return null;
  const wait = conn.retryAt ? Math.max(0, Math.ceil((conn.retryAt - Date.now()) / 1000)) : 0;
  return html`<div class="conn-banner" role="status">
    <span class="dot dot-error" aria-hidden="true"></span>
    <span>${conn.state === 'connecting' ? 'Verbinde …' : `Verbindung getrennt – neuer Versuch in ${wait} s`}</span>
    <span class="muted">Kurse und Status werden bis dahin alle 15 s abgefragt.</span>
    ${conn.state === 'closed' && html`<button type="button" class="btn btn-sm" onClick=${reconnectNow}>Jetzt verbinden</button>`}
  </div>`;
}

function Clock() {
  useTick(1);
  const now = serverNow();
  return html`<span class="clock" title="Ihre Ortszeit"><span class="weekday">${fmtWeekday(now)} </span>${fmtTime(now)}</span>`;
}

function ThemeToggle() {
  const theme = useStore(themeStore, (s) => s.theme);
  const toLight = theme === 'dark';
  return html`<button type="button" class="icon-btn" onClick=${toggleTheme}
      aria-label=${toLight ? 'Helles Design' : 'Dunkles Design'} title=${toLight ? 'Helles Design' : 'Dunkles Design'}>
    ${toLight
      ? html`<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="3.6" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M10 1.8v2.2M10 16v2.2M1.8 10H4M16 10h2.2M4.2 4.2l1.6 1.6M14.2 14.2l1.6 1.6M4.2 15.8l1.6-1.6M14.2 5.8l1.6-1.6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`
      : html`<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M15.5 12.6A6.5 6.5 0 0 1 7.4 4.5a6.5 6.5 0 1 0 8.1 8.1z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>`}
  </button>`;
}

function BrandMark() {
  return html`<svg class="brand-mark" viewBox="0 0 18 18" aria-hidden="true">
    <rect x="0" y="2" width="8" height="3.2" rx="0.6" fill="rgba(var(--session), 0.9)"/>
    <rect x="5" y="7.4" width="9" height="3.2" rx="0.6" fill="rgba(var(--session), 0.9)"/>
    <rect x="10" y="12.8" width="8" height="3.2" rx="0.6" fill="rgba(var(--session), 0.9)"/>
  </svg>`;
}

/** @param {{route: import('../lib/router.js').Route}} props */
export function TopBar({ route }) {
  const status = useStore(app, (s) => s.status);
  const statusError = useStore(app, (s) => s.statusError);
  useTick(5);
  const states = indicatorStates(status, serverNow(), statusError);
  return html`<header class="topbar">
    <div class="topbar-inner">
      <a class="brand" href="#/"><${BrandMark} /><span class="brand-text">Trading Local</span></a>
      <nav class="nav" aria-label="Hauptnavigation">
        ${NAV.map((n) => html`<a href=${`#${n.path}`} aria-current=${n.routes.includes(route.name) ? 'page' : undefined}>${n.label}</a>`)}
      </nav>
      <div class="topbar-right">
        <div class="indicators" aria-label="Systemzustand">
          <${Indicator} id="data" name="Daten" state=${states.data} />
          <${Indicator} id="llm" name="KI" state=${states.llm} />
          <${Indicator} id="mail" name="Mail" state=${states.mail} />
        </div>
        <${Clock} />
        <a class="icon-btn" href="#/einstellungen" aria-label="Einstellungen" title="Einstellungen"
          aria-current=${route.name === 'settings' ? 'page' : undefined}>
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3 5.5h8M15 5.5h2M3 10h2M9 10h8M3 14.5h6M13 14.5h4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="13" cy="5.5" r="1.8" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="7" cy="10" r="1.8" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="11" cy="14.5" r="1.8" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>
        </a>
        <${ThemeToggle} />
      </div>
    </div>
    <${Connection} />
  </header>
  ${status?.offline && html`<div class="offline-banner" role="status"><span class="dot dot-off" aria-hidden="true"></span>
    Offline-Modus: keine Datenabrufe. Die Oberfläche zeigt die gespeicherten Daten${status.started_ts ? `, Server läuft seit ${fmtDuration(serverNow() - status.started_ts)}` : ''}.</div>`}`;
}
