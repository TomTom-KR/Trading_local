// #/system - status tiles, jobs, feeds, LLM calls, activity log (live), mail log and 'Über'.

import { html, useEffect, useRef, useState } from '../vendor/preact-htm.mjs';
import { api, errorText } from '../lib/api.js';
import { app } from '../lib/data.js';
import {
  DASH, fmtAge, fmtCompact, fmtDate, fmtDuration, fmtMs, fmtNum, fmtShare, fmtTime, fmtWhen, isNum,
} from '../lib/format.js';
import {
  ACTIVITY_CATEGORY_LABELS, LEVEL_LABELS, NEWS_REGION_LABELS, activityCategoryLabel, levelLabel, llmKindLabel,
  mailKindLabel, mailModeLabel, mailStatusLabel,
} from '../lib/labels.js';
import { useLive } from '../lib/live.js';
import { useResource } from '../lib/resource.js';
import { useInterval, useStore, useTick } from '../lib/store.js';
import { serverNow } from '../lib/time.js';
import { indicatorStates } from '../components/topbar.js';
import { Empty, ErrorState, Loading } from '../components/ui.js';

const ACTIVITY_PAGE = 60;
const MAX_ACTIVITY_ITEMS = 500; // upper bound of the list while live entries keep arriving

function Tile({ title, state, children }) {
  return html`<section class="tile" aria-label=${title}>
    <h2 class="tile-title"><span class=${`dot dot-${state.tone}`} aria-hidden="true"></span>${title}<span class="tile-word">${state.word}</span></h2>
    <dl class="tile-kv">${children}</dl>
  </section>`;
}
const Row = ({ k, children }) => html`<div><dt>${k}</dt><dd>${children}</dd></div>`;

function MailTest() {
  const [state, setState] = useState(null);
  const send = async () => {
    setState({ busy: true });
    try {
      const out = await api.post('/mail/test');
      setState({ ok: out.ok, text: out.detail_de || mailStatusLabel(out.status) });
    } catch (err) {
      setState({ ok: false, text: errorText(err) });
    }
  };
  return html`<div class="tile-action">
    <button type="button" class="btn btn-sm" onClick=${send} disabled=${state?.busy}>${state?.busy ? 'Wird gesendet …' : 'Testmail senden'}</button>
    ${state && !state.busy && html`<p class=${state.ok ? 'small' : 'field-error'} role="status">${state.text}</p>`}
  </div>`;
}

function Tiles({ status }) {
  const now = serverNow();
  const st = indicatorStates(status, now);
  const d = status.data;
  const l = status.llm;
  const n = status.news;
  const m = status.mail;
  const db = status.db;
  const newsState = n.feeds_failing > 0 ? { tone: 'warn', word: `${n.feeds_failing} Feed gestört` } : { tone: 'ok', word: 'läuft' };
  return html`<div class="tiles">
    <${Tile} title="Daten" state=${st.data}>
      <${Row} k="Kursstrom">${d.stream_connected ? 'verbunden' : 'getrennt'}</${Row}>
      <${Row} k="Letzter Tick">${d.last_tick_ts ? fmtAge(d.last_tick_ts, now) : 'noch keiner'}</${Row}>
      <${Row} k="Letzte REST-Abfrage">${d.last_rest_ok_ts ? fmtAge(d.last_rest_ok_ts, now) : 'noch keine'}</${Row}>
      <${Row} k="Drosselung">${d.rate_limited_until && d.rate_limited_until > now ? `bis ${fmtTime(d.rate_limited_until)}` : 'keine'}</${Row}>
      <${Row} k="Historie">${isNum(d.backfill_progress) ? `wird geladen: ${fmtShare(d.backfill_progress)}` : 'vollständig'}</${Row}>
      <${Row} k="Werte ohne frische Daten">${fmtNum(d.symbols_stale, 0)} von ${fmtNum(d.symbols_active, 0)}</${Row}>
    </${Tile}>
    <${Tile} title="KI (lokales Modell)" state=${st.llm}>
      <${Row} k="Modell">${l.model}</${Row}>
      <${Row} k="Zustand">${!l.enabled ? 'ausgeschaltet' : !l.available ? 'nicht erreichbar' : l.busy ? 'arbeitet' : 'frei'}</${Row}>
      <${Row} k="Warteschlange">${fmtNum(l.queue_depth, 0)}</${Row}>
      <${Row} k="Letzte Antwortzeit">${fmtMs(l.last_latency_ms)}</${Row}>
      <${Row} k="Letzte Stunde">${fmtNum(l.calls_last_hour, 0)} Aufrufe, ${fmtNum(l.errors_last_hour, 0)} Fehler</${Row}>
      ${l.last_error && html`<${Row} k="Letzter Fehler">${l.last_error}</${Row}>`}
    </${Tile}>
    <${Tile} title="Nachrichten" state=${newsState}>
      <${Row} k="Feeds">${fmtNum(n.feeds_total, 0)}, davon ${fmtNum(n.feeds_failing, 0)} gestört</${Row}>
      <${Row} k="Letzte Stunde">${fmtNum(n.items_last_hour, 0)} Meldungen, ${fmtNum(n.analyzed_last_hour, 0)} analysiert</${Row}>
      <${Row} k="Warten auf Analyse">${fmtNum(n.backlog, 0)}</${Row}>
      <${Row} k="Letzte Abfrage">${n.last_poll_ts ? fmtAge(n.last_poll_ts, now) : 'noch keine'}</${Row}>
    </${Tile}>
    <${Tile} title="Mail" state=${st.mail}>
      <${Row} k="Modus">${mailModeLabel(m.mode)}</${Row}>
      <${Row} k="Empfänger">${m.recipient_masked || 'nicht eingetragen'}</${Row}>
      <${Row} k="Letzte Stunde">${fmtNum(m.sent_last_hour, 0)} versendet</${Row}>
      <${Row} k="Letzter Fehler">${m.last_error || 'keiner'}</${Row}>
      <${MailTest} />
    </${Tile}>
    <${Tile} title="Datenbank" state=${{ tone: 'ok', word: `${fmtNum(db.size_mb, 0)} MB` }}>
      <${Row} k="Minutenkerzen">${fmtCompact(db.bars_1m)}</${Row}>
      <${Row} k="Zeitraum">${db.first_bar_ts ? `${fmtDate(db.first_bar_ts)} bis ${fmtDate(db.last_bar_ts)}` : 'noch leer'}</${Row}>
      <${Row} k="Nachrichten">${fmtCompact(db.news)}</${Row}>
      <${Row} k="Signale">${fmtCompact(db.signals)}</${Row}>
      <${Row} k="Server läuft seit">${fmtDuration(now - status.started_ts)}</${Row}>
    </${Tile}>
  </div>`;
}

function SimpleTable({ title, res, columns, empty, rowKey }) {
  return html`<section class="panel" aria-label=${title}>
    <div class="panel-head"><h2>${title}</h2>${res.data && html`<div class="aside">${fmtNum(res.data.length, 0)} Einträge</div>`}</div>
    ${res.loading ? html`<${Loading} lines=${3} />`
      : res.error ? html`<${ErrorState} error=${res.error} onRetry=${res.reload} />`
        : !res.data?.length ? html`<p class="small muted">${empty}</p>`
          : html`<div class="table-wrap table-scroll"><table class="table stack-mobile">
              <thead><tr>${columns.map((c) => html`<th scope="col" class=${c.r ? 'r' : ''}>${c.label}</th>`)}</tr></thead>
              <tbody>${res.data.map((row) => html`<tr key=${rowKey(row)}>${columns.map((c) => html`<td data-label=${c.label} class=${c.r ? 'r' : ''}>${c.cell(row)}</td>`)}</tr>`)}</tbody>
            </table></div>`}
  </section>`;
}

function ActivityLog() {
  const [level, setLevel] = useState('');
  const [category, setCategory] = useState('');
  const [state, setState] = useState({ items: [], loading: true, error: null, done: false });
  const req = useRef(0);
  const now = useTick(30);
  const fetchPage = async (beforeId) => {
    const id = ++req.current;
    setState((s) => ({ ...s, loading: true }));
    try {
      const items = await api.get('/activity', { limit: ACTIVITY_PAGE, level: level || null, category: category || null, before_id: beforeId ?? null });
      if (id !== req.current) return;
      setState((s) => ({ items: beforeId ? [...s.items, ...items] : items, loading: false, error: null, done: items.length < ACTIVITY_PAGE }));
    } catch (err) {
      if (id === req.current) setState((s) => ({ ...s, loading: false, error: errorText(err) }));
    }
  };
  useEffect(() => { fetchPage(null); }, [level, category]);
  useLive('activity', (a) => {
    if ((level && a.level !== level) || (category && a.category !== category)) return;
    setState((s) => {
      if (s.items.some((x) => x.id === a.id)) return s;
      // live entries are capped; whatever falls off the end can be paged in again ('Ältere Einträge')
      const items = [a, ...s.items];
      return items.length > MAX_ACTIVITY_ITEMS ? { ...s, items: items.slice(0, MAX_ACTIVITY_ITEMS), done: false } : { ...s, items };
    });
  });
  return html`<section class="panel" aria-labelledby="act-title">
    <div class="panel-head"><h2 id="act-title">Aktivitätsprotokoll</h2>
      <div class="aside">
        <label class="field-inline"><span class="sr-only">Stufe</span>
          <select class="select" value=${level} onChange=${(e) => setLevel(e.currentTarget.value)}>
            <option value="">alle Stufen</option>${Object.entries(LEVEL_LABELS).map(([k, v]) => html`<option value=${k}>${v}</option>`)}
          </select></label>
        <label class="field-inline"><span class="sr-only">Bereich</span>
          <select class="select" value=${category} onChange=${(e) => setCategory(e.currentTarget.value)}>
            <option value="">alle Bereiche</option>${Object.entries(ACTIVITY_CATEGORY_LABELS).map(([k, v]) => html`<option value=${k}>${v}</option>`)}
          </select></label>
      </div></div>
    ${state.error && !state.items.length ? html`<${ErrorState} error=${state.error} onRetry=${() => fetchPage(null)} />`
      : state.loading && !state.items.length ? html`<${Loading} lines=${5} />`
        : !state.items.length ? html`<${Empty} title="Keine Einträge für diese Auswahl.">Neue Ereignisse erscheinen hier live.</${Empty}>`
          : html`<ol class="activity">${state.items.map((a) => html`<li key=${a.id} class=${`act act-${a.level}`}>
              <span class="act-time num">${fmtWhen(a.ts, now)}</span>
              <span class=${`act-level act-level-${a.level}`}>${levelLabel(a.level)}</span>
              <span class="act-cat muted">${activityCategoryLabel(a.category)}</span>
              <span class="act-msg">${a.message}${a.ref_type === 'signal' && a.ref_id ? html` <a href=${`#/signale/${a.ref_id}`}>Signal #${a.ref_id}</a>` : ''}</span>
            </li>`)}</ol>`}
    <div class="list-more">
      ${!state.done && state.items.length > 0 && html`<button type="button" class="btn btn-sm" disabled=${state.loading}
        onClick=${() => fetchPage(state.items[state.items.length - 1].id)}>Ältere Einträge</button>`}
    </div>
  </section>`;
}

function About({ status }) {
  return html`<section class="panel about" aria-labelledby="about-title">
    <div class="panel-head"><h2 id="about-title">Über</h2></div>
    <dl class="kv">
      <div><dt>Version</dt><dd>${status?.version || DASH}</dd></div>
      <div><dt>Engine</dt><dd>${status?.engine_version || DASH}</dd></div>
      <div><dt>Basiswährung</dt><dd>${status?.base_currency || DASH}</dd></div>
    </dl>
    <div class="about-text prose small">
      <p>Kursdaten: Yahoo Finance, teilweise verzögert (die Verzögerung steht bei jedem Wert); Kryptopaare in EUR in Echtzeit von Kraken (öffentliche Marktdaten, ohne API-Schlüssel). Keine Anlageberatung – alle Depots sind Papierdepots.</p>
      <p>Die Strategieparameter sind feste Lehrbuchwerte und werden nicht auf die gespeicherte Historie angepasst; jedes Ergebnis wird außerhalb der Stichprobe verfolgt.</p>
      <p>Charts: TradingView Lightweight Charts™ – Copyright © 2025 TradingView, Inc.${' '}
        (<a href="https://www.tradingview.com/" target="_blank" rel="noopener noreferrer">tradingview.com</a>), Apache-Lizenz 2.0.</p>
      <p>Oberfläche: Preact und htm (MIT bzw. Apache-Lizenz 2.0). Schrift: Archivo (SIL Open Font License 1.1). Die Lizenztexte liegen in den Ordnern <code>static/vendor</code> und <code>static/fonts</code>.</p>
    </div>
  </section>`;
}

export function SystemView() {
  const status = useStore(app, (s) => s.status);
  const statusError = useStore(app, (s) => s.statusError);
  const jobs = useResource('/system/jobs', null, []);
  const feeds = useResource('/system/feeds', null, []);
  const feedsSorted = { ...feeds, data: feeds.data ? [...feeds.data].sort((a, b) => b.fail_count - a.fail_count || a.name.localeCompare(b.name)) : null };
  const llm = useResource('/system/llm-calls', null, []);
  const mails = useResource('/mail/log', { limit: 50 }, []);
  useInterval(() => { jobs.reload(); llm.reload(); }, 30000);
  useTick(15);
  const now = serverNow();
  return html`<div>
    <div class="page-head"><div><h1>System</h1><p class="lede">Zustand der Hintergrunddienste. Fehler einzelner Feeds oder Werte halten den Minutentakt nicht an.</p></div>
      <a class="btn" href="#/einstellungen">Einstellungen</a></div>
    ${status && statusError && html`<p class="notice" role="alert">${statusError} Die Kacheln zeigen den letzten bekannten Stand von ${fmtTime(status.server_time)}.</p>`}
    ${status ? html`<${Tiles} status=${status} />` : statusError ? html`<${ErrorState} error=${statusError} />` : html`<${Loading} lines=${5} />`}
    <div class="grid-2 sys-grid">
      <${SimpleTable} title="Hintergrundjobs" res=${jobs} rowKey=${(j) => j.job} empty="Noch keine Jobs gelaufen."
        columns=${[
          { label: 'Job', cell: (j) => j.label_de || j.job },
          { label: 'Zuletzt', cell: (j) => (j.last_started_ts ? fmtWhen(j.last_started_ts, now) : DASH) },
          { label: 'Dauer', r: true, cell: (j) => fmtMs(j.last_duration_ms) },
          { label: 'Ergebnis', cell: (j) => html`<span class="row" style=${{ gap: '6px', display: 'inline-flex' }}><span class=${`dot ${j.ok ? 'dot-ok' : 'dot-error'}`} aria-hidden="true"></span>${j.ok ? 'ok' : 'Fehler'}</span>${j.message && html`<div class="tiny muted">${j.message}</div>`}` },
        ]} />
      <${SimpleTable} title="KI-Aufrufe" res=${llm} rowKey=${(c) => c.id} empty="Noch keine KI-Aufrufe."
        columns=${[
          { label: 'Zeit', cell: (c) => fmtWhen(c.ts, now) },
          { label: 'Art', cell: (c) => llmKindLabel(c.kind) },
          { label: 'Antwort', r: true, cell: (c) => fmtMs(c.latency_ms) },
          { label: 'Wartezeit', r: true, cell: (c) => fmtMs(c.queue_wait_ms) },
          { label: 'Tokens', r: true, cell: (c) => (isNum(c.prompt_tokens) ? `${fmtNum(c.prompt_tokens, 0)} / ${fmtNum(c.output_tokens, 0)}` : DASH) },
          { label: 'Ergebnis', cell: (c) => (c.ok ? 'ok' : html`<span class="field-error">${c.error || 'Fehler'}</span>`) },
        ]} />
    </div>
    <div class="sys-stack stack">
      <${SimpleTable} title="Nachrichten-Feeds" res=${feedsSorted} rowKey=${(f) => f.feed_id} empty="Keine Feeds konfiguriert."
        columns=${[
          { label: 'Feed', cell: (f) => html`<span class="row" style=${{ gap: '6px', display: 'inline-flex' }}><span class=${`dot ${f.fail_count > 0 ? 'dot-error' : 'dot-ok'}`} aria-hidden="true"></span>${f.name}</span>` },
          { label: 'Region', cell: (f) => NEWS_REGION_LABELS[f.region] || f.region },
          { label: 'Sprache', cell: (f) => f.language },
          { label: 'Zuletzt ok', cell: (f) => (f.last_ok_ts ? fmtAge(f.last_ok_ts, now) : 'noch nie') },
          { label: 'Fehler', cell: (f) => (f.fail_count ? html`<span>${f.fail_count}× · <span class="muted">${f.last_error}</span></span>` : 'keine') },
          { label: 'Meldungen', r: true, cell: (f) => fmtNum(f.items_total, 0) },
        ]} />
      <${ActivityLog} />
      <${SimpleTable} title="Mail-Protokoll" res=${mails} rowKey=${(m) => m.id} empty="Noch keine Mails versendet."
        columns=${[
          { label: 'Zeit', cell: (m) => fmtWhen(m.ts, now) },
          { label: 'Art', cell: (m) => mailKindLabel(m.kind) },
          { label: 'Betreff', cell: (m) => html`${m.subject}${m.signal_id ? html` <a href=${`#/signale/${m.signal_id}`}>#${m.signal_id}</a>` : ''}` },
          { label: 'Empfänger', cell: (m) => m.recipient },
          { label: 'Status', cell: (m) => html`${mailStatusLabel(m.status)}${m.detail && html`<div class="tiny muted">${m.detail}</div>`}` },
        ]} />
      <${About} status=${status} />
    </div>
  </div>`;
}
