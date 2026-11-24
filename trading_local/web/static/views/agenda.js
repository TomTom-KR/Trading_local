// #/termine - agenda: macro data, central-bank decisions and earnings with forecast, actual value and surprise.
// Grouped by local day; filters for importance, kind, period and instrument. Entries whose entry block is
// active are marked; the list reloads every minute (actual values arrive seconds after a release).

import { html, useMemo, useState } from '../vendor/preact-htm.mjs';
import { app } from '../lib/data.js';
import { dayTitle, figure, groupByDay } from '../lib/agenda.js';
import { DASH, fmtNum, fmtTime } from '../lib/format.js';
import { AGENDA_KIND_LABELS, IMPORTANCE_LABELS, SURPRISE_LABELS } from '../lib/labels.js';
import { useResource } from '../lib/resource.js';
import { symbolHref } from '../lib/router.js';
import { useInterval, useStore, useTick } from '../lib/store.js';
import { Empty, ErrorState, Loading, Segmented } from '../components/ui.js';

const PERIODS = { week: 7, month: 28 };
const IMPORTANCE_OPTIONS = [
  { value: '3', label: 'hoch' }, { value: '2', label: 'ab mittel' }, { value: '1', label: 'alle' },
];
const PERIOD_OPTIONS = [{ value: 'week', label: '7 Tage' }, { value: 'month', label: '4 Wochen' }];

function Importance({ level }) {
  const dots = [1, 2, 3].map((i) => (i <= level ? '●' : '○')).join('');
  return html`<span class=${`imp imp-${level}`} title=${`Wichtigkeit ${IMPORTANCE_LABELS[level] || level}`}
    aria-label=${`Wichtigkeit ${IMPORTANCE_LABELS[level] || level}`}>${dots}</span>`;
}

function Surprise({ e }) {
  if (!e.actual_text) return html`<span class="faint">${DASH}</span>`;
  if (!e.surprise_label) return html`<span class="muted">ohne Prognose</span>`;
  // only for earnings is the side also good or bad news; for macro data it depends on the indicator
  const tone = e.kind === 'earnings' && e.surprise_label !== 'inline'
    ? (e.surprise_label === 'above' ? 'chip-up' : 'chip-down') : 'chip-solid';
  const detail = e.kind === 'earnings' && e.surprise !== null
    ? ` (${e.surprise > 0 ? '+' : ''}${fmtNum(e.surprise, 1)} %)`
    : e.surprise_z !== null ? ` (${e.surprise_z > 0 ? '+' : ''}${fmtNum(e.surprise_z, 1)} σ)` : '';
  return html`<span class=${`chip ${tone}`}>${SURPRISE_LABELS[e.surprise_label]}${detail}</span>`;
}

function EventName({ e }) {
  if (e.kind === 'earnings') {
    return html`<a href=${symbolHref(e.symbol)}>${e.name || e.symbol}</a> <span class="muted">Quartalszahlen</span>`;
  }
  return html`<span class="chip chip-quiet agenda-ccy">${e.currency || '?'}</span> ${e.title}
    ${e.kind === 'central_bank' && html` <span class="chip chip-solid">Notenbank</span>`}`;
}

function BlockNote({ e, now }) {
  if (e.blocks_entries) return html` <span class="chip chip-down" title="Keine neuen Einstiege für die betroffenen Werte">Einstiege gesperrt</span>`;
  if (e.block_from_ts !== null && e.block_from_ts > now) return html` · sperrt ab ${fmtTime(e.block_from_ts)}`;
  return null;
}

function EventRow({ e, now }) {
  const past = e.event_ts < now;
  const affected = e.affected === 1 ? '1 handelbarer Wert' : `${fmtNum(e.affected, 0)} handelbare Werte`;
  return html`<tr class=${past ? 'agenda-past' : ''}>
    <td data-label="Zeit" class="num">${e.time_known ? fmtTime(e.event_ts) : html`<span class="muted">ganztägig</span>`}</td>
    <td data-label="Termin"><div class="agenda-name"><${EventName} e=${e} /></div>
      <div class="small faint">${AGENDA_KIND_LABELS[e.kind]} · betrifft ${affected}<${BlockNote} e=${e} now=${now} /></div></td>
    <td data-label="Wichtigkeit"><${Importance} level=${e.importance} /></td>
    <td data-label="Prognose" class="r num">${figure(e.forecast_text)}</td>
    <td data-label="Vorwert" class="r num">${figure(e.previous_text)}</td>
    <td data-label="Ist" class="r num"><strong>${figure(e.actual_text)}</strong></td>
    <td data-label="Überraschung"><${Surprise} e=${e} /></td>
  </tr>`;
}

export function AgendaView() {
  const instruments = useStore(app, (s) => s.instruments);
  const [minImportance, setMinImportance] = useState('2');
  const [kind, setKind] = useState('');
  const [period, setPeriod] = useState('week');
  const [symbol, setSymbol] = useState('');
  const now = useTick(30);
  const from = useMemo(() => Math.floor(now / 3600) * 3600 - 86400, [Math.floor(now / 3600)]);
  const to = from + 86400 + PERIODS[period] * 86400;
  const res = useResource('/agenda', {
    from_ts: from, to_ts: to, min_importance: minImportance, kind: kind || null, symbol: symbol || null,
  }, [from, to, minImportance, kind, symbol]);
  const settings = useResource('/settings', null, []);
  useInterval(() => res.reload(), 60000);

  const days = useMemo(() => groupByDay(res.data || []), [res.data]);
  const blocking = (res.data || []).filter((e) => e.blocks_entries);
  const block = settings.data;
  const tradable = useMemo(() => [...instruments].filter((i) => i.tradable && i.active)
    .sort((a, b) => a.symbol.localeCompare(b.symbol)), [instruments]);

  return html`<div>
    <div class="page-head"><div><h1>Termine</h1>
      <p class="lede">Konjunkturdaten, Notenbankentscheidungen und Quartalszahlen mit Prognose, Istwert und Überraschung. Vor wichtigen Terminen gibt die Engine keine neuen Einstiege aus.</p></div>
      ${blocking.length > 0 && html`<p class="small"><span class="chip chip-down">Sperre aktiv</span> ${blocking.map((e) => e.title).join(', ')}</p>`}
    </div>
    <div class="settings-grid"><div>
    <section class="panel" aria-label="Terminliste">
      <div class="filters">
        <div class="field-inline"><span class="field-label">Wichtigkeit</span>
          <${Segmented} label="Wichtigkeit" options=${IMPORTANCE_OPTIONS} value=${minImportance} onChange=${setMinImportance} /></div>
        <div class="field-inline"><span class="field-label">Zeitraum</span>
          <${Segmented} label="Zeitraum" options=${PERIOD_OPTIONS} value=${period} onChange=${setPeriod} /></div>
        <label class="field-inline"><span class="field-label">Art</span>
          <select class="select" value=${kind} onChange=${(e) => setKind(e.currentTarget.value)}>
            <option value="">alle</option>
            ${Object.entries(AGENDA_KIND_LABELS).map(([k, label]) => html`<option value=${k}>${label}</option>`)}
          </select></label>
        <label class="field-inline"><span class="field-label">Wert</span>
          <select class="select" value=${symbol} onChange=${(e) => setSymbol(e.currentTarget.value)}>
            <option value="">alle</option>
            ${tradable.map((i) => html`<option value=${i.symbol}>${i.symbol} – ${i.name}</option>`)}
          </select></label>
      </div>
      ${res.error && !res.data ? html`<${ErrorState} error=${res.error} onRetry=${res.reload} />`
        : res.loading && !res.data ? html`<${Loading} lines=${8} />`
          : !days.length ? html`<${Empty} title="Keine Termine für diese Auswahl.">
              Der Kalender der laufenden Woche wird stündlich geladen, die Quartalstermine nach dem Start nach und nach für alle Aktien.</${Empty}>`
            : days.map((d) => html`<div class="agenda-day">
                <h2 class="agenda-day-h">${dayTitle(d.ts, now)}</h2>
                <div class="table-wrap"><table class="table stack-mobile agenda-table">
                  <colgroup><col class="ag-time" /><col /><col class="ag-imp" /><col class="ag-num" /><col class="ag-num" /><col class="ag-num" /><col class="ag-surprise" /></colgroup>
                  <thead><tr><th>Zeit</th><th>Termin</th><th>Wichtigkeit</th><th class="r">Prognose</th><th class="r">Vorwert</th><th class="r">Ist</th><th>Überraschung</th></tr></thead>
                  <tbody>${d.events.map((e) => html`<${EventRow} key=${e.id} e=${e} now=${now} />`)}</tbody>
                </table></div>
              </div>`)}
    </section>
    </div>
    <aside class="panel small muted settings-aside agenda-aside">
      <h2 class="side-h">Sperre vor Terminen</h2>
      ${block && (block.event_block_enabled
        ? html`<p>Aktiv: ${fmtNum(block.event_block_before_min, 0)} Minuten vor wichtigen Konjunkturdaten und Notenbankentscheidungen in der Währung eines Werts (US-Termine auch für Kryptowährungen) gibt es keinen neuen Einstieg. Eigene Quartalszahlen sperren, wenn sie vor den geplanten Ausstieg fallen.</p>`
        : html`<p>Ausgeschaltet. Einschalten unter <a href="#/einstellungen">Einstellungen → Signale</a>.</p>`)}
      <p>Gesperrte Kandidaten werden als verworfen mit dem Grund „Termin“ weiter verfolgt. So zeigt das Signal-Journal später, ob die Sperre Verluste vermieden oder Gewinne gekostet hat.</p>
      <h2 class="side-h">Überraschung</h2>
      <p>Istwert minus Prognose. Gibt es mindestens sechs frühere Veröffentlichungen desselben Termins, steht sie zusätzlich in Standardabweichungen (σ) der früheren Überraschungen. Ob „über Erwartung“ gut oder schlecht ist, hängt vom Indikator ab – bei Quartalszahlen ist es eine gute Nachricht.</p>
      <h2 class="side-h">Quellen</h2>
      <p>Wochenkalender von ForexFactory (stündlich), Istwerte aus den FinancialJuice-Meldungen (Sekunden nach der Veröffentlichung), Quartalstermine von Yahoo Finance.</p>
    </aside>
    </div>
  </div>`;
}
