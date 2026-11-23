// #/signale - statistics of all tracked signals, the signal table with infinite paging, and
// #/signale/:id - the detail drawer over the list (deep-linkable from the alert e-mail).

import { html, useEffect, useMemo, useRef, useState } from '../vendor/preact-htm.mjs';
import { api, errorText, isAbort } from '../lib/api.js';
import { app, instrumentOf } from '../lib/data.js';
import {
  DASH, fmtDateTime, fmtFlex, fmtNum, fmtPrice, fmtPctSigned, fmtR, fmtShare, fmtWhen, isNum, priceUnit,
} from '../lib/format.js';
import {
  componentLabel, directionLabel, featureLabel, regionLabel, shadowReasonLabel, statusReasonLabel, verdictLabel,
} from '../lib/labels.js';
import { useLive } from '../lib/live.js';
import { navigate, symbolHref } from '../lib/router.js';
import { useResource } from '../lib/resource.js';
import {
  OPEN_STATUSES, holdingPeriodText, signalLevels, timeframeLabel, upsertSignal, withLiveR,
} from '../lib/signals.js';
import { useStore, useTick } from '../lib/store.js';
import { serverNow } from '../lib/time.js';
import { NewsItem } from '../components/news-item.js';
import { Drawer } from '../components/overlay.js';
import { PriceChartView } from '../components/price-chart.js';
import { CalibrationChart, GroupTable, StatBlockView } from '../components/stats.js';
import { followSignal } from '../components/ticket-drawer.js';
import {
  Confidence, DirectionBadge, Empty, ErrorState, Loading, RValue, Segmented, SignalStatus, Tabs, Verdict,
} from '../components/ui.js';

const PAGE_SIZE = 40;
const SCOPES = [{ value: 'open', label: 'Offen' }, { value: 'closed', label: 'Abgeschlossen' }, { value: 'all', label: 'Alle' }];
const KINDS = [
  { value: 'true', label: 'Empfehlungen' }, { value: 'false', label: 'Verworfene Kandidaten' }, { value: 'all', label: 'Alle' },
];

function StatsSection() {
  const stats = useResource('/signals/stats', null, []);
  if (stats.loading) return html`<section class="panel"><${Loading} lines=${4} /></section>`;
  if (stats.error) return html`<section class="panel"><${ErrorState} error=${stats.error} onRetry=${stats.reload} /></section>`;
  const s = stats.data;
  return html`<section class="panel sstats" aria-labelledby="st-title">
    <div class="panel-head"><h2 id="st-title">Bilanz aller verfolgten Signale</h2>
      <div class="aside"><span>Ergebnisse in R nach Slippage, ohne Ordergebühren</span>${s.engine_version && html`<span>Engine ${s.engine_version}</span>`}</div></div>
    ${s.note_de && html`<p class="notice notice-strong">${s.note_de}</p>`}
    <div class="sstats-grid">
      <${StatBlockView} title="Empfehlungen" stats=${s.actionable} />
      <${StatBlockView} title="Verworfene Kandidaten" stats=${s.shadow} />
      <div class="sstats-calib">
        <h3 class="statblock-title">Kalibrierung der Konfidenz</h3>
        <${CalibrationChart} bins=${s.calibration} />
      </div>
    </div>
    <details class="sstats-groups" open>
      <summary>Aufschlüsselung nach Setup, Region, Richtung und KI-Prüfung</summary>
      <div class="grid-4">
        <${GroupTable} title="Setup" groups=${s.by_setup} />
        <${GroupTable} title="Region" groups=${s.by_region} />
        <${GroupTable} title="Richtung" groups=${s.by_direction} />
        <${GroupTable} title="KI-Prüfung" groups=${s.by_llm_verdict} />
      </div>
    </details>
  </section>`;
}

function SignalTable() {
  const [scope, setScope] = useState('open');
  const [kind, setKind] = useState('true');
  const [symbol, setSymbol] = useState('');
  const [state, setState] = useState({ items: [], loading: true, error: null, done: false, more: false });
  const instruments = useStore(app, (s) => s.instruments);
  const quotes = useStore(app, (s) => s.quotes);
  const sentinel = useRef(null);
  const now = useTick(30);
  const reqRef = useRef(0);
  const stateRef = useRef(state);
  stateRef.current = state;

  const fetchPage = async (beforeId) => {
    const req = ++reqRef.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const items = await api.get('/signals', {
        scope, actionable: kind, symbol: symbol || null, limit: PAGE_SIZE, before_id: beforeId ?? null,
      });
      if (req !== reqRef.current) return;
      setState((s) => ({
        items: beforeId ? [...s.items, ...items] : items, loading: false, error: null, done: items.length < PAGE_SIZE, more: false,
      }));
    } catch (err) {
      if (req === reqRef.current && !isAbort(err)) setState((s) => ({ ...s, loading: false, error: errorText(err) }));
    }
  };

  useEffect(() => { fetchPage(null); }, [scope, kind, symbol]);

  // infinite paging: load the next page when the sentinel scrolls into view
  useEffect(() => {
    const el = sentinel.current;
    if (!el || typeof IntersectionObserver === 'undefined') return undefined;
    const io = new IntersectionObserver((entries) => {
      const s = stateRef.current;
      if (entries.some((e) => e.isIntersecting) && !s.loading && !s.done && s.items.length && !s.error) {
        fetchPage(s.items[s.items.length - 1].id);
      }
    }, { rootMargin: '300px' });
    io.observe(el);
    return () => io.disconnect();
  }, [scope, kind, symbol, state.items.length]);

  useLive('signal', (sig) => {
    const matchesKind = kind === 'all' || String(sig.actionable) === kind;
    const isOpen = OPEN_STATUSES.has(sig.status);
    const matchesScope = scope === 'all' || (scope === 'open' ? isOpen : !isOpen);
    setState((s) => {
      const known = s.items.some((x) => x.id === sig.id);
      if (!matchesKind || (symbol && sig.symbol !== symbol) || !matchesScope) {
        return known ? { ...s, items: s.items.filter((x) => x.id !== sig.id) } : s;
      }
      return { ...s, items: upsertSignal(s.items, sig, Math.max(s.items.length + 1, PAGE_SIZE)) };
    });
  });

  const items = useMemo(() => withLiveR(state.items, quotes), [state.items, quotes]);
  const symbols = useMemo(() => [...instruments].sort((a, b) => a.symbol.localeCompare(b.symbol)), [instruments]);
  return html`<section class="panel" aria-labelledby="sl-title">
    <div class="panel-head"><h2 id="sl-title">Signale</h2></div>
    <${Tabs} tabs=${SCOPES} value=${scope} onChange=${setScope} label="Signalstatus" />
    <div class="filters">
      <${Segmented} options=${KINDS} value=${kind} onChange=${setKind} label="Art" />
      <label class="field-inline"><span class="field-label">Symbol</span>
        <select class="select" value=${symbol} onChange=${(e) => setSymbol(e.currentTarget.value)}>
          <option value="">alle</option>
          ${symbols.map((i) => html`<option value=${i.symbol}>${i.symbol} – ${i.name}</option>`)}
        </select>
      </label>
    </div>
    ${state.error && !items.length ? html`<${ErrorState} error=${state.error} onRetry=${() => fetchPage(null)} />`
      : !state.loading && !items.length ? html`<${Empty} title=${scope === 'open' ? 'Gerade ist kein Signal offen.' : 'Keine Signale für diese Auswahl.'}>
          ${scope === 'open' ? 'Die Engine prüft jede Minute alle offenen Märkte – neue Signale erscheinen hier sofort.' : 'Andere Filter wählen oder später wiederkommen: abgeschlossene Signale sammeln sich mit der Zeit.'}</${Empty}>`
        : html`<div class="table-wrap"><table class="table stack-mobile sigtable">
          <thead><tr>
            <th scope="col">Zeit</th><th scope="col">Richtung</th><th scope="col">Wert</th><th scope="col">Setup</th>
            <th scope="col" class="r">Referenz</th><th scope="col" class="r">Stopp / Ziel</th><th scope="col" class="r">C/R</th>
            <th scope="col">Konfidenz</th><th scope="col">KI</th><th scope="col">Status</th>
          </tr></thead>
          <tbody>${items.map((s) => {
            const inst = instrumentOf(s.symbol);
            const { basis, stop, target } = signalLevels(s);
            return html`<tr class="clickable" key=${s.id} onClick=${() => navigate(`/signale/${s.id}`)}>
              <td data-label="Zeit" class="nowrap"><a href=${`#/signale/${s.id}`} onClick=${(e) => e.stopPropagation()}>${fmtWhen(s.created_ts, now)}</a></td>
              <td data-label="Richtung"><${DirectionBadge} direction=${s.direction} /></td>
              <td data-label="Wert"><span class="sym">${s.symbol}</span> <span class="name">${s.name}</span></td>
              <td data-label="Setup">${s.setup_label_de || s.setup}${!s.actionable && html`<div class="tiny faint">${shadowReasonLabel(s.shadow_reason) || 'verworfen'}</div>`}</td>
              <td data-label="Referenz" class="r">${fmtPrice(basis, inst?.asset_class)}</td>
              <td data-label="Stopp / Ziel" class="r nowrap">${fmtPrice(stop, inst?.asset_class)} / ${fmtPrice(target, inst?.asset_class)}</td>
              <td data-label="C/R" class="r">${fmtNum(s.rr, 1)}</td>
              <td data-label="Konfidenz"><${Confidence} value=${s.confidence} /></td>
              <td data-label="KI">${verdictLabel(s.llm_verdict)}</td>
              <td data-label="Status"><${SignalStatus} signal=${s} /></td>
            </tr>`;
          })}</tbody>
        </table></div>`}
    <div ref=${sentinel} class="sentinel" aria-hidden="true"></div>
    ${state.loading && html`<${Loading} lines=${2} label="Weitere Signale werden geladen" />`}
    ${state.error && items.length > 0 && html`<${ErrorState} error=${state.error} onRetry=${() => fetchPage(items[items.length - 1].id)} title="Weitere Signale konnten nicht geladen werden." />`}
    ${state.done && items.length > 0 && html`<p class="small faint list-end">Ende der Liste · ${fmtNum(items.length, 0)} Signale</p>`}
  </section>`;
}

/** A stored feature for the reader: flags as ja/nein, ISO dates as German dates, numbers without noise. */
function featureValue(key, v) {
  if (key === 'in_session') return v ? 'ja' : 'nein';
  if (typeof v === 'string') return /^\d{4}-\d{2}-\d{2}$/.test(v) ? v.split('-').reverse().join('.') : v;
  return Number.isInteger(v) ? fmtNum(v, 0) : fmtFlex(v, Math.abs(v) >= 100 ? 2 : 4);
}

/** Score component bars 0..1 with the neutral 0.5 marked. */
function ComponentBars({ components }) {
  const entries = Object.entries(components || {});
  if (!entries.length) return html`<p class="small muted">Keine Bestandteile gespeichert.</p>`;
  return html`<ul class="cbars">
    ${entries.map(([k, v]) => html`<li>
      <span class="cbars-label">${componentLabel(k)}</span>
      <span class="cbars-track" role="img" aria-label=${`${componentLabel(k)}: ${fmtNum(v, 2)} (neutral 0,50)`}>
        <span class="cbars-fill" style=${{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }}></span>
        <span class="cbars-mid"></span>
      </span>
      <span class="cbars-v num">${fmtNum(v, 2)}</span>
    </li>`)}
  </ul>`;
}

function Timeline({ s }) {
  const verdictText = s.llm_verdict ? `KI: ${verdictLabel(s.llm_verdict)}${isNum(s.llm_conviction) ? `, Überzeugung ${fmtShare(s.llm_conviction)}` : ''}` : 'ohne KI-Prüfung';
  const closed = !OPEN_STATUSES.has(s.status);
  const reason = statusReasonLabel(s.status_reason);
  const steps = [
    { label: 'erkannt', ts: s.created_ts, text: `${s.setup_label_de || s.setup}, Daten bis ${fmtDateTime(s.data_ts)}` },
    { label: 'geprüft', ts: s.status === 'review' ? null : s.issued_ts ?? s.created_ts, text: s.status === 'review' ? 'Prüfung läuft' : verdictText },
    { label: 'ausgegeben', ts: s.issued_ts, text: s.actionable ? (s.mailed_ts ? 'als Empfehlung, per Mail gemeldet' : 'als Empfehlung') : `nicht ausgegeben: ${shadowReasonLabel(s.shadow_reason) || 'verworfener Kandidat'}`, skip: !s.actionable },
    { label: 'Einstieg', ts: s.entry_ts, text: isNum(s.entry_price) ? `zu ${fmtPrice(s.entry_price)}` : s.status === 'cancelled' ? `kein Einstieg${reason ? `: ${reason}` : ''}` : 'wartet auf den nächsten Kurs' },
    { label: 'Ausstieg', ts: s.exit_ts, text: isNum(s.exit_price) ? `zu ${fmtPrice(s.exit_price)}${reason ? ` · ${reason}` : ''}` : closed ? reason || 'beendet' : `spätestens ${s.expires_ts ? fmtDateTime(s.expires_ts) : 'nach Ablauf der Haltedauer'}` },
  ];
  return html`<ol class="tl">
    ${steps.map((st) => html`<li class=${st.ts ? 'done' : st.skip ? 'skip' : ''}>
      <span class="tl-dot" aria-hidden="true"></span>
      <span class="tl-label">${st.label}</span>
      <span class="tl-time num">${st.ts ? fmtDateTime(st.ts) : '–'}</span>
      <span class="tl-text">${st.text}</span>
    </li>`)}
  </ol>`;
}

function SignalDrawer({ id }) {
  const res = useResource(`/signals/${encodeURIComponent(id)}`, null, [id]);
  const quotes = useStore(app, (s) => s.quotes);
  useLive('signal', (sig) => { if (String(sig.id) === String(id)) res.set((d) => (d ? { ...d, ...sig } : d)); });
  const close = () => navigate('/signale');
  const s = res.data ? withLiveR([res.data], quotes)[0] : null;
  const inst = s ? instrumentOf(s.symbol) : null;
  const unit = s ? priceUnit(s.currency || inst?.currency || '', inst?.price_scale ?? 1) : '';
  const head = s ? html`<div class="row"><${DirectionBadge} direction=${s.direction} /><h2>${s.symbol}</h2>
      <span class="muted truncate">${s.name}</span><${SignalStatus} signal=${s} /></div>` : html`<h2>Signal #${id}</h2>`;
  return html`<${Drawer} title=${`Signal #${id}`} onClose=${close} head=${head}>
    ${res.loading ? html`<${Loading} lines=${8} />`
      : res.error ? html`<${ErrorState} error=${res.error} onRetry=${res.reload} title=${`Signal #${id} konnte nicht geladen werden.`} />`
        : html`<${SignalDetail} s=${s} inst=${inst} unit=${unit} />`}
  </${Drawer}>`;
}

function SignalDetail({ s, inst, unit }) {
  const ac = inst?.asset_class;
  const { basis, stop, target } = signalLevels(s);
  const pct = (lvl) => (basis ? fmtPctSigned(((lvl - basis) / basis) * 100, 1) : DASH);
  const markers = useMemo(() => {
    const long = s.direction === 'LONG';
    const m = [];
    if (s.entry_ts) m.push({ ts: s.entry_ts, position: long ? 'belowBar' : 'aboveBar', shape: long ? 'arrowUp' : 'arrowDown', color: long ? 'up' : 'down', text: 'Einstieg' });
    else m.push({ ts: s.created_ts, position: long ? 'belowBar' : 'aboveBar', shape: long ? 'arrowUp' : 'arrowDown', color: 'ink2', text: 'erkannt' });
    if (s.exit_ts) m.push({ ts: s.exit_ts, position: long ? 'aboveBar' : 'belowBar', shape: 'circle', color: 'ink2', text: `Ausstieg ${fmtR(s.outcome_r)}`, size: 0.7 });
    return m;
  }, [s.id, s.entry_ts, s.exit_ts]);
  const lines = useMemo(() => [
    { price: basis, color: 'ink2', style: 'dashed', title: s.entry_price ? 'Einstieg' : 'Referenz' },
    { price: stop, color: 'down', style: 'dashed', title: 'Stopp' },
    { price: target, color: 'up', style: 'dashed', title: 'Ziel' },
  ], [basis, stop, target]);
  const closed = !OPEN_STATUSES.has(s.status);
  const before = closed && (s.exit_ts || s.expires_ts) ? (s.exit_ts || s.expires_ts) + 6 * 3600 : null;
  const canFollow = OPEN_STATUSES.has(s.status) && s.actionable && inst?.tradable !== false;
  const features = Object.entries(s.features || {}).filter(([, v]) => typeof v === 'number' || typeof v === 'string');

  return html`<div class="sdetail">
    <p class="small muted">${s.setup_label_de || s.setup} · ${timeframeLabel(s.timeframe)} · ${regionLabel(s.region)} · erkannt ${fmtDateTime(s.created_ts)}
      ${s.delay_min > 0 ? ` · Daten ${s.delay_min} Min verzögert` : ''}</p>
    ${!s.actionable && html`<p class="notice small">Verworfener Kandidat – nicht ausgegeben${s.shadow_reason ? `: ${shadowReasonLabel(s.shadow_reason)}` : ''}. Er wird trotzdem verfolgt, damit die Auswahl überprüfbar bleibt.</p>`}
    <dl class="kv sdetail-kv">
      <div><dt>${s.entry_price ? 'Einstieg' : 'Referenzkurs'}</dt><dd>${fmtPrice(basis, ac)} ${unit}</dd></div>
      <div><dt>Stopp</dt><dd>${fmtPrice(stop, ac)} <span class="muted">${pct(stop)}</span></dd></div>
      <div><dt>Ziel</dt><dd>${fmtPrice(target, ac)} <span class="muted">${pct(target)}</span></dd></div>
      <div><dt>Chance/Risiko</dt><dd>${fmtNum(s.rr, 2)}</dd></div>
      <div><dt>Konfidenz</dt><dd><${Confidence} value=${s.confidence} /></dd></div>
      <div><dt>Score</dt><dd>${fmtNum(s.score, 2)}</dd></div>
      <div><dt>Erwartetes Ergebnis</dt><dd><${RValue} value=${s.expected_r} digits=${2} /></dd></div>
      <div><dt>ATR</dt><dd>${fmtPrice(s.atr, ac)}</dd></div>
      <div><dt>Haltedauer</dt><dd>${holdingPeriodText(s)}${s.status === 'active' && s.expires_ts ? html`<div class="small muted">spätestens ${fmtDateTime(s.expires_ts)}</div>` : ''}</dd></div>
      <div><dt>Ergebnis</dt><dd>${s.status === 'active' ? html`<${RValue} value=${s.open_r} /> <span class="muted">offen</span>` : html`<${RValue} value=${s.outcome_r} />`}</dd></div>
      <div><dt>Bester Verlauf (MFE)</dt><dd>${fmtR(s.mfe_r)}</dd></div>
      <div><dt>Schlechtester Verlauf (MAE)</dt><dd>${fmtR(s.mae_r)}</dd></div>
    </dl>
    <div class="row sdetail-actions">
      ${canFollow && html`<button type="button" class="btn btn-primary" onClick=${() => followSignal(s)}>Übernehmen</button>`}
      <a class="btn" href=${symbolHref(s.symbol)}>Zum Chart von ${s.symbol}</a>
    </div>

    <h3 class="section-title">Verlauf um das Signal (15 Min)</h3>
    <${PriceChartView} symbol=${s.symbol} instrument=${inst} timeframe="15m" markers=${markers} priceLines=${lines}
      height=${280} limit=${120} before=${before} compactLegend=${true} live=${!closed} label=${s.name || s.symbol} volumeHeight=${48} />

    <h3 class="section-title">Lebenslauf</h3>
    <${Timeline} s=${s} />

    <h3 class="section-title">Bestandteile des Scores</h3>
    <${ComponentBars} components=${s.components} />

    <h3 class="section-title">KI-Prüfung</h3>
    <p class="row"><${Verdict} verdict=${s.llm_verdict} conviction=${s.llm_conviction} />
      ${isNum(s.llm_conviction) && html`<span class="small muted">Überzeugung ${fmtShare(s.llm_conviction)}</span>`}</p>
    ${s.rationale_de && html`<h4 class="sub">Begründung</h4><p class="prose">${s.rationale_de}</p>`}
    ${s.risks_de && html`<h4 class="sub">Risiken</h4><p class="prose">${s.risks_de}</p>`}

    ${features.length > 0 && html`<h3 class="section-title">Messwerte zum Entscheidungszeitpunkt</h3>
      <dl class="kv">${features.map(([k, v]) => html`<div><dt>${featureLabel(k)}</dt><dd>${featureValue(k, v)}</dd></div>`)}</dl>`}

    <h3 class="section-title">Nachrichten zum Signal</h3>
    ${s.news?.length ? html`<div class="news-list news-list-narrow">${s.news.map((n) => html`<${NewsItem} key=${n.id} item=${n} now=${serverNow()} />`)}</div>`
      : html`<p class="small muted">Keine Nachrichten flossen in dieses Signal ein.</p>`}
    <p class="tiny faint sdetail-foot">Signal #${s.id} · Engine ${s.engine_version || DASH} · ${directionLabel(s.direction)}</p>
  </div>`;
}

export function SignalsView({ id }) {
  return html`<div>
    <div class="page-head"><div><h1>Signale</h1>
      <p class="lede">Jedes Signal wird bis zum Ende verfolgt – auch die verworfenen Kandidaten. So bleibt sichtbar, ob die Auswahl der Engine etwas taugt.</p></div></div>
    <div class="stack">
      <${StatsSection} />
      <${SignalTable} />
    </div>
    ${id && html`<${SignalDrawer} id=${id} />`}
  </div>`;
}
