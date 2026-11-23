// #/markt/:symbol - one instrument: header with live price, chart with signals and levels, technical
// readings, levels, KI chart analysis, news, signals and the order ticket.

import {
  html, useEffect, useMemo, useRef, useState,
} from '../vendor/preact-htm.mjs';
import { api, errorText } from '../lib/api.js';
import { app } from '../lib/data.js';
import {
  fmtAge, fmtDuration, fmtNum, fmtPrice, fmtQty, fmtR, fmtWhen, isNum, priceUnit,
} from '../lib/format.js';
import {
  assetClassLabel, biasLabel, directionLabel, phaseLabel, regionLabel,
} from '../lib/labels.js';
import { useLive } from '../lib/live.js';
import { useResource } from '../lib/resource.js';
import { OPEN_STATUSES, signalLevels, upsertSignal, withLiveR } from '../lib/signals.js';
import { useInterval, useStore, useTick } from '../lib/store.js';
import { serverNow } from '../lib/time.js';
import { NewsItem } from '../components/news-item.js';
import { OrderTicket } from '../components/order-ticket.js';
import { PriceChartView } from '../components/price-chart.js';
import {
  Change, DirectionBadge, Empty, ErrorState, FlashValue, Loading, PhaseDot, Segmented, SignalStatus,
} from '../components/ui.js';

const TIMEFRAMES = [
  { value: '1m', label: '1 Min' }, { value: '5m', label: '5 Min' }, { value: '15m', label: '15 Min' },
  { value: '1h', label: '1 Std' }, { value: '1d', label: '1 Tag' },
];
const STANCE = { bullish: { glyph: '▲', cls: 'glyph-up', word: 'positiv' }, bearish: { glyph: '▼', cls: 'glyph-down', word: 'negativ' }, neutral: { glyph: '●', cls: 'faint', word: 'neutral' } };

/**
 * 'schließt in 1 Std 12 Min' / 'öffnet in 6 Std 40 Min' for the instrument's venue. The 24/7 crypto venues
 * (CRYPTO, KRAKEN) have no lane in the session overview: an open crypto instrument trades around the clock.
 */
function venueCountdown(sessions, inst, now) {
  if (!sessions) return '';
  for (const region of sessions.regions) {
    const ex = region.exchanges.find((e) => e.exchange === inst.exchange);
    if (!ex) continue;
    if (ex.phase === 'open') return isNum(ex.next_close) ? `schließt in ${fmtDuration(ex.next_close - now)}` : 'handelt rund um die Uhr';
    if (ex.phase === 'break' && isNum(ex.until)) return `Mittagspause bis ${new Date(ex.until * 1000).toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' })}`;
    if (isNum(ex.next_open)) return `öffnet in ${fmtDuration(ex.next_open - now)}${ex.note ? ` · ${ex.note}` : ''}`;
  }
  return inst.asset_class === 'crypto' && inst.phase === 'open' ? 'handelt rund um die Uhr' : '';
}

function Header({ inst, quote }) {
  const sessions = useStore(app, (s) => s.sessions);
  useTick(15);
  const unit = priceUnit(inst.currency, inst.price_scale);
  return html`<header class="ins-head">
    <div class="ins-id">
      <h1>${inst.name}</h1>
      <div class="ins-meta">
        <span class="ins-sym">${inst.symbol}</span>
        <span>${inst.exchange_name || inst.exchange}</span>
        <span>${assetClassLabel(inst.asset_class)} · ${regionLabel(inst.region)}</span>
        <span class="row" style=${{ gap: '6px', display: 'inline-flex' }}><${PhaseDot} phase=${inst.phase} />${phaseLabel(inst.phase)}</span>
        <span>${venueCountdown(sessions, inst, serverNow())}</span>
      </div>
    </div>
    <div class="ins-price">
      <${FlashValue} value=${quote?.price} class="figure figure-40">${quote ? fmtPrice(quote.price, inst.asset_class) : '–'}</${FlashValue}>
      <div class="ins-price-sub">
        <span class="muted">${unit}</span>
        <${Change} value=${quote?.change_pct} digits=${2} />
        ${quote && html`<span class="faint small">Stand ${fmtWhen(quote.ts, serverNow())}</span>`}
      </div>
      ${inst.delay_min > 0 && html`<div class="small muted">Daten ${inst.delay_min} Min verzögert</div>`}
      ${inst.price_scale !== 1 && html`<div class="small muted">Kurs in ${unit} (${unit === 'GBp' ? 'Notierung in Pence, ' : ''}keine Umrechnung)</div>`}
    </div>
  </header>`;
}

function Readings({ analysis }) {
  if (analysis.loading) return html`<${Loading} lines=${4} />`;
  if (analysis.error) return html`<${ErrorState} error=${analysis.error} onRetry=${analysis.reload} />`;
  const a = analysis.data;
  if (!a?.available) return html`<${Empty} title="Noch zu wenig Kursdaten.">${a?.note_de || 'Die technische Lage erscheint, sobald genug Minutenkerzen gespeichert sind.'}</${Empty}>`;
  return html`<ul class="readings">
    ${a.readings.map((r) => {
      const s = STANCE[r.stance] || STANCE.neutral;
      return html`<li><span class="muted">${r.label_de}</span><span class="reading-v">${r.value_de}</span>
        <span class=${s.cls} role="img" aria-label=${s.word} title=${s.word}>${s.glyph}</span></li>`;
    })}
  </ul>
  ${a.data_ts && html`<p class="tiny faint">Stand der Daten: ${fmtWhen(a.data_ts, serverNow())}${a.note_de ? ` · ${a.note_de}` : ''}</p>`}`;
}

function Levels({ analysis, assetClass, unit }) {
  const levels = analysis.data?.levels || [];
  if (!levels.length) return html`<p class="small muted">Keine Marken berechnet.</p>`;
  return html`<ul class="levels num">
    ${[...levels].sort((a, b) => b.price - a.price).map((l) => html`<li><span>${l.label_de}</span><span>${fmtPrice(l.price, assetClass)} <span class="faint">${unit}</span></span></li>`)}
  </ul>`;
}

// The engine gives up on a chart analysis after 10 minutes (plus up to 1 minute to render the chart); the
// button waits a little longer for the result and then becomes usable again.
const CHART_ANALYSIS_WAIT_MS = 12 * 60 * 1000;
const CHART_NOT_QUEUED_DE = 'Analyse derzeit nicht möglich (KI aus, zu wenig Kursdaten oder bereits eine andere Analyse aktiv).';
const CHART_NO_RESULT_DE = 'Keine Chartanalyse erhalten – die KI hat nicht rechtzeitig geantwortet. Bitte später erneut versuchen.';

function ChartAnalysis({ symbol }) {
  const res = useResource(`/chart-analyses/${encodeURIComponent(symbol)}`, { limit: 3 }, [symbol]);
  // {state: 'sending' | 'waiting' | 'info', text}; only 'sending' and 'waiting' block the button
  const [request, setRequest] = useState(null);
  const waitTimer = useRef(null);
  const symbolRef = useRef(symbol);
  symbolRef.current = symbol;
  const stopWaiting = () => clearTimeout(waitTimer.current);
  useEffect(() => {
    setRequest(null);
    return stopWaiting;
  }, [symbol]);
  useLive('chart_analysis', (ca) => {
    if (ca.symbol !== symbol) return;
    res.set((d) => [ca, ...(d || []).filter((x) => x.id !== ca.id)]);
    stopWaiting();
    setRequest(null);
  });
  const ask = async () => {
    const asked = symbol;
    stopWaiting();
    setRequest({ state: 'sending', text: '…' });
    try {
      const out = await api.post(`/analysis/${encodeURIComponent(asked)}/chart`);
      if (asked !== symbolRef.current) return; // the user moved on to another instrument meanwhile
      if (!out?.queued) {
        setRequest({ state: 'info', text: out?.detail_de || CHART_NOT_QUEUED_DE });
        return;
      }
      setRequest({ state: 'waiting', text: 'in Warteschlange' });
      waitTimer.current = setTimeout(() => setRequest({ state: 'info', text: CHART_NO_RESULT_DE }), CHART_ANALYSIS_WAIT_MS);
    } catch (err) {
      if (asked === symbolRef.current) setRequest({ state: 'info', text: errorText(err) });
    }
  };
  const busy = request?.state === 'sending' || request?.state === 'waiting';
  const latest = res.data?.[0];
  return html`<section class="panel" aria-labelledby="ca-title">
    <div class="panel-head"><h2 id="ca-title">KI-Chartanalyse</h2>
      <div class="aside">
        ${request && html`<span role="status">${request.text}</span>`}
        <button type="button" class="btn btn-sm" onClick=${ask} disabled=${busy}>Neu analysieren</button>
      </div>
    </div>
    ${res.loading ? html`<${Loading} lines=${2} />`
      : res.error ? html`<${ErrorState} error=${res.error} onRetry=${res.reload} />`
        : !latest ? html`<${Empty} title="Noch keine Chartanalyse.">Die KI schaut sich offene Märkte regelmäßig an; mit „Neu analysieren“ kommt dieser Wert sofort in die Warteschlange.</${Empty}>`
          : html`<div class="ca">
              <div class="ca-head">
                <span class="chip chip-solid">Tendenz: ${biasLabel(latest.bias)}</span>
                <span class="chip">Stärke ${latest.strength} von 5</span>
                ${latest.pattern && html`<span class="chip">${latest.pattern}</span>`}
                <span class="small muted">${fmtAge(latest.ts, serverNow())} · ${latest.timeframe} · ${latest.model}</span>
              </div>
              <p class="prose">${latest.comment_de}</p>
              ${(isNum(latest.support) || isNum(latest.resistance)) && html`<p class="small muted num">
                ${isNum(latest.support) ? `Unterstützung ${fmtPrice(latest.support)}` : ''}${isNum(latest.support) && isNum(latest.resistance) ? ' · ' : ''}${isNum(latest.resistance) ? `Widerstand ${fmtPrice(latest.resistance)}` : ''}
              </p>`}
            </div>`}
  </section>`;
}

function SymbolNews({ symbol }) {
  const res = useResource('/news', { symbol, limit: 8 }, [symbol]);
  const now = useTick(30);
  return html`<section class="panel" aria-labelledby="sn-title">
    <div class="panel-head"><h2 id="sn-title">Nachrichten zu ${symbol}</h2></div>
    ${res.loading ? html`<${Loading} lines=${3} />`
      : res.error ? html`<${ErrorState} error=${res.error} onRetry=${res.reload} />`
        : !res.data?.length ? html`<${Empty} title="Keine Meldungen zu diesem Wert.">Neue Nachrichten, die die KI diesem Wert zuordnet, erscheinen hier.</${Empty}>`
          : html`<div class="news-list news-list-narrow">${res.data.map((n) => html`<${NewsItem} key=${n.id} item=${n} now=${now} />`)}</div>`}
  </section>`;
}

function SymbolSignals({ signals, inst }) {
  const now = useTick(30);
  return html`<section class="panel" aria-labelledby="ss-title">
    <div class="panel-head"><h2 id="ss-title">Signale zu ${inst.symbol}</h2><div class="aside"><a href="#/signale">Alle Signale</a></div></div>
    ${signals.loading ? html`<${Loading} lines=${3} />`
      : signals.error ? html`<${ErrorState} error=${signals.error} onRetry=${signals.reload} />`
        : !signals.data?.length ? html`<${Empty} title="Noch keine Signale zu diesem Wert.">Sobald ein Setup auftritt, erscheint es hier und als Pfeil im Chart.</${Empty}>`
          : html`<div class="table-wrap"><table class="table stack-mobile">
              <thead><tr><th>Zeit</th><th>Richtung</th><th>Setup</th><th class="r">Referenz</th><th>Status</th><th class="r">Konfidenz</th></tr></thead>
              <tbody>${signals.data.map((s) => html`<tr class="clickable" onClick=${() => { window.location.hash = `#/signale/${s.id}`; }}>
                <td data-label="Zeit"><a href=${`#/signale/${s.id}`}>${fmtWhen(s.created_ts, now)}</a></td>
                <td data-label="Richtung"><${DirectionBadge} direction=${s.direction} /></td>
                <td data-label="Setup">${s.setup_label_de || s.setup}${s.actionable ? '' : html` <span class="faint">(verworfen)</span>`}</td>
                <td data-label="Referenz" class="r">${fmtPrice(s.ref_price, inst.asset_class)}</td>
                <td data-label="Status"><${SignalStatus} signal=${s} /></td>
                <td data-label="Konfidenz" class="r">${fmtNum(s.confidence * 100, 0)} %</td>
              </tr>`)}</tbody>
            </table></div>`}
  </section>`;
}

/** Chart markers: arrows at entries, small circles at exits (actionable signals only). */
function signalMarkers(signals) {
  const out = [];
  for (const s of signals) {
    if (!s.actionable) continue;
    const long = s.direction === 'LONG';
    const at = s.entry_ts ?? s.issued_ts ?? s.created_ts;
    out.push({ ts: at, position: long ? 'belowBar' : 'aboveBar', shape: long ? 'arrowUp' : 'arrowDown', color: long ? 'up' : 'down', text: directionLabel(s.direction) });
    if (s.exit_ts) out.push({ ts: s.exit_ts, position: long ? 'aboveBar' : 'belowBar', shape: 'circle', color: 'ink2', text: fmtR(s.outcome_r), size: 0.6 });
  }
  return out;
}

export function InstrumentView({ symbol }) {
  const loaded = useStore(app, (s) => s.instrumentsLoaded);
  const inst = useStore(app, (s) => s.instruments.find((i) => i.symbol === symbol) ?? null);
  const quote = useStore(app, (s) => s.quotes[symbol]);
  const quotes = useStore(app, (s) => s.quotes);
  const [tf, setTf] = useState('15m');
  const signals = useResource('/signals', { symbol, scope: 'all', actionable: 'all', limit: 40 }, [symbol]);
  const analysis = useResource(`/analysis/${encodeURIComponent(symbol)}`, null, [symbol]);
  const me = useResource('/portfolios/me', null, [symbol]);
  useInterval(() => analysis.reload(), 60000);
  useLive('signal', (s) => { if (s.symbol === symbol) signals.set((d) => upsertSignal(d || [], s, 40)); });
  useLive('portfolio', (p) => { if (p?.key === 'me') me.reload(); });

  const liveSignals = useMemo(() => withLiveR(signals.data || [], quotes), [signals.data, quotes]);
  const markers = useMemo(() => signalMarkers(signals.data || []), [signals.data]);
  const priceLines = useMemo(() => {
    const lines = [];
    const open = (signals.data || []).find((s) => s.actionable && OPEN_STATUSES.has(s.status));
    if (open) {
      const { basis, stop, target } = signalLevels(open);
      lines.push({ price: basis, color: 'ink2', style: 'dashed', title: open.entry_price ? 'Einstieg' : 'Referenz' });
      lines.push({ price: stop, color: 'down', style: 'dashed', title: 'Stopp' });
      lines.push({ price: target, color: 'up', style: 'dashed', title: 'Ziel' });
    }
    const pos = (me.data?.positions || []).find((p) => p.symbol === symbol);
    if (pos) {
      lines.push({ price: pos.entry_price, color: 'ink1', style: 'solid', title: 'Mein Einstieg' });
      if (pos.stop_price) lines.push({ price: pos.stop_price, color: 'down', style: 'dotted', title: 'Mein Stopp' });
      if (pos.target_price) lines.push({ price: pos.target_price, color: 'up', style: 'dotted', title: 'Mein Ziel' });
    }
    return lines;
  }, [signals.data, me.data, symbol]);

  if (!loaded) return html`<${Loading} lines=${6} />`;
  if (!inst) {
    return html`<${Empty} title=${`„${symbol}“ ist nicht in der Beobachtungsliste.`}>
      Unter <a href="#/maerkte">Märkte</a> lässt sich ein Wert per Symbol hinzufügen; danach lädt die Plattform seine Kurshistorie.</${Empty}>`;
  }
  const unit = priceUnit(inst.currency, inst.price_scale);
  const position = (me.data?.positions || []).find((p) => p.symbol === symbol);
  return html`<div class="ins">
    <${Header} inst=${inst} quote=${quote} />
    <div class="ins-grid">
      <div class="stack">
        <section class="panel ins-chart" aria-label="Chart">
          <div class="row-between ins-toolbar">
            <${Segmented} options=${TIMEFRAMES} value=${tf} onChange=${setTf} label="Zeitrahmen" />
            ${position && html`<span class="small muted">Meine Position: ${directionLabel(position.side)} ${fmtQty(position.qty, inst.asset_class)} Stück zu ${fmtPrice(position.entry_price, inst.asset_class)} ${unit}</span>`}
          </div>
          <${PriceChartView} symbol=${symbol} instrument=${inst} timeframe=${tf} markers=${markers} priceLines=${priceLines} height=${440} label=${inst.name} />
        </section>
        <div class="grid-2 ins-lower">
          <${ChartAnalysis} symbol=${symbol} />
          <${SymbolNews} symbol=${symbol} />
        </div>
        <${SymbolSignals} signals=${{ ...signals, data: liveSignals }} inst=${inst} />
      </div>
      <aside class="stack ins-side">
        <section class="panel" aria-labelledby="tk-title">
          <div class="panel-head"><h2 id="tk-title">Order für mein Depot</h2></div>
          ${inst.tradable ? html`<${OrderTicket} prefill=${{ symbol }} lockSymbol=${true} idPrefix="ins-ot" />`
            : html`<p class="muted">${inst.name} dient nur als Umrechnungskurs oder Vergleichswert und ist nicht handelbar.</p>`}
        </section>
        <section class="panel" aria-labelledby="ta-title">
          <div class="panel-head"><h2 id="ta-title">Technische Lage</h2></div>
          <${Readings} analysis=${analysis} />
        </section>
        <section class="panel" aria-labelledby="lv-title">
          <div class="panel-head"><h2 id="lv-title">Marken</h2></div>
          <${Levels} analysis=${analysis} assetClass=${inst.asset_class} unit=${unit} />
        </section>
      </aside>
    </div>
  </div>`;
}

