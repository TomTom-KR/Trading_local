// #/news - all news with filters (relevance, symbol, analysed only, region of the source) and paging.
// Items whose analysis is still running are replaced live by 'news' events.

import { html, useEffect, useMemo, useRef, useState } from '../vendor/preact-htm.mjs';
import { api, errorText } from '../lib/api.js';
import { app } from '../lib/data.js';
import { fmtNum } from '../lib/format.js';
import { NEWS_REGION_LABELS } from '../lib/labels.js';
import { useLive } from '../lib/live.js';
import { useResource } from '../lib/resource.js';
import { useStore, useTick } from '../lib/store.js';
import { NewsItem } from '../components/news-item.js';
import { Empty, ErrorState, Loading } from '../components/ui.js';

const PAGE = 30;
const MAX_ITEMS = 300; // upper bound of the list while live items keep arriving

export function NewsView() {
  const instruments = useStore(app, (s) => s.instruments);
  const feeds = useResource('/system/feeds', null, []);
  const [minRel, setMinRel] = useState('');
  const [symbol, setSymbol] = useState('');
  const [analyzedOnly, setAnalyzedOnly] = useState(false);
  const [region, setRegion] = useState('');
  const [state, setState] = useState({ items: [], loading: true, error: null, done: false });
  const now = useTick(30);
  const req = useRef(0);

  const fetchPage = async (beforeId) => {
    const id = ++req.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const items = await api.get('/news', {
        limit: PAGE, before_id: beforeId ?? null, symbol: symbol || null, min_relevance: minRel || null,
        analyzed_only: analyzedOnly ? 'true' : null,
      });
      if (id !== req.current) return;
      setState((s) => ({ items: beforeId ? [...s.items, ...items] : items, loading: false, error: null, done: items.length < PAGE }));
    } catch (err) {
      if (id === req.current) setState((s) => ({ ...s, loading: false, error: errorText(err) }));
    }
  };
  useEffect(() => { fetchPage(null); }, [minRel, symbol, analyzedOnly]);

  useLive('news', (n) => {
    setState((s) => {
      const known = s.items.some((x) => x.id === n.id);
      if (known) return { ...s, items: s.items.map((x) => (x.id === n.id ? n : x)) };
      const rel = n.analysis?.relevance ?? 0;
      if ((minRel && rel < Number(minRel)) || (analyzedOnly && !n.analysis)) return s;
      if (symbol && !(n.analysis?.symbols || []).some((x) => x.symbol === symbol) && !n.hinted_symbols.includes(symbol)) return s;
      // live items are capped; whatever falls off the end can be paged in again ('Ältere Meldungen laden')
      const items = [n, ...s.items];
      return items.length > MAX_ITEMS ? { ...s, items: items.slice(0, MAX_ITEMS), done: false } : { ...s, items };
    });
  });

  const regionOf = useMemo(() => Object.fromEntries((feeds.data || []).map((f) => [f.feed_id, f.region])), [feeds.data]);
  const regions = useMemo(() => [...new Set((feeds.data || []).map((f) => f.region))].sort(), [feeds.data]);
  const visible = region ? state.items.filter((n) => regionOf[n.source] === region) : state.items;
  const pendingCount = visible.filter((n) => !n.analysis && (n.status === 'new' || n.status === 'queued')).length;

  return html`<div>
    <div class="page-head"><div><h1>Nachrichten</h1>
      <p class="lede">Alle gesammelten Meldungen. Die lokale KI bewertet Relevanz und Stimmung für die Werte der Beobachtungsliste und fasst sie auf Deutsch zusammen.</p></div>
      ${pendingCount > 0 && html`<p class="small muted">${fmtNum(pendingCount, 0)} Analysen laufen</p>`}
    </div>
    <section class="panel" aria-label="Nachrichtenliste">
      <div class="filters">
        <label class="field-inline"><span class="field-label">Relevanz ab</span>
          <select class="select" value=${minRel} onChange=${(e) => setMinRel(e.currentTarget.value)}>
            <option value="">alle</option>${[1, 2, 3, 4, 5].map((n) => html`<option value=${n}>${n} von 5</option>`)}
          </select></label>
        <label class="field-inline"><span class="field-label">Symbol</span>
          <select class="select" value=${symbol} onChange=${(e) => setSymbol(e.currentTarget.value)}>
            <option value="">alle</option>
            ${[...instruments].sort((a, b) => a.symbol.localeCompare(b.symbol)).map((i) => html`<option value=${i.symbol}>${i.symbol} – ${i.name}</option>`)}
          </select></label>
        <label class="field-inline"><span class="field-label">Region der Quelle</span>
          <select class="select" value=${region} onChange=${(e) => setRegion(e.currentTarget.value)} disabled=${!regions.length}>
            <option value="">alle</option>${regions.map((r) => html`<option value=${r}>${NEWS_REGION_LABELS[r] || r}</option>`)}
          </select></label>
        <label class="check small"><input type="checkbox" checked=${analyzedOnly} onChange=${(e) => setAnalyzedOnly(e.currentTarget.checked)} />nur analysierte</label>
      </div>
      ${state.error && !state.items.length ? html`<${ErrorState} error=${state.error} onRetry=${() => fetchPage(null)} />`
        : state.loading && !state.items.length ? html`<${Loading} lines=${6} />`
          : !visible.length ? html`<${Empty} title="Keine Meldungen für diese Auswahl.">
              Die Feeds werden jede Minute abgefragt. Filter lockern oder etwas später wiederkommen.</${Empty}>`
            : html`<div class="news-list news-list-full">${visible.map((n) => html`<${NewsItem} key=${n.id} item=${n} now=${now} full=${true} />`)}</div>`}
      ${state.error && state.items.length > 0 && html`<${ErrorState} error=${state.error} title="Weitere Meldungen konnten nicht geladen werden." />`}
      <div class="list-more">
        ${!state.done && state.items.length > 0 && html`<button type="button" class="btn" disabled=${state.loading}
          onClick=${() => fetchPage(state.items[state.items.length - 1].id)}>${state.loading ? 'Wird geladen …' : 'Ältere Meldungen laden'}</button>`}
        ${state.done && state.items.length > 0 && html`<span class="small faint">Keine älteren Meldungen.</span>`}
      </div>
    </section>
  </div>`;
}
