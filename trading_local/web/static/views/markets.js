// #/maerkte - the full watchlist: filters, search, add an instrument, activate/deactivate.

import { html, useMemo, useState } from '../vendor/preact-htm.mjs';
import { api, errorText } from '../lib/api.js';
import { app, loadInstruments } from '../lib/data.js';
import {
  fmtPrice, fmtWhen, isTradableClass, priceUnit,
} from '../lib/format.js';
import {
  ASSET_CLASS_LABELS, REGION_LABELS, assetClassLabel, phaseLabel,
} from '../lib/labels.js';
import { navigate, symbolHref } from '../lib/router.js';
import { useStore, useTick } from '../lib/store.js';
import { serverNow } from '../lib/time.js';
import { Sparkline } from '../components/sparkline.js';
import {
  Change, DirectionBadge, Empty, ErrorState, FlashValue, Loading, PhaseDot, Toast,
} from '../components/ui.js';

const REGION_ORDER = ['ASIA', 'EU', 'US', 'GLOBAL'];
const PHASE_RANK = { open: 0, break: 1, pre: 2, post: 3, closed: 4 };
const SORT_STORAGE_KEY = 'tl-markets-sort';
const RESORT_SECONDS = 10; // live quotes re-order a price/change sort at most this often (rows must not jump)

// Sortable columns. ``first`` is the direction of the first click: text A-Z, numbers and flags
// with the largest/true values first. Values that are unknown (no quote yet) always go last.
const SORT_COLUMNS = {
  phase: { label: 'Handelsstatus', first: 'asc', value: (i) => PHASE_RANK[i.phase] ?? 9 },
  symbol: { label: 'Symbol', first: 'asc', value: (i) => i.symbol, text: true },
  name: { label: 'Name', first: 'asc', value: (i) => i.name, text: true },
  exchange: { label: 'Börse', first: 'asc', value: (i) => i.exchange_name || i.exchange, text: true },
  asset: { label: 'Klasse', first: 'asc', value: (i) => assetClassLabel(i.asset_class), text: true },
  price: { label: 'Kurs', first: 'desc', value: (i, q) => q?.price ?? null },
  change: { label: 'Veränderung', first: 'desc', value: (i, q) => q?.change_pct ?? null },
  signal: { label: 'Signal', first: 'desc', value: (i) => (i.open_signal ? (i.open_signal.direction === 'LONG' ? 2 : 1) : 0) },
  active: { label: 'Aktiv', first: 'desc', value: (i) => (i.active ? 1 : 0) },
  tradable: { label: 'Handelbar', first: 'desc', value: (i) => (i.tradable ? 1 : 0) },
};

function readSort() {
  try {
    const raw = JSON.parse(window.localStorage.getItem(SORT_STORAGE_KEY) || 'null');
    return raw && SORT_COLUMNS[raw.key] && (raw.dir === 'asc' || raw.dir === 'desc') ? raw : null;
  } catch {
    return null;
  }
}

function storeSort(sort) {
  try {
    if (sort) window.localStorage.setItem(SORT_STORAGE_KEY, JSON.stringify(sort));
    else window.localStorage.removeItem(SORT_STORAGE_KEY);
  } catch {
    /* private mode: the choice is simply not remembered */
  }
}

/** Next sort state after a click on ``key``: first direction, then the other one, then back to the default order. */
function nextSort(current, key) {
  const { first } = SORT_COLUMNS[key];
  if (!current || current.key !== key) return { key, dir: first };
  if (current.dir === first) return { key, dir: first === 'asc' ? 'desc' : 'asc' };
  return null;
}

function compareRows(a, b, sort, quotes) {
  const col = SORT_COLUMNS[sort.key];
  const va = col.value(a, quotes[a.symbol]);
  const vb = col.value(b, quotes[b.symbol]);
  const missingA = va === null || va === undefined || (typeof va === 'number' && !Number.isFinite(va));
  const missingB = vb === null || vb === undefined || (typeof vb === 'number' && !Number.isFinite(vb));
  if (missingA || missingB) return missingA === missingB ? a.symbol.localeCompare(b.symbol) : missingA ? 1 : -1;
  const order = col.text ? String(va).localeCompare(String(vb), 'de', { sensitivity: 'base' }) : va - vb;
  return (sort.dir === 'asc' ? order : -order) || a.symbol.localeCompare(b.symbol);
}

function SortHeader({ id, sort, onSort, className, children }) {
  const active = sort?.key === id;
  const ariaSort = active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none';
  const next = nextSort(sort, id);
  const hint = next ? `nach ${SORT_COLUMNS[id].label} ${next.dir === 'asc' ? 'aufsteigend' : 'absteigend'} sortieren` : 'Standardreihenfolge wiederherstellen';
  return html`<th scope="col" class=${className} aria-sort=${ariaSort}>
    <button type="button" class=${`th-sort ${active ? 'on' : ''}`} onClick=${() => onSort(id)} title=${hint} aria-label=${`${SORT_COLUMNS[id].label}: ${hint}`}>
      ${children}<span class="sort-ind" aria-hidden="true">${active ? (sort.dir === 'asc' ? '▲' : '▼') : '↕'}</span>
    </button>
  </th>`;
}

function AddInstrument({ notify }) {
  const [symbol, setSymbol] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const submit = async (e) => {
    e.preventDefault();
    const value = symbol.trim().toUpperCase();
    if (!value) {
      setError('Bitte ein Symbol eingeben, z. B. ASML.AS oder 7203.T.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const inst = await api.post('/instruments', { symbol: value });
      setSymbol('');
      notify(`${inst.name} (${inst.symbol}) hinzugefügt – die Kurshistorie wird im Hintergrund geladen.`);
      loadInstruments();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };
  return html`<form class="add-inst" onSubmit=${submit} noValidate>
    <div class="field">
      <label for="add-symbol">Wert hinzufügen</label>
      <div class="row">
        <input id="add-symbol" class="input" value=${symbol} placeholder="z. B. ASML.AS"
          autocomplete="off" spellcheck="false" aria-invalid=${error ? 'true' : 'false'} aria-describedby="add-symbol-msg"
          onInput=${(e) => setSymbol(e.currentTarget.value)} />
        <button type="submit" class="btn btn-primary" disabled=${busy}>${busy ? 'Wird geprüft …' : 'Hinzufügen'}</button>
      </div>
      <span id="add-symbol-msg" class=${error ? 'field-error' : 'field-help'} role=${error ? 'alert' : undefined}>
        ${error || 'Symbol in der Schreibweise von Yahoo Finance (z. B. ASML.AS). Es wird bei der Datenquelle geprüft; Börse, Währung und Handelszeiten werden übernommen. Die Kryptopaare in EUR kommen von Kraken.'}
      </span>
    </div>
  </form>`;
}

export function MarketsView() {
  const instruments = useStore(app, (s) => s.instruments);
  const loaded = useStore(app, (s) => s.instrumentsLoaded);
  const error = useStore(app, (s) => s.instrumentsError);
  const quotes = useStore(app, (s) => s.quotes);
  const sparks = useStore(app, (s) => s.sparklines);
  const [region, setRegion] = useState('');
  const [asset, setAsset] = useState('');
  const [query, setQuery] = useState('');
  const [showInactive, setShowInactive] = useState(true);
  const [toast, setToast] = useState(null);
  const [pending, setPending] = useState(null);
  const [sort, setSortState] = useState(readSort);
  const now = useTick(RESORT_SECONDS);
  const resortEpoch = Math.floor(now / RESORT_SECONDS);
  const setSort = (value) => {
    setSortState(value);
    storeSort(value);
  };
  const onSort = (key) => setSort(nextSort(sort, key));

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = instruments
      .filter((i) => (!region || i.region === region) && (!asset || i.asset_class === asset) && (showInactive || i.active))
      .filter((i) => !q || i.symbol.toLowerCase().includes(q) || i.name.toLowerCase().includes(q) || (i.exchange_name || '').toLowerCase().includes(q));
    if (sort) return filtered.sort((a, b) => compareRows(a, b, sort, quotes));
    return filtered.sort((a, b) => REGION_ORDER.indexOf(a.region) - REGION_ORDER.indexOf(b.region) || a.sort_order - b.sort_order || a.symbol.localeCompare(b.symbol));
    // quotes are read through resortEpoch on purpose: a live tick must not re-order the rows under the cursor
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [instruments, region, asset, query, showInactive, sort, resortEpoch]);

  const patch = async (inst, body, done) => {
    setPending(`${inst.symbol}:${Object.keys(body)[0]}`);
    try {
      await api.patch(`/instruments/${encodeURIComponent(inst.symbol)}`, body);
      setToast(done);
      await loadInstruments();
    } catch (err) {
      setToast(errorText(err));
    } finally {
      setPending(null);
    }
  };

  const regionCounts = REGION_ORDER.map((r) => ({ r, n: instruments.filter((i) => i.region === r && i.active).length }));
  return html`<div>
    <div class="page-head"><div><h1>Märkte</h1>
      <p class="lede">Die Beobachtungsliste: Für aktive Werte werden Minutenkerzen gespeichert, Nachrichten zugeordnet und Signale gesucht.</p></div>
      <p class="small muted">${regionCounts.filter((x) => x.n).map((x) => `${REGION_LABELS[x.r]} ${x.n}`).join(' · ')}</p>
    </div>
    <div class="markets-grid">
      <section class="panel" aria-label="Beobachtungsliste">
        <div class="filters">
          <label class="field-inline"><span class="field-label">Suche</span>
            <input class="input" type="search" value=${query} placeholder="Symbol, Name oder Börse" onInput=${(e) => setQuery(e.currentTarget.value)} /></label>
          <label class="field-inline"><span class="field-label">Region</span>
            <select class="select" value=${region} onChange=${(e) => setRegion(e.currentTarget.value)}>
              <option value="">alle</option>${REGION_ORDER.map((r) => html`<option value=${r}>${REGION_LABELS[r]}</option>`)}
            </select></label>
          <label class="field-inline"><span class="field-label">Anlageklasse</span>
            <select class="select" value=${asset} onChange=${(e) => setAsset(e.currentTarget.value)}>
              <option value="">alle</option>${Object.entries(ASSET_CLASS_LABELS).map(([k, v]) => html`<option value=${k}>${v}</option>`)}
            </select></label>
          <label class="field-inline sort-select"><span class="field-label">Sortierung</span>
            <select class="select" value=${sort ? `${sort.key}:${sort.dir}` : ''}
              onChange=${(e) => { const [key, dir] = e.currentTarget.value.split(':'); setSort(key ? { key, dir } : null); }}>
              <option value="">Region (Standard)</option>
              ${Object.entries(SORT_COLUMNS).flatMap(([key, col]) => {
                const dirs = col.first === 'asc' ? ['asc', 'desc'] : ['desc', 'asc'];
                return dirs.map((dir) => html`<option value=${`${key}:${dir}`}>${col.label} ${dir === 'asc' ? (col.text ? 'A–Z' : 'aufsteigend') : (col.text ? 'Z–A' : 'absteigend')}</option>`);
              })}
            </select></label>
          <label class="check small"><input type="checkbox" checked=${showInactive} onChange=${(e) => setShowInactive(e.currentTarget.checked)} />inaktive zeigen</label>
        </div>
        ${!loaded ? html`<${Loading} lines=${8} />`
          : error && !instruments.length ? html`<${ErrorState} error=${error} onRetry=${loadInstruments} />`
            : !rows.length ? html`<${Empty} title="Kein Wert passt zu diesen Filtern.">Filter zurücksetzen oder rechts einen neuen Wert hinzufügen.</${Empty}>`
              : html`<div class="table-wrap"><table class="table stack-mobile markets-table">
                <thead><tr>
                  <${SortHeader} id="phase" sort=${sort} onSort=${onSort}><span class="sr-only">Handelsstatus</span></${SortHeader}>
                  <${SortHeader} id="symbol" sort=${sort} onSort=${onSort}>Symbol</${SortHeader}>
                  <${SortHeader} id="name" sort=${sort} onSort=${onSort}>Name</${SortHeader}>
                  <${SortHeader} id="exchange" sort=${sort} onSort=${onSort}>Börse</${SortHeader}>
                  <${SortHeader} id="asset" sort=${sort} onSort=${onSort}>Klasse</${SortHeader}>
                  <${SortHeader} id="price" sort=${sort} onSort=${onSort} className="r">Kurs</${SortHeader}>
                  <${SortHeader} id="change" sort=${sort} onSort=${onSort} className="r">Veränd.</${SortHeader}>
                  <th scope="col">Heute</th>
                  <${SortHeader} id="signal" sort=${sort} onSort=${onSort}>Signal</${SortHeader}>
                  <${SortHeader} id="active" sort=${sort} onSort=${onSort}>Aktiv</${SortHeader}>
                  <${SortHeader} id="tradable" sort=${sort} onSort=${onSort}>Handelbar</${SortHeader}>
                </tr></thead>
                <tbody>${rows.map((i, idx) => {
                  const q = quotes[i.symbol];
                  const groupHead = !sort && !region && (idx === 0 || rows[idx - 1].region !== i.region);
                  return html`${groupHead && html`<tr class="group-row" key=${`g-${i.region}`}><th colspan="11" scope="colgroup">${REGION_LABELS[i.region]}</th></tr>`}
                  <tr key=${i.symbol} class=${`clickable ${i.active ? '' : 'inactive'}`} onClick=${(e) => { if (!e.target.closest('input,button,a,label')) navigate(`/markt/${encodeURIComponent(i.symbol)}`); }}>
                    <td data-label="Status"><${PhaseDot} phase=${i.phase} /></td>
                    <td data-label="Symbol"><a class="sym" href=${symbolHref(i.symbol)}>${i.symbol}</a></td>
                    <td data-label="Name" class="name">${i.name}</td>
                    <td data-label="Börse" class="nowrap">${i.exchange_name || i.exchange}<div class="tiny faint">${phaseLabel(i.phase)}${i.delay_min ? ` · ${i.delay_min} Min verzögert` : ''}</div></td>
                    <td data-label="Klasse">${assetClassLabel(i.asset_class)}</td>
                    <td data-label="Kurs" class="r nowrap"><${FlashValue} value=${q?.price}>${q ? fmtPrice(q.price, i.asset_class) : '–'}</${FlashValue}>
                      <div class="tiny faint">${priceUnit(i.currency, i.price_scale)}${q ? ` · ${fmtWhen(q.ts, serverNow())}` : ''}</div></td>
                    <td data-label="Veränd." class="r"><${Change} value=${q?.change_pct} digits=${2} /></td>
                    <td data-label="Heute"><${Sparkline} points=${sparks[i.symbol]} prevClose=${q?.prev_close} width=${84} height=${24} /></td>
                    <td data-label="Signal">${i.open_signal ? html`<a href=${`#/signale/${i.open_signal.id}`} class="nowrap"><${DirectionBadge} direction=${i.open_signal.direction} /></a>` : html`<span class="faint">–</span>`}</td>
                    <td data-label="Aktiv"><label class="switch"><input type="checkbox" checked=${i.active} disabled=${pending === `${i.symbol}:active`}
                      aria-label=${`${i.symbol} beobachten`}
                      onChange=${(e) => patch(i, { active: e.currentTarget.checked }, `${i.symbol} wird ${e.currentTarget.checked ? 'wieder beobachtet' : 'nicht mehr beobachtet'}.`)} /></label></td>
                    <td data-label="Handelbar"><label class="switch" title=${!isTradableClass(i.asset_class) && !i.tradable ? 'Umrechnungskurs oder Vergleichswert – im Papierdepot nicht handelbar' : undefined}>
                      <input type="checkbox" checked=${i.tradable}
                      disabled=${pending === `${i.symbol}:tradable` || (!isTradableClass(i.asset_class) && !i.tradable)}
                      aria-label=${`${i.symbol} handelbar`}
                      onChange=${(e) => patch(i, { tradable: e.currentTarget.checked }, `${i.symbol} ist ${e.currentTarget.checked ? 'jetzt handelbar' : 'nicht mehr handelbar'}.`)} /></label></td>
                  </tr>`;
                })}</tbody>
              </table></div>`}
      </section>
      <aside class="stack">
        <section class="panel"><${AddInstrument} notify=${setToast} /></section>
        <section class="panel small muted">
          <h2 class="side-h">Hinweise</h2>
          <p>Aktien, Indizes, Devisen und Futures kommen von Yahoo Finance und sind teilweise verzögert – die Verzögerung steht bei jedem Wert. Die Kryptopaare in EUR kommen in Echtzeit von Kraken.</p>
          <p>Londoner Aktien notieren in Pence (GBp); die Plattform rechnet sie nicht stillschweigend um.</p>
          <p>„Handelbar“ aus: Der Wert bleibt sichtbar, aber weder die KI noch Sie können ihn im Papierdepot handeln. Devisenpaare und Futures dienen nur als Umrechnungskurs oder Vergleichswert und lassen sich nicht handelbar schalten.</p>
        </section>
      </aside>
    </div>
    <${Toast} message=${toast} onDone=${() => setToast(null)} />
  </div>`;
}
