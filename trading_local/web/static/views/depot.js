// #/depot - KI gegen mich: side-by-side summary, equity chart, positions / orders / trades of both
// portfolios, the order ticket for 'Mein Depot' and the reset of both portfolios.

import { html, useEffect, useMemo, useState } from '../vendor/preact-htm.mjs';
import { api, errorText } from '../lib/api.js';
import { app, instrumentOf, loadPortfolios } from '../lib/data.js';
import {
  DASH, fmtDate, fmtMoney, fmtMoneySigned, fmtNum, fmtPct, fmtPrice, fmtQty, fmtShare, fmtWhen, isNum, parseDecimal,
  toInput,
} from '../lib/format.js';
import { exitReasonLabel, orderOriginLabel, orderStatusLabel } from '../lib/labels.js';
import { useLive } from '../lib/live.js';
import { symbolHref } from '../lib/router.js';
import {
  useInterval, useSoon, useStore, useTick,
} from '../lib/store.js';
import { serverNow } from '../lib/time.js';
import { LineChartView } from '../components/line-chart.js';
import { OrderTicket } from '../components/order-ticket.js';
import { Dialog } from '../components/overlay.js';
import {
  Change, DirectionBadge, Empty, ErrorState, Loading, Segmented, Tabs, Toast,
} from '../components/ui.js';

const RANGES = [
  { value: '1d', label: '1T', seconds: 86400 }, { value: '1w', label: '1W', seconds: 7 * 86400 },
  { value: '1m', label: '1M', seconds: 30 * 86400 }, { value: 'all', label: 'Alles', seconds: null },
];
const KEYS = ['ai', 'me'];
const NAMES = { ai: 'KI-Depot', me: 'Mein Depot' };

function SummaryTable({ ai, me }) {
  const cur = ai.base_currency;
  const money = (v) => fmtMoney(v, cur, 0);
  const rows = [
    ['Rendite seit Start', (p) => html`<${Change} value=${p.total_return_pct} digits=${2} />`],
    ['Heute', (p) => html`<${Change} value=${p.day_return_pct} digits=${2} /> <span class="muted">${fmtMoneySigned(p.day_pnl, cur, 0)}</span>`],
    ['Realisiert', (p) => fmtMoneySigned(p.realized_pnl, cur, 0)],
    ['Unrealisiert', (p) => fmtMoneySigned(p.unrealized_pnl, cur, 0)],
    ['Gebühren', (p) => money(p.fees_paid)],
    ['Trefferquote', (p) => (p.closed_trades ? `${fmtShare(p.win_rate)} (${p.wins} von ${p.closed_trades})` : 'noch keine Trades')],
    ['Profitfaktor', (p) => (isNum(p.profit_factor) ? fmtNum(p.profit_factor, 2) : DASH)],
    ['Max. Drawdown', (p) => (isNum(p.max_drawdown_pct) ? fmtPct(-Math.abs(p.max_drawdown_pct), 1) : DASH)],
    ['Sharpe', (p) => (isNum(p.sharpe) ? fmtNum(p.sharpe, 2) : html`<span title="zu wenig Historie" aria-label="zu wenig Historie">${DASH}</span>`)],
    ['Offene Positionen', (p) => `${p.open_positions}${p.pending_orders ? ` · ${p.pending_orders} Order offen` : ''}`],
    ['Exposure long / short', (p) => `${money(p.long_exposure)} / ${money(p.short_exposure)}`],
    ['Kaufkraft', (p) => money(p.buying_power)],
    ['Cash', (p) => money(p.cash)],
  ];
  return html`<section class="panel vsum" aria-label="Vergleich KI-Depot und Mein Depot">
    <table class="vsum-table">
      <thead><tr><th scope="col"><span class="sr-only">Kennzahl</span></th>
        ${[ai, me].map((p) => html`<th scope="col">
          <span class="vs-name"><span class=${`swatch swatch-${p.key}`} aria-hidden="true"></span>${NAMES[p.key]}</span>
          <span class="figure figure-28 vsum-equity">${money(p.equity)}</span>
          <span class="small muted">Start ${money(p.start_cash)} am ${fmtDate(p.reset_ts)}</span>
        </th>`)}</tr></thead>
      <tbody>${rows.map(([label, fn]) => html`<tr><th scope="row">${label}</th><td>${fn(ai)}</td><td>${fn(me)}</td></tr>`)}</tbody>
    </table>
  </section>`;
}

function EquityPanel({ instruments }) {
  const [range, setRange] = useState('all');
  const [state, setState] = useState({ data: null, error: null });
  const load = () => {
    const r = RANGES.find((x) => x.value === range);
    const since = r.seconds ? Math.floor(serverNow() - r.seconds) : null;
    api.get('/portfolios/equity', { since_ts: since, max_points: 600 })
      .then((data) => setState({ data, error: null }))
      .catch((err) => setState((s) => ({ ...s, error: errorText(err) })));
  };
  useEffect(load, [range]);
  useInterval(load, 60000);
  useLive('portfolio', useSoon(load));
  const d = state.data;
  const series = useMemo(() => {
    if (!d) return [];
    const out = [{ id: 'ai', points: d.ai, color: 'ai' }, { id: 'me', points: d.me, color: 'me' }];
    for (const [k, pts] of Object.entries(d.benchmarks || {})) out.push({ id: k, points: pts, color: 'ink3', style: 'dashed', width: 1 });
    return out;
  }, [d]);
  const benchNames = Object.keys(d?.benchmarks || {}).map((k) => instruments.find((i) => i.symbol === k)?.name || k);
  const empty = d && d.ai.length < 2 && d.me.length < 2;
  return html`<section class="panel" aria-labelledby="eq-title">
    <div class="panel-head"><h2 id="eq-title">Depotwert im Verlauf</h2>
      <div class="aside"><${Segmented} options=${RANGES} value=${range} onChange=${setRange} label="Zeitraum" /></div></div>
    ${state.error && !d ? html`<${ErrorState} error=${state.error} onRetry=${load} />`
      : !d ? html`<${Loading} lines=${5} />`
        : empty ? html`<${Empty} title="Noch kein Verlauf.">Der Depotwert wird jede Minute festgehalten, solange ein Markt offen ist – die Kurve wächst ab dem nächsten Handelstag.</${Empty}>`
          : html`<${LineChartView} series=${series} height=${300} format=${(v) => fmtMoney(v, d.base_currency, 0)}
              label="Depotwert von KI-Depot und Mein Depot im Verlauf, Vergleichsindizes gestrichelt" />`}
    <div class="legend">
      <span><i class="swatch swatch-ai"></i>KI-Depot</span><span><i class="swatch swatch-me"></i>Mein Depot</span>
      ${benchNames.map((n) => html`<span><i class="swatch swatch-bench"></i>${n} (gekauft und gehalten)</span>`)}
    </div>
  </section>`;
}

function LevelsForm({ pos, onDone }) {
  const inst = instrumentOf(pos.symbol);
  const [stop, setStop] = useState(toInput(pos.stop_price));
  const [target, setTarget] = useState(toInput(pos.target_price));
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const save = async (e) => {
    e.preventDefault();
    const s = stop.trim() ? parseDecimal(stop) : null;
    const t = target.trim() ? parseDecimal(target) : null;
    if ((s !== null && !(s > 0)) || (t !== null && !(t > 0))) {
      setError('Bitte gültige Kurse eingeben oder das Feld leeren, um die Marke zu entfernen.');
      return;
    }
    setBusy(true);
    try {
      await api.patch(`/portfolios/me/positions/${pos.id}`, { stop_price: s, target_price: t });
      onDone(true);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };
  return html`<form class="levels-form" onSubmit=${save}>
    <label class="field-inline"><span class="field-label">Stopp</span>
      <input class="input input-num" inputmode="decimal" value=${stop} onInput=${(e) => setStop(e.currentTarget.value)} data-autofocus /></label>
    <label class="field-inline"><span class="field-label">Ziel</span>
      <input class="input input-num" inputmode="decimal" value=${target} onInput=${(e) => setTarget(e.currentTarget.value)} /></label>
    <span class="small muted">Kurs ${fmtPrice(pos.last_price, inst?.asset_class)} · leeres Feld entfernt die Marke</span>
    <button type="submit" class="btn btn-sm btn-primary" disabled=${busy}>Speichern</button>
    <button type="button" class="btn btn-sm" onClick=${() => onDone(false)}>Abbrechen</button>
    ${error && html`<span class="field-error" role="alert">${error}</span>`}
  </form>`;
}

function PositionsTable({ portfolioKey, positions, onChanged, notify }) {
  const [editing, setEditing] = useState(null);
  const [busy, setBusy] = useState(null);
  const cur = useStore(app, (s) => s.status?.base_currency) || 'EUR';
  const mine = portfolioKey === 'me';
  const close = async (pos) => {
    setBusy(pos.id);
    try {
      await api.post(`/portfolios/me/positions/${pos.id}/close`);
      notify(`Schließorder für ${pos.symbol} angenommen – Ausführung zum nächsten verfügbaren Kurs.`);
      onChanged();
    } catch (err) {
      notify(errorText(err));
    } finally {
      setBusy(null);
    }
  };
  if (!positions.length) {
    return html`<p class="small muted table-empty">${mine ? 'Keine offenen Positionen. Das Orderticket rechts eröffnet eine.' : 'Die KI hält gerade keine Position.'}</p>`;
  }
  return html`<div class="table-wrap"><table class="table stack-mobile">
    <thead><tr><th>Wert</th><th>Richtung</th><th class="r">Stück</th><th class="r">Einstieg</th><th class="r">Kurs</th>
      <th class="r">Stopp</th><th class="r">Ziel</th><th class="r">Marktwert</th><th class="r">G/V</th><th>seit</th>${mine && html`<th><span class="sr-only">Aktionen</span></th>`}</tr></thead>
    <tbody>${positions.map((p) => {
      const ac = instrumentOf(p.symbol)?.asset_class;
      return html`<tr key=${p.id}>
        <td data-label="Wert"><a class="sym" href=${symbolHref(p.symbol)}>${p.symbol}</a> <span class="name">${p.name}</span></td>
        <td data-label="Richtung"><${DirectionBadge} direction=${p.side} /></td>
        <td data-label="Stück" class="r">${fmtQty(p.qty, ac)}</td>
        <td data-label="Einstieg" class="r">${fmtPrice(p.entry_price, ac)}</td>
        <td data-label="Kurs" class="r">${fmtPrice(p.last_price, ac)}</td>
        <td data-label="Stopp" class="r">${fmtPrice(p.stop_price, ac)}</td>
        <td data-label="Ziel" class="r">${fmtPrice(p.target_price, ac)}</td>
        <td data-label="Marktwert" class="r">${fmtMoney(p.exposure_base, cur, 0)}</td>
        <td data-label="G/V" class="r nowrap"><${Change} value=${p.unrealized_pnl} suffix=${(v) => fmtMoneySigned(v, cur, 0)} />
          <span class="pnl-pct"><${Change} value=${p.unrealized_pct} digits=${1} /></span></td>
        <td data-label="seit" class="nowrap">${fmtWhen(p.opened_ts, serverNow())}${p.signal_id ? html` · <a href=${`#/signale/${p.signal_id}`}>Signal</a>` : ''}</td>
        ${mine && html`<td data-label="Aktionen"><div class="row-actions">
          <button type="button" class="btn btn-sm" onClick=${() => setEditing(editing === p.id ? null : p.id)} aria-expanded=${editing === p.id ? 'true' : 'false'}>Stopp/Ziel ändern</button>
          <button type="button" class="btn btn-sm" disabled=${busy === p.id} onClick=${() => close(p)}>Schließen</button>
        </div></td>`}
      </tr>
      ${editing === p.id && html`<tr class="edit-row"><td colspan="11"><${LevelsForm} pos=${p} onDone=${(saved) => {
        setEditing(null);
        if (saved) { notify(`Stopp und Ziel für ${p.symbol} gespeichert.`); onChanged(); }
      }} /></td></tr>`}`;
    })}</tbody>
  </table></div>`;
}

function OrdersTable({ portfolioKey, orders, onChanged, notify }) {
  const mine = portfolioKey === 'me';
  const cur = useStore(app, (s) => s.status?.base_currency) || 'EUR';
  const cancel = async (o) => {
    try {
      await api.del(`/portfolios/me/orders/${o.id}`);
      notify(`Order #${o.id} storniert.`);
      onChanged();
    } catch (err) {
      notify(errorText(err));
    }
  };
  const sorted = [...orders].sort((a, b) => (b.status === 'pending') - (a.status === 'pending') || b.created_ts - a.created_ts);
  if (!sorted.length) return html`<p class="small muted table-empty">Noch keine Orders.</p>`;
  return html`<div class="table-wrap"><table class="table stack-mobile">
    <thead><tr><th>Zeit</th><th>Wert</th><th>Art</th><th>Richtung</th><th class="r">Menge</th><th>Herkunft</th><th>Status</th><th class="r">Ausführung</th>${mine && html`<th><span class="sr-only">Aktionen</span></th>`}</tr></thead>
    <tbody>${sorted.map((o) => {
      const ac = instrumentOf(o.symbol)?.asset_class;
      return html`<tr key=${o.id}>
        <td data-label="Zeit" class="nowrap">${fmtWhen(o.created_ts, serverNow())}</td>
        <td data-label="Wert"><a class="sym" href=${symbolHref(o.symbol)}>${o.symbol}</a></td>
        <td data-label="Art">${o.intent === 'open' ? 'Eröffnung' : 'Schließung'}</td>
        <td data-label="Richtung"><${DirectionBadge} direction=${o.side} /></td>
        <td data-label="Menge" class="r">${isNum(o.qty) ? `${fmtQty(o.qty, ac)} Stück` : isNum(o.notional_base) ? fmtMoney(o.notional_base, cur, 0) : DASH}</td>
        <td data-label="Herkunft">${orderOriginLabel(o.origin)}</td>
        <td data-label="Status"><span class=${`chip ${o.status === 'pending' ? 'chip-solid' : ''}`} title=${o.status_reason}>${orderStatusLabel(o.status)}</span>
          ${o.status_reason && html`<div class="tiny faint">${o.status_reason}</div>`}</td>
        <td data-label="Ausführung" class="r nowrap">${isNum(o.fill_price) ? `${fmtPrice(o.fill_price, ac)} · ${fmtWhen(o.filled_ts, serverNow())}` : DASH}</td>
        ${mine && html`<td>${o.status === 'pending' && html`<button type="button" class="btn btn-sm" onClick=${() => cancel(o)}>Stornieren</button>`}</td>`}
      </tr>`;
    })}</tbody>
  </table></div>`;
}

function TradesTable({ trades }) {
  const cur = useStore(app, (s) => s.status?.base_currency) || 'EUR';
  if (!trades.length) return html`<p class="small muted table-empty">Noch keine abgeschlossenen Trades.</p>`;
  return html`<div class="table-wrap"><table class="table stack-mobile">
    <thead><tr><th>Geschlossen</th><th>Wert</th><th>Richtung</th><th class="r">Stück</th><th class="r">Einstieg</th><th class="r">Ausstieg</th><th>Grund</th><th class="r">Gebühren</th><th class="r">G/V</th></tr></thead>
    <tbody>${trades.map((t) => {
      const ac = instrumentOf(t.symbol)?.asset_class;
      return html`<tr key=${t.id}>
        <td data-label="Geschlossen" class="nowrap">${fmtWhen(t.closed_ts, serverNow())}</td>
        <td data-label="Wert"><a class="sym" href=${symbolHref(t.symbol)}>${t.symbol}</a> <span class="name">${t.name}</span></td>
        <td data-label="Richtung"><${DirectionBadge} direction=${t.side} /></td>
        <td data-label="Stück" class="r">${fmtQty(t.qty, ac)}</td>
        <td data-label="Einstieg" class="r">${fmtPrice(t.entry_price, ac)}</td>
        <td data-label="Ausstieg" class="r">${fmtPrice(t.exit_price, ac)}</td>
        <td data-label="Grund">${exitReasonLabel(t.exit_reason)}</td>
        <td data-label="Gebühren" class="r">${fmtMoney(t.fees_base, cur, 2)}</td>
        <td data-label="G/V" class="r nowrap"><${Change} value=${t.realized_pnl} suffix=${(v) => fmtMoneySigned(v, cur, 2)} />
          <span class="pnl-pct"><${Change} value=${t.realized_pct} digits=${1} /></span></td>
      </tr>`;
    })}</tbody>
  </table></div>`;
}

function Books({ notify }) {
  const [tab, setTab] = useState('positions');
  const [data, setData] = useState({ ai: null, me: null, error: null });
  const load = async () => {
    try {
      const out = {};
      await Promise.all(KEYS.map(async (k) => {
        const [detail, orders, trades] = await Promise.all([
          api.get(`/portfolios/${k}`), api.get(`/portfolios/${k}/orders`, { limit: 100 }), api.get(`/portfolios/${k}/trades`, { limit: 100 }),
        ]);
        out[k] = { detail, orders, trades };
      }));
      setData({ ...out, error: null });
    } catch (err) {
      setData((d) => ({ ...d, error: errorText(err) }));
    }
  };
  const loadSoon = useSoon(() => { load(); loadPortfolios(); }, 400);
  useEffect(() => { load(); }, []);
  useInterval(load, 30000);
  useLive('portfolio', loadSoon);
  useLive('order', loadSoon);

  if (data.error && !data.ai) return html`<section class="panel"><${ErrorState} error=${data.error} onRetry=${load} /></section>`;
  if (!data.ai || !data.me) return html`<section class="panel"><${Loading} lines=${6} /></section>`;
  const count = (k, what) => (what === 'positions' ? data[k].detail.positions.length
    : what === 'orders' ? data[k].orders.filter((o) => o.status === 'pending').length : data[k].trades.length);
  const tabs = [
    { value: 'positions', label: 'Positionen', count: count('ai', 'positions') + count('me', 'positions') },
    { value: 'orders', label: 'Orders', count: count('ai', 'orders') + count('me', 'orders') },
    { value: 'trades', label: 'Trades', count: count('ai', 'trades') + count('me', 'trades') },
  ];
  return html`<section class="panel" aria-label="Positionen, Orders und Trades">
    <${Tabs} tabs=${tabs} value=${tab} onChange=${setTab} label="Depotbuch" />
    ${KEYS.map((k) => html`<div class="book">
      <h3 class="book-title"><span class=${`swatch swatch-${k}`} aria-hidden="true"></span>${NAMES[k]}</h3>
      ${tab === 'positions' && html`<${PositionsTable} portfolioKey=${k} positions=${data[k].detail.positions} onChanged=${loadSoon} notify=${notify} />`}
      ${tab === 'orders' && html`<${OrdersTable} portfolioKey=${k} orders=${data[k].orders} onChanged=${loadSoon} notify=${notify} />`}
      ${tab === 'trades' && html`<${TradesTable} trades=${data[k].trades} />`}
    </div>`)}
  </section>`;
}

function ResetAction({ notify, startCash, currency }) {
  const [open, setOpen] = useState(false);
  const [capital, setCapital] = useState(toInput(startCash, 0));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const confirm = async () => {
    const value = parseDecimal(capital, { grouping: true });
    if (!(value > 0)) {
      setError('Bitte ein Startkapital größer als 0 eingeben.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.post('/portfolios/reset', { confirm: true, start_capital: value });
      setOpen(false);
      notify(`Beide Depots starten neu mit ${fmtMoney(value, currency, 0)}.`);
      loadPortfolios();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };
  return html`<section class="panel" aria-labelledby="rs-title">
    <div class="panel-head"><h2 id="rs-title">Neuer Vergleich</h2></div>
    <p class="small muted">Beide Depots mit gleichem Kapital neu starten, zum Beispiel nach einer Änderung der Engine.</p>
    <button type="button" class="btn btn-danger reset-btn" onClick=${() => setOpen(true)}>Neu starten …</button>
    ${open && html`<${Dialog} title="Beide Depots neu starten?" onClose=${() => setOpen(false)}>
      <p class="prose">Alle Positionen, Orders und Trades beider Depots werden ohne Abrechnung gelöscht, ebenso der bisherige
        Depotverlauf. Danach starten beide mit demselben Kapital, und der Vergleich „KI gegen mich“ beginnt von vorn.
        Das Signal-Journal und die Statistik der Empfehlungen bleiben erhalten.</p>
      <div class="field reset-field">
        <label for="reset-capital">Startkapital</label>
        <div class="input-affix"><input id="reset-capital" class="input input-num" inputmode="decimal" value=${capital}
          onInput=${(e) => setCapital(e.currentTarget.value)} /><span class="affix">${currency}</span></div>
      </div>
      ${error && html`<p class="field-error" role="alert">${error}</p>`}
      <div class="dialog-actions">
        <button type="button" class="btn" onClick=${() => setOpen(false)} data-autofocus>Abbrechen</button>
        <button type="button" class="btn btn-primary" onClick=${confirm} disabled=${busy}>${busy ? 'Wird neu gestartet …' : 'Beide Depots neu starten'}</button>
      </div>
    </${Dialog}>`}
  </section>`;
}

export function DepotView() {
  const portfolios = useStore(app, (s) => s.portfolios);
  const error = useStore(app, (s) => s.portfoliosError);
  const instruments = useStore(app, (s) => s.instruments);
  const [toast, setToast] = useState(null);
  useTick(30);
  const ai = portfolios.find((p) => p.key === 'ai');
  const me = portfolios.find((p) => p.key === 'me');
  return html`<div>
    <div class="page-head"><div><h1>Depot – KI gegen mich</h1>
      <p class="lede">Zwei Papierdepots mit identischen Regeln und Kosten: Die KI handelt ihre eigenen Empfehlungen, Sie entscheiden selbst.</p></div></div>
    ${!ai || !me ? (error ? html`<${ErrorState} error=${error} onRetry=${loadPortfolios} />` : html`<${Loading} lines=${6} />`) : html`
    <div class="depot-grid">
      <div class="stack">
        <${SummaryTable} ai=${ai} me=${me} />
        <${EquityPanel} instruments=${instruments} />
        <${Books} notify=${setToast} />
      </div>
      <aside class="stack">
        <section class="panel" aria-labelledby="dt-title">
          <div class="panel-head"><h2 id="dt-title">Order für mein Depot</h2></div>
          <${OrderTicket} idPrefix="depot-ot" />
        </section>
        <${ResetAction} notify=${setToast} startCash=${me.start_cash} currency=${me.base_currency} />
      </aside>
    </div>`}
    <${Toast} message=${toast} onDone=${() => setToast(null)} />
  </div>`;
}
