// #/ Übersicht: session ribbon (hero), recommendations, 'KI gegen mich', compact watchlist, news.

import {
  html, useEffect, useMemo, useRef, useState,
} from '../vendor/preact-htm.mjs';
import { api, errorText, isAbort } from '../lib/api.js';
import { app } from '../lib/data.js';
import {
  fmtDate, fmtMoney, fmtPrice, fmtSigned,
} from '../lib/format.js';
import { REGION_LABELS } from '../lib/labels.js';
import { useLive } from '../lib/live.js';
import { symbolHref } from '../lib/router.js';
import { upsertSignal, withLiveR } from '../lib/signals.js';
import {
  useInterval, useSoon, useStore, useTick,
} from '../lib/store.js';
import { LineChartView } from '../components/line-chart.js';
import { NewsItem } from '../components/news-item.js';
import { SessionRibbon } from '../components/ribbon.js';
import { SignalTicket } from '../components/signal-ticket.js';
import { Sparkline } from '../components/sparkline.js';
import {
  Change, Empty, ErrorState, FlashValue, Loading, PhaseDot,
} from '../components/ui.js';

const REGION_ORDER = ['ASIA', 'EU', 'US', 'GLOBAL'];

function Recommendations() {
  const [showShadow, setShowShadow] = useState(false);
  const [state, setState] = useState({ items: [], loading: true, error: null });
  const [fresh, setFresh] = useState(() => new Set());
  const quotes = useStore(app, (s) => s.quotes);
  const now = useTick(15);

  const generation = useRef(0); // a refresh that answers after the filter changed must not win
  const load = (signal) => {
    const mine = ++generation.current;
    return api.get('/signals', { scope: 'all', actionable: showShadow ? 'all' : 'true', limit: 8 }, { signal })
      .then((items) => { if (mine === generation.current) setState({ items, loading: false, error: null }); })
      .catch((err) => {
        if (!isAbort(err) && mine === generation.current) setState((s) => ({ ...s, loading: false, error: errorText(err) }));
      });
  };

  useEffect(() => {
    const c = new AbortController();
    setState((s) => ({ ...s, loading: true }));
    load(c.signal);
    return () => c.abort();
  }, [showShadow]);
  useInterval(() => load(), 60000);
  useLive('signal', (sig) => {
    if (!showShadow && !sig.actionable) return;
    setState((s) => {
      if (!s.items.some((x) => x.id === sig.id)) setFresh((f) => new Set(f).add(sig.id));
      return { ...s, items: upsertSignal(s.items, sig, 8) };
    });
  });

  const items = useMemo(() => withLiveR(state.items, quotes), [state.items, quotes]);
  const open = items.filter((s) => ['review', 'pending_entry', 'active'].includes(s.status)).length;
  return html`<section class="panel ov-signals" aria-labelledby="rec-title">
    <div class="panel-head">
      <h2 id="rec-title">Empfehlungen</h2>
      <div class="aside">
        <span>${open} offen</span>
        <label class="switch small"><input type="checkbox" checked=${showShadow} onChange=${(e) => setShowShadow(e.currentTarget.checked)} />
          auch verworfene Kandidaten zeigen</label>
        <a href="#/signale">Alle Signale</a>
      </div>
    </div>
    ${state.loading && !state.items.length ? html`<${Loading} lines=${5} />`
      : state.error && !state.items.length ? html`<${ErrorState} error=${state.error} onRetry=${() => load()} />`
        : !items.length ? html`<${Empty} title="Noch keine Empfehlungen.">
            Die Engine prüft jede Minute alle offenen Märkte – Signale erscheinen hier, sobald ein Setup auftritt.</${Empty}>`
          : html`<div class="sig-list">${items.map((s) => html`<${SignalTicket} key=${s.id} signal=${s} now=${now} fresh=${fresh.has(s.id)} />`)}</div>`}
  </section>`;
}

function VersusPanel() {
  const portfolios = useStore(app, (s) => s.portfolios);
  const error = useStore(app, (s) => s.portfoliosError);
  const [curves, setCurves] = useState(null);
  const loadCurves = () => api.get('/portfolios/equity', { max_points: 240 }).then(setCurves).catch(() => {});
  useEffect(() => { loadCurves(); }, []);
  useInterval(loadCurves, 60000);
  useLive('portfolio', useSoon(loadCurves));

  const ai = portfolios.find((p) => p.key === 'ai');
  const me = portfolios.find((p) => p.key === 'me');
  const series = useMemo(() => {
    if (!curves) return [];
    const out = [
      { id: 'ai', points: curves.ai, color: 'ai', width: 2 },
      { id: 'me', points: curves.me, color: 'me', width: 2 },
    ];
    for (const [k, pts] of Object.entries(curves.benchmarks || {})) out.push({ id: k, points: pts, color: 'ink3', style: 'dashed', width: 1 });
    return out;
  }, [curves]);

  if (!ai || !me) {
    return html`<section class="panel" aria-labelledby="vs-title"><div class="panel-head"><h2 id="vs-title">KI gegen mich</h2></div>
      ${error ? html`<${ErrorState} error=${error} />` : html`<${Loading} lines=${4} />`}</section>`;
  }
  const cur = ai.base_currency;
  const diff = ai.equity - me.equity;
  const leader = Math.abs(diff) < 0.5 ? null : diff > 0 ? 'Die KI' : 'Sie';
  const benchNames = Object.keys(curves?.benchmarks || {}).map((k) => app.get().instruments.find((i) => i.symbol === k)?.name || k);
  return html`<section class="panel vs" aria-labelledby="vs-title">
    <div class="panel-head">
      <h2 id="vs-title">KI gegen mich</h2>
      <div class="aside"><span>seit ${fmtDate(ai.reset_ts)}</span><a href="#/depot">Depot</a></div>
    </div>
    <div class="vs-figures">
      ${[ai, me].map((p) => html`<div class="vs-col">
        <div class="vs-name"><span class=${`swatch swatch-${p.key}`} aria-hidden="true"></span>${p.key === 'ai' ? 'KI-Depot' : 'Mein Depot'}</div>
        <div class="figure figure-28">${fmtMoney(p.equity, cur, 0)}</div>
        <div class="small"><${Change} value=${p.total_return_pct} digits=${2} /> <span class="muted">seit Start</span>
          · <${Change} value=${p.day_return_pct} digits=${2} /> <span class="muted">heute</span></div>
      </div>`)}
    </div>
    <p class="vs-diff small">
      ${leader ? html`${leader} ${leader === 'Sie' ? 'liegen' : 'liegt'} <strong class="num">${fmtMoney(Math.abs(diff), cur, 0)}</strong> vorn
        (<span class="num">${fmtSigned(ai.total_return_pct - me.total_return_pct, 2).replace(/^[+−]/, '')} Prozentpunkte</span>).`
        : 'Beide Depots liegen gleichauf.'}
    </p>
    ${series.length > 0 && html`<${LineChartView} series=${series} height=${150} compact=${true}
      format=${(v) => fmtMoney(v, cur, 0)} label="Depotwert KI und ich seit dem Start" />`}
    <div class="legend">
      <span><i class="swatch swatch-ai"></i>KI</span><span><i class="swatch swatch-me"></i>Ich</span>
      ${benchNames.map((n) => html`<span><i class="swatch swatch-bench"></i>${n}</span>`)}
    </div>
  </section>`;
}

function WatchPanel() {
  const instruments = useStore(app, (s) => s.instruments);
  const loaded = useStore(app, (s) => s.instrumentsLoaded);
  const error = useStore(app, (s) => s.instrumentsError);
  const quotes = useStore(app, (s) => s.quotes);
  const sparks = useStore(app, (s) => s.sparklines);
  const groups = useMemo(() => REGION_ORDER.map((r) => ({
    region: r,
    // conversion rates (non-tradable FX pairs) stay on the full watchlist only
    items: instruments.filter((i) => i.active && i.region === r && !(i.asset_class === 'fx' && !i.tradable))
      .sort((a, b) => a.sort_order - b.sort_order),
  })).filter((g) => g.items.length), [instruments]);

  return html`<section class="panel watch" aria-labelledby="watch-title">
    <div class="panel-head"><h2 id="watch-title">Märkte</h2><div class="aside"><a href="#/maerkte">Beobachtungsliste</a></div></div>
    ${!loaded ? html`<${Loading} lines=${6} />`
      : error && !instruments.length ? html`<${ErrorState} error=${error} />`
        : !groups.length ? html`<${Empty} title="Die Beobachtungsliste ist leer.">Unter Märkte lassen sich Werte per Symbol hinzufügen.</${Empty}>`
          : groups.map((g) => html`<div class="watch-group">
              <h3 class="watch-region">${REGION_LABELS[g.region]}</h3>
              <ul class="watch-list">
                ${g.items.map((i) => {
                  const q = quotes[i.symbol];
                  return html`<li><a class="wrow" href=${symbolHref(i.symbol)}>
                    <${PhaseDot} phase=${i.phase} />
                    <span class="wrow-sym">${i.symbol}</span>
                    <span class="wrow-name truncate">${i.name}</span>
                    <${Sparkline} points=${sparks[i.symbol]} prevClose=${q?.prev_close} width=${64} height=${20} />
                    <${FlashValue} value=${q?.price} class="wrow-price num">${q ? fmtPrice(q.price, i.asset_class) : '–'}</${FlashValue}>
                    <span class="wrow-chg"><${Change} value=${q?.change_pct} digits=${2} /></span>
                  </a></li>`;
                })}
              </ul>
            </div>`)}
  </section>`;
}

function NewsPanel() {
  const [state, setState] = useState({ items: [], loading: true, error: null });
  const now = useTick(30);
  const load = () => api.get('/news', { limit: 8, min_relevance: 3 })
    .then((items) => setState({ items, loading: false, error: null }))
    .catch((err) => setState((s) => ({ ...s, loading: false, error: errorText(err) })));
  useEffect(() => { load(); }, []);
  useInterval(load, 120000);
  useLive('news', (n) => {
    if (!n.analysis || n.analysis.relevance < 3) return;
    setState((s) => ({ ...s, items: [n, ...s.items.filter((x) => x.id !== n.id)].slice(0, 8) }));
  });
  return html`<section class="panel ov-news" aria-labelledby="news-title">
    <div class="panel-head"><h2 id="news-title">Nachrichten</h2>
      <div class="aside"><span>Relevanz ab 3 von 5</span><a href="#/news">Alle Nachrichten</a></div></div>
    ${state.loading ? html`<${Loading} lines=${4} />`
      : state.error && !state.items.length ? html`<${ErrorState} error=${state.error} onRetry=${load} />`
        : !state.items.length ? html`<${Empty} title="Noch keine relevanten Nachrichten.">
            Die Feeds werden jede Minute abgefragt; die KI bewertet neue Meldungen nacheinander.</${Empty}>`
          : html`<div class="news-list">${state.items.map((n) => html`<${NewsItem} key=${n.id} item=${n} now=${now} />`)}</div>`}
  </section>`;
}

export function OverviewView() {
  const sessions = useStore(app, (s) => s.sessions);
  const sessionsError = useStore(app, (s) => s.sessionsError);
  const instruments = useStore(app, (s) => s.instruments);
  const around = useMemo(() => instruments
    .filter((i) => i.active && i.asset_class === 'crypto' && i.is_open)
    .map((i) => i.name), [instruments]);
  return html`<div class="overview">
    ${sessions ? html`<${SessionRibbon} sessions=${sessions} around=${around} />`
      : html`<section class="ribbon ribbon-empty">
          ${sessionsError ? html`<${ErrorState} error=${sessionsError} title="Handelszeiten konnten nicht geladen werden." />` : html`<${Loading} lines=${3} label="Handelszeiten werden geladen" />`}
        </section>`}
    <div class="ov-grid">
      <${Recommendations} />
      <div class="stack">
        <${VersusPanel} />
        <${WatchPanel} />
      </div>
    </div>
    <${NewsPanel} />
  </div>`;
}

