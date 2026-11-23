// Candlestick chart of one symbol: loads bars for a timeframe, lazily loads older bars when the user
// scrolls left, follows live 'bar' events (aggregated into the shown timeframe) and 'quotes', and shows
// an OHLCV legend for the bar under the crosshair (or the newest bar).

import { html, useEffect, useRef, useState } from '../vendor/preact-htm.mjs';
import { api, errorText, isAbort } from '../lib/api.js';
import { OVERLAY_LEGEND, PriceChart } from '../lib/charts.js';
import {
  fmtCompact, fmtDateTime, fmtPrice, fmtSessionDate, priceUnit,
} from '../lib/format.js';
import { liveBarAction, TF_SECONDS } from '../lib/live-bars.js';
import { onLive, subscribeBars } from '../lib/live.js';
import { useStore } from '../lib/store.js';
import { themeStore } from '../lib/theme.js';
import { Change } from './ui.js';

const PAGE = { '1m': 600, '5m': 500, '15m': 400, '1h': 400, '1d': 400 };
const DAILY_REFRESH_MS = 30000; // at most one reload of the newest daily candles per 30 s

/**
 * @param {{symbol: string, instrument?: import('../lib/types.js').InstrumentOut | null, timeframe: string,
 *   markers?: import('../lib/charts.js').MarkerSpec[], priceLines?: import('../lib/charts.js').PriceLineSpec[],
 *   live?: boolean, height?: number, limit?: number, compactLegend?: boolean, label?: string, before?: number | null}} props
 *   ``before``: show the bars that opened before this time (a historical window; live updates are off)
 *   ``volumeHeight``: height of the volume pane in px (default 96)
 */
export function PriceChartView({
  symbol, instrument, timeframe, markers = [], priceLines = [], live = true, height = 460, limit, compactLegend = false,
  label, before = null, volumeHeight,
}) {
  const hostRef = useRef(null);
  const chartRef = useRef(null);
  const [state, setState] = useState({ loading: true, error: null, empty: false });
  const [hover, setHover] = useState(null);
  const [meta, setMeta] = useState(null);
  const [, setLegendTick] = useState(0); // re-render the legend after live updates
  const lastBarRef = useRef(null);
  const theme = useStore(themeStore, (s) => s.theme);
  const daily = timeframe === '1d';
  const assetClass = instrument?.asset_class;

  // create / destroy the chart per symbol + timeframe and load the first page
  useEffect(() => {
    const controller = new AbortController();
    const chart = new PriceChart(hostRef.current, {
      daily,
      assetClass,
      volumeHeight,
      onCrosshair: (bar) => setHover(bar),
      onNeedOlder: async (beforeTs) => {
        try {
          const older = await api.get(`/bars/${encodeURIComponent(symbol)}`,
            { tf: timeframe, limit: PAGE[timeframe], before: beforeTs }, { signal: controller.signal });
          chart.prepend(older);
        } catch (err) {
          if (!isAbort(err)) chart.olderFailed();
        }
      },
    });
    chartRef.current = chart;
    setState({ loading: true, error: null, empty: false });
    setHover(null);
    api.get(`/bars/${encodeURIComponent(symbol)}`, { tf: timeframe, limit: limit ?? PAGE[timeframe], before },
      { signal: controller.signal })
      .then((out) => {
        chart.setData(out);
        lastBarRef.current = out.bars[out.bars.length - 1] ?? null;
        setMeta({ currency: out.currency, priceScale: out.price_scale, delay: out.delay_min, overlays: Object.keys(out.overlays || {}) });
        setState({ loading: false, error: null, empty: out.bars.length === 0 });
      })
      .catch((err) => {
        if (!isAbort(err)) setState({ loading: false, error: errorText(err), empty: false });
      });
    return () => {
      controller.abort();
      chart.destroy();
      chartRef.current = null;
    };
  }, [symbol, timeframe, before]);

  useEffect(() => chartRef.current?.setMarkers(markers), [markers, state.loading]);
  useEffect(() => chartRef.current?.setPriceLines(priceLines), [priceLines, state.loading]);
  useEffect(() => chartRef.current?.applyTheme(), [theme]);

  // live updates: 1-minute bars aggregated into the shown timeframe, plus quotes for the last close
  useEffect(() => {
    if (!live || before) return undefined;
    const tfSec = TF_SECONDS[timeframe];
    // volume of the shown bucket = volume already in the loaded bar + the live minutes seen since
    const minuteVolumes = new Map();
    let bucketTs = null;
    let baseVolume = 0;
    const controller = new AbortController();
    let refreshTimer = null;
    let lastRefresh = 0;
    const apply = (bar) => {
      const chart = chartRef.current;
      chart.updateBar(bar);
      lastBarRef.current = chart.bars[chart.bars.length - 1];
      setLegendTick((n) => n + 1);
    };
    // Daily candles carry the venue's trading date, which only the server knows (a Sydney session starts on
    // the previous UTC day, a session can open while the chart is shown). Every new minute bar therefore
    // reloads the newest daily candles (today's is built by the server from the closed minutes) instead of
    // merging them by guesswork (lib/live-bars.js); quotes keep the close current in between.
    const refreshDaily = () => {
      if (refreshTimer !== null) return;
      const wait = Math.max(0, lastRefresh + DAILY_REFRESH_MS - Date.now());
      refreshTimer = setTimeout(async () => {
        lastRefresh = Date.now();
        try {
          const out = await api.get(`/bars/${encodeURIComponent(symbol)}`, { tf: '1d', limit: 2 }, { signal: controller.signal });
          if (chartRef.current) out.bars.forEach(apply);
        } catch {
          // the next minute bar tries again; the chart keeps what it shows
        } finally {
          refreshTimer = null;
        }
      }, wait);
    };
    const onBar = (data) => {
      const chart = chartRef.current;
      if (!data || data.symbol !== symbol || !Array.isArray(data.bar) || !chart || !chart.bars.length) return;
      const [ts, o, h, l, c, v] = data.bar;
      const last = chart.bars[chart.bars.length - 1];
      const { action, bucket } = liveBarAction(timeframe, ts, last[0]);
      if (action === 'refresh') refreshDaily();
      if (action !== 'merge') return;
      if (tfSec === 60) {
        apply([ts, o, h, l, c, v]);
        return;
      }
      if (bucket !== bucketTs) {
        bucketTs = bucket;
        minuteVolumes.clear();
        baseVolume = bucket === last[0] ? last[5] : 0;
      }
      minuteVolumes.set(ts, v);
      let volume = baseVolume;
      for (const x of minuteVolumes.values()) volume += x;
      if (bucket === last[0]) apply([bucket, last[1], Math.max(last[2], h), Math.min(last[3], l), c, volume]);
      else apply([bucket, o, h, l, c, volume]);
    };
    const onQuotes = (data) => {
      const q = data?.items?.find((x) => x.symbol === symbol);
      const chart = chartRef.current;
      if (!q || !chart || !chart.bars.length) return;
      const last = chart.bars[chart.bars.length - 1];
      const span = daily ? 86400 : tfSec;
      if (q.ts < last[0] || q.ts >= last[0] + span) return;
      apply([last[0], last[1], Math.max(last[2], q.price), Math.min(last[3], q.price), q.price, last[5]]);
    };
    const unsubscribe = subscribeBars(symbol);
    const offBar = onLive('bar', onBar);
    const offQuotes = onLive('quotes', onQuotes);
    return () => {
      unsubscribe();
      offBar();
      offQuotes();
      controller.abort();
      clearTimeout(refreshTimer);
    };
  }, [symbol, timeframe, live, before]);

  const legendBar = hover || lastBarRef.current;
  const unit = meta ? priceUnit(meta.currency, meta.priceScale) : '';
  return html`<div class="pchart">
    <div class="pchart-legend" aria-live="off">
      ${legendBar ? html`<${OhlcLegend} bar=${legendBar} daily=${daily} assetClass=${assetClass} unit=${unit} compact=${compactLegend} />`
        : html`<span class="muted small">${label || symbol}</span>`}
      ${!compactLegend && meta && html`<span class="pchart-overlays">
        ${meta.overlays.filter((k) => OVERLAY_LEGEND[k]).map((k) => html`<span class=${`ov ov-${k}`}><i aria-hidden="true"></i>${OVERLAY_LEGEND[k].label}</span>`)}
      </span>`}
    </div>
    <div class="pchart-host" ref=${hostRef} style=${{ height: `${height}px` }} role="img"
      aria-label=${`Kursverlauf ${label || symbol} (${timeframe === '1d' ? 'Tageskerzen' : timeframe})`}></div>
    ${state.loading && html`<div class="pchart-state">Kurse werden geladen …</div>`}
    ${state.error && html`<div class="pchart-state pchart-error" role="alert">${state.error}</div>`}
    ${state.empty && html`<div class="pchart-state">Für diesen Zeitraum liegen noch keine Kursdaten vor. Neue Minutenkerzen erscheinen, sobald der Handelsplatz öffnet.</div>`}
  </div>`;
}

function OhlcLegend({ bar, daily, assetClass, unit, compact }) {
  const [ts, o, h, l, c, v] = bar;
  const p = (x) => fmtPrice(x, assetClass);
  return html`<span class="ohlc">
    <span class="ohlc-time">${daily ? fmtSessionDate(ts) : fmtDateTime(ts)}</span>
    <span><span class="k">E</span> ${p(o)}</span>
    <span><span class="k">H</span> ${p(h)}</span>
    <span><span class="k">T</span> ${p(l)}</span>
    <span><span class="k">S</span> ${p(c)}</span>
    <${Change} value=${o ? ((c - o) / o) * 100 : null} digits=${2} />
    ${!compact && v > 0 && html`<span><span class="k">Vol</span> ${fmtCompact(v)}</span>`}
    ${!compact && unit && html`<span class="faint">${unit}</span>`}
  </span>`;
}
