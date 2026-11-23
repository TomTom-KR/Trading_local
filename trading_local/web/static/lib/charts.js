// lightweight-charts (v5) wrappers: a price chart (candles, volume pane, overlays, markers, price lines,
// live updates, lazy loading of older bars) and a line chart (equity curves, cumulative R).
//
// Time axis: the library has no time zones. Intraday data keeps UTC seconds and every label is
// formatted in the browser's local zone; daily bars (ts = session date 00:00 UTC) are passed as
// 'YYYY-MM-DD' business days so they never shift a day. Colours come from the CSS tokens, given as
// token names ('up', 'down', 'ink2', ...) so a theme switch can re-apply them.
// The library's attribution logo is switched off because it injects an inline <style> element that
// a strict Content-Security-Policy blocks; the TradingView notice and link are on the system page.

import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  LineStyle,
  TickMarkType,
  createChart,
  createSeriesMarkers,
} from '../vendor/lightweight-charts.mjs';
import { fmtCompact, fmtNum, fmtPrice, priceDecimals } from './format.js';
import { cssVar } from './theme.js';

const TOKENS = {
  bg: '--surface-1', bg2: '--surface-2', line: '--line', ink1: '--ink-1', ink2: '--ink-2', ink3: '--ink-3',
  up: '--up', down: '--down', ai: '--ai', me: '--me',
};

/** Current colour of a token name (or a literal colour passed through). */
function tokenColor(name) {
  return TOKENS[name] ? cssVar(TOKENS[name]) : name;
}

/** '#rrggbb' + alpha -> 'rgba(...)'. */
function withAlpha(hex, alpha) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

const LINE_STYLES = { solid: LineStyle.Solid, dotted: LineStyle.Dotted, dashed: LineStyle.Dashed };

const fmtCache = new Map();
function dtf(options) {
  const key = JSON.stringify(options);
  if (!fmtCache.has(key)) fmtCache.set(key, new Intl.DateTimeFormat('de-DE', { hourCycle: 'h23', ...options }));
  return fmtCache.get(key);
}

/** Library time (UTC seconds, business-day string or object) -> Date. */
function timeToDate(time) {
  if (typeof time === 'number') return new Date(time * 1000);
  if (typeof time === 'string') return new Date(`${time}T00:00:00Z`);
  return new Date(Date.UTC(time.year, time.month - 1, time.day));
}
const isDay = (time) => typeof time !== 'number';

function tickMarkFormatter(time, type) {
  const d = timeToDate(time);
  const tz = isDay(time) ? { timeZone: 'UTC' } : {};
  switch (type) {
    case TickMarkType.Year:
      return dtf({ year: 'numeric', ...tz }).format(d);
    case TickMarkType.Month:
      return dtf({ month: 'short', ...tz }).format(d).replace('.', '');
    case TickMarkType.DayOfMonth:
      return dtf({ day: '2-digit', month: '2-digit', ...tz }).format(d);
    case TickMarkType.TimeWithSeconds:
      return dtf({ hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(d);
    default:
      return dtf({ hour: '2-digit', minute: '2-digit' }).format(d);
  }
}

function crosshairTimeFormatter(time) {
  const d = timeToDate(time);
  if (isDay(time)) return dtf({ weekday: 'short', day: '2-digit', month: '2-digit', year: 'numeric', timeZone: 'UTC' }).format(d);
  return dtf({ weekday: 'short', day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }).format(d);
}

/** Bar timestamp -> library time ('YYYY-MM-DD' for daily bars). */
function chartTime(ts, daily) {
  return daily ? new Date(ts * 1000).toISOString().slice(0, 10) : ts;
}

/** Chart options shared by all charts; value formatting is set per series (priceFormat 'custom'). */
function baseOptions() {
  const line = tokenColor('line');
  return {
    autoSize: true,
    layout: {
      background: { type: ColorType.Solid, color: tokenColor('bg') },
      textColor: tokenColor('ink3'),
      fontFamily: getComputedStyle(document.body).fontFamily,
      fontSize: 12,
      attributionLogo: false,
      panes: { separatorColor: line, separatorHoverColor: line, enableResize: false },
    },
    grid: {
      vertLines: { color: withAlpha(line, 0.55) },
      horzLines: { color: withAlpha(line, 0.55) },
    },
    crosshair: {
      mode: CrosshairMode.Normal,
      vertLine: { color: tokenColor('ink3'), labelBackgroundColor: tokenColor('bg2') },
      horzLine: { color: tokenColor('ink3'), labelBackgroundColor: tokenColor('bg2') },
    },
    rightPriceScale: { borderColor: line },
    timeScale: { borderColor: line, rightOffset: 4, secondsVisible: false, tickMarkFormatter },
    localization: { locale: 'de-DE', timeFormatter: crosshairTimeFormatter },
  };
}

/**
 * Break a line where it restarts (session-anchored VWAP) so no jump is drawn across the night. The
 * library draws straight over whitespace items, but it colours the segment that STARTS at a point with
 * that point's colour: the last point before a gap gets a transparent colour. No point is added or
 * removed, so the time scale (and the logical range of the newest bars) stays unchanged.
 */
function withBreaks(points, step) {
  return points.map((p, i) => {
    const next = points[i + 1];
    return next && next.time - p.time > 2 * step ? { ...p, color: 'transparent' } : p;
  });
}

/** Smallest positive spacing of consecutive bar times in seconds (60 when it cannot be told). */
function minStep(bars) {
  let step = Infinity;
  for (let i = 1; i < bars.length; i += 1) {
    const d = bars[i][0] - bars[i - 1][0];
    if (d > 0 && d < step) step = d;
  }
  return Number.isFinite(step) ? step : 60;
}

// ------------------------------------------------------------------------------------------------
// Price chart
// ------------------------------------------------------------------------------------------------
const OVERLAY_STYLE = {
  ema20: { color: 'ink2', width: 1, style: 'solid', label: 'EMA 20' },
  ema50: { color: 'ink3', width: 2, style: 'solid', label: 'EMA 50' },
  ema200: { color: 'ink3', width: 2, style: 'dashed', label: 'EMA 200' },
  vwap: { color: 'ink2', width: 1, style: 'dotted', label: 'VWAP' },
};
export const OVERLAY_LEGEND = OVERLAY_STYLE;

/**
 * @typedef {{ts: number, position: 'aboveBar' | 'belowBar' | 'inBar', shape: 'arrowUp' | 'arrowDown' | 'circle' | 'square', color: string, text?: string, size?: number}} MarkerSpec
 * @typedef {{price: number, color: string, style?: 'solid' | 'dashed' | 'dotted', title: string, width?: number}} PriceLineSpec
 */

export class PriceChart {
  /**
   * @param {HTMLElement} el
   * @param {{daily?: boolean, assetClass?: string, volumeHeight?: number, onCrosshair?: (bar: number[] | null) => void, onNeedOlder?: (beforeTs: number) => void}} opts
   */
  constructor(el, opts = {}) {
    this.daily = Boolean(opts.daily);
    this.volumeHeight = opts.volumeHeight ?? 96;
    this.assetClass = opts.assetClass;
    this.onCrosshair = opts.onCrosshair;
    this.onNeedOlder = opts.onNeedOlder;
    /** @type {number[][]} */ this.bars = [];
    this.overlays = {};
    this.hasVolume = false;
    this.markerSpecs = [];
    this.lineSpecs = [];
    this.priceLines = [];
    this.loadingOlder = false;
    this.noMoreOlder = false;

    this.chart = createChart(el, {
      ...baseOptions(),
      timeScale: { ...baseOptions().timeScale, timeVisible: !this.daily },
    });
    this.candles = this.chart.addSeries(CandlestickSeries, { borderVisible: false, priceLineVisible: true });
    this.volume = null;
    this.lines = {};
    this.markers = createSeriesMarkers(this.candles, []);
    this.applyTheme();

    this.chart.subscribeCrosshairMove((p) => {
      if (!this.onCrosshair) return;
      if (!p || p.time === undefined || !p.point) {
        this.onCrosshair(null);
        return;
      }
      const idx = this.indexOfTime(p.time);
      this.onCrosshair(idx >= 0 ? this.bars[idx] : null);
    });
    this.chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (!range || !this.onNeedOlder || this.loadingOlder || this.noMoreOlder || !this.bars.length) return;
      if (range.from < 15) {
        this.loadingOlder = true;
        this.onNeedOlder(this.bars[0][0]);
      }
    });
  }

  indexOfTime(time) {
    const target = typeof time === 'number' ? time : null;
    if (target === null) {
      const day = typeof time === 'string' ? time : `${time.year}-${String(time.month).padStart(2, '0')}-${String(time.day).padStart(2, '0')}`;
      return this.bars.findIndex((b) => chartTime(b[0], true) === day);
    }
    let lo = 0;
    let hi = this.bars.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const t = this.bars[mid][0];
      if (t === target) return mid;
      if (t < target) lo = mid + 1;
      else hi = mid - 1;
    }
    return -1;
  }

  candle(b) {
    return { time: chartTime(b[0], this.daily), open: b[1], high: b[2], low: b[3], close: b[4] };
  }

  volumeItem(b) {
    const up = b[4] >= b[1];
    return { time: chartTime(b[0], this.daily), value: b[5], color: withAlpha(tokenColor(up ? 'up' : 'down'), 0.45) };
  }

  ensureVolumePane() {
    if (this.hasVolume && !this.volume) {
      this.volume = this.chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'custom', formatter: fmtCompact, minMove: 1 }, priceLineVisible: false, lastValueVisible: false,
      }, 1);
      const panes = this.chart.panes();
      if (panes[1]) panes[1].setHeight(this.volumeHeight);
    }
  }

  renderOverlays() {
    for (const [key, points] of Object.entries(this.overlays)) {
      const style = OVERLAY_STYLE[key];
      if (!style) continue;
      if (!this.lines[key]) {
        this.lines[key] = this.chart.addSeries(LineSeries, {
          lineWidth: style.width, lineStyle: LINE_STYLES[style.style], priceLineVisible: false,
          lastValueVisible: false, crosshairMarkerVisible: false, color: tokenColor(style.color),
        });
      }
      const data = points.map(([t, v]) => ({ time: chartTime(t, this.daily), value: v }));
      this.lines[key].setData(key === 'vwap' ? withBreaks(data, this.barStep()) : data);
    }
    for (const key of Object.keys(this.lines)) {
      if (!this.overlays[key]) {
        this.chart.removeSeries(this.lines[key]);
        delete this.lines[key];
      }
    }
  }

  /** Spacing of the bars in seconds (the smallest one: session gaps and weekends do not count). */
  barStep() {
    return minStep(this.bars);
  }

  /** Index of the newest bar that opened at or before ``ts`` (-1 when ``ts`` is before the first bar). */
  barIndexAtOrBefore(ts) {
    let lo = 0;
    let hi = this.bars.length - 1;
    let found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (this.bars[mid][0] <= ts) {
        found = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return found;
  }

  /** Replace all data. @param {import('./types.js').BarsOut} out */
  setData(out) {
    this.bars = out.bars.slice();
    this.overlays = out.overlays || {};
    this.hasVolume = this.bars.some((b) => b[5] > 0);
    this.noMoreOlder = false;
    this.loadingOlder = false;
    const decimals = this.bars.length ? priceDecimals(this.bars[this.bars.length - 1][4], this.assetClass) : 2;
    this.candles.applyOptions({
      priceFormat: { type: 'custom', formatter: (p) => fmtPrice(p, this.assetClass), minMove: 10 ** -decimals },
    });
    this.candles.setData(this.bars.map((b) => this.candle(b)));
    this.ensureVolumePane();
    if (this.volume) this.volume.setData(this.bars.map((b) => this.volumeItem(b)));
    this.renderOverlays();
    this.renderMarkers();
    const visible = this.daily ? 160 : 180;
    if (this.bars.length > visible) {
      this.chart.timeScale().setVisibleLogicalRange({ from: this.bars.length - visible, to: this.bars.length + 3 });
    } else {
      this.chart.timeScale().fitContent();
    }
  }

  /** Prepend older bars (lazy loading) and keep the visible window where it was. */
  prepend(out) {
    this.loadingOlder = false;
    const firstTs = this.bars.length ? this.bars[0][0] : Infinity;
    const older = out.bars.filter((b) => b[0] < firstTs);
    if (!older.length) {
      this.noMoreOlder = true;
      return;
    }
    const range = this.chart.timeScale().getVisibleLogicalRange();
    this.bars = older.concat(this.bars);
    for (const [key, points] of Object.entries(out.overlays || {})) {
      const current = this.overlays[key] || [];
      const firstOverlay = current.length ? current[0][0] : Infinity;
      this.overlays[key] = points.filter((p) => p[0] < firstOverlay).concat(current);
    }
    this.candles.setData(this.bars.map((b) => this.candle(b)));
    if (this.volume) this.volume.setData(this.bars.map((b) => this.volumeItem(b)));
    this.renderOverlays();
    this.renderMarkers();
    if (range) {
      this.chart.timeScale().setVisibleLogicalRange({ from: range.from + older.length, to: range.to + older.length });
    }
  }

  /** Signal that no older data could be loaded (stop asking). */
  olderFailed() {
    this.loadingOlder = false;
    this.noMoreOlder = true;
  }

  /** Live update of the newest bar (same ts replaces, later ts appends; older ts is ignored). */
  updateBar(b) {
    if (!this.bars.length) return;
    const last = this.bars[this.bars.length - 1];
    if (b[0] < last[0]) return;
    const appended = b[0] !== last[0];
    if (appended) this.bars.push(b);
    else this.bars[this.bars.length - 1] = b;
    this.candles.update(this.candle(b));
    if (this.volume) this.volume.update(this.volumeItem(b));
    // a marker of an event newer than the loaded bars appears once its bar exists
    if (appended && this.markerSpecs.length) this.renderMarkers();
  }

  /** @param {MarkerSpec[]} specs */
  setMarkers(specs) {
    this.markerSpecs = specs;
    this.renderMarkers();
  }

  renderMarkers() {
    if (!this.bars.length) {
      this.markers.setMarkers([]);
      return;
    }
    // snap every marker to the bar that contains its time (entry/exit times are bar open times; a
    // 15-minute or daily bar contains many of them); events outside the loaded bars are left out
    const lastIndex = this.bars.length - 1;
    const span = this.daily ? 86400 : this.barStep();
    const items = [];
    for (const m of this.markerSpecs) {
      const idx = this.barIndexAtOrBefore(m.ts);
      if (idx < 0 || (idx === lastIndex && m.ts >= this.bars[lastIndex][0] + span)) continue;
      items.push({
        time: chartTime(this.bars[idx][0], this.daily), position: m.position, shape: m.shape, color: tokenColor(m.color),
        text: m.text, size: m.size ?? 1,
      });
    }
    items.sort((a, b) => (a.time < b.time ? -1 : a.time > b.time ? 1 : 0));
    this.markers.setMarkers(items);
  }

  /** @param {PriceLineSpec[]} specs */
  setPriceLines(specs) {
    this.lineSpecs = specs;
    for (const pl of this.priceLines) this.candles.removePriceLine(pl);
    this.priceLines = specs.map((s) => this.candles.createPriceLine({
      price: s.price, color: tokenColor(s.color), lineWidth: s.width ?? 1, lineStyle: LINE_STYLES[s.style || 'dashed'],
      axisLabelVisible: true, title: s.title,
    }));
  }

  /** Re-read the CSS tokens (after a theme switch). */
  applyTheme() {
    const opts = baseOptions();
    this.chart.applyOptions({ layout: opts.layout, grid: opts.grid, crosshair: opts.crosshair,
      rightPriceScale: opts.rightPriceScale, timeScale: { borderColor: opts.timeScale.borderColor } });
    const up = tokenColor('up');
    const down = tokenColor('down');
    this.candles.applyOptions({ upColor: up, downColor: down, wickUpColor: up, wickDownColor: down });
    for (const [key, series] of Object.entries(this.lines)) series.applyOptions({ color: tokenColor(OVERLAY_STYLE[key].color) });
    if (this.volume && this.bars.length) this.volume.setData(this.bars.map((b) => this.volumeItem(b)));
    this.renderMarkers();
    this.setPriceLines(this.lineSpecs);
  }

  destroy() {
    this.chart.remove();
  }
}

// ------------------------------------------------------------------------------------------------
// Line chart (equity curves, cumulative R)
// ------------------------------------------------------------------------------------------------
/**
 * @typedef {{id: string, points: Array<[number, number]>, color: string, style?: 'solid' | 'dashed' | 'dotted', width?: number}} LineSpec
 */

export class LineChart {
  /**
   * @param {HTMLElement} el
   * @param {{format?: (v: number) => string, compact?: boolean}} [opts]
   */
  constructor(el, opts = {}) {
    this.format = opts.format || ((v) => fmtNum(v, 0));
    this.compact = Boolean(opts.compact);
    this.chart = createChart(el, baseOptions());
    this.chart.applyOptions({
      timeScale: { timeVisible: true, rightOffset: 2, fixLeftEdge: true, fixRightEdge: true },
      handleScroll: !this.compact,
      handleScale: !this.compact,
      rightPriceScale: { scaleMargins: { top: 0.12, bottom: 0.08 } },
    });
    /** @type {Map<string, any>} */ this.series = new Map();
    this.specs = [];
  }

  /** @param {LineSpec[]} specs */
  setSeries(specs) {
    this.specs = specs;
    const wanted = new Set(specs.map((s) => s.id));
    for (const [id, s] of this.series) {
      if (!wanted.has(id)) {
        this.chart.removeSeries(s);
        this.series.delete(id);
      }
    }
    for (const spec of specs) {
      let s = this.series.get(spec.id);
      if (!s) {
        s = this.chart.addSeries(LineSeries, {
          priceLineVisible: false, lastValueVisible: !this.compact, crosshairMarkerVisible: true,
          crosshairMarkerRadius: 3, priceFormat: { type: 'custom', formatter: this.format, minMove: 0.01 },
        });
        this.series.set(spec.id, s);
      }
      s.applyOptions({ color: tokenColor(spec.color), lineWidth: spec.width ?? 2, lineStyle: LINE_STYLES[spec.style || 'solid'] });
      const seen = new Set();
      const data = [];
      for (const [t, v] of spec.points) {
        if (seen.has(t)) continue;
        seen.add(t);
        data.push({ time: t, value: v });
      }
      s.setData(data);
    }
    this.chart.timeScale().fitContent();
  }

  applyTheme() {
    const opts = baseOptions();
    this.chart.applyOptions({ layout: opts.layout, grid: opts.grid, crosshair: opts.crosshair,
      rightPriceScale: opts.rightPriceScale, timeScale: { borderColor: opts.timeScale.borderColor } });
    this.setSeries(this.specs);
  }

  destroy() {
    this.chart.remove();
  }
}
