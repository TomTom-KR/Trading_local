// Session ribbon: a 24-hour timeline (now - 6 h ... now + 18 h) with one lane per region. Regular
// sessions are solid bars, pre/post market hatched, lunch breaks gaps. Venues with identical windows
// share one row ('Xetra · Paris · Zürich'). Each lane says what happens next: 'Xetra schließt in
// 1 Std 12 Min' / 'Tokio öffnet in 6 Std 40 Min'. On narrow screens the ribbon scrolls horizontally
// and starts scrolled so that the 'jetzt' line is in view.
//
// Row labels sit on their bar or right beside it, never clipped at the timeline's ends (lib/ribbon-layout.js).
// Their real widths are measured from hidden copies in the ribbon's own font, so the placement does not
// rely on an estimated character width.

import {
  html, useEffect, useLayoutEffect, useMemo, useRef, useState,
} from '../vendor/preact-htm.mjs';
import {
  fmtDuration, fmtHour, fmtList, fmtTime, fmtWeekday, isNum,
} from '../lib/format.js';
import { labelCandidates, labelPlacement } from '../lib/ribbon-layout.js';
import { useTick } from '../lib/store.js';
import { serverNow } from '../lib/time.js';

const KIND_LABEL = { regular: 'Handel', pre: 'Vorbörse', post: 'Nachbörse' };
const LABEL_CHAR_PX = 6.6; // estimate of a 12 px semibold condensed character until the widths are measured
const LABEL_PAD_PX = 16;

/** Group the venues of a region into rows of identical segment patterns. */
function laneRows(region, timeline) {
  const rows = new Map();
  for (const ex of region.exchanges) {
    const segs = timeline.filter((s) => s.exchange === ex.exchange).sort((a, b) => a.start - b.start);
    if (!segs.length) continue;
    const signature = segs.map((s) => `${s.kind}:${s.start}-${s.end}`).join('|');
    const row = rows.get(signature);
    if (row) row.names.push(ex.name);
    else rows.set(signature, { key: signature, names: [ex.name], segments: segs });
  }
  return [...rows.values()].map((row) => ({ ...row, candidates: labelCandidates(row.names) }));
}

/** What happens next in a region, as one German sentence. */
function laneSentence(region, now) {
  const exs = region.exchanges;
  const open = exs.filter((e) => e.phase === 'open');
  if (open.length) {
    const closing = open.filter((e) => isNum(e.next_close)).sort((a, b) => a.next_close - b.next_close)[0];
    if (!closing) return `${open[0].name} handelt rund um die Uhr`;
    return `${closing.name} schließt in ${fmtDuration(closing.next_close - now)}`;
  }
  const pause = exs.filter((e) => e.phase === 'break' && isNum(e.until)).sort((a, b) => a.until - b.until)[0];
  if (pause) return `${pause.name}: Mittagspause bis ${fmtTime(pause.until)}`;
  const next = exs.filter((e) => isNum(e.next_open)).sort((a, b) => a.next_open - b.next_open)[0];
  if (next) return `${next.name} öffnet in ${fmtDuration(next.next_open - now)}`;
  return 'Keine Handelszeiten bekannt';
}

/** Local full hours between t0 and t1 (epoch seconds). */
function hourTicks(t0, t1) {
  const d = new Date(t0 * 1000);
  d.setMinutes(0, 0, 0);
  if (d.getTime() / 1000 < t0) d.setHours(d.getHours() + 1);
  const ticks = [];
  while (d.getTime() / 1000 <= t1) {
    ticks.push(d.getTime() / 1000);
    d.setHours(d.getHours() + 1);
  }
  return ticks;
}

/**
 * @param {{sessions: import('../lib/types.js').SessionsOut, around?: string[]}} props
 *   ``around``: names of instruments that trade around the clock (crypto of the 24/7 venues CRYPTO and KRAKEN,
 *   which have no lane of their own; shown as a short footnote, the full list in its tooltip)
 */
export function SessionRibbon({ sessions, around = [] }) {
  useTick(15);
  const scrollerRef = useRef(null);
  const axisRef = useRef(null);
  const measureRef = useRef(null);
  const [trackPx, setTrackPx] = useState(1000);
  const [labelWidths, setLabelWidths] = useState(() => new Map());
  const t0 = sessions.timeline_start;
  const t1 = sessions.timeline_end;
  const span = Math.max(1, t1 - t0);
  const now = Math.min(t1, Math.max(t0, serverNow()));
  const pct = (ts) => ((Math.min(t1, Math.max(t0, ts)) - t0) / span) * 100;
  const nowPct = pct(now);

  const lanes = useMemo(
    () => sessions.regions.map((region) => ({ region, rows: laneRows(region, sessions.timeline) })),
    [sessions],
  );
  const ticks = useMemo(() => hourTicks(t0, t1), [t0, t1]);
  const labelTexts = useMemo(
    () => [...new Set(lanes.flatMap(({ rows }) => rows.flatMap((row) => row.candidates)))],
    [lanes],
  );
  const textKey = labelTexts.join('\n');

  // measure every label text in the label font (again once the web font has loaded)
  useLayoutEffect(() => {
    const measure = () => {
      const host = measureRef.current;
      if (!host) return;
      const widths = new Map();
      for (const el of host.children) widths.set(el.dataset.text, el.offsetWidth);
      setLabelWidths(widths);
    };
    measure();
    let alive = true;
    document.fonts?.ready?.then(() => { if (alive) measure(); });
    return () => { alive = false; };
  }, [textKey]);
  const widthPx = (text) => labelWidths.get(text) || text.length * LABEL_CHAR_PX + LABEL_PAD_PX;

  // keep the 'jetzt' line in view when the ribbon scrolls (narrow screens); remember the track width
  const centre = () => {
    if (axisRef.current) setTrackPx(axisRef.current.offsetWidth);
    const el = scrollerRef.current;
    if (!el || el.scrollWidth <= el.clientWidth) return;
    const label = el.querySelector('.rib-label')?.offsetWidth ?? 0;
    const track = el.scrollWidth - label;
    const visible = el.clientWidth - label;
    el.scrollLeft = Math.max(0, label + (nowPct / 100) * track - label - visible * 0.2);
  };
  useLayoutEffect(centre, []);
  useEffect(() => {
    const el = scrollerRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(centre);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return html`<section class="ribbon" aria-labelledby="rib-title">
    <div class="rib-head">
      <h2 id="rib-title">Der Handelstag wandert um die Welt</h2>
      <span class="rib-sub">Handelszeiten in Ihrer Ortszeit, ${fmtTime(t0)} bis ${fmtWeekday(t1)} ${fmtTime(t1)}</span>
      <span class="rib-legend" aria-hidden="true">
        <span><i class="key key-regular"></i>Handel</span>
        <span><i class="key key-ext"></i>Vor- und Nachbörse</span>
        <span><i class="key key-gap"></i>Pause</span>
      </span>
    </div>
    <div class="rib-scroll" ref=${scrollerRef}>
      <div class="rib-grid" style=${{ gridTemplateRows: `28px repeat(${lanes.length}, auto) 26px` }}>
        <div class="rib-corner"></div>
        <div class="rib-toptrack">
          <span class="rib-nowpill" style=${{ left: `${nowPct}%` }}>jetzt ${fmtTime(now)}</span>
        </div>
        ${lanes.map(({ region, rows }) => html`
          <div class="rib-label" key=${`l-${region.region}`}>
            <div class="rib-region">
              <span class=${`dot ${region.is_open ? 'dot-open' : 'dot-closed'}`} aria-hidden="true"></span>
              <span class="rib-region-name">${region.label}</span>
              <span class="rib-state">${region.is_open ? 'geöffnet' : 'geschlossen'}</span>
            </div>
            <div class="rib-count">${laneSentence(region, now)}</div>
          </div>
          <div class="rib-track" key=${`t-${region.region}`} role="list" aria-label=${`Handelszeiten ${region.label}`}>
            ${rows.length === 0 && html`<div class="rib-none">Keine Sitzung in diesem Zeitfenster</div>`}
            ${rows.map((row) => html`<${RibbonRow} key=${row.key} row=${row} pct=${pct} now=${now} trackPx=${trackPx} widthPx=${widthPx} />`)}
          </div>`)}
        <div class="rib-corner"></div>
        <div class="rib-axis" aria-hidden="true" ref=${axisRef}>
          ${ticks.map((t) => {
            const midnight = new Date(t * 1000).getHours() === 0;
            const show = midnight || new Date(t * 1000).getHours() % 2 === 0;
            return show ? html`<span class=${`rib-hour ${midnight ? 'rib-day' : ''}`} style=${{ left: `${pct(t)}%` }}>
              ${midnight ? fmtWeekday(t) : fmtHour(t)}</span>` : null;
          })}
        </div>
        <div class="rib-layer" aria-hidden="true">
          ${ticks.map((t) => html`<i class=${`rib-tick ${new Date(t * 1000).getHours() === 0 ? 'rib-tick-day' : ''}`} style=${{ left: `${pct(t)}%` }}></i>`)}
          <i class="rib-past" style=${{ width: `${nowPct}%` }}></i>
          <i class="rib-now" style=${{ left: `${nowPct}%` }}></i>
        </div>
      </div>
    </div>
    ${around.length > 0 && html`<p class="rib-foot" title=${around.join(', ')}>Rund um die Uhr handelbar: ${fmtList(around)}</p>`}
    <div class="rib-measure" aria-hidden="true" ref=${measureRef}>
      ${labelTexts.map((text) => html`<span class="seg-label" data-text=${text} key=${text}>${text}</span>`)}
    </div>
  </section>`;
}

function RibbonRow({ row, pct, now, trackPx, widthPx }) {
  const names = row.names.join(' · ');
  const describe = (s) => `${names}: ${KIND_LABEL[s.kind] || s.kind} ${fmtTime(s.start)}–${fmtTime(s.end)}`;
  const place = labelPlacement(row.segments, row.candidates, widthPx, pct, now, trackPx);
  const labelClass = `seg-label${place.outside ? ' seg-label-out' : ''}${place.align === 'end' ? ' seg-label-end' : ''}`;
  return html`<div class="rib-row" role="listitem" aria-label=${row.segments.map(describe).join('; ')}>
    ${row.segments.map((s) => {
      const active = s.start <= now && now < s.end;
      const left = pct(s.start);
      const width = Math.max(0.15, pct(s.end) - left);
      return html`<span class=${`seg seg-${s.kind} ${active ? 'seg-active' : ''}`} title=${describe(s)}
        style=${{ left: `${left}%`, width: `${width}%` }}></span>`;
    })}
    <span class=${labelClass} title=${place.text === names ? undefined : names}
      style=${{ left: `${place.left}%`, width: `${place.width}%` }}>${place.text}</span>
  </div>`;
}
