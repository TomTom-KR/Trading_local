// Label placement of the session ribbon (pure functions).
//
// A ribbon row is one or more venues with identical sessions ('Xetra · London · Paris'). Its label goes on
// its bar when the bar is wide enough; otherwise right next to the bar (after it, or before it and aligned
// to the bar's start) inside the visible timeline; when the full list of names fits nowhere, a shorter
// label ('Xetra · London +4') is tried. Widths are measured in px by the caller, positions are percent of
// the track. A label is never pushed beyond either end of the timeline.

const SLACK_PX = 1; // sub-pixel rounding of the measured width must not trigger the ellipsis

/**
 * Label texts of a row, longest first: all names, then the first names with '+N' for the rest, down to the
 * first name alone ('Xetra · London · Paris', 'Xetra · London +1', 'Xetra +2', 'Xetra').
 * @param {string[]} names
 */
export function labelCandidates(names) {
  const out = [names.join(' · ')];
  for (let shown = names.length - 1; shown >= 1; shown -= 1) {
    out.push(`${names.slice(0, shown).join(' · ')} +${names.length - shown}`);
  }
  if (names.length > 1) out.push(names[0]);
  return [...new Set(out)];
}

/**
 * Where a row's label goes.
 * @param {Array<{start: number, end: number, kind: string}>} segments the row's segments (epoch seconds)
 * @param {string[]} candidates label texts, longest first (``labelCandidates``)
 * @param {(text: string) => number} widthPx rendered width of a label text in px, padding included
 * @param {(ts: number) => number} pct time -> percent of the track, clamped to 0..100
 * @param {number} now epoch seconds
 * @param {number} trackPx width of the track in px
 * @returns {{text: string, left: number, width: number, outside: boolean, align: 'start' | 'end'}}
 *   ``left`` / ``width`` in percent of the track; ``align: 'end'`` puts the text against the right edge
 *   (labels before their bar end exactly at the bar's start)
 */
export function labelPlacement(segments, candidates, widthPx, pct, now, trackPx) {
  const regular = segments.filter((s) => s.kind === 'regular');
  const pool = regular.length ? regular : segments;
  // obstacles for labels beside a bar: the row's other bars (hatched pre/post windows may be written over)
  const bars = pool.map((s) => ({ seg: s, start: pct(s.start), end: pct(s.end) }));
  const ahead = (s) => s.end - Math.max(s.start, Math.min(now, s.end)); // seconds of the segment still to come
  // host order: most time still to come, then the longer segment, then the earlier one (stable sort)
  const hosts = [...bars].sort((a, b) => ahead(b.seg) - ahead(a.seg) || (b.seg.end - b.seg.start) - (a.seg.end - a.seg.start));
  const nowPct = pct(now);
  const px = Math.max(1, trackPx);
  const need = (text) => ((widthPx(text) + SLACK_PX) / px) * 100;

  for (const text of candidates) {
    const width = need(text);
    for (const host of hosts) {
      const spot = spotBeside(host, bars, width, nowPct);
      if (spot) return { text, width, ...spot };
    }
  }
  // nothing fits: the shortest text at its host, shifted so it stays inside the timeline
  const text = candidates[candidates.length - 1];
  const width = Math.min(100, need(text));
  const host = hosts[0];
  const anchor = host ? Math.min(host.end, Math.max(host.start, nowPct)) : 0;
  const at = Math.min(Math.max(0, anchor), 100 - width);
  return { text, left: at, width, outside: !host || at < host.start || at + width > host.end, align: 'start' };
}

/** On the bar (preferably its part from 'now' on), after it, or before it; null when there is no room. */
function spotBeside(host, bars, width, nowPct) {
  const from = Math.min(host.end, Math.max(host.start, nowPct));
  if (host.end - from >= width) return { left: from, outside: false, align: 'start' };
  if (host.end - host.start >= width) return { left: host.end - width, outside: false, align: 'start' };
  const others = bars.filter((b) => b !== host);
  const until = Math.min(100, ...others.filter((b) => b.start >= host.end).map((b) => b.start));
  if (until - host.end >= width) return { left: host.end, outside: true, align: 'start' };
  const since = Math.max(0, ...others.filter((b) => b.end <= host.start).map((b) => b.end));
  if (host.start - since >= width) return { left: host.start - width, outside: true, align: 'end' };
  return null;
}
