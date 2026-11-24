// Pure helpers of the agenda view (#/termine): grouping by local day, day titles, published figures.

import { DASH, fmtDate, fmtWeekday } from './format.js';

/** Local calendar day of a timestamp as 'YYYY-M-D' (grouping key). */
function dayKey(ts) {
  const d = new Date(ts * 1000);
  return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`;
}

/**
 * Events grouped by local day, in time order: [{key, ts, events}].
 * @param {import('./types.js').AgendaEventOut[]} events
 */
export function groupByDay(events) {
  const days = [];
  for (const e of [...events].sort((a, b) => a.event_ts - b.event_ts || a.id - b.id)) {
    const key = dayKey(e.event_ts);
    if (!days.length || days[days.length - 1].key !== key) days.push({ key, ts: e.event_ts, events: [] });
    days[days.length - 1].events.push(e);
  }
  return days;
}

/** Local midnight of a timestamp, in milliseconds. */
function startOfDay(ts) {
  const d = new Date(ts * 1000);
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
}

/** German heading of a day relative to today: 'Heute, Di 24.11.2026'. */
export function dayTitle(ts, now) {
  const offset = Math.round((startOfDay(ts) - startOfDay(now)) / 86400000);
  const word = { [-1]: 'Gestern', 0: 'Heute', 1: 'Morgen' }[offset];
  return `${word ? `${word}, ` : ''}${fmtWeekday(ts)} ${fmtDate(ts)}`;
}

/** The value a published figure is shown with: '0,3 %' style German decimals, '–' when missing. */
export function figure(text) {
  if (!text) return DASH;
  return text.replace(/(\d)\.(\d)/g, '$1,$2').replace(/(\d)%/, '$1 %');
}
