// Signal helpers shared by the views (pure functions).

import { DASH, fmtDuration, isNum } from './format.js';

export const OPEN_STATUSES = new Set(['review', 'pending_entry', 'active']);

// Regular trading minutes per session the engine assumes when it knows no session length
// (trading_local/analysis/params.py default_session_minutes).
const DEFAULT_SESSION_MINUTES = 390;

/**
 * Maximum holding period in words. ``horizon_min`` counts TRADING minutes (the time stop skips nights,
 * weekends and lunch breaks), so it is never shown as wall-clock time: daily signals in trading days of the
 * decision day's session length ('ca. 5 Handelstage'), intraday signals as trading time.
 * @param {import('./types.js').SignalOut & {features?: Record<string, unknown>}} signal
 */
export function holdingPeriodText(signal) {
  const horizon = signal.horizon_min;
  if (!isNum(horizon)) return DASH;
  if (signal.intraday) return `${fmtDuration(horizon * 60)} Handelszeit, intraday`;
  const f = signal.features || {};
  const session = Number(f.minutes_since_open) + Number(f.minutes_to_close);
  const days = Math.max(1, Math.round(horizon / (isNum(session) && session > 0 ? session : DEFAULT_SESSION_MINUTES)));
  return days === 1 ? 'ca. 1 Handelstag' : `ca. ${days} Handelstage`;
}

/** German label of a signal timeframe. */
export const timeframeLabel = (tf) => ({ '1m': '1 Min', '5m': '5 Min', '15m': '15 Min', '1h': '1 Std', '1d': 'Tag' }[tf] || tf);

/**
 * Planned levels of a signal: from the entry once entered, otherwise from the reference price.
 * @param {import('./types.js').SignalOut} signal
 */
export function signalLevels(signal) {
  const sign = signal.direction === 'SHORT' ? -1 : 1;
  const basis = signal.entry_price ?? signal.ref_price;
  return {
    basis,
    stop: signal.stop_price ?? basis - sign * signal.stop_dist,
    target: signal.target_price ?? basis + sign * signal.target_dist,
  };
}

/**
 * Signals with the unrealised result of ACTIVE ones recomputed from the latest quotes (before costs).
 * @param {import('./types.js').SignalOut[]} signals
 * @param {Record<string, import('./types.js').QuoteOut>} quotes
 */
export function withLiveR(signals, quotes) {
  return signals.map((s) => {
    const q = quotes[s.symbol];
    if (s.status !== 'active' || !isNum(s.entry_price) || !q || !(s.stop_dist > 0)) return s;
    const sign = s.direction === 'SHORT' ? -1 : 1;
    return { ...s, last_price: q.price, open_r: ((q.price - s.entry_price) * sign) / s.stop_dist };
  });
}

/** Insert or replace a signal in a newest-first list, keeping at most ``limit`` entries. */
export function upsertSignal(list, signal, limit = 50) {
  const rest = list.filter((s) => s.id !== signal.id);
  return [signal, ...rest].sort((a, b) => b.created_ts - a.created_ts || b.id - a.id).slice(0, limit);
}
