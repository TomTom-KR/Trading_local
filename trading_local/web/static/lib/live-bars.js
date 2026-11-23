// What a live 1-minute bar does to a candlestick chart (pure function, used by components/price-chart.js).

/** Bucket length in seconds of the intraday timeframes. */
export const TF_SECONDS = { '1m': 60, '5m': 300, '15m': 900, '1h': 3600 };

/**
 * How a live 1-minute bar that opened at ``barTs`` changes a chart of ``timeframe`` whose newest candle opened
 * at ``lastTs``:
 * - 'refresh' (daily chart): reload the newest daily candles from the server. A daily candle carries the venue's
 *   trading date at 00:00 UTC, which only the server knows; the first hour of a Sydney session lies before that
 *   date's UTC midnight, so the bar's time says nothing about the candle it belongs to.
 * - 'ignore' (intraday): the bar belongs to a bucket older than the newest candle.
 * - 'merge' (intraday): the bar extends the newest bucket or starts the next one (``bucket`` = its start).
 * @param {string} timeframe '1m' | '5m' | '15m' | '1h' | '1d'
 * @param {number} barTs open time of the minute bar (epoch seconds)
 * @param {number} lastTs open time of the chart's newest candle (epoch seconds)
 * @returns {{action: 'refresh' | 'ignore' | 'merge', bucket: number | null}}
 */
export function liveBarAction(timeframe, barTs, lastTs) {
  const step = TF_SECONDS[timeframe];
  if (!step) return { action: 'refresh', bucket: null };
  const bucket = Math.floor(barTs / step) * step;
  return { action: bucket < lastTs ? 'ignore' : 'merge', bucket };
}
