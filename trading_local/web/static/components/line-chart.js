// Preact wrapper around LineChart (equity curves, cumulative R).

import { html, useEffect, useRef } from '../vendor/preact-htm.mjs';
import { LineChart } from '../lib/charts.js';
import { useStore } from '../lib/store.js';
import { themeStore } from '../lib/theme.js';

/**
 * @param {{series: import('../lib/charts.js').LineSpec[], height?: number, format?: (v: number) => string,
 *   compact?: boolean, label: string}} props
 */
export function LineChartView({ series, height = 280, format, compact = false, label }) {
  const hostRef = useRef(null);
  const chartRef = useRef(null);
  const theme = useStore(themeStore, (s) => s.theme);

  useEffect(() => {
    const chart = new LineChart(hostRef.current, { format, compact });
    chartRef.current = chart;
    return () => {
      chart.destroy();
      chartRef.current = null;
    };
  }, []);
  useEffect(() => chartRef.current?.setSeries(series), [series]);
  useEffect(() => chartRef.current?.applyTheme(), [theme]);

  return html`<div class="lchart" ref=${hostRef} style=${{ height: `${height}px` }} role="img" aria-label=${label}></div>`;
}
