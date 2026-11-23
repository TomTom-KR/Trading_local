// Tiny inline SVG line of the current session's closes; the dashed base line is the previous close.

import { html } from '../vendor/preact-htm.mjs';
import { isNum } from '../lib/format.js';

/**
 * @param {{points?: Array<[number, number]>, prevClose?: number | null, width?: number, height?: number}} props
 */
export function Sparkline({ points, prevClose, width = 72, height = 22 }) {
  if (!points || points.length < 2) return html`<svg class="spark" width=${width} height=${height} aria-hidden="true"></svg>`;
  const values = points.map((p) => p[1]);
  const ref = isNum(prevClose) ? prevClose : values[0];
  const lo = Math.min(ref, ...values);
  const hi = Math.max(ref, ...values);
  const range = hi - lo || 1;
  const t0 = points[0][0];
  const tspan = points[points.length - 1][0] - t0 || 1;
  const x = (t) => ((t - t0) / tspan) * (width - 2) + 1;
  const y = (v) => height - 2 - ((v - lo) / range) * (height - 4);
  const last = values[values.length - 1];
  const stroke = last > ref ? 'var(--up)' : last < ref ? 'var(--down)' : 'var(--ink-3)';
  return html`<svg class="spark" width=${width} height=${height} viewBox=${`0 0 ${width} ${height}`} aria-hidden="true">
    <line class="base" x1="0" x2=${width} y1=${y(ref)} y2=${y(ref)} />
    <polyline points=${points.map((p) => `${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(' ')} stroke=${stroke} />
  </svg>`;
}
