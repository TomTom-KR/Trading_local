// Outcome statistics: StatBlock summary, group tables, calibration chart and per-fold bars (SVG).

import { html } from '../vendor/preact-htm.mjs';
import {
  DASH, fmtNum, fmtR, fmtShare, isNum,
} from '../lib/format.js';
import { RValue } from './ui.js';

const interval = (lo, hi, fmt) => (isNum(lo) && isNum(hi) ? `${fmt(lo)} bis ${fmt(hi)}` : null);

/** Key figures of a set of closed signals; intervals are 95 %. */
export function StatBlockView({ title, stats, identity }) {
  const s = stats || {};
  const wr = interval(s.win_rate_lo, s.win_rate_hi, (v) => fmtShare(v));
  const ar = interval(s.avg_r_lo, s.avg_r_hi, (v) => fmtR(v, 2));
  return html`<section class="statblock" aria-label=${title}>
    <h3 class="statblock-title">${identity && html`<span class=${`swatch swatch-${identity}`} aria-hidden="true"></span>`}${title}</h3>
    <div class="statblock-n"><span class="figure figure-28">${fmtNum(s.n ?? 0, 0)}</span> <span class="muted">abgeschlossen</span></div>
    <dl class="statblock-grid">
      <div><dt>Trefferquote</dt><dd><span class="num">${fmtShare(s.win_rate)}</span>${wr && html`<span class="ci">95 %: ${wr}</span>`}</dd></div>
      <div><dt>Ø Ergebnis</dt><dd><${RValue} value=${s.avg_r} digits=${2} />${ar && html`<span class="ci">95 %: ${ar}</span>`}</dd></div>
      <div><dt>Summe</dt><dd><${RValue} value=${s.total_r} digits=${1} /></dd></div>
      <div><dt>Profitfaktor</dt><dd class="num">${isNum(s.profit_factor) ? fmtNum(s.profit_factor, 2) : DASH}</dd></div>
    </dl>
  </section>`;
}

/** Table of group statistics (per setup, region, direction, KI verdict). */
export function GroupTable({ title, groups }) {
  return html`<div class="gtable">
    <h3 class="gtable-title">${title}</h3>
    ${!groups?.length ? html`<p class="small muted">Noch keine abgeschlossenen Signale.</p>` : html`
    <table class="table">
      <thead><tr><th scope="col">Gruppe</th><th scope="col" class="r">n</th><th scope="col" class="r">Treffer</th><th scope="col" class="r">Ø R</th></tr></thead>
      <tbody>
        ${groups.map((g) => html`<tr>
          <td>${g.label_de || g.key}</td>
          <td class="r">${fmtNum(g.stats.n, 0)}</td>
          <td class="r">${fmtShare(g.stats.win_rate)}</td>
          <td class="r"><${RValue} value=${g.stats.avg_r} digits=${2} /></td>
        </tr>`)}
      </tbody>
    </table>`}
  </div>`;
}

/**
 * Calibration chart: predicted win probability (x) against observed win rate (y) per bin, with the
 * diagonal of perfect calibration; point area grows with n, n is written next to each point.
 * @param {{bins: import('../lib/types.js').CalibrationBin[], label?: string}} props
 */
export function CalibrationChart({ bins, label = 'Kalibrierung' }) {
  const W = 330;
  const H = 240;
  const m = { l: 44, r: 22, t: 12, b: 36 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const used = (bins || []).filter((b) => b.n > 0 && isNum(b.observed));
  const maxN = Math.max(1, ...used.map((b) => b.n));
  const x = (v) => m.l + v * iw;
  const y = (v) => m.t + (1 - v) * ih;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const summary = used.map((b) => `${fmtShare(b.predicted)} erwartet, ${fmtShare(b.observed)} eingetreten (n = ${b.n})`).join('; ');
  return html`<figure class="calib">
    <svg viewBox=${`0 0 ${W} ${H}`} role="img" aria-label=${`${label}: ${summary || 'noch keine Daten'}`}>
      ${ticks.map((t) => html`<g>
        <line class="grid" x1=${x(0)} x2=${x(1)} y1=${y(t)} y2=${y(t)} />
        <line class="grid" x1=${x(t)} x2=${x(t)} y1=${y(0)} y2=${y(1)} />
        <text class="axis" x=${m.l - 6} y=${y(t) + 4} text-anchor="end">${fmtShare(t)}</text>
        <text class="axis" x=${x(t)} y=${H - m.b + 16} text-anchor="middle">${fmtShare(t)}</text>
      </g>`)}
      <line class="diag" x1=${x(0)} y1=${y(0)} x2=${x(1)} y2=${y(1)} />
      ${used.map((b) => {
        const r = 3 + 6 * Math.sqrt(b.n / maxN);
        return html`<g>
          <circle class="pt" cx=${x(b.predicted)} cy=${y(b.observed)} r=${r} />
          <text class="nlabel" x=${x(b.predicted) + r + 3} y=${y(b.observed) + 4}>n = ${b.n}</text>
        </g>`;
      })}
      <text class="axis-title" x=${m.l + iw / 2} y=${H - 4} text-anchor="middle">erwartete Trefferquote</text>
      <text class="axis-title" transform=${`translate(11 ${m.t + ih / 2}) rotate(-90)`} text-anchor="middle">eingetreten</text>
    </svg>
    ${!used.length && html`<figcaption class="small muted">Noch keine abgeschlossenen Signale für die Kalibrierung.</figcaption>`}
  </figure>`;
}

/**
 * Vertical bars of a signed value per category (Ø R per fold) with n under each bar.
 * @param {{items: Array<{label: string, value: number | null, n: number}>, label: string}} props
 */
export function SignedBars({ items, label }) {
  const W = 420;
  const H = 220;
  const m = { l: 44, r: 8, t: 20, b: 44 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const values = items.map((i) => (isNum(i.value) ? i.value : 0));
  // headroom so value labels never collide with the axis labels
  const ext = Math.max(0.25, ...values.map((v) => Math.abs(v))) * 1.35;
  const y = (v) => m.t + ih / 2 - (v / ext) * (ih / 2);
  const bw = Math.min(48, (iw / Math.max(1, items.length)) * 0.6);
  const step = iw / Math.max(1, items.length);
  return html`<svg class="sbars" viewBox=${`0 0 ${W} ${H}`} role="img"
      aria-label=${`${label}: ${items.map((i) => `${i.label} ${isNum(i.value) ? fmtR(i.value, 2) : DASH} (n = ${i.n})`).join('; ')}`}>
    <line class="grid" x1=${m.l} x2=${W - m.r} y1=${y(ext)} y2=${y(ext)} />
    <line class="grid" x1=${m.l} x2=${W - m.r} y1=${y(-ext)} y2=${y(-ext)} />
    <text class="axis" x=${m.l - 6} y=${y(ext) + 4} text-anchor="end">${fmtR(ext, 1)}</text>
    <text class="axis" x=${m.l - 6} y=${y(-ext) + 4} text-anchor="end">${fmtR(-ext, 1)}</text>
    <line class="zero" x1=${m.l} x2=${W - m.r} y1=${y(0)} y2=${y(0)} />
    ${items.map((it, i) => {
      const v = isNum(it.value) ? it.value : 0;
      const cx = m.l + step * i + step / 2;
      const top = Math.min(y(v), y(0));
      const h = Math.max(1, Math.abs(y(v) - y(0)));
      return html`<g>
        <rect class=${v >= 0 ? 'bar-up' : 'bar-down'} x=${cx - bw / 2} y=${top} width=${bw} height=${h} rx="2" />
        <text class="vlabel" x=${cx} y=${v >= 0 ? top - 4 : top + h + 12} text-anchor="middle">${isNum(it.value) ? fmtR(it.value, 2) : DASH}</text>
        <text class="axis" x=${cx} y=${H - m.b + 18} text-anchor="middle">${it.label}</text>
        <text class="axis" x=${cx} y=${H - m.b + 32} text-anchor="middle">n = ${it.n}</text>
      </g>`;
    })}
  </svg>`;
}
