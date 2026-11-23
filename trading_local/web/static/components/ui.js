// Small presentational building blocks shared by all views.

import { html, useEffect, useRef, useState } from '../vendor/preact-htm.mjs';
import { errorText } from '../lib/api.js';
import {
  DASH, fmtPctSigned, fmtR, fmtShare, isNum,
} from '../lib/format.js';
import {
  directionGlyph, directionLabel, phaseLabel, statusReasonLabel, verdictLabel,
} from '../lib/labels.js';

/** '▲ Long' / '▼ Short' badge: direction is never colour alone. */
export function DirectionBadge({ direction }) {
  const short = direction === 'SHORT';
  return html`<span class=${`dir ${short ? 'dir-short' : 'dir-long'}`}>
    <span class="glyph" aria-hidden="true">${directionGlyph(direction)}</span>${directionLabel(direction)}
  </span>`;
}

/** Signed change in percent, coloured and with its sign (the only text that wears a series colour). */
export function Change({ value, digits = 1, suffix }) {
  if (!isNum(value)) return html`<span class="muted">${DASH}</span>`;
  const cls = value > 0 ? 'up' : value < 0 ? 'down' : 'muted';
  return html`<span class=${`num ${cls}`}>${suffix ? suffix(value) : fmtPctSigned(value, digits)}</span>`;
}

/** Signed R value, coloured. */
export function RValue({ value, digits = 1 }) {
  return html`<${Change} value=${value} suffix=${(v) => fmtR(v, digits)} />`;
}

/** Session state dot with an accessible label. */
export function PhaseDot({ phase }) {
  return html`<span class=${`dot dot-${phase || 'closed'}`} role="img" aria-label=${phaseLabel(phase)} title=${phaseLabel(phase)}></span>`;
}

/** Thin 0..1 bar with the percentage in front. */
export function Confidence({ value }) {
  if (!isNum(value)) return html`<span class="muted">${DASH}</span>`;
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return html`<span class="row" style=${{ gap: '8px', display: 'inline-flex' }}>
    <span class="num">${fmtShare(value)}</span>
    <span class="meter" aria-hidden="true"><span style=${{ width: `${pct}%` }}></span></span>
  </span>`;
}

/** LLM relevance 1..5 as five small pips. */
export function Pips({ value }) {
  const n = Math.max(0, Math.min(5, Math.round(value || 0)));
  return html`<span class="pips" role="img" aria-label=${`Relevanz ${n} von 5`} title=${`Relevanz ${n} von 5`}>
    ${[1, 2, 3, 4, 5].map((i) => html`<i class=${i <= n ? 'on' : ''}></i>`)}
  </span>`;
}

/** Sentiment -1..1 as a short diverging bar around zero. */
export function Sentiment({ value }) {
  if (!isNum(value)) return html`<span class="muted">${DASH}</span>`;
  const v = Math.max(-1, Math.min(1, value));
  const width = Math.abs(v) * 50;
  const style = v >= 0 ? { left: '50%', width: `${width}%` } : { left: `${50 - width}%`, width: `${width}%` };
  const word = v > 0.15 ? 'positiv' : v < -0.15 ? 'negativ' : 'neutral';
  const text = `Stimmung ${word} (${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(2).replace('.', ',')})`;
  return html`<span class="senti" role="img" aria-label=${text} title=${text}>
    <span class=${v >= 0 ? 'pos' : 'neg'} style=${style}></span>
  </span>`;
}

/** KI review verdict as a quiet chip. */
export function Verdict({ verdict, conviction }) {
  const label = verdictLabel(verdict);
  const extra = isNum(conviction) ? ` · Überzeugung ${fmtShare(conviction)}` : '';
  return html`<span class="chip chip-quiet" title=${`KI-Prüfung: ${label}${extra}`}>KI: ${label}</span>`;
}

/** Status text of a signal including its result in R where that makes sense. */
function signalStatusText(signal) {
  switch (signal.status) {
    case 'review':
      return 'Prüfung läuft';
    case 'pending_entry':
      return 'wartet auf Einstieg';
    case 'active':
      return isNum(signal.open_r) ? `aktiv ${fmtR(signal.open_r)}` : 'aktiv';
    case 'target':
      return `Ziel erreicht ${fmtR(signal.outcome_r)}`;
    case 'stopped':
      return `Stopp ${fmtR(signal.outcome_r)}`;
    case 'expired':
      return isNum(signal.outcome_r) ? `abgelaufen ${fmtR(signal.outcome_r)}` : 'abgelaufen';
    case 'cancelled':
      return 'verworfen';
    default:
      return signal.status;
  }
}

/** Status chip of a signal: tinted by the sign of its (open) result, the text says it in words. */
export function SignalStatus({ signal }) {
  const r = signal.status === 'active' ? signal.open_r : signal.outcome_r;
  const tone = ['active', 'target', 'stopped', 'expired'].includes(signal.status) && isNum(r)
    ? r > 0 ? 'chip-up' : r < 0 ? 'chip-down' : 'chip-solid'
    : 'chip-solid';
  return html`<span class=${`chip ${tone}`} title=${statusReasonLabel(signal.status_reason)}>${signalStatusText(signal)}</span>`;
}

/** Empty state written as direction: what will appear here and when. */
export function Empty({ title, children }) {
  return html`<div class="state" role="status"><strong>${title}</strong>${children}</div>`;
}

/** Error state with the German message and an optional retry button. */
export function ErrorState({ error, onRetry, title = 'Daten konnten nicht geladen werden.' }) {
  return html`<div class="state state-error" role="alert">
    <strong>${title}</strong>
    <span>${typeof error === 'string' ? error : errorText(error)}</span>
    ${onRetry && html`<div><button type="button" class="btn btn-sm" onClick=${onRetry}>Erneut versuchen</button></div>`}
  </div>`;
}

/** Placeholder lines while the first load is running. */
export function Loading({ lines = 3, label = 'Wird geladen …' }) {
  return html`<div aria-busy="true" aria-label=${label}>
    ${Array.from({ length: lines }, (_, i) => html`<div class="skeleton" style=${{ width: `${92 - i * 14}%` }}></div>`)}
  </div>`;
}

/** Tabs with arrow-key navigation (role=tablist). */
export function Tabs({ tabs, value, onChange, label }) {
  const onKey = (e) => {
    const i = tabs.findIndex((t) => t.value === value);
    let next = null;
    if (e.key === 'ArrowRight') next = tabs[(i + 1) % tabs.length];
    if (e.key === 'ArrowLeft') next = tabs[(i - 1 + tabs.length) % tabs.length];
    if (next) {
      e.preventDefault();
      onChange(next.value);
      // currentTarget is reset to null when the dispatch ends: keep the list for the deferred focus move
      const list = e.currentTarget;
      requestAnimationFrame(() => list.querySelector('[aria-selected="true"]')?.focus());
    }
  };
  return html`<div class="tabs" role="tablist" aria-label=${label} onKeyDown=${onKey}>
    ${tabs.map((t) => html`<button type="button" role="tab" aria-selected=${t.value === value ? 'true' : 'false'}
        tabindex=${t.value === value ? 0 : -1} onClick=${() => onChange(t.value)}>
        ${t.label}${isNum(t.count) && html`<span class="count">${t.count}</span>`}
      </button>`)}
  </div>`;
}

/** Segmented buttons (aria-pressed) for small exclusive choices. */
export function Segmented({ options, value, onChange, label }) {
  return html`<div class="segmented" role="group" aria-label=${label}>
    ${options.map((o) => html`<button type="button" aria-pressed=${o.value === value ? 'true' : 'false'}
        onClick=${() => onChange(o.value)} title=${o.title || ''}>${o.label}</button>`)}
  </div>`;
}

/** A price cell that tints toward up/down for 600 ms whenever the value changes. */
export function FlashValue({ value, children, class: cls = '' }) {
  const prev = useRef(value);
  const [flash, setFlash] = useState('');
  useEffect(() => {
    if (isNum(prev.current) && isNum(value) && value !== prev.current) {
      setFlash(value > prev.current ? 'flash-up' : 'flash-down');
      const t = setTimeout(() => setFlash(''), 650);
      prev.current = value;
      return () => clearTimeout(t);
    }
    prev.current = value;
    return undefined;
  }, [value]);
  return html`<span class=${`${cls} ${flash}`}>${children}</span>`;
}

/** Short-lived confirmation message. */
export function Toast({ message, onDone, ms = 3500 }) {
  useEffect(() => {
    if (!message) return undefined;
    const t = setTimeout(onDone, ms);
    return () => clearTimeout(t);
  }, [message]);
  return message ? html`<div class="toast" role="status">${message}</div>` : null;
}
