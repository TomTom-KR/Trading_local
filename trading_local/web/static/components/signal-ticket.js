// One recommendation as a ticket: direction, instrument, levels with distances, chance/risk,
// confidence, KI review, status, age, delay note, 'Übernehmen' and a link to the detail.

import { html } from '../vendor/preact-htm.mjs';
import { instrumentOf } from '../lib/data.js';
import {
  fmtAge, fmtNum, fmtPctSigned, fmtPrice, priceUnit,
} from '../lib/format.js';
import { regionLabel, shadowReasonLabel } from '../lib/labels.js';
import { symbolHref } from '../lib/router.js';
import { OPEN_STATUSES, signalLevels, timeframeLabel } from '../lib/signals.js';
import { followSignal } from './ticket-drawer.js';
import {
  Confidence, DirectionBadge, SignalStatus, Verdict,
} from './ui.js';

/** @param {{signal: import('../lib/types.js').SignalOut, now: number, fresh?: boolean}} props */
export function SignalTicket({ signal, now, fresh = false }) {
  const inst = instrumentOf(signal.symbol);
  const assetClass = inst?.asset_class;
  const unit = priceUnit(signal.currency || inst?.currency || '', inst?.price_scale ?? 1);
  const { basis, stop, target } = signalLevels(signal);
  const pctOf = (lvl) => (basis ? ((lvl - basis) / basis) * 100 : null);
  const open = OPEN_STATUSES.has(signal.status);
  const canFollow = open && signal.actionable && inst?.tradable !== false;
  return html`<article class=${`sig ${fresh ? 'sig-fresh' : ''} ${signal.actionable ? '' : 'sig-shadow'}`} aria-label=${`Signal ${signal.symbol}`}>
    <header class="sig-head">
      <${DirectionBadge} direction=${signal.direction} />
      <a class="sig-sym" href=${symbolHref(signal.symbol)}>${signal.symbol}</a>
      <span class="sig-name truncate">${signal.name}</span>
      <span class="sig-age faint small" title=${new Date(signal.created_ts * 1000).toLocaleString('de-DE')}>${fmtAge(signal.created_ts, now)}</span>
    </header>
    <div class="sig-meta small muted">
      ${signal.setup_label_de || signal.setup} · ${timeframeLabel(signal.timeframe)} · ${regionLabel(signal.region)}
      ${signal.delay_min > 0 && ` · Daten ${signal.delay_min} Min verzögert`}
      ${!signal.actionable && html` · <span class="sig-shadow-note">verworfener Kandidat${signal.shadow_reason ? `: ${shadowReasonLabel(signal.shadow_reason)}` : ''}</span>`}
    </div>
    <dl class="sig-levels num">
      <div><dt>${signal.entry_price ? 'Einstieg' : 'Referenz'}</dt><dd>${fmtPrice(basis, assetClass)} <span class="faint">${unit}</span></dd></div>
      <div><dt>Stopp</dt><dd>${fmtPrice(stop, assetClass)} <span class="muted">${fmtPctSigned(pctOf(stop), 1)}</span></dd></div>
      <div><dt>Ziel</dt><dd>${fmtPrice(target, assetClass)} <span class="muted">${fmtPctSigned(pctOf(target), 1)}</span></dd></div>
      <div><dt>Chance/Risiko</dt><dd>${fmtNum(signal.rr, 1)}</dd></div>
      <div><dt>Konfidenz</dt><dd><${Confidence} value=${signal.confidence} /></dd></div>
    </dl>
    <footer class="sig-foot">
      <${SignalStatus} signal=${signal} />
      <${Verdict} verdict=${signal.llm_verdict} conviction=${signal.llm_conviction} />
      <span class="sig-actions">
        <a class="btn btn-sm" href=${`#/signale/${signal.id}`}>Details</a>
        ${canFollow && html`<button type="button" class="btn btn-sm btn-primary" onClick=${() => followSignal(signal)}>Übernehmen</button>`}
      </span>
    </footer>
  </article>`;
}
