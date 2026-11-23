// One news item: time, source, linked title, LLM relevance pips, sentiment bar, affected symbols and
// the German summary. Items whose analysis is still pending say so.

import { html } from '../vendor/preact-htm.mjs';
import { fmtWhen } from '../lib/format.js';
import { NEWS_STATUS_PENDING, horizonLabel, newsCategoryLabel } from '../lib/labels.js';
import { symbolHref } from '../lib/router.js';
import { Pips, Sentiment } from './ui.js';

/** @param {{item: import('../lib/types.js').NewsOut, now: number, full?: boolean}} props */
export function NewsItem({ item, now, full = false }) {
  const a = item.analysis;
  const when = item.published_ts ?? item.fetched_ts;
  const pending = !a && NEWS_STATUS_PENDING.has(item.status);
  const symbols = a?.symbols?.length ? a.symbols : item.hinted_symbols.map((s) => ({ symbol: s, sentiment: null }));
  return html`<article class="news">
    <div class="news-when small">
      <time datetime=${new Date(when * 1000).toISOString()}>${fmtWhen(when, now)}</time>
      <span class="muted truncate">${item.source_name || item.source}</span>
    </div>
    <div class="news-main">
      <h3 class="news-title"><a href=${item.url} target="_blank" rel="noopener noreferrer">${item.title}</a></h3>
      ${a && html`<p class=${`news-summary prose ${full ? '' : 'clamp-2'}`}>${a.summary_de}</p>`}
      ${pending && html`<p class="news-pending small muted"><span class="dot dot-warn" aria-hidden="true"></span>Analyse läuft</p>`}
      ${!a && !pending && html`<p class="small faint">Nicht analysiert${item.status === 'skipped' ? ' (für die Beobachtungsliste nicht relevant)' : ''}.</p>`}
      ${full && !a && item.summary && html`<p class="news-summary prose muted">${item.summary}</p>`}
    </div>
    <div class="news-marks">
      ${a && html`<${Pips} value=${a.relevance} /><${Sentiment} value=${a.sentiment} />
        <span class="small muted">${newsCategoryLabel(a.category)} · ${horizonLabel(a.horizon)}</span>`}
    </div>
    <div class="news-syms">
      ${symbols.slice(0, full ? 8 : 4).map((s) => html`<a class=${`chip ${s.sentiment > 0.15 ? 'chip-up' : s.sentiment < -0.15 ? 'chip-down' : ''}`}
          href=${symbolHref(s.symbol)} title=${s.sentiment === null ? 'erwähnt' : `Stimmung ${s.sentiment > 0 ? '+' : ''}${s.sentiment.toFixed(2).replace('.', ',')}`}>
          ${s.sentiment > 0.15 ? html`<span class="glyph-up" aria-hidden="true">▲</span>` : s.sentiment < -0.15 ? html`<span class="glyph-down" aria-hidden="true">▼</span>` : ''}${s.symbol}</a>`)}
    </div>
  </article>`;
}
