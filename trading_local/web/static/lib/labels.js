// German labels for the API's enum values. Unknown values fall back to the raw value, so a new backend
// value shows up readable instead of breaking the page.

const pick = (table, key, fallback) => table[key] ?? fallback ?? key ?? '';

export const REGION_LABELS = { ASIA: 'Asien', EU: 'Europa', US: 'USA', GLOBAL: 'Global' };
export const regionLabel = (r) => pick(REGION_LABELS, r);

export const ASSET_CLASS_LABELS = {
  equity: 'Aktie', index: 'Index', crypto: 'Krypto', fx: 'Devisen', future: 'Future',
};
export const assetClassLabel = (a) => pick(ASSET_CLASS_LABELS, a);

const PHASE_LABELS = {
  open: 'geöffnet', break: 'Mittagspause', pre: 'Vorbörse', post: 'Nachbörse', closed: 'geschlossen',
};
export const phaseLabel = (p) => pick(PHASE_LABELS, p);

export const directionLabel = (d) => (d === 'SHORT' ? 'Short' : 'Long');
export const directionGlyph = (d) => (d === 'SHORT' ? '▼' : '▲');
export const directionVerb = (d) => (d === 'SHORT' ? 'Leerverkaufen' : 'Kaufen');

const VERDICT_LABELS = {
  confirm: 'bestätigt', reduce: 'reduziert', veto: 'Veto', unavailable: 'ohne Prüfung',
};
/** KI review verdict; a missing verdict means the review has not happened (yet). */
export const verdictLabel = (v) => (v ? pick(VERDICT_LABELS, v) : 'ohne Prüfung');

const ORDER_STATUS_LABELS = {
  pending: 'offen', filled: 'ausgeführt', cancelled: 'storniert', rejected: 'abgelehnt',
};
export const orderStatusLabel = (s) => pick(ORDER_STATUS_LABELS, s);

const ORDER_ORIGIN_LABELS = {
  manual: 'manuell', signal: 'KI-Signal', stop: 'Stopp', target: 'Ziel', time_stop: 'Zeitstopp',
  session_close: 'Handelsschluss', reset: 'Neustart',
};
export const orderOriginLabel = (o) => pick(ORDER_ORIGIN_LABELS, o);

const EXIT_REASON_LABELS = {
  manual: 'manuell geschlossen', stop: 'Stopp ausgelöst', target: 'Ziel erreicht', time_stop: 'Haltedauer abgelaufen',
  session_close: 'zum Handelsschluss glattgestellt', reset: 'beim Neustart geschlossen',
};
export const exitReasonLabel = (r) => pick(EXIT_REASON_LABELS, r, r || '–');

// Why a candidate was not issued (Signal.shadow_reason, trading_local/analysis/engine.py SHADOW_*).
const SHADOW_REASON_LABELS = {
  unter_schwelle: 'Score unter der Schwelle',
  erwartungswert: 'Erwartungswert nach Slippage nicht positiv',
  llm_veto: 'Veto der KI-Prüfung',
  llm_nicht_verfuegbar: 'KI-Prüfung nicht verfügbar',
  markt_geschlossen: 'Börse bei Freigabe geschlossen',
  daten_veraltet: 'Kursdaten bei Freigabe veraltet',
  termin: 'Wichtiger Termin steht an',
};
/** German reason of a shadow candidate; '' when there is none. */
export const shadowReasonLabel = (r) => (r ? pick(SHADOW_REASON_LABELS, r) : '');

// How a signal ended (Signal.status_reason: the exit reasons of trading_local/execution.py and the
// cancellation reasons of trading_local/analysis/tracking.py).
const STATUS_REASON_LABELS = {
  stop: 'Stopp ausgelöst',
  target: 'Ziel erreicht',
  time_stop: 'Haltedauer abgelaufen',
  session_close: 'vor Handelsschluss glattgestellt',
  kein_einstieg: 'kein Einstieg rechtzeitig',
  kurs_entlaufen: 'Kurs vor dem Einstieg davongelaufen',
};
/** German end reason of a signal; '' when there is none. */
export const statusReasonLabel = (r) => (r ? pick(STATUS_REASON_LABELS, r) : '');

const NEWS_CATEGORY_LABELS = {
  earnings: 'Zahlen', guidance: 'Prognose', m_and_a: 'Übernahme', analyst: 'Analysten', macro: 'Konjunktur',
  regulation_legal: 'Recht & Regulierung', product: 'Produkt', management: 'Management', other: 'Sonstiges',
};
export const newsCategoryLabel = (c) => pick(NEWS_CATEGORY_LABELS, c);

const HORIZON_LABELS = { intraday: 'heute', days: 'Tage', weeks: 'Wochen' };
export const horizonLabel = (h) => pick(HORIZON_LABELS, h);

export const NEWS_REGION_LABELS = { ASIA: 'Asien', DE: 'Deutschland', EU: 'Europa', US: 'USA', GLOBAL: 'Global' };

// Score components of the signal engine (trading_local/analysis/params.py DEFAULT_WEIGHTS).
const COMPONENT_LABELS = {
  trend: 'Trend', momentum: 'Momentum', volume: 'Volumen', structure: 'Kursstruktur', market: 'Marktumfeld',
  news: 'Nachrichtenlage', volatility: 'Volatilität',
};
export const componentLabel = (k) => pick(COMPONENT_LABELS, k);

// Feature snapshot stored with every signal (trading_local/analysis/features.py FEATURE_COLUMNS; the
// engine decides on 5-minute buckets and reads the trend from 15-minute bars).
const FEATURE_LABELS = {
  price: 'Kurs', session: 'Handelstag', minutes_since_open: 'Minuten seit Eröffnung',
  minutes_to_close: 'Minuten bis Handelsschluss', in_session: 'In der Handelszeit',
  atr_fast: 'ATR 14 (5 Min)', atr_slow: 'ATR 14 (15 Min)', atr_daily: 'ATR 14 (Tag)',
  ema_fast_f: 'EMA 20 (5 Min)', ema_slow_f: 'EMA 50 (5 Min)', ema_fast_s: 'EMA 20 (15 Min)', ema_slow_s: 'EMA 50 (15 Min)',
  ema_fast_s_slope: 'Steigung EMA 20 (15 Min)', rsi_fast: 'RSI 14 (5 Min)', rsi_slow: 'RSI 14 (15 Min)',
  rsi_daily: 'RSI 14 (Tag)', macd_hist_s: 'MACD-Histogramm (15 Min)', adx_slow: 'ADX 14 (15 Min)',
  bb_mid_s: 'Bollinger-Mitte (15 Min)', bb_upper_s: 'Bollinger oben (15 Min)', bb_lower_s: 'Bollinger unten (15 Min)',
  vwap: 'VWAP', prev_high_f: 'Hoch der Vorkerze (5 Min)', prev_low_f: 'Tief der Vorkerze (5 Min)',
  prev_price_f: 'Schluss der Vorkerze (5 Min)', high_f: 'Hoch (5 Min)', low_f: 'Tief (5 Min)', open_f: 'Eröffnung (5 Min)',
  pullback_low: 'Tief des Rücksetzers', pullback_high: 'Hoch des Rücksetzers',
  donchian_upper_s: 'Donchian oben (15 Min)', donchian_lower_s: 'Donchian unten (15 Min)',
  day_high: 'Tageshoch', day_low: 'Tagestief', or_high: 'Hoch der Eröffnungsspanne', or_low: 'Tief der Eröffnungsspanne',
  prev_day_high: 'Vortageshoch', prev_day_low: 'Vortagestief', prev_day_close: 'Vortagesschluss',
  gap_pct: 'Kurslücke zur Eröffnung in %', ema_daily_fast: 'EMA 50 (Tag)', ema_daily_slow: 'EMA 200 (Tag)',
  ema_daily_pullback: 'EMA 20 (Tag)', rel_volume: 'Volumen ggü. üblich', atr_pct_rank: 'Rang der Schwankung',
  ret_60: 'Veränderung 60 Min', bench_trend: 'Trend des Vergleichsindex', bench_ret_60: 'Vergleichsindex 60 Min',
  rel_strength_60: 'Relative Stärke 60 Min',
  event_next: 'Nächster Termin', event_next_min: 'Minuten bis zum nächsten Termin', event_last: 'Letzter Termin',
  event_last_surprise: 'Überraschung des letzten Termins', event_last_surprise_z: 'Überraschung (σ früherer Termine)',
  event_block: 'Gesperrt wegen Termin', event_block_ts: 'Zeitpunkt des Termins', event_block_kind: 'Art des Termins',
  deriv_funding_pa: 'Funding p.a. in %', deriv_funding_pred_pa: 'Funding-Prognose p.a. in %',
  deriv_funding_z: 'Funding ggü. 30 Tagen (σ)', deriv_oi_1h_pct: 'Open Interest 1 Std in %',
  deriv_oi_24h_pct: 'Open Interest 24 Std in %', deriv_basis_pct: 'Basis Perpetual in %',
  deriv_liq_60m_usd: 'Liquidationen 1 Std in USD', deriv_liq_5m_ratio: 'Liquidationen 5 Min ggü. üblich',
};
export const featureLabel = (k) => pick(FEATURE_LABELS, k);

export const ACTIVITY_CATEGORY_LABELS = {
  data: 'Daten', news: 'Nachrichten', llm: 'KI', signal: 'Signale', sim: 'Depot', mail: 'Mail', system: 'System',
};
export const activityCategoryLabel = (c) => pick(ACTIVITY_CATEGORY_LABELS, c);

export const LEVEL_LABELS = { info: 'Info', warning: 'Warnung', error: 'Fehler' };
export const levelLabel = (l) => pick(LEVEL_LABELS, l);

const MAIL_STATUS_LABELS = { sent: 'gesendet', file: 'als Datei', skipped: 'übersprungen', failed: 'fehlgeschlagen' };
export const mailStatusLabel = (s) => pick(MAIL_STATUS_LABELS, s);

const MAIL_KIND_LABELS = { signal: 'Signal', exit: 'Ausstieg', news: 'Nachricht', test: 'Test', digest: 'Übersicht' };
export const mailKindLabel = (k) => pick(MAIL_KIND_LABELS, k);

const MAIL_MODE_LABELS = { smtp: 'Versand per SMTP', file: 'nur als Datei (Outbox)', off: 'ausgeschaltet' };
export const mailModeLabel = (m) => pick(MAIL_MODE_LABELS, m);

const LLM_KIND_LABELS = { news: 'Nachricht', trade_review: 'Signalprüfung', chart: 'Chartanalyse' };
export const llmKindLabel = (k) => pick(LLM_KIND_LABELS, k);

const BIAS_LABELS = { bullish: 'steigend', bearish: 'fallend', neutral: 'neutral' };
export const biasLabel = (b) => pick(BIAS_LABELS, b);

// Agenda (trading_local/agenda: kinds, importance, surprise side)
export const AGENDA_KIND_LABELS = { macro: 'Konjunkturdaten', central_bank: 'Notenbank', earnings: 'Quartalszahlen' };
export const IMPORTANCE_LABELS = { 1: 'gering', 2: 'mittel', 3: 'hoch' };
export const SURPRISE_LABELS = { above: 'über Erwartung', below: 'unter Erwartung', inline: 'wie erwartet' };

export const NEWS_STATUS_PENDING = new Set(['new', 'queued']);
