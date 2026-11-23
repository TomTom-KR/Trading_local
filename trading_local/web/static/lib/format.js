// German number, money, time and duration formatting (de-DE, 24 h clock, browser time zone).
//
// Conventions used everywhere in the GUI
// * decimal comma and dot grouping: 1.234,56
// * percent with a non-breaking space: '58 %'
// * signed values use a real minus sign (U+2212) and an explicit plus: '+1,2 %', '−0,8 %'
// * all timestamps are epoch SECONDS (UTC); they are shown in the browser's local time zone
// * missing values render as an en dash '–'

export const DASH = '–';
const MINUS = '−';
const NBSP = ' ';

const numberFormats = new Map();

/** Cached Intl.NumberFormat with a fixed number of fraction digits. */
function nf(digits, grouping = true) {
  const key = `${digits}|${grouping}`;
  let f = numberFormats.get(key);
  if (!f) {
    f = new Intl.NumberFormat('de-DE', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
      useGrouping: grouping,
    });
    numberFormats.set(key, f);
  }
  return f;
}

/** @param {unknown} v */
export function isNum(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

/** Plain number, e.g. fmtNum(1234.5, 2) -> '1.234,50'; a value that rounds to zero never shows a sign. */
export function fmtNum(v, digits = 2) {
  if (!isNum(v)) return DASH;
  const text = nf(digits).format(v);
  return /[1-9]/.test(text) ? text.replace('-', MINUS) : text.replace('-', '');
}

const flexFormats = new Map();

/** Number with at most ``maxDigits`` decimals and no trailing zeros: fmtFlex(0.35) -> '0,35'. */
export function fmtFlex(v, maxDigits = 4) {
  if (!isNum(v)) return DASH;
  let f = flexFormats.get(maxDigits);
  if (!f) {
    f = new Intl.NumberFormat('de-DE', { maximumFractionDigits: maxDigits });
    flexFormats.set(maxDigits, f);
  }
  const text = f.format(v);
  return /[1-9]/.test(text) ? text.replace('-', MINUS) : text.replace('-', '');
}

/** Number with explicit sign: '+1,20' / '−0,80' / '0,00'. */
export function fmtSigned(v, digits = 2) {
  if (!isNum(v)) return DASH;
  const text = nf(digits).format(Math.abs(v));
  if (Number(text.replace(/\./g, '').replace(',', '.')) === 0) return text;
  return (v > 0 ? '+' : MINUS) + text;
}

/**
 * Sensible number of decimals for a price: FX pairs and small prices need more digits.
 * @param {number} price
 * @param {string} [assetClass]
 */
export function priceDecimals(price, assetClass) {
  const p = Math.abs(price);
  if (assetClass === 'fx') return p >= 20 ? 3 : 4;
  if (p >= 1) return 2;
  if (p >= 0.01) return 4;
  return 6;
}

// Lot sizes of the paper broker (trading_local/sim/sizing.py _LOTS): shares trade in whole units, indices in
// steps of 0.01, crypto in steps of 0.000001. FX pairs and futures are not tradable and count as whole units.
const QTY_DECIMALS = { equity: 0, index: 2, crypto: 6 };

/**
 * True for asset classes the paper broker trades (sizing.is_tradable_class); FX pairs and futures are only
 * conversion rates or reference values.
 * @param {string} [assetClass]
 */
export function isTradableClass(assetClass) {
  return Object.hasOwn(QTY_DECIMALS, assetClass ?? '');
}

/**
 * Decimals of a tradable quantity of the asset class (the broker's lot size).
 * @param {string} [assetClass]
 */
export function qtyDecimals(assetClass) {
  return QTY_DECIMALS[assetClass] ?? 0;
}

/**
 * ``qty`` floored to whole lots of the asset class, exactly like the broker (0.03 / 0.01 counts as three
 * lots although it is 2.9999999999999996 in binary floating point). 0 for non-positive or invalid input.
 * @param {number} qty
 * @param {string} [assetClass]
 */
export function floorToLot(qty, assetClass) {
  if (!isNum(qty) || qty <= 0) return 0;
  const digits = qtyDecimals(assetClass);
  const lots = Math.floor(Number((qty * 10 ** digits).toFixed(9)));
  return Number((lots / 10 ** digits).toFixed(digits));
}

/**
 * A quantity with the decimals its asset class trades in, trailing zeros dropped: fmtQty(0.08, 'index') ->
 * '0,08', fmtQty(12, 'equity') -> '12'. Without an asset class up to six decimals are shown.
 * @param {number} qty
 * @param {string} [assetClass]
 */
export function fmtQty(qty, assetClass) {
  return fmtFlex(qty, assetClass ? qtyDecimals(assetClass) : 6);
}

/**
 * A short German enumeration: up to ``shown`` names, the rest counted -
 * fmtList(['Bitcoin', 'Ethereum', 'Solana', 'XRP', 'Cardano', 'Dogecoin'], 4) -> 'Bitcoin, Ethereum, Solana, XRP und 2 weitere'.
 * One extra name is shown instead of 'und 1 weitere'.
 * @param {string[]} names
 * @param {number} [shown]
 */
export function fmtList(names, shown = 4) {
  if (names.length <= shown + 1) return names.join(', ');
  return `${names.slice(0, shown).join(', ')} und ${names.length - shown} weitere`;
}

/** Price in source units with automatic decimals: fmtPrice(142.351) -> '142,35'. */
export function fmtPrice(v, assetClass) {
  if (!isNum(v)) return DASH;
  return fmtNum(v, priceDecimals(v, assetClass));
}

/** Percent with space: fmtPct(58) -> '58 %'. ``v`` is already in percent. */
export function fmtPct(v, digits = 0) {
  if (!isNum(v)) return DASH;
  return `${fmtNum(v, digits)}${NBSP}%`;
}

/** Signed percent: '+1,2 %' / '−0,8 %'. ``v`` is already in percent. */
export function fmtPctSigned(v, digits = 1) {
  if (!isNum(v)) return DASH;
  return `${fmtSigned(v, digits)}${NBSP}%`;
}

/** Share 0..1 as percent: fmtShare(0.58) -> '58 %'. */
export function fmtShare(v, digits = 0) {
  return isNum(v) ? fmtPct(v * 100, digits) : DASH;
}

/** Money amount with currency code after the number: '1.840 EUR'. */
export function fmtMoney(v, currency = 'EUR', digits = 2) {
  if (!isNum(v)) return DASH;
  return `${fmtNum(v, digits)}${NBSP}${currency}`;
}

/** Signed money: '+412,50 EUR' / '−37,00 EUR'. */
export function fmtMoneySigned(v, currency = 'EUR', digits = 2) {
  if (!isNum(v)) return DASH;
  return `${fmtSigned(v, digits)}${NBSP}${currency}`;
}

/** Result in R multiples: '+0,4 R' / '−1,0 R'. */
export function fmtR(v, digits = 1) {
  if (!isNum(v)) return DASH;
  return `${fmtSigned(v, digits)}${NBSP}R`;
}

/** Large counts compactly: 1.234 / 12,3 Tsd. / 1,2 Mio. / 3,4 Mrd. */
export function fmtCompact(v) {
  if (!isNum(v)) return DASH;
  const a = Math.abs(v);
  if (a >= 1e9) return `${fmtNum(v / 1e9, 1)}${NBSP}Mrd.`;
  if (a >= 1e6) return `${fmtNum(v / 1e6, 1)}${NBSP}Mio.`;
  if (a >= 1e5) return `${fmtNum(v / 1e3, 0)}${NBSP}Tsd.`;
  return fmtNum(v, 0);
}

/**
 * Display unit of a price in source units. London shares are quoted in pence (price_scale 0.01):
 * they are shown as 'GBp' and never converted silently.
 * @param {string} currency
 * @param {number} [priceScale]
 */
export function priceUnit(currency, priceScale = 1) {
  if (!currency) return '';
  if (!priceScale || priceScale === 1) return currency;
  if (currency === 'GBP' && Math.abs(priceScale - 0.01) < 1e-12) return 'GBp';
  return `${currency}${NBSP}×${NBSP}${fmtNum(priceScale, 4).replace(/0+$/, '').replace(/,$/, '')}`;
}

// ------------------------------------------------------------------------------------------------
// Time
// ------------------------------------------------------------------------------------------------
const timeFmt = new Intl.DateTimeFormat('de-DE', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
const dayFmt = new Intl.DateTimeFormat('de-DE', { day: '2-digit', month: '2-digit' });
const dateFmt = new Intl.DateTimeFormat('de-DE', { day: '2-digit', month: '2-digit', year: 'numeric' });
const weekdayFmt = new Intl.DateTimeFormat('de-DE', { weekday: 'short' });
const utcDateFmt = new Intl.DateTimeFormat('de-DE', {
  day: '2-digit', month: '2-digit', year: 'numeric', timeZone: 'UTC',
});

const toDate = (ts) => new Date(ts * 1000);

/** '15:42' */
export function fmtTime(ts) {
  return isNum(ts) ? timeFmt.format(toDate(ts)) : DASH;
}

/** '20.11.2026' */
export function fmtDate(ts) {
  return isNum(ts) ? dateFmt.format(toDate(ts)) : DASH;
}

/** Weekday abbreviation without the trailing dot: 'Fr'. */
export function fmtWeekday(ts) {
  return isNum(ts) ? weekdayFmt.format(toDate(ts)).replace('.', '') : DASH;
}

/** Two-digit local hour without 'Uhr': '09'. */
export function fmtHour(ts) {
  return isNum(ts) ? String(toDate(ts).getHours()).padStart(2, '0') : DASH;
}

/** True when both timestamps fall on the same local calendar day. */
function sameLocalDay(a, b) {
  const x = toDate(a);
  const y = toDate(b);
  return x.getFullYear() === y.getFullYear() && x.getMonth() === y.getMonth() && x.getDate() === y.getDate();
}

/**
 * Compact date-time relative to ``now``: '15:42' today, 'Mi 15:42' within the week, '12.11. 15:42' otherwise.
 */
export function fmtWhen(ts, now) {
  if (!isNum(ts)) return DASH;
  if (isNum(now) && sameLocalDay(ts, now)) return fmtTime(ts);
  if (isNum(now) && Math.abs(now - ts) < 6 * 86400) return `${fmtWeekday(ts)} ${fmtTime(ts)}`;
  return `${dayFmt.format(toDate(ts))} ${fmtTime(ts)}`;
}

/** Full date and time: 'Fr 20.11.2026, 15:42'. */
export function fmtDateTime(ts) {
  return isNum(ts) ? `${fmtWeekday(ts)} ${fmtDate(ts)}, ${fmtTime(ts)}` : DASH;
}

/** A daily bar's session date (ts = session date 00:00 UTC): '20.11.2026' regardless of time zone. */
export function fmtSessionDate(ts) {
  return isNum(ts) ? utcDateFmt.format(toDate(ts)) : DASH;
}

/**
 * Duration in German short units: '45 Min', '1 Std 12 Min', '2 T 9 Std', 'unter 1 Min'.
 * @param {number} seconds
 */
export function fmtDuration(seconds) {
  if (!isNum(seconds)) return DASH;
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return 'unter 1 Min';
  const minutes = Math.floor(s / 60);
  const days = Math.floor(minutes / 1440);
  const hours = Math.floor((minutes % 1440) / 60);
  const mins = minutes % 60;
  if (days > 0) return hours > 0 ? `${days} T ${hours} Std` : `${days} T`;
  if (hours > 0) return mins > 0 ? `${hours} Std ${mins} Min` : `${hours} Std`;
  return `${mins} Min`;
}

/** Age relative to ``now``: 'gerade eben', 'vor 3 Min', 'vor 2 Std', 'vor 3 T'. */
export function fmtAge(ts, now) {
  if (!isNum(ts) || !isNum(now)) return DASH;
  const d = now - ts;
  if (d < 45) return 'gerade eben';
  if (d < 3600) return `vor ${Math.max(1, Math.round(d / 60))} Min`;
  if (d < 86400) return `vor ${Math.round(d / 3600)} Std`;
  return `vor ${Math.round(d / 86400)} T`;
}

/** Milliseconds as '850 ms' / '2,4 s'. */
export function fmtMs(ms) {
  if (!isNum(ms)) return DASH;
  return ms < 1000 ? `${fmtNum(ms, 0)}${NBSP}ms` : `${fmtNum(ms / 1000, 1)}${NBSP}s`;
}

/**
 * Parse a German or English decimal typed by the user ('1.234,5', '1234.5', '0,5'); NaN when invalid.
 * With a comma the dots are thousands separators. Without a comma a single dot is a decimal point
 * ('142.35'), unless ``grouping`` is set and the text is written in German thousands groups: then
 * '1.500' and '100.000' mean 1500 and 100000. Use ``grouping`` for amounts and whole-share quantities,
 * where nobody types three decimals, never for prices ('1.085' is an FX rate).
 * @param {string} text
 * @param {{grouping?: boolean}} [opts]
 */
export function parseDecimal(text, { grouping = false } = {}) {
  const t = String(text ?? '').trim().replace(/[\s  ]/g, '');
  if (!t) return NaN;
  let normalised = t;
  if (t.includes(',')) normalised = t.replace(/\./g, '').replace(',', '.');
  else if (grouping && /^[-+]?[1-9]\d{0,2}(\.\d{3})+$/.test(t)) normalised = t.replace(/\./g, '');
  if (!/^[-+]?\d*\.?\d+$/.test(normalised)) return NaN;
  return Number(normalised);
}

/** Input value for a number field (German decimal comma, no grouping). */
export function toInput(v, digits) {
  if (!isNum(v)) return '';
  return digits === undefined ? String(v).replace('.', ',') : nf(digits, false).format(v);
}
