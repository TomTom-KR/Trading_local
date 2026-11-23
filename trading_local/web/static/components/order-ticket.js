// Order ticket for the user's paper portfolio ('Mein Depot'): symbol search, side, amount in the base
// currency or a quantity, optional stop and target with live distance and reward/risk, a preview line
// and the note that a market order fills at the next available price.

import { html, useEffect, useMemo, useState } from '../vendor/preact-htm.mjs';
import { api, errorText } from '../lib/api.js';
import { app, refreshPortfoliosSoon } from '../lib/data.js';
import {
  floorToLot, fmtMoney, fmtNum, fmtPct, fmtPrice, fmtQty, isNum, parseDecimal, priceDecimals, priceUnit, qtyDecimals,
  toInput,
} from '../lib/format.js';
import { directionGlyph, directionVerb } from '../lib/labels.js';
import { useStore } from '../lib/store.js';

/**
 * Units of the base currency per ONE unit of ``currency`` from the FX quotes in the watchlist
 * ('EURUSD=X' quotes USD per EUR). Null when no rate is known.
 */
function fxRate(currency, base, quotes) {
  if (!currency || currency === base) return 1;
  const direct = quotes[`${base}${currency}=X`]?.price;
  if (isNum(direct) && direct > 0) return 1 / direct;
  const inverse = quotes[`${currency}${base}=X`]?.price;
  if (isNum(inverse) && inverse > 0) return inverse;
  return null;
}

/** A prefilled price as input text, rounded to the decimals the price needs. */
const priceInput = (v) => (isNum(v) ? toInput(v, priceDecimals(v)) : '');

/**
 * @param {{prefill?: {symbol?: string, side?: 'LONG'|'SHORT', stop?: number|null, target?: number|null, signalId?: number|null},
 *   onDone?: (order: import('../lib/types.js').OrderOut) => void, lockSymbol?: boolean, idPrefix?: string}} props
 */
export function OrderTicket({ prefill = {}, onDone, lockSymbol = false, idPrefix = 'ot' }) {
  const instruments = useStore(app, (s) => s.instruments);
  const quotes = useStore(app, (s) => s.quotes);
  const base = useStore(app, (s) => s.status?.base_currency) || 'EUR';
  const me = useStore(app, (s) => s.portfolios.find((p) => p.key === 'me'));

  const [symbol, setSymbol] = useState(prefill.symbol || '');
  const [side, setSide] = useState(prefill.side || 'LONG');
  const [mode, setMode] = useState('amount');
  const [amount, setAmount] = useState('2000');
  const [qty, setQty] = useState('');
  const [stop, setStop] = useState(priceInput(prefill.stop));
  const [target, setTarget] = useState(priceInput(prefill.target));
  const [signalId, setSignalId] = useState(prefill.signalId ?? null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [done, setDone] = useState(null);

  useEffect(() => {
    setSymbol(prefill.symbol || '');
    setSide(prefill.side || 'LONG');
    setStop(priceInput(prefill.stop));
    setTarget(priceInput(prefill.target));
    setSignalId(prefill.signalId ?? null);
    setError(null);
    setDone(null);
  }, [prefill.symbol, prefill.side, prefill.stop, prefill.target, prefill.signalId]);

  const tradables = useMemo(() => instruments.filter((i) => i.tradable && i.active), [instruments]);
  const inst = useMemo(
    () => instruments.find((i) => i.symbol.toUpperCase() === symbol.trim().toUpperCase()) ?? null,
    [instruments, symbol],
  );
  const price = inst ? quotes[inst.symbol]?.price ?? inst.quote?.price ?? null : null;
  const unit = inst ? priceUnit(inst.currency, inst.price_scale) : '';
  const scale = inst?.price_scale || 1;
  const fx = inst ? fxRate(inst.currency, base, quotes) : null;
  const sign = side === 'LONG' ? 1 : -1;

  // amounts and whole-share quantities accept German thousands groups ('1.500'); index and crypto
  // quantities (lots of 0.01 / 0.000001, like the broker) do not
  const assetClass = inst?.asset_class;
  const amountVal = parseDecimal(amount, { grouping: true });
  const qtyVal = parseDecimal(qty, { grouping: qtyDecimals(assetClass) === 0 });
  const stopVal = stop.trim() ? parseDecimal(stop) : null;
  const targetVal = target.trim() ? parseDecimal(target) : null;

  const unitValueBase = isNum(price) && isNum(fx) ? price * scale * fx : null;
  // the broker floors every quantity to whole lots; the preview shows what it will trade
  let estQty = null;
  if (mode === 'qty' && isNum(qtyVal) && qtyVal > 0) estQty = floorToLot(qtyVal, assetClass);
  if (mode === 'amount' && isNum(amountVal) && amountVal > 0 && isNum(unitValueBase) && unitValueBase > 0) {
    estQty = floorToLot(amountVal / unitValueBase, assetClass);
  }
  const minLot = fmtQty(10 ** -qtyDecimals(assetClass), assetClass);

  // field validation (hints shown at the fields; the server checks again)
  const errors = {};
  if (!inst) errors.symbol = symbol.trim() ? 'Unbekanntes Symbol – bitte aus der Liste wählen.' : 'Bitte ein Symbol wählen.';
  else if (!inst.tradable) errors.symbol = `${inst.name} ist nicht handelbar (nur Umrechnungskurs).`;
  if (mode === 'amount' && !(isNum(amountVal) && amountVal > 0)) errors.amount = `Bitte einen Betrag in ${base} größer als 0 eingeben.`;
  if (mode === 'qty' && !(isNum(qtyVal) && qtyVal > 0)) errors.qty = 'Bitte eine Stückzahl größer als 0 eingeben.';
  else if (mode === 'qty' && isNum(estQty) && estQty <= 0) errors.qty = `Die kleinste handelbare Menge ist ${minLot} Stück.`;
  if (mode === 'amount' && isNum(estQty) && estQty <= 0) {
    errors.amount = qtyDecimals(assetClass) === 0
      ? 'Der Betrag reicht nicht für ein Stück.'
      : `Der Betrag reicht nicht für die kleinste handelbare Menge (${minLot} Stück).`;
  }
  if (stopVal !== null) {
    if (!(isNum(stopVal) && stopVal > 0)) errors.stop = 'Ungültiger Stoppkurs.';
    else if (isNum(price) && (stopVal - price) * sign >= 0) {
      errors.stop = side === 'LONG' ? 'Bei Long muss der Stopp unter dem aktuellen Kurs liegen.' : 'Bei Short muss der Stopp über dem aktuellen Kurs liegen.';
    }
  }
  if (targetVal !== null) {
    if (!(isNum(targetVal) && targetVal > 0)) errors.target = 'Ungültiger Zielkurs.';
    else if (isNum(price) && (targetVal - price) * sign <= 0) {
      errors.target = side === 'LONG' ? 'Bei Long muss das Ziel über dem aktuellen Kurs liegen.' : 'Bei Short muss das Ziel unter dem aktuellen Kurs liegen.';
    }
  }
  const valid = Object.keys(errors).length === 0;

  const dist = (lvl) => (isNum(lvl) && isNum(price) && price > 0 ? ((lvl - price) / price) * 100 : null);
  const stopDist = !errors.stop ? dist(stopVal) : null;
  const targetDist = !errors.target ? dist(targetVal) : null;
  const rr = isNum(stopDist) && isNum(targetDist) && stopDist !== 0 ? Math.abs(targetDist / stopDist) : null;
  const riskBase = isNum(stopVal) && !errors.stop && isNum(estQty) && isNum(fx) && isNum(price)
    ? Math.abs(price - stopVal) * scale * fx * estQty : null;
  const notional = isNum(estQty) && isNum(unitValueBase) ? estQty * unitValueBase : null;

  const submit = async (e) => {
    e.preventDefault();
    if (!valid || busy) return;
    setBusy(true);
    setError(null);
    setDone(null);
    const body = {
      symbol: inst.symbol,
      side,
      ...(mode === 'amount' ? { amount_base: amountVal } : { qty: qtyVal }),
      ...(stopVal !== null ? { stop_price: stopVal } : {}),
      ...(targetVal !== null ? { target_price: targetVal } : {}),
      ...(signalId ? { signal_id: signalId } : {}),
    };
    try {
      const order = await api.post('/portfolios/me/orders', body);
      const what = `${directionVerb(order.side)} ${isNum(order.qty) ? `${fmtQty(order.qty, assetClass)} ` : ''}${order.symbol}`;
      setDone(order.status === 'filled'
        ? `Order ausgeführt: ${what} zu ${fmtPrice(order.fill_price, inst.asset_class)} ${unit}.`
        : `Order angenommen: ${what} – wird zum nächsten verfügbaren Kurs ausgeführt.`);
      refreshPortfoliosSoon();
      onDone?.(order);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const id = (s) => `${idPrefix}-${s}`;
  const delay = inst?.delay_min || 0;
  return html`<form class="ticket" onSubmit=${submit} noValidate>
    ${signalId && html`<p class="ticket-signal small">Übernimmt KI-Signal <a href=${`#/signale/${signalId}`}>#${signalId}</a>
      <button type="button" class="btn-link small" onClick=${() => setSignalId(null)}>Verknüpfung lösen</button></p>`}
    <div class="field">
      <label for=${id('symbol')}>Wertpapier</label>
      <input id=${id('symbol')} class="input" list=${id('list')} value=${symbol} disabled=${lockSymbol}
        autocomplete="off" spellcheck="false" placeholder="Symbol, z. B. SAP.DE"
        aria-invalid=${symbol && errors.symbol ? 'true' : 'false'} aria-describedby=${id('symbol-msg')}
        onInput=${(e) => setSymbol(e.currentTarget.value)} />
      <datalist id=${id('list')}>
        ${tradables.map((i) => html`<option value=${i.symbol}>${i.name}</option>`)}
      </datalist>
      <span id=${id('symbol-msg')} class=${symbol && errors.symbol ? 'field-error' : 'field-help'}>
        ${symbol && errors.symbol ? errors.symbol
          : inst ? html`${inst.name} · ${isNum(price) ? `${fmtPrice(price, inst.asset_class)} ${unit}` : 'noch kein Kurs'}${delay ? ` · Daten ${delay} Min verzögert` : ''}`
            : 'Nur handelbare Werte der Beobachtungsliste.'}
      </span>
    </div>

    <div class="field">
      <span class="field-label" id=${id('side-l')}>Richtung</span>
      <div class="side-toggle" role="group" aria-labelledby=${id('side-l')}>
        ${['LONG', 'SHORT'].map((d) => html`<button type="button" class=${`side side-${d.toLowerCase()}`}
            aria-pressed=${side === d ? 'true' : 'false'} onClick=${() => setSide(d)}>
            <span class="glyph" aria-hidden="true">${directionGlyph(d)}</span>
            ${d === 'LONG' ? 'Long' : 'Short'} <span class="muted">(${directionVerb(d)})</span>
          </button>`)}
      </div>
    </div>

    <div class="field">
      <div class="row-between">
        <label class="field-label" for=${id(mode)}>${mode === 'amount' ? `Betrag in ${base}` : 'Stückzahl'}</label>
        <div class="segmented" role="group" aria-label="Mengenangabe">
          <button type="button" aria-pressed=${mode === 'amount' ? 'true' : 'false'} onClick=${() => setMode('amount')}>${base}</button>
          <button type="button" aria-pressed=${mode === 'qty' ? 'true' : 'false'} onClick=${() => setMode('qty')}>Stück</button>
        </div>
      </div>
      ${mode === 'amount'
        ? html`<div class="input-affix"><input id=${id('amount')} class="input input-num" inputmode="decimal" value=${amount}
            aria-invalid=${errors.amount ? 'true' : 'false'} onInput=${(e) => setAmount(e.currentTarget.value)} /><span class="affix">${base}</span></div>`
        : html`<div class="input-affix"><input id=${id('qty')} class="input input-num" inputmode="decimal" value=${qty}
            aria-invalid=${errors.qty ? 'true' : 'false'} onInput=${(e) => setQty(e.currentTarget.value)} /><span class="affix">Stück</span></div>`}
      ${(errors.amount || errors.qty) && (amount || qty) && html`<span class="field-error">${errors.amount || errors.qty}</span>`}
    </div>

    <div class="ticket-levels">
      <div class="field">
        <label for=${id('stop')}>Stopp <span class="faint">(optional)</span></label>
        <div class="input-affix"><input id=${id('stop')} class="input input-num" inputmode="decimal" value=${stop}
          aria-invalid=${errors.stop ? 'true' : 'false'} onInput=${(e) => setStop(e.currentTarget.value)} /><span class="affix">${unit || '–'}</span></div>
        ${errors.stop ? html`<span class="field-error">${errors.stop}</span>`
          : isNum(stopDist) && html`<span class="field-help num">${fmtPct(stopDist, 1)} vom Kurs</span>`}
      </div>
      <div class="field">
        <label for=${id('target')}>Ziel <span class="faint">(optional)</span></label>
        <div class="input-affix"><input id=${id('target')} class="input input-num" inputmode="decimal" value=${target}
          aria-invalid=${errors.target ? 'true' : 'false'} onInput=${(e) => setTarget(e.currentTarget.value)} /><span class="affix">${unit || '–'}</span></div>
        ${errors.target ? html`<span class="field-error">${errors.target}</span>`
          : isNum(targetDist) && html`<span class="field-help num">${fmtPct(targetDist, 1)} vom Kurs${isNum(rr) ? ` · Chance/Risiko ${fmtNum(rr, 1)}` : ''}</span>`}
      </div>
    </div>

    <p class="ticket-preview num" aria-live="polite">
      ${inst && isNum(estQty) && isNum(price)
        ? html`<span>ca. <strong>${fmtQty(estQty, assetClass)} Stück</strong> à ${fmtPrice(price, inst.asset_class)} ${unit}${
          isNum(notional) ? html` ≈ <strong>${fmtMoney(notional, base, 0)}</strong>` : ` (Umrechnung in ${base} nicht verfügbar)`}${
          isNum(riskBase) ? `, Risiko bis Stopp ≈ ${fmtMoney(riskBase, base, 0)}` : ''}</span>`
        : html`<span class="muted">Die Vorschau erscheint, sobald Wertpapier und Menge feststehen.</span>`}
      ${me && html`<span class="ticket-bp muted">Kaufkraft: ${fmtMoney(me.buying_power, base, 0)}</span>`}
    </p>
    <p class="ticket-note small muted">
      Marktorder: Ausführung zum nächsten verfügbaren Kurs (Eröffnung der nächsten Minutenkerze)${delay ? `, bei ${inst.name} mit ${delay} Minuten Datenverzögerung` : ''}.
      Ist der Handelsplatz geschlossen, bleibt die Order bis zur Eröffnung offen.
    </p>
    ${error && html`<div class="field-error" role="alert">${error}</div>`}
    ${done && html`<div class="ticket-done" role="status">${done}</div>`}
    <button type="submit" class="btn btn-primary ticket-submit" disabled=${!valid || busy}>
      ${busy ? 'Wird gesendet …' : `${directionVerb(side)}${inst ? ` ${inst.symbol}` : ''}`}
    </button>
  </form>`;
}
