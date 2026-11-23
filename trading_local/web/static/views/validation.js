// #/validierung - replay validation: start a run, history of runs, and the full result with folds,
// groups, calibration, the random-entry baseline and the cumulative R curve.

import { html, useEffect, useMemo, useState } from '../vendor/preact-htm.mjs';
import { api, errorText } from '../lib/api.js';
import {
  DASH, fmtDate, fmtNum, fmtPct, fmtR, fmtWhen, isNum,
} from '../lib/format.js';
import { useLive } from '../lib/live.js';
import { useResource } from '../lib/resource.js';
import { useTick } from '../lib/store.js';
import { serverNow } from '../lib/time.js';
import { LineChartView } from '../components/line-chart.js';
import {
  CalibrationChart, GroupTable, SignedBars, StatBlockView,
} from '../components/stats.js';
import { Empty, ErrorState, Loading } from '../components/ui.js';

const DAY_OPTIONS = [5, 10, 20, 30, 60, 90, 180];
const STATUS_LABELS = { running: 'läuft', done: 'fertig', failed: 'fehlgeschlagen' };

function Baseline({ baseline, engineAvg }) {
  if (!baseline) return html`<p class="small muted">Für diesen Lauf wurde kein Zufallsvergleich berechnet.</p>`;
  const { mean_avg_r: mean, p95_avg_r: p95, p_value: p, n_sims: n } = baseline;
  const lo = Math.min(mean, p95, engineAvg ?? mean) - 0.1;
  const hi = Math.max(mean, p95, engineAvg ?? mean) + 0.1;
  const x = (v) => `${((v - lo) / (hi - lo)) * 100}%`;
  const beats = isNum(p) && p < 0.05;
  return html`<div class="baseline">
    <div class="baseline-scale" role="img" aria-label=${`Zufall im Mittel ${fmtR(mean, 2)}, 95 % der Zufallsläufe unter ${fmtR(p95, 2)}, Engine ${fmtR(engineAvg, 2)}`}>
      <span class="bl-band" style=${{ width: x(p95) }}></span>
      <span class="bl-mark bl-mean" style=${{ left: x(mean) }}><i></i><b>Zufall Ø ${fmtR(mean, 2)}</b></span>
      <span class="bl-mark bl-p95" style=${{ left: x(p95) }}><i></i><b>95 % darunter: ${fmtR(p95, 2)}</b></span>
      ${isNum(engineAvg) && html`<span class="bl-mark bl-engine" style=${{ left: x(engineAvg) }}><i></i><b>Engine ${fmtR(engineAvg, 2)}</b></span>`}
    </div>
    <p class="prose small">
      ${fmtNum(n, 0)} Durchläufe mit zufälligen Einstiegen, aber identischen Stopps, Zielen, Haltedauern und identischer Slippage.
      ${isNum(p) ? html` <strong>p = ${fmtNum(p, 2)}</strong>: In ${fmtPct(p * 100, 0)} der Zufallsdurchläufe war das Ergebnis mindestens so gut wie das der Engine.
        ${beats ? ' Das spricht für einen echten Vorteil – sofern sich das Ergebnis in weiteren Zeiträumen wiederholt.'
          : ' Das ist kein belastbarer Vorteil gegenüber Zufall (üblich ist eine Schwelle von 5 %).'}` : ''}
    </p>
  </div>`;
}

function Result({ run }) {
  const r = run.result;
  const folds = useMemo(() => (r?.folds || []).map((f) => ({
    label: `${fmtDate(f.start_ts).slice(0, 6)}`, value: f.stats.avg_r, n: f.stats.n,
  })), [r]);
  const curve = useMemo(() => (r?.equity_r?.length ? [{ id: 'r', points: r.equity_r, color: 'ink1', width: 2 }] : []), [r]);
  if (run.status === 'running') {
    return html`<${Empty} title="Die Validierung läuft.">Die Wiederholung läuft in einem eigenen Prozess; das Ergebnis erscheint hier automatisch, sobald sie fertig ist.</${Empty}>`;
  }
  if (run.status === 'failed' || !r) {
    return html`<${ErrorState} title="Dieser Lauf ist fehlgeschlagen." error=${run.message || 'Keine Details gespeichert.'} />`;
  }
  return html`<div class="vresult stack">
    <section class="panel">
      <p class="notice notice-strong vverdict">${r.verdict_de || 'Kein Urteil gespeichert.'}</p>
      ${r.warnings_de?.length > 0 && html`<ul class="warnings">${r.warnings_de.map((w) => html`<li>${w}</li>`)}</ul>`}
      <dl class="kv vfacts">
        <div><dt>Zeitraum</dt><dd>${fmtDate(r.period_start_ts)} bis ${fmtDate(r.period_end_ts)}</dd></div>
        <div><dt>Handelstage</dt><dd>${fmtNum(r.trading_days, 0)}</dd></div>
        <div><dt>Werte</dt><dd>${fmtNum(r.symbols, 0)}</dd></div>
        <div><dt>Entscheidungspunkte</dt><dd>${fmtNum(r.decisions, 0)}</dd></div>
        <div><dt>Engine</dt><dd>${run.engine_version || DASH}</dd></div>
      </dl>
    </section>
    <div class="grid-2">
      <section class="panel"><${StatBlockView} title="Gesamtergebnis" stats=${r.overall} />
        <p class="tiny faint">Ergebnisse in R nach Slippage, ohne Ordergebühren (die feste Gebühr je Order fällt nur in den Depots an).</p></section>
      <section class="panel">
        <h3 class="statblock-title">Zufallsvergleich</h3>
        <${Baseline} baseline=${r.baseline} engineAvg=${r.overall?.avg_r} />
      </section>
    </div>
    <div class="grid-2">
      <section class="panel">
        <h3 class="statblock-title">Stabilität über die Zeit: Ø R je Zeitblock</h3>
        ${folds.length ? html`<${SignedBars} items=${folds} label="Durchschnittliches Ergebnis je Zeitblock" />` : html`<p class="small muted">Keine Zeitblöcke.</p>`}
      </section>
      <section class="panel">
        <h3 class="statblock-title">Kalibrierung (nur auf früheren Blöcken angepasst)</h3>
        <${CalibrationChart} bins=${r.calibration} />
      </section>
    </div>
    <section class="panel">
      <h3 class="statblock-title">Kumuliertes Ergebnis in R</h3>
      ${curve.length ? html`<${LineChartView} series=${curve} height=${240} format=${(v) => fmtR(v, 1)} label="Kumuliertes Ergebnis der Validierung in R" />`
        : html`<p class="small muted">Keine Trades im Zeitraum.</p>`}
    </section>
    <section class="panel">
      <div class="grid-3">
        <${GroupTable} title="Setup" groups=${r.by_setup} />
        <${GroupTable} title="Region" groups=${r.by_region} />
        <${GroupTable} title="Richtung" groups=${r.by_direction} />
      </div>
    </section>
  </div>`;
}

export function ValidationView() {
  const runs = useResource('/validation', null, []);
  const [days, setDays] = useState(30);
  const [selected, setSelected] = useState(null);
  const [msg, setMsg] = useState(null);
  const [busy, setBusy] = useState(false);
  useTick(30);
  useLive('validation', () => runs.reload());

  useEffect(() => {
    if (!selected && runs.data?.length) setSelected((runs.data.find((r) => r.status === 'done') || runs.data[0]).id);
  }, [runs.data]);

  const start = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const run = await api.post('/validation/run', { days });
      setMsg({ ok: true, text: `Validierung über ${days} Tage gestartet.` });
      setSelected(run.id);
      runs.reload();
    } catch (err) {
      setMsg({ ok: false, text: err.status === 409 ? 'Eine Validierung läuft bereits – bitte warten, bis sie fertig ist.' : errorText(err) });
    } finally {
      setBusy(false);
    }
  };
  const run = runs.data?.find((r) => r.id === selected) || null;
  return html`<div>
    <div class="page-head"><div><h1>Validierung</h1>
      <p class="lede">Die Validierung spielt die gespeicherten Minutenkerzen noch einmal ab und lässt die Engine mit ihren festen Regeln entscheiden, als wäre es live – ohne Blick in die Zukunft und mit derselben Slippage wie im Depot. Die Ergebnisse sind in R nach Slippage angegeben, ohne Ordergebühren.
      Ein gutes Ergebnis zeigt nur, dass die Regeln in diesem Zeitraum funktioniert hätten; es beweist keinen künftigen Gewinn, und ein kurzer Zeitraum kann reiner Zufall sein.</p></div></div>
    <div class="valid-grid">
      <aside class="stack">
        <section class="panel">
          <div class="panel-head"><h2>Neuer Lauf</h2></div>
          <div class="field">
            <label for="val-days">Zeitraum</label>
            <select id="val-days" class="select" value=${days} onChange=${(e) => setDays(Number(e.currentTarget.value))}>
              ${DAY_OPTIONS.map((d) => html`<option value=${d}>letzte ${d} Tage</option>`)}
            </select>
            <span class="field-help">Längere Zeiträume dauern länger, sagen aber mehr aus.</span>
          </div>
          <button type="button" class="btn btn-primary val-start" onClick=${start} disabled=${busy}>${busy ? 'Wird gestartet …' : 'Validierung starten'}</button>
          ${msg && html`<p class=${msg.ok ? 'small' : 'field-error'} role="status">${msg.text}</p>`}
        </section>
        <section class="panel">
          <div class="panel-head"><h2>Bisherige Läufe</h2></div>
          ${runs.loading ? html`<${Loading} lines=${3} />`
            : runs.error ? html`<${ErrorState} error=${runs.error} onRetry=${runs.reload} />`
              : !runs.data?.length ? html`<p class="small muted">Noch keine Läufe.</p>`
                : html`<ul class="runs">${runs.data.map((r) => html`<li>
                    <button type="button" class="run" aria-pressed=${r.id === selected ? 'true' : 'false'} onClick=${() => setSelected(r.id)}>
                      <span class=${`dot ${r.status === 'done' ? 'dot-ok' : r.status === 'running' ? 'dot-warn' : 'dot-error'}`} aria-hidden="true"></span>
                      <span class="run-main"><span>#${r.id} · ${isNum(r.params?.days) ? `${r.params.days} Tage` : 'Lauf'}</span>
                        <span class="tiny muted">${fmtWhen(r.ts, serverNow())} · ${STATUS_LABELS[r.status] || r.status}</span></span>
                      ${r.result && html`<span class="run-r num">${fmtR(r.result.overall?.avg_r, 2)}</span>`}
                    </button></li>`)}</ul>`}
        </section>
      </aside>
      <div>
        ${run ? html`<${Result} run=${run} />`
          : !runs.loading && html`<${Empty} title="Noch kein Ergebnis.">Einen Lauf starten: Er nutzt nur Daten, die bereits gespeichert sind, und dauert je nach Zeitraum einige Minuten.</${Empty}>`}
      </div>
    </div>
  </div>`;
}
