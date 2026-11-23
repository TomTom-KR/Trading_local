// #/einstellungen - every RuntimeSettings field, grouped, with German help texts, client-side bounds
// (mirroring the pydantic constraints) and server validation errors shown at the field.

import { html, useEffect, useMemo, useState } from '../vendor/preact-htm.mjs';
import { api, errorText } from '../lib/api.js';
import { app } from '../lib/data.js';
import { fmtNum } from '../lib/format.js';
import { REGION_LABELS } from '../lib/labels.js';
import { useResource } from '../lib/resource.js';
import { useStore } from '../lib/store.js';
import { ErrorState, Loading, Toast } from '../components/ui.js';

const REGIONS = ['ASIA', 'EU', 'US', 'GLOBAL'];
const SETUPS = {
  trend_pullback: 'Trend-Rücksetzer', breakout: 'Ausbruch', mean_reversion: 'Gegenbewegung',
  news_momentum: 'Nachrichten-Impuls', daily_trend: 'Tagestrend',
};

/**
 * Field definitions. Bounds: min/max inclusive unless minExclusive (pydantic gt).
 * kind: bool | int | float | regions | setups | benchmarks
 */
const GROUPS = [
  {
    title: 'E-Mail-Alarme',
    intro: 'Wann eine Empfehlung per Mail kommt. Alle Signale bleiben unabhängig davon in der Oberfläche sichtbar.',
    fields: [
      { key: 'mail_enabled', kind: 'bool', label: 'Mail-Alarme senden', help: 'Aus: Es werden keine Signal- und Ausstiegs-Mails verschickt; die Testmail funktioniert weiter.' },
      { key: 'mail_regions', kind: 'regions', label: 'Regionen für Mails', help: 'Nur Signale aus diesen Regionen lösen eine Mail aus – etwa Asien abwählen, wenn nachts Ruhe sein soll.' },
      { key: 'mail_max_per_hour', kind: 'int', min: 0, max: 120, unit: 'pro Stunde', label: 'Höchstens Mails je Stunde', help: 'Darüber hinaus werden Mails übersprungen und im Mail-Protokoll vermerkt. 0 schaltet Mails praktisch ab.' },
      { key: 'mail_symbol_cooldown_min', kind: 'int', min: 0, max: 1440, unit: 'Min', label: 'Pause je Wert', help: 'Nach einer Mail zu einem Wert kommt frühestens nach dieser Zeit die nächste zum selben Wert.' },
      { key: 'mail_on_exit', kind: 'bool', label: 'Auch Ausstiege melden', help: 'Mail, wenn eine gemeldete Empfehlung ihr Ziel, den Stopp oder das Ende der Haltedauer erreicht.' },
      { key: 'mail_attach_chart', kind: 'bool', label: 'Chart anhängen', help: 'Hängt ein Chartbild mit Einstieg, Stopp und Ziel an die Mail an (etwa 100 KB je Mail).' },
    ],
  },
  {
    title: 'Signale',
    intro: 'Welche Signale die Engine sucht und ab wann sie als Empfehlung ausgegeben werden. Die Strategieregeln selbst sind fest und nicht einstellbar.',
    fields: [
      { key: 'signals_enabled', kind: 'bool', label: 'Signalsuche', help: 'Aus: Es entstehen keine neuen Signale; offene werden weiter verfolgt.' },
      { key: 'signal_regions', kind: 'regions', label: 'Regionen', help: 'Nur in diesen Regionen sucht die Engine nach Setups.' },
      { key: 'disabled_setups', kind: 'setups', label: 'Setups', help: 'Abgewählte Setups erzeugen keine Signale mehr. Das ändert nicht die Regeln, nur welche genutzt werden.' },
      { key: 'allow_long', kind: 'bool', label: 'Long-Signale', help: 'Empfehlungen auf steigende Kurse.' },
      { key: 'allow_short', kind: 'bool', label: 'Short-Signale', help: 'Empfehlungen auf fallende Kurse (Leerverkauf im Papierdepot).' },
      { key: 'min_score_review', kind: 'float', min: 0, max: 1, step: 0.01, label: 'Mindestscore für die KI-Prüfung', help: 'Kandidaten darunter werden nur still als verworfene Kandidaten verfolgt.' },
      { key: 'min_score_actionable', kind: 'float', min: 0, max: 1, step: 0.01, label: 'Mindestscore für eine Empfehlung', help: 'Erst ab diesem Score wird ein Kandidat ausgegeben, gemeldet und vom KI-Depot gehandelt. Sinnvoll: mindestens so hoch wie der Score für die Prüfung.' },
      { key: 'llm_review', kind: 'bool', label: 'KI prüft Kandidaten', help: 'Die lokale KI kann bestätigen, die Größe reduzieren oder ein Veto einlegen. Aus: Signale gehen ohne Prüfung hinaus.' },
      { key: 'llm_review_timeout_s', kind: 'int', min: 5, max: 300, unit: 's', label: 'Wartezeit auf die KI-Prüfung', help: 'Antwortet die KI nicht rechtzeitig, geht das Signal „ohne Prüfung“ weiter. Länger warten heißt späterer Einstieg.' },
      { key: 'signal_cooldown_min', kind: 'int', min: 0, max: 1440, unit: 'Min', label: 'Pause nach einem Signal', help: 'So lange sucht die Engine nach dem Ende eines Signals keinen neuen Einstieg im selben Wert.' },
    ],
  },
  {
    title: 'Risiko KI-Depot',
    intro: 'Wie das KI-Depot Positionen bemisst. Die Grenzen für den Hebel gelten für beide Depots.',
    fields: [
      { key: 'risk_per_trade_pct', kind: 'float', min: 0, minExclusive: true, max: 5, step: 0.05, unit: '%', label: 'Risiko je Trade', help: 'Anteil des Depotwerts, der zwischen Einstieg und Stopp verloren gehen darf. 0,5 % heißt: 200 Stopps in Folge bis zum Totalverlust.' },
      { key: 'max_position_pct', kind: 'float', min: 0, minExclusive: true, max: 100, step: 0.5, unit: '%', label: 'Höchstens je Position', help: 'Obergrenze für den Wert einer Position, auch wenn der Stopp sehr eng liegt.' },
      { key: 'max_open_positions', kind: 'int', min: 1, max: 100, label: 'Höchstens offene Positionen', help: 'Weitere Empfehlungen werden vom KI-Depot übersprungen, bis eine Position geschlossen ist.' },
      { key: 'max_gross_leverage', kind: 'float', min: 0, minExclusive: true, max: 5, step: 0.1, unit: '×', label: 'Höchster Hebel (brutto)', help: 'Long- plus Short-Exposure geteilt durch den Depotwert, für beide Depots. 1 heißt: kein Kredit.' },
    ],
  },
  {
    title: 'KI-Hintergrundarbeit',
    intro: 'Die lokale KI arbeitet eine Anfrage nach der anderen ab; Signalprüfungen haben immer Vorrang.',
    fields: [
      { key: 'news_llm_enabled', kind: 'bool', label: 'Nachrichten analysieren', help: 'Relevanz, Stimmung und deutsche Zusammenfassung für neue Meldungen. Aus: Meldungen erscheinen ohne Bewertung.' },
      { key: 'chart_scan_enabled', kind: 'bool', label: 'Regelmäßige Chartanalysen', help: 'Die KI schaut sich die Charts offener Märkte in festen Abständen an.' },
      { key: 'chart_scan_interval_min', kind: 'int', min: 5, max: 1440, unit: 'Min', label: 'Abstand der Chartanalysen', help: 'Je Wert, solange sein Handelsplatz geöffnet ist. Kürzer heißt mehr Last auf der Grafikkarte.' },
    ],
  },
  {
    title: 'Vergleich',
    intro: 'Vergleichsindizes in der Depotkurve: als hätte man am Start für dasselbe Kapital den Index gekauft.',
    fields: [
      { key: 'benchmarks', kind: 'benchmarks', label: 'Vergleichsindizes', help: 'Werden in der Kurve „KI gegen mich“ grau gestrichelt gezeigt.' },
    ],
  },
];
const FIELDS = GROUPS.flatMap((g) => g.fields);

/** Client-side check of one value; German message or null. */
function checkField(f, value) {
  if (f.kind === 'int' || f.kind === 'float') {
    if (typeof value !== 'number' || Number.isNaN(value)) return 'Bitte eine Zahl eingeben.';
    if (f.kind === 'int' && !Number.isInteger(value)) return 'Bitte eine ganze Zahl eingeben.';
    if (f.minExclusive ? value <= f.min : value < f.min) return `Der Wert muss ${f.minExclusive ? 'größer als' : 'mindestens'} ${fmtNum(f.min, f.kind === 'int' ? 0 : 2)} sein.`;
    if (value > f.max) return `Der Wert darf höchstens ${fmtNum(f.max, f.kind === 'int' ? 0 : 2)} sein.`;
  }
  if (f.kind === 'regions' && f.key === 'signal_regions' && !value.length) return 'Ohne Region sucht die Engine keine Signale – dafür lieber die Signalsuche ausschalten.';
  return null;
}

/** Map a server error to fields: structured errors first, otherwise field keys or labels named in the text. */
function serverFieldErrors(err) {
  const out = { ...(err.fields || {}) };
  if (!Object.keys(out).length && err.message) {
    for (const f of FIELDS) {
      if (err.message.includes(f.key) || err.message.includes(f.label)) out[f.key] = err.message;
    }
  }
  return out;
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

function FieldControl({ f, value, onChange, error, indexes }) {
  const id = `set-${f.key}`;
  const describedBy = `${id}-help`;
  if (f.kind === 'bool') {
    return html`<div class="set-row">
      <div class="set-text"><label for=${id} class="set-label">${f.label}</label><p id=${describedBy} class="field-help">${f.help}</p></div>
      <label class="switch"><input id=${id} type="checkbox" checked=${value} aria-describedby=${describedBy} onChange=${(e) => onChange(e.currentTarget.checked)} /></label>
    </div>`;
  }
  if (f.kind === 'int' || f.kind === 'float') {
    return html`<div class="set-row">
      <div class="set-text"><label for=${id} class="set-label">${f.label}</label><p id=${describedBy} class="field-help">${f.help}</p>
        ${error && html`<p class="field-error" role="alert">${error}</p>`}</div>
      <div class="input-affix set-num">
        <input id=${id} class="input input-num" type="number" min=${f.min} max=${f.max} step=${f.step || 1} value=${value}
          aria-invalid=${error ? 'true' : 'false'} aria-describedby=${describedBy}
          onInput=${(e) => onChange(e.currentTarget.value === '' ? Number.NaN : e.currentTarget.valueAsNumber)} />
        ${f.unit && html`<span class="affix">${f.unit}</span>`}
      </div>
    </div>`;
  }
  const options = f.kind === 'regions' ? REGIONS.map((r) => [r, REGION_LABELS[r]])
    : f.kind === 'setups' ? Object.entries(SETUPS)
      : indexes.map((i) => [i.symbol, i.name]);
  // setups: the setting stores the DISABLED ones, the form shows the enabled ones as checked
  const isChecked = (k) => (f.kind === 'setups' ? !value.includes(k) : value.includes(k));
  const toggle = (k, on) => {
    const enabledList = f.kind === 'setups' ? !on : on;
    onChange(enabledList ? [...value.filter((x) => x !== k), k] : value.filter((x) => x !== k));
  };
  return html`<fieldset class="set-row set-multi" aria-describedby=${describedBy}>
    <div class="set-text"><legend class="set-label">${f.label}</legend><p id=${describedBy} class="field-help">${f.help}</p>
      ${error && html`<p class="field-error" role="alert">${error}</p>`}</div>
    <div class="set-checks">
      ${options.map(([k, label]) => html`<label class="check"><input type="checkbox" checked=${isChecked(k)} onChange=${(e) => toggle(k, e.currentTarget.checked)} />${label}</label>`)}
      ${f.kind === 'benchmarks' && value.filter((k) => !options.some(([o]) => o === k)).map((k) => html`<label class="check"><input type="checkbox" checked onChange=${() => toggle(k, false)} />${k}</label>`)}
    </div>
  </fieldset>`;
}

export function SettingsView() {
  const res = useResource('/settings', null, []);
  const instruments = useStore(app, (s) => s.instruments);
  const [form, setForm] = useState(null);
  const [serverErrors, setServerErrors] = useState({});
  const [formError, setFormError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState(null);
  useEffect(() => { if (res.data) setForm(res.data); }, [res.data]);
  const indexes = useMemo(() => instruments.filter((i) => i.asset_class === 'index'), [instruments]);

  if (res.loading) return html`<${Loading} lines=${8} />`;
  if (res.error) return html`<${ErrorState} error=${res.error} onRetry=${res.reload} />`;
  if (!form) return null;

  const clientErrors = Object.fromEntries(FIELDS.map((f) => [f.key, checkField(f, form[f.key])]).filter(([, v]) => v));
  const changed = FIELDS.filter((f) => !same(form[f.key], res.data[f.key])).map((f) => f.key);
  const scoreHint = form.min_score_actionable < form.min_score_review
    ? 'Der Score für eine Empfehlung liegt unter dem für die Prüfung: Kandidaten zwischen beiden Werten würden ungeprüft ausgegeben.' : null;
  const set = (key) => (v) => {
    setForm((f) => ({ ...f, [key]: v }));
    setServerErrors((e) => ({ ...e, [key]: undefined }));
  };
  const save = async (e) => {
    e.preventDefault();
    if (Object.keys(clientErrors).length || !changed.length) return;
    setBusy(true);
    setFormError(null);
    try {
      const patch = Object.fromEntries(changed.map((k) => [k, form[k]]));
      const saved = await api.put('/settings', patch);
      res.set(() => saved);
      setForm(saved);
      setServerErrors({});
      setToast('Gespeichert. Die Änderungen gelten ab dem nächsten Minutentakt.');
    } catch (err) {
      setServerErrors(serverFieldErrors(err));
      setFormError(errorText(err));
    } finally {
      setBusy(false);
    }
  };
  return html`<form class="settings" onSubmit=${save} noValidate>
    <div class="page-head"><div><h1>Einstellungen</h1>
      <p class="lede">Vorlieben für Alarme, Signale und das KI-Depot. Gespeichert gelten sie ab dem nächsten Minutentakt.</p></div></div>
    <div class="settings-grid"><div>
    ${GROUPS.map((g, gi) => html`<section class="panel set-group" aria-labelledby=${`set-grp-${gi}`}>
      <div class="set-group-head"><h2 id=${`set-grp-${gi}`}>${g.title}</h2><p class="small muted prose">${g.intro}</p></div>
      ${g.fields.map((f) => html`<${FieldControl} f=${f} value=${form[f.key]} onChange=${set(f.key)} indexes=${indexes}
          error=${clientErrors[f.key] || serverErrors[f.key] || (f.key === 'min_score_actionable' ? scoreHint : null)} />`)}
    </section>`)}
    </div>
    <aside class="panel small muted settings-aside">
      <h2 class="side-h">Bewusst nicht einstellbar</h2>
      <p>Indikatorperioden, Setup-Regeln und Score-Gewichte sind feste Lehrbuchwerte der Engine. Sie werden nie an die gespeicherte Historie angepasst – das wäre genau die Überanpassung, die diese Plattform vermeiden soll.</p>
      <p>Ändern sich diese Regeln, bekommt die Engine eine neue Version; Statistiken und Validierungen weisen die Version aus.</p>
      <p>Zugangsdaten für den Mailversand, Datenordner und Port stehen in der Datei <code>.env</code> und werden hier nie angezeigt.</p>
    </aside>
    </div>
    <div class=${`set-bar ${changed.length ? 'set-bar-dirty' : ''}`} role="region" aria-label="Speichern">
      ${formError && html`<p class="field-error" role="alert">${formError}</p>`}
      <span class="small muted">${changed.length ? `${changed.length} ${changed.length === 1 ? 'Änderung' : 'Änderungen'} nicht gespeichert` : 'Alle Änderungen gespeichert'}</span>
      <button type="button" class="btn" disabled=${!changed.length || busy} onClick=${() => { setForm(res.data); setServerErrors({}); setFormError(null); }}>Verwerfen</button>
      <button type="submit" class="btn btn-primary" disabled=${!changed.length || busy || Object.keys(clientErrors).length > 0}>${busy ? 'Wird gespeichert …' : 'Speichern'}</button>
    </div>
    <${Toast} message=${toast} onDone=${() => setToast(null)} />
  </form>`;
}
