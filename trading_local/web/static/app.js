// Bootstrap: theme, data, WebSocket, router and the app frame.

import { html, render, useEffect, useErrorBoundary, useRef } from './vendor/preact-htm.mjs';
import { startData } from './lib/data.js';
import { startLive } from './lib/live.js';
import { useRoute } from './lib/router.js';
import { initTheme } from './lib/theme.js';
import { TicketDrawer } from './components/ticket-drawer.js';
import { TopBar } from './components/topbar.js';
import { AgendaView } from './views/agenda.js';
import { DepotView } from './views/depot.js';
import { InstrumentView } from './views/instrument.js';
import { MarketsView } from './views/markets.js';
import { NewsView } from './views/news.js';
import { OverviewView } from './views/overview.js';
import { SettingsView } from './views/settings.js';
import { SignalsView } from './views/signals.js';
import { SystemView } from './views/system.js';
import { ValidationView } from './views/validation.js';

const TITLES = {
  overview: 'Übersicht', instrument: 'Markt', markets: 'Märkte', signals: 'Signale', depot: 'Depot',
  news: 'Nachrichten', agenda: 'Termine', validation: 'Validierung', system: 'System', settings: 'Einstellungen', notfound: 'Übersicht',
};

/** Keeps one broken view from blanking the whole app; offers a reload. */
function ViewBoundary({ children, routeKey }) {
  const [error, resetError] = useErrorBoundary((err) => console.error('view failed', err));
  useEffect(() => { if (error) resetError(); }, [routeKey]);
  if (error) {
    return html`<div class="state state-error" role="alert">
      <strong>Diese Ansicht konnte nicht angezeigt werden.</strong>
      <span>Ein Fehler in der Oberfläche ist aufgetreten. Andere Ansichten funktionieren weiter; ein Neuladen der Seite behebt das meist.</span>
      <div><button type="button" class="btn btn-sm" onClick=${() => window.location.reload()}>Seite neu laden</button></div>
    </div>`;
  }
  return children;
}

function currentView(route) {
  switch (route.name) {
    case 'instrument':
      return html`<${InstrumentView} symbol=${route.params.symbol} />`;
    case 'markets':
      return html`<${MarketsView} />`;
    case 'signals':
      return html`<${SignalsView} id=${route.params.id || null} />`;
    case 'depot':
      return html`<${DepotView} />`;
    case 'news':
      return html`<${NewsView} />`;
    case 'agenda':
      return html`<${AgendaView} />`;
    case 'validation':
      return html`<${ValidationView} />`;
    case 'system':
      return html`<${SystemView} />`;
    case 'settings':
      return html`<${SettingsView} />`;
    default:
      return html`<${OverviewView} />`;
  }
}

function App() {
  const route = useRoute();
  const mainRef = useRef(null);
  // the signals drawer is part of the signals page: opening it must not scroll the list away
  const pageKey = route.name === 'instrument' ? `instrument:${route.params.symbol}` : route.name;
  useEffect(() => {
    document.title = `${TITLES[route.name] || 'Übersicht'}${route.params.symbol ? ` ${route.params.symbol}` : ''} · Trading Local`;
  }, [route.path]);
  useEffect(() => { window.scrollTo(0, 0); }, [pageKey]);
  const skip = () => mainRef.current?.focus();
  return html`
    <button type="button" class="skip-link" onClick=${skip}>Zum Inhalt springen</button>
    <${TopBar} route=${route} />
    <main id="main" class=${`main ${route.name === 'overview' || route.name === 'notfound' ? 'main-full' : ''}`} tabindex="-1" ref=${mainRef}>
      <${ViewBoundary} routeKey=${pageKey}>${currentView(route)}</${ViewBoundary}>
    </main>
    <${TicketDrawer} />`;
}

function mount() {
  const root = document.getElementById('app');
  root.textContent = '';
  render(html`<${App} />`, root);
}

initTheme();
startData();
startLive();
// canvas charts draw text once: wait (briefly) for the web font so axis labels use Archivo
const fontReady = document.fonts ? document.fonts.load('400 12px Archivo').catch(() => null) : Promise.resolve();
Promise.race([fontReady, new Promise((resolve) => { setTimeout(resolve, 1200); })]).then(mount);
