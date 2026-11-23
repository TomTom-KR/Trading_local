// Shared application data: system status, sessions, instruments, quotes, sparklines and the two
// portfolio summaries. Loaded once at start-up, kept current by WebSocket events, and polled over REST
// while the WebSocket is down (quotes and status every 15 s).
//
// Session phases change during the day: the hub sends a 'sessions' message whenever a venue's phase
// changes, which patches the instruments at once and reloads sessions and instruments; both are also
// reloaded every 60 s (and after every reconnect) as a fallback, which keeps the instruments' open
// signals current as well.

import { api, errorText } from './api.js';
import { isLive, onLive } from './live.js';
import { applyPhases, phasesFromSessions } from './sessions.js';
import { createStore } from './store.js';
import { syncServerTime } from './time.js';

const FALLBACK_POLL_MS = 15000;
const SLOW_POLL_MS = 60000;
const SIGNAL_RELOAD_DELAY_MS = 2000; // a burst of signal updates causes one instrument reload

/**
 * @typedef {import('./types.js').SystemStatusOut} SystemStatusOut
 * @typedef {import('./types.js').SessionsOut} SessionsOut
 * @typedef {import('./types.js').InstrumentOut} InstrumentOut
 * @typedef {import('./types.js').QuoteOut} QuoteOut
 * @typedef {import('./types.js').PortfolioSummaryOut} PortfolioSummaryOut
 */

export const app = createStore({
  /** @type {SystemStatusOut | null} */ status: null,
  /** @type {string | null} */ statusError: null,
  /** @type {SessionsOut | null} */ sessions: null,
  /** @type {string | null} */ sessionsError: null,
  /** @type {InstrumentOut[]} */ instruments: [],
  instrumentsLoaded: false,
  /** @type {string | null} */ instrumentsError: null,
  /** @type {Record<string, QuoteOut>} */ quotes: {},
  /** @type {Record<string, Array<[number, number]>>} */ sparklines: {},
  /** @type {PortfolioSummaryOut[]} */ portfolios: [],
  /** @type {string | null} */ portfoliosError: null,
});

/** Instrument by symbol (null when unknown). */
export function instrumentOf(symbol) {
  return app.get().instruments.find((i) => i.symbol === symbol) ?? null;
}

async function loadStatus() {
  try {
    const status = await api.get('/status');
    syncServerTime(status.server_time);
    app.set({ status, statusError: null });
  } catch (err) {
    app.set({ statusError: errorText(err) });
  }
}

/** Copy venue phases onto the instruments (no store change when nothing differs). */
function patchPhases(phases) {
  const { instruments } = app.get();
  const next = applyPhases(instruments, phases);
  if (next !== instruments) app.set({ instruments: next });
}

async function loadSessions() {
  try {
    const sessions = await api.get('/sessions');
    syncServerTime(sessions.now);
    app.set({ sessions, sessionsError: null });
    patchPhases(phasesFromSessions(sessions));
  } catch (err) {
    app.set({ sessionsError: errorText(err) });
  }
}

export async function loadInstruments() {
  try {
    const instruments = await api.get('/instruments', { include_inactive: true });
    const quotes = { ...app.get().quotes };
    for (const inst of instruments) {
      if (inst.quote && !quotes[inst.symbol]) quotes[inst.symbol] = inst.quote;
    }
    app.set({ instruments, instrumentsLoaded: true, instrumentsError: null, quotes });
  } catch (err) {
    app.set({ instrumentsError: errorText(err), instrumentsLoaded: true });
  }
}

let instrumentsTimer = null;
/** Reload the instruments shortly after a burst of events. */
function refreshInstrumentsSoon(ms = 400) {
  clearTimeout(instrumentsTimer);
  instrumentsTimer = setTimeout(loadInstruments, ms);
}

/** Merge fresh quotes into the store (keeps quotes of other symbols). */
function mergeQuotes(items) {
  if (!Array.isArray(items) || !items.length) return;
  const quotes = { ...app.get().quotes };
  for (const q of items) {
    const old = quotes[q.symbol];
    if (!old || q.ts >= old.ts) quotes[q.symbol] = q;
  }
  app.set({ quotes });
}

async function loadQuotes() {
  try {
    mergeQuotes(await api.get('/quotes'));
  } catch {
    // the status indicator and the connection banner already explain a dead server
  }
}

async function loadSparklines() {
  try {
    const out = await api.get('/sparklines');
    app.set({ sparklines: out.series || {} });
  } catch {
    // sparklines are decoration of the watchlist; keep the previous ones
  }
}

export async function loadPortfolios() {
  try {
    app.set({ portfolios: await api.get('/portfolios'), portfoliosError: null });
  } catch (err) {
    app.set({ portfoliosError: errorText(err) });
  }
}

let portfolioTimer = null;
/** Re-fetch the summaries shortly after a burst of 'portfolio'/'order' events. */
export function refreshPortfoliosSoon() {
  clearTimeout(portfolioTimer);
  portfolioTimer = setTimeout(loadPortfolios, 400);
}

let sessionsTimer = null;
function refreshSessionsSoon() {
  clearTimeout(sessionsTimer);
  sessionsTimer = setTimeout(loadSessions, 400);
}

/** Load everything once and wire live updates and the polling fallback. */
export function startData() {
  loadStatus();
  loadSessions();
  loadInstruments();
  loadQuotes();
  loadSparklines();
  loadPortfolios();

  onLive('status', (status) => app.set({ status, statusError: null }));
  onLive('quotes', (data) => mergeQuotes(data?.items));
  onLive('sessions', (data) => {
    if (data?.exchange && data?.phase) patchPhases({ [data.exchange]: data.phase });
    refreshSessionsSoon();
    refreshInstrumentsSoon();
  });
  onLive('signal', () => refreshInstrumentsSoon(SIGNAL_RELOAD_DELAY_MS)); // open signal per instrument
  onLive('portfolio', refreshPortfoliosSoon);
  onLive('order', refreshPortfoliosSoon);
  onLive('open', () => {
    loadStatus();
    loadQuotes();
    loadSessions();
    loadInstruments();
    loadPortfolios();
  });
  // connection lost: ask over REST at once, so a dead server shows up without waiting for the next poll
  onLive('close', () => {
    loadStatus();
    loadQuotes();
  });

  setInterval(() => {
    if (!isLive()) {
      loadQuotes();
      loadStatus();
    }
  }, FALLBACK_POLL_MS);
  setInterval(() => {
    loadSessions();
    loadInstruments();
    loadSparklines();
    loadPortfolios();
  }, SLOW_POLL_MS);
}
