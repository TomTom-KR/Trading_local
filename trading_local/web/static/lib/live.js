// WebSocket client for /ws: reconnects with exponential backoff plus jitter, re-subscribes the bar
// symbols after every reconnect and exposes the connection state for the top bar.
//
// Server -> client messages are {type, data}; components listen with onLive(type, fn) / useLive(type, fn).
// Bar subscriptions are reference-counted: every chart calls subscribeBars(symbol) and the client sends
// the union (replace semantics, at most 20 symbols, most recently requested first).

import { useEffect, useRef } from '../vendor/preact-htm.mjs';
import { createStore } from './store.js';
import { syncServerTime } from './time.js';

const MAX_BAR_SYMBOLS = 20;
const BACKOFF_BASE_MS = 1000;
const BACKOFF_MAX_MS = 30000;
const SILENCE_LIMIT_MS = 30000; // the server sends status every 5 s: silence this long means a dead link

/** @typedef {'connecting' | 'open' | 'closed'} ConnState */
/** @type {ReturnType<typeof createStore<{state: ConnState, retryAt: number | null, attempts: number}>>} */
export const connection = createStore({ state: 'connecting', retryAt: null, attempts: 0 });

/** @type {Map<string, Set<(data: any) => void>>} */
const listeners = new Map();
/** @type {Map<string, number>} symbol -> number of active subscribers (insertion order = recency) */
const barRefs = new Map();

let socket = null;
let retryTimer = null;
let attempts = 0;
let lastMessageAt = 0;
let started = false;

/**
 * Listen to one message type; returns the unsubscribe function.
 * @param {string} type
 * @param {(data: any) => void} fn
 */
export function onLive(type, fn) {
  let set = listeners.get(type);
  if (!set) {
    set = new Set();
    listeners.set(type, set);
  }
  set.add(fn);
  return () => set.delete(fn);
}

/** Hook form of onLive; the newest ``fn`` is always called. */
export function useLive(type, fn) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => onLive(type, (data) => ref.current(data)), [type]);
}

function dispatch(type, data) {
  const set = listeners.get(type);
  if (!set) return;
  for (const fn of [...set]) {
    try {
      fn(data);
    } catch (err) {
      console.warn(`live: handler for '${type}' failed`, err);
    }
  }
}

function send(message) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

function sendSubscription() {
  const symbols = [...barRefs.keys()].reverse().slice(0, MAX_BAR_SYMBOLS);
  send({ type: 'subscribe', symbols });
}

/**
 * Receive 'bar' events for ``symbol`` while the returned function has not been called.
 * @param {string} symbol
 */
export function subscribeBars(symbol) {
  const count = barRefs.get(symbol) ?? 0;
  barRefs.delete(symbol); // re-insert: most recent last
  barRefs.set(symbol, count + 1);
  sendSubscription();
  let active = true;
  return () => {
    if (!active) return;
    active = false;
    const left = (barRefs.get(symbol) ?? 1) - 1;
    if (left <= 0) barRefs.delete(symbol);
    else barRefs.set(symbol, left);
    sendSubscription();
  };
}

function scheduleReconnect() {
  clearTimeout(retryTimer);
  attempts += 1;
  const base = Math.min(BACKOFF_MAX_MS, BACKOFF_BASE_MS * 2 ** (attempts - 1));
  const delay = Math.round(base * (0.7 + Math.random() * 0.6)); // +-30 % jitter
  connection.set({ state: 'closed', retryAt: Date.now() + delay, attempts });
  retryTimer = setTimeout(connect, delay);
}

function connect() {
  clearTimeout(retryTimer);
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  let ws;
  try {
    ws = new WebSocket(`${proto}//${window.location.host}/ws`);
  } catch {
    scheduleReconnect();
    return;
  }
  socket = ws;
  connection.set({ state: 'connecting', retryAt: null, attempts });
  ws.onopen = () => {
    attempts = 0;
    lastMessageAt = Date.now();
    connection.set({ state: 'open', retryAt: null, attempts: 0 });
    sendSubscription();
    dispatch('open', null);
  };
  ws.onmessage = (event) => {
    lastMessageAt = Date.now();
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    if (!msg || typeof msg.type !== 'string') return;
    if (msg.type === 'hello' || msg.type === 'status') syncServerTime(msg.data?.server_time);
    dispatch(msg.type, msg.data);
  };
  ws.onclose = () => {
    if (socket !== ws) return;
    socket = null;
    dispatch('close', null);
    scheduleReconnect();
  };
  ws.onerror = () => {
    // 'close' always follows 'error'; reconnecting happens there.
  };
}

/** Try to connect immediately (button in the top bar). */
export function reconnectNow() {
  attempts = 0;
  if (socket) {
    const old = socket;
    socket = null;
    old.close();
  }
  connect();
}

/** Open the connection once (called by the app bootstrap). */
export function startLive() {
  if (started) return;
  started = true;
  connect();
  setInterval(() => {
    if (socket && socket.readyState === WebSocket.OPEN && Date.now() - lastMessageAt > SILENCE_LIMIT_MS) {
      const dead = socket;
      socket = null;
      dead.close();
      dispatch('close', null);
      scheduleReconnect();
    }
  }, 5000);
}

/** True while the WebSocket is connected (REST polling takes over otherwise). */
export function isLive() {
  return connection.get().state === 'open';
}
