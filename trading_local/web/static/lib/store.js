// A tiny observable store and the hooks that connect Preact components to it.
//
// createStore(initial) -> { get, set, update, subscribe }. Subscribers run synchronously after every
// change; components use useStore(store, selector) and re-render only when the selected value changes
// (compared with Object.is), so selectors should return stored objects rather than fresh ones.

import {
  useCallback, useEffect, useRef, useState,
} from '../vendor/preact-htm.mjs';
import { serverNow } from './time.js';

/**
 * @template T
 * @param {T} initial
 */
export function createStore(initial) {
  let state = initial;
  /** @type {Set<(s: T) => void>} */
  const listeners = new Set();
  return {
    /** @returns {T} */
    get: () => state,
    /** Shallow-merge a patch into the state. @param {Partial<T>} patch */
    set(patch) {
      state = { ...state, ...patch };
      listeners.forEach((fn) => fn(state));
    },
    /** Replace the state through a function of the previous state. @param {(s: T) => T} fn */
    update(fn) {
      state = fn(state);
      listeners.forEach((l) => l(state));
    },
    /** @param {(s: T) => void} fn @returns {() => void} unsubscribe */
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}

/**
 * Subscribe a component to a slice of a store.
 * @template T, S
 * @param {{get: () => T, subscribe: (fn: (s: T) => void) => () => void}} store
 * @param {(s: T) => S} selector
 * @returns {S}
 */
export function useStore(store, selector) {
  const [value, setValue] = useState(() => selector(store.get()));
  const selectorRef = useRef(selector);
  selectorRef.current = selector;
  useEffect(() => {
    const check = (s) => {
      const next = selectorRef.current(s);
      setValue((prev) => (Object.is(prev, next) ? prev : next));
    };
    check(store.get());
    return store.subscribe(check);
  }, [store]);
  return value;
}

// ------------------------------------------------------------------------------------------------
// Shared ticker: one interval for the whole app instead of one per countdown.
// ------------------------------------------------------------------------------------------------
const tickStore = createStore({ second: Math.floor(Date.now() / 1000) });
setInterval(() => tickStore.set({ second: Math.floor(Date.now() / 1000) }), 1000);

/**
 * Re-render the calling component every ``everySeconds`` seconds; returns the current SERVER time in
 * whole seconds (use it for ages and countdowns).
 * @param {number} [everySeconds]
 */
export function useTick(everySeconds = 1) {
  useStore(tickStore, (s) => Math.floor(s.second / everySeconds));
  return Math.floor(serverNow());
}

/**
 * A trigger that runs ``fn`` once, ``ms`` milliseconds after its LAST call (a burst of live events
 * causes one reload). A pending run is dropped when the component unmounts.
 */
export function useSoon(fn, ms = 500) {
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const timer = useRef(null);
  useEffect(() => () => clearTimeout(timer.current), []);
  return useCallback(() => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => fnRef.current(), ms);
  }, [ms]);
}

/**
 * Run ``fn`` every ``ms`` milliseconds while the component is mounted (and once immediately when
 * ``immediate`` is true). The latest ``fn`` is always used, so it may close over fresh props.
 */
export function useInterval(fn, ms, immediate = false) {
  const fnRef = useRef(fn);
  fnRef.current = fn;
  useEffect(() => {
    if (immediate) fnRef.current();
    if (!ms) return undefined;
    const id = setInterval(() => fnRef.current(), ms);
    return () => clearInterval(id);
  }, [ms, immediate]);
}
