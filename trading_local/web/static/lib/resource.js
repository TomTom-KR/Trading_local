// useResource: load one GET endpoint for a component, cancel on unmount or when the dependencies change.

import { useEffect, useRef, useState } from '../vendor/preact-htm.mjs';
import { api, errorText, isAbort } from './api.js';

/**
 * @template T
 * @param {string | null} path API path ('/signals'); null skips loading
 * @param {Record<string, unknown> | null} params query parameters
 * @param {unknown[]} deps reload when these change
 * @returns {{data: T | null, loading: boolean, error: string | null, reload: () => Promise<void>, set: (fn: (d: T | null) => T) => void}}
 */
export function useResource(path, params, deps) {
  const [state, setState] = useState({ data: null, loading: Boolean(path), error: null });
  const args = useRef({ path, params });
  args.current = { path, params };
  // only the newest request may write the state: an older reload that answers late is discarded
  const generation = useRef(0);

  const load = (signal) => {
    const { path: p, params: q } = args.current;
    if (!p) return Promise.resolve();
    const mine = ++generation.current;
    return api.get(p, q, { signal })
      .then((data) => {
        if (mine === generation.current) setState({ data, loading: false, error: null });
      })
      .catch((err) => {
        if (!isAbort(err) && mine === generation.current) setState((s) => ({ ...s, loading: false, error: errorText(err) }));
      });
  };

  useEffect(() => {
    const c = new AbortController();
    setState({ data: null, loading: Boolean(args.current.path), error: null });
    load(c.signal);
    return () => c.abort();
  }, deps);

  return {
    ...state,
    reload: () => load(),
    set: (fn) => setState((s) => ({ ...s, data: fn(s.data) })),
  };
}
