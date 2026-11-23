// Hash router. Routes:
//   #/                   overview            #/markt/:symbol      instrument
//   #/maerkte            watchlist           #/signale            signals
//   #/signale/:id        signal drawer       #/depot              portfolios
//   #/news               news                #/validierung        validation
//   #/system             system              #/einstellungen      settings
// Unknown hashes fall back to the overview (name 'notfound' so the view can say so).

import { useEffect, useState } from '../vendor/preact-htm.mjs';

/**
 * @typedef {{name: string, params: Record<string, string>, path: string}} Route
 */

/** @returns {Route} */
function parseHash(hash = window.location.hash) {
  const path = (hash.replace(/^#/, '') || '/').split('?')[0];
  const parts = path.split('/').filter(Boolean).map((p) => {
    try {
      return decodeURIComponent(p);
    } catch {
      return p;
    }
  });
  const [head, arg] = parts;
  const route = (name, params = {}) => ({ name, params, path });
  if (!head) return route('overview');
  switch (head) {
    case 'markt':
      return arg ? route('instrument', { symbol: arg }) : route('markets');
    case 'maerkte':
      return route('markets');
    case 'signale':
      return arg ? route('signals', { id: arg }) : route('signals');
    case 'depot':
      return route('depot');
    case 'news':
      return route('news');
    case 'validierung':
      return route('validation');
    case 'system':
      return route('system');
    case 'einstellungen':
      return route('settings');
    default:
      return route('notfound');
  }
}

/** Hash link for a path: href('/markt/SAP.DE') -> '#/markt/SAP.DE'. */
export function href(path) {
  return `#${path}`;
}

/** Link to an instrument page (symbols contain '^', '=' and '.'). */
export function symbolHref(symbol) {
  return `#/markt/${encodeURIComponent(symbol)}`;
}

/** Navigate programmatically. */
export function navigate(path) {
  window.location.hash = path;
}

/** Current route; re-renders on every hash change. */
export function useRoute() {
  const [route, setRoute] = useState(() => parseHash());
  useEffect(() => {
    const onChange = () => setRoute(parseHash());
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);
  return route;
}
