// REST client: JSON in and out, German error messages, timeouts and cancellation.
//
// Every state-changing request carries 'Content-Type: application/json' and a JSON body ('{}' when the
// call has no payload): the server rejects anything else. Failures throw ApiError whose ``message`` is a
// German sentence that can be shown to the user verbatim.

const DEFAULT_TIMEOUT_MS = 15000;

const OFFLINE_MESSAGE = 'Keine Verbindung zum Server. Läuft „python -m trading_local“?';

export class ApiError extends Error {
  /**
   * @param {string} message German text for the user
   * @param {number} status HTTP status (0 = network error or timeout)
   * @param {Record<string, string>} [fields] field name -> German message (validation errors)
   */
  constructor(message, status, fields = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.fields = fields;
  }
}

/**
 * Build '/api/...' URL with query parameters; null/undefined/'' parameters are left out.
 * @param {string} path
 * @param {Record<string, unknown>} [params]
 */
function apiUrl(path, params) {
  const query = new URLSearchParams();
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null && v !== '') query.set(k, String(v));
  }
  const qs = query.toString();
  return `/api${path}${qs ? `?${qs}` : ''}`;
}

/** Turn a FastAPI/pydantic 'detail' (string or list of errors) into a message plus per-field messages. */
function parseDetail(detail, status) {
  if (typeof detail === 'string' && detail.trim()) return { message: detail, fields: {} };
  if (Array.isArray(detail)) {
    const fields = {};
    const parts = [];
    for (const item of detail) {
      const loc = Array.isArray(item?.loc) ? item.loc.filter((p) => p !== 'body') : [];
      const msg = String(item?.msg ?? 'ungültiger Wert');
      if (loc.length) fields[String(loc[0])] = msg;
      parts.push(loc.length ? `${loc.join('.')}: ${msg}` : msg);
    }
    return { message: parts.join(' · ') || `Ungültige Eingabe (HTTP ${status}).`, fields };
  }
  return { message: statusMessage(status), fields: {} };
}

function statusMessage(status) {
  if (status === 404) return 'Nicht gefunden.';
  if (status === 409) return 'Konflikt mit dem aktuellen Zustand.';
  if (status === 415) return 'Der Server erwartet JSON.';
  if (status === 422) return 'Ungültige Eingabe.';
  if (status >= 500) return `Interner Serverfehler (HTTP ${status}). Details stehen im Server-Log.`;
  return `Anfrage fehlgeschlagen (HTTP ${status}).`;
}

/** The error a cancelled request rejects with (see isAbort). */
const abortError = () => new DOMException('Anfrage abgebrochen.', 'AbortError');

/**
 * Send one request and read the whole response. The timeout and the caller's AbortSignal cover the body
 * as well as the headers, and a request the caller cancelled always rejects with an AbortError - never
 * with data that arrived after the cancellation.
 * @param {string} method
 * @param {string} url
 * @param {{body?: unknown, signal?: AbortSignal, timeout?: number}} [opts]
 */
async function request(method, url, opts = {}) {
  if (opts.signal?.aborted) throw abortError();
  const controller = new AbortController();
  const timeoutMs = opts.timeout ?? DEFAULT_TIMEOUT_MS;
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onOuterAbort = () => controller.abort();
  opts.signal?.addEventListener('abort', onOuterAbort, { once: true });

  /** @type {RequestInit} */
  const init = { method, signal: controller.signal, headers: { Accept: 'application/json' } };
  if (method !== 'GET') {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(opts.body ?? {});
  }
  let response;
  let text;
  try {
    response = await fetch(url, init);
    text = await response.text();
  } catch {
    if (opts.signal?.aborted) throw abortError();
    if (timedOut) {
      throw new ApiError(`Der Server antwortet nicht (keine Antwort nach ${Math.round(timeoutMs / 1000)} s).`, 0);
    }
    throw new ApiError(OFFLINE_MESSAGE, 0);
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener('abort', onOuterAbort);
  }
  if (opts.signal?.aborted) throw abortError();

  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }
  if (!response.ok) {
    const { message, fields } = parseDetail(data?.detail, response.status);
    throw new ApiError(message, response.status, fields);
  }
  return data;
}

export const api = {
  /** @param {string} path @param {Record<string, unknown>} [params] @param {{signal?: AbortSignal, timeout?: number}} [opts] */
  get: (path, params, opts) => request('GET', apiUrl(path, params), opts),
  post: (path, body, opts) => request('POST', apiUrl(path), { ...opts, body }),
  put: (path, body, opts) => request('PUT', apiUrl(path), { ...opts, body }),
  patch: (path, body, opts) => request('PATCH', apiUrl(path), { ...opts, body }),
  del: (path, opts) => request('DELETE', apiUrl(path), opts),
};

/** True for the error a cancelled request throws (ignore it, the component has moved on). */
export function isAbort(err) {
  return err?.name === 'AbortError';
}

/** German message for anything thrown by the client. */
export function errorText(err) {
  if (err instanceof ApiError) return err.message;
  return 'Unerwarteter Fehler in der Oberfläche. Bitte die Seite neu laden.';
}
