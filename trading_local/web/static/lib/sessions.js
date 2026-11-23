// Session phases per venue for the instrument list (pure functions).
//
// GET /api/instruments reports every instrument's phase at the time of the request. The phases change during
// the day (Xetra opens, Tokyo goes to lunch); the session overview and the hub's 'sessions' messages carry the
// current phase per venue, and these helpers copy it onto the instruments, so the watchlist dots, the markets
// table and the instrument header switch without waiting for the next instrument reload.

/**
 * Venue -> phase from a SessionsOut payload (venues the overview does not list are missing).
 * @param {import('./types.js').SessionsOut | null | undefined} sessions
 * @returns {Record<string, string>}
 */
export function phasesFromSessions(sessions) {
  const phases = {};
  for (const region of sessions?.regions || []) {
    for (const ex of region.exchanges || []) {
      if (ex?.exchange && ex.phase) phases[ex.exchange] = ex.phase;
    }
  }
  return phases;
}

/**
 * Instruments with ``phase`` / ``is_open`` taken from ``phases`` (venue -> phase). Instruments of venues not
 * in ``phases`` stay as they are; when nothing changes the SAME array is returned (store subscribers compare
 * with Object.is and do not re-render).
 * @param {import('./types.js').InstrumentOut[]} instruments
 * @param {Record<string, string>} phases
 */
export function applyPhases(instruments, phases) {
  let changed = false;
  const next = instruments.map((inst) => {
    const phase = phases[inst.exchange];
    if (!phase || (phase === inst.phase && inst.is_open === (phase === 'open'))) return inst;
    changed = true;
    return { ...inst, phase, is_open: phase === 'open' };
  });
  return changed ? next : instruments;
}
