// Server clock. Ages, countdowns and the 'jetzt' line use the SERVER's time, so they stay right when the
// browser clock is off (and when the development mock runs on a demo date). The offset is learnt from
// every payload that carries the server time (hello, status, sessions).

let offsetSeconds = 0;

/** Remember the server's current time (epoch seconds). */
export function syncServerTime(serverTs) {
  if (typeof serverTs === 'number' && Number.isFinite(serverTs)) {
    offsetSeconds = serverTs - Date.now() / 1000;
  }
}

/** Current server time in epoch seconds (fractional). */
export function serverNow() {
  return Date.now() / 1000 + offsetSeconds;
}
