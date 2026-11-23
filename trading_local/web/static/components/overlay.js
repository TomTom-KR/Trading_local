// Drawer and modal dialog: focus moves inside on open, Tab is trapped, Escape closes, focus returns
// to the element that opened the overlay. Overlays can be stacked (the order ticket opened from the
// signal drawer): only the topmost one reacts to the keyboard, and the page scrolls again only after
// the last one has closed.

import { html, useLayoutEffect, useRef } from '../vendor/preact-htm.mjs';

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Open overlays, oldest first (one token object per mounted overlay). */
const openOverlays = [];

function useFocusTrap(ref, onClose) {
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  // layout effect: focus moves in and Escape works immediately after the overlay renders
  useLayoutEffect(() => {
    const token = {};
    openOverlays.push(token);
    const previous = document.activeElement;
    const el = ref.current;
    const first = el?.querySelector('[data-autofocus]') || el?.querySelector(FOCUSABLE);
    (first || el)?.focus();
    const onKey = (e) => {
      if (openOverlays[openOverlays.length - 1] !== token) return;
      if (e.key === 'Escape') {
        e.preventDefault();
        closeRef.current();
        return;
      }
      if (e.key !== 'Tab' || !el) return;
      const items = [...el.querySelectorAll(FOCUSABLE)].filter((n) => n.offsetParent !== null);
      if (!items.length) return;
      const firstItem = items[0];
      const lastItem = items[items.length - 1];
      if (e.shiftKey && document.activeElement === firstItem) {
        e.preventDefault();
        lastItem.focus();
      } else if (!e.shiftKey && document.activeElement === lastItem) {
        e.preventDefault();
        firstItem.focus();
      }
    };
    document.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      openOverlays.splice(openOverlays.indexOf(token), 1);
      if (!openOverlays.length) document.body.style.overflow = '';
      if (previous && typeof previous.focus === 'function') previous.focus();
    };
  }, []);
}

/** Side drawer (slides in from the right; full width on phones). */
export function Drawer({ title, onClose, children, head }) {
  const ref = useRef(null);
  useFocusTrap(ref, onClose);
  return html`<div class="backdrop" onClick=${onClose}></div>
    <aside class="drawer" ref=${ref} role="dialog" aria-modal="true" aria-label=${title} tabindex="-1">
      <div class="drawer-head">
        <div class="drawer-title">${head || html`<h2>${title}</h2>`}</div>
        <button type="button" class="icon-btn" onClick=${onClose} aria-label="Schließen (Esc)" title="Schließen (Esc)">
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5L5 15" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>
        </button>
      </div>
      <div class="drawer-body">${children}</div>
    </aside>`;
}

/** Modal dialog with a title; the content provides its own action buttons. */
export function Dialog({ title, onClose, children }) {
  const ref = useRef(null);
  useFocusTrap(ref, onClose);
  return html`<div class="backdrop" onClick=${onClose}></div>
    <div class="dialog" ref=${ref} role="dialog" aria-modal="true" aria-labelledby="dlg-title" tabindex="-1">
      <h2 id="dlg-title">${title}</h2>
      ${children}
    </div>`;
}
