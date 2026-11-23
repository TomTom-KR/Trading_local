// The order ticket as a drawer that every 'Übernehmen' button can open (prefilled from a signal).

import { html } from '../vendor/preact-htm.mjs';
import { signalLevels } from '../lib/signals.js';
import { createStore, useStore } from '../lib/store.js';
import { OrderTicket } from './order-ticket.js';
import { Drawer } from './overlay.js';

const ticketStore = createStore({ open: false, prefill: {} });

/** Open the ticket prefilled for the user's portfolio, linked to ``signal``. */
export function followSignal(signal) {
  const { stop, target } = signalLevels(signal);
  ticketStore.set({
    open: true,
    prefill: { symbol: signal.symbol, side: signal.direction, stop, target, signalId: signal.id },
  });
}

export function TicketDrawer() {
  const { open, prefill } = useStore(ticketStore, (s) => s);
  if (!open) return null;
  const close = () => ticketStore.set({ open: false, prefill: {} });
  return html`<${Drawer} title="Order für mein Depot" onClose=${close}>
    <${OrderTicket} prefill=${prefill} idPrefix="drawer-ot" />
  </${Drawer}>`;
}
