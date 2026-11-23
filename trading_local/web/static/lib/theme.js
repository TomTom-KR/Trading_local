// Light/dark theme. Dark is the default; without an explicit choice the operating system's
// prefers-color-scheme decides. The choice is kept per browser (localStorage, best effort).

import { createStore } from './store.js';

const STORAGE_KEY = 'tl-theme';
const media = window.matchMedia('(prefers-color-scheme: light)');

function readStored() {
  try {
    const v = window.localStorage.getItem(STORAGE_KEY);
    return v === 'light' || v === 'dark' ? v : null;
  } catch {
    return null;
  }
}

const effective = (choice) => choice ?? (media.matches ? 'light' : 'dark');

/** {choice: 'light'|'dark'|null, theme: 'light'|'dark'} - ``theme`` is what is shown. */
export const themeStore = createStore({ choice: readStored(), theme: effective(readStored()) });

function apply() {
  const { choice } = themeStore.get();
  if (choice) document.documentElement.dataset.theme = choice;
  else delete document.documentElement.dataset.theme;
}

media.addEventListener('change', () => {
  const { choice } = themeStore.get();
  themeStore.set({ theme: effective(choice) });
});

/** Switch between light and dark and remember the choice. */
export function toggleTheme() {
  const next = themeStore.get().theme === 'dark' ? 'light' : 'dark';
  try {
    window.localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // storage blocked (private mode): the choice lasts for this page only
  }
  themeStore.set({ choice: next, theme: next });
  apply();
}

/** Apply the stored choice to <html> (called once at start-up). */
export function initTheme() {
  apply();
}

/** Current value of a CSS custom property, e.g. cssVar('--up'). */
export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
