/**
 * Keyboard shortcut catalog for the help modal (technical-spec §6.3).
 * Kept as data so M2 can append edit-related entries (e/s/d) without
 * touching rendering code.
 */
export interface ShortcutEntry {
  keys: string;
  action: string;
}

export const SHORTCUTS: ShortcutEntry[] = [
  { keys: "/", action: "search focus" },
  { keys: "j / k", action: "next / previous record" },
  { keys: "esc", action: "close modal / cancel" },
  { keys: "?", action: "shortcut help" },
];
