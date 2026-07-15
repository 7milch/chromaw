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

/**
 * Edit-mode shortcuts (technical-spec §6.3 e/d rows), write mode only.
 *
 * Deviation from spec: §6.3 assigns `d` to "show diff" and doesn't mention
 * delete. In this implementation the diff is already shown automatically as
 * part of the edit -> confirm -> save flow (MetadataEditor/DocumentEditor),
 * so there is no standalone "show diff" action left to bind. `d` is instead
 * bound to open the delete-record confirmation modal, matching the M2-9
 * task spec. `s` / `cmd+s` ("save edit") is intentionally omitted: saving
 * already goes through an explicit two-step Save -> Confirm button flow in
 * the editors, and there's no single unambiguous action for a keyboard
 * shortcut to trigger across both steps.
 */
export const EDIT_SHORTCUTS: ShortcutEntry[] = [
  { keys: "e", action: "edit metadata" },
  { keys: "d", action: "delete record" },
];
