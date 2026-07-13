import { useEffect } from "react";
import type { RefObject } from "react";

/**
 * Global keyboard shortcuts (technical-spec §6.3):
 *  - "/"   focus the search input (prevents the browser's quick-find)
 *  - j/k   move the record selection down/up within the currently
 *          displayed page (no auto paging at page edges)
 *  - esc   blur the search input; also closes the help modal if open
 *  - ?     toggle the shortcut help modal
 *
 * "/", j/k and "?" are suppressed while focus is on an input/textarea/
 * contenteditable element so normal typing isn't hijacked; esc's blur
 * behavior stays active regardless.
 */
interface UseKeyboardShortcutsOptions {
  searchInputRef: RefObject<HTMLInputElement | null>;
  recordIds: string[];
  selectedRecordId: string | null;
  onSelectRecordId: (id: string) => void;
  helpOpen: boolean;
  onSetHelpOpen: (open: boolean) => void;
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || target.isContentEditable;
}

export function useKeyboardShortcuts({
  searchInputRef,
  recordIds,
  selectedRecordId,
  onSelectRecordId,
  helpOpen,
  onSetHelpOpen,
}: UseKeyboardShortcutsOptions): void {
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.isComposing || e.keyCode === 229) return;

      if (e.key === "Escape") {
        searchInputRef.current?.blur();
        if (helpOpen) onSetHelpOpen(false);
        return;
      }

      if (isEditableTarget(document.activeElement)) return;

      if (e.key === "/") {
        e.preventDefault();
        searchInputRef.current?.focus();
        return;
      }

      if (e.key === "?") {
        onSetHelpOpen(!helpOpen);
        return;
      }

      if (e.key === "j" || e.key === "k") {
        if (recordIds.length === 0) return;
        const currentIndex = selectedRecordId
          ? recordIds.indexOf(selectedRecordId)
          : -1;
        let nextIndex: number;
        if (currentIndex === -1) {
          nextIndex = 0;
        } else if (e.key === "j") {
          nextIndex = Math.min(currentIndex + 1, recordIds.length - 1);
        } else {
          nextIndex = Math.max(currentIndex - 1, 0);
        }
        onSelectRecordId(recordIds[nextIndex]);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [searchInputRef, recordIds, selectedRecordId, onSelectRecordId, helpOpen, onSetHelpOpen]);
}
