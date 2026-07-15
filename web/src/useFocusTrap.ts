import { useEffect } from "react";
import type { RefObject } from "react";

/**
 * Simple focus trap for modal dialogs (M1-5 roadmap carry-over): autofocuses
 * the first focusable element inside ``containerRef`` on mount (unless focus
 * is already inside, e.g. an input with its own ``autoFocus``) and keeps
 * Tab / shift+Tab cycling within the container instead of escaping to the
 * page behind the modal.
 */
export function useFocusTrap(containerRef: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    function getFocusable(): HTMLElement[] {
      if (!container) return [];
      return Array.from(
        container.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
        )
      );
    }

    if (!container.contains(document.activeElement)) {
      const focusable = getFocusable();
      focusable[0]?.focus();
    }

    function handleKeyDown(e: KeyboardEvent) {
      if (e.key !== "Tab") return;
      const els = getFocusable();
      if (els.length === 0) return;
      const first = els[0];
      const last = els[els.length - 1];
      if (e.shiftKey) {
        if (document.activeElement === first || !container!.contains(document.activeElement)) {
          e.preventDefault();
          last.focus();
        }
      } else if (document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }

    container.addEventListener("keydown", handleKeyDown);
    return () => container.removeEventListener("keydown", handleKeyDown);
  }, [containerRef]);
}
