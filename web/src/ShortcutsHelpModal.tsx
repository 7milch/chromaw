import { SHORTCUTS } from "./shortcuts";

interface ShortcutsHelpModalProps {
  onClose: () => void;
}

/**
 * Shortcut help modal (technical-spec §6.3). Closes on esc (handled by
 * useKeyboardShortcuts), background click, or the close button.
 */
function ShortcutsHelpModal({ onClose }: ShortcutsHelpModalProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        onClick={(e) => e.stopPropagation()}
        className="w-80 rounded border border-slate-700 bg-slate-900 p-4 text-slate-100 shadow-lg"
      >
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">
            Keyboard shortcuts
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded px-1.5 py-0.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
          >
            ×
          </button>
        </div>
        <table className="w-full text-sm">
          <tbody>
            {SHORTCUTS.map((s) => (
              <tr key={s.keys} className="border-t border-slate-800 first:border-t-0">
                <td className="py-1.5 pr-3 align-top">
                  <kbd className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-xs text-slate-200">
                    {s.keys}
                  </kbd>
                </td>
                <td className="py-1.5 align-top text-slate-300">{s.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default ShortcutsHelpModal;
