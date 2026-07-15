import { useRef, useState } from "react";
import { useFocusTrap } from "./useFocusTrap";

interface RenameCollectionModalProps {
  currentName: string;
  onConfirm: (newName: string) => Promise<void>;
  onCancel: () => void;
}

/**
 * Two-step rename flow (technical-spec §3.2, §5.2, §6.5, roadmap M2-7):
 * first collect the new name, then require the user to type the
 * collection's *current* name to confirm -- matching
 * ``PATCH /api/collections/{name}``'s ``confirm`` field, which the server
 * checks against the current name, not the new one.
 */
export default function RenameCollectionModal({
  currentName,
  onConfirm,
  onCancel,
}: RenameCollectionModalProps) {
  const [step, setStep] = useState<"name" | "confirm">("name");
  const [newName, setNewName] = useState("");
  const [confirmText, setConfirmText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  useFocusTrap(containerRef);

  const matches = confirmText === currentName;

  async function handleConfirm() {
    if (!matches || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await onConfirm(newName);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div
        ref={containerRef}
        role="dialog"
        aria-modal="true"
        aria-label="Rename collection"
        className="w-full max-w-sm rounded border border-slate-700 bg-slate-900 p-4 shadow-xl"
      >
        <h2 className="mb-2 text-sm font-semibold text-slate-100">Rename collection</h2>

        {step === "name" && (
          <>
            <p className="mb-2 text-xs text-slate-400">
              Renaming <code className="text-slate-200">{currentName}</code>
            </p>
            <input
              type="text"
              autoFocus
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && newName.trim()) setStep("confirm");
                if (e.key === "Escape") onCancel();
              }}
              placeholder="new name"
              className="mb-3 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-100"
            />
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={onCancel}
                className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={!newName.trim() || newName === currentName}
                onClick={() => setStep("confirm")}
                className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </>
        )}

        {step === "confirm" && (
          <>
            <p className="mb-3 text-xs text-slate-400">
              <code className="text-slate-200">{currentName}</code> →{" "}
              <code className="text-slate-200">{newName}</code>
            </p>
            <p className="mb-2 text-xs text-slate-400">
              Type{" "}
              <code className="rounded bg-slate-800 px-1 py-0.5 text-slate-200">
                {currentName}
              </code>{" "}
              to confirm.
            </p>
            <input
              type="text"
              autoFocus
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleConfirm();
                if (e.key === "Escape") onCancel();
              }}
              className="mb-3 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-100"
            />
            {error && <p className="mb-2 text-xs text-red-400">{error}</p>}
            <div className="flex justify-end gap-2">
              <button
                type="button"
                disabled={submitting}
                onClick={() => setStep("name")}
                className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800 disabled:opacity-50"
              >
                Back
              </button>
              <button
                type="button"
                disabled={submitting}
                onClick={onCancel}
                className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={!matches || submitting}
                onClick={handleConfirm}
                className="rounded border border-red-700 bg-red-900/60 px-2 py-1 text-xs text-red-200 hover:bg-red-900 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {submitting ? "Renaming..." : "Rename"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
