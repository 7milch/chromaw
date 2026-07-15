import { useRef, useState } from "react";
import { useFocusTrap } from "./useFocusTrap";

interface ConfirmNameModalProps {
  /** Short action label shown in the header, e.g. "Delete record". */
  title: string;
  /** Human-readable description of what's about to happen. */
  description: string;
  /** The exact string the user must type to enable the confirm button. */
  expected: string;
  /** Label for the confirm button, e.g. "Delete". */
  confirmLabel: string;
  onConfirm: (typed: string) => Promise<void>;
  onCancel: () => void;
}

/**
 * Modal used by the delete-record / delete-collection / rename-collection
 * flows (technical-spec §3.2, §6.5, roadmap M2-7): the confirm button stays
 * disabled until the user types the target name/id exactly, matching the
 * spec's "Type `delete memory` to confirm" example. The server performs the
 * authoritative confirm check again -- this is a UX gate, not the source of
 * truth for whether the action is allowed.
 */
export default function ConfirmNameModal({
  title,
  description,
  expected,
  confirmLabel,
  onConfirm,
  onCancel,
}: ConfirmNameModalProps) {
  const [typed, setTyped] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  useFocusTrap(containerRef);

  const matches = typed === expected;

  async function handleConfirm() {
    if (!matches || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await onConfirm(typed);
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
        aria-label={title}
        className="w-full max-w-sm rounded border border-slate-700 bg-slate-900 p-4 shadow-xl"
      >
        <h2 className="mb-2 text-sm font-semibold text-slate-100">{title}</h2>
        <p className="mb-3 text-xs text-slate-400">{description}</p>
        <p className="mb-2 text-xs text-slate-400">
          Type <code className="rounded bg-slate-800 px-1 py-0.5 text-slate-200">{expected}</code>{" "}
          to confirm.
        </p>
        <input
          type="text"
          autoFocus
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
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
            {submitting ? "Working..." : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
