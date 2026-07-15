import { useRef, useState } from "react";
import { useFocusTrap } from "./useFocusTrap";

interface BulkPatchModalProps {
  /** Number of records the patch would be applied to (for the description). */
  selectedCount: number;
  /** The collection name the user must type to confirm (technical-spec §6.5). */
  collectionName: string;
  onConfirm: (metadata: Record<string, unknown>) => Promise<void>;
  onCancel: () => void;
}

/**
 * Modal for the "Edit metadata (N)" bulk-patch flow (technical-spec §3.3,
 * §5.4, §6.5, roadmap M4-4): the user supplies a JSON object of metadata to
 * merge into every selected record, then types the collection's name to
 * confirm -- same "type the target name" convention as
 * ``ConfirmNameModal``'s bulk-delete usage, since a bulk operation has no
 * single record id to type. The server performs the authoritative
 * confirm/validation check again; this is only a UX gate.
 */
export default function BulkPatchModal({
  selectedCount,
  collectionName,
  onConfirm,
  onCancel,
}: BulkPatchModalProps) {
  const [metadataText, setMetadataText] = useState("{\n  \n}");
  const [typed, setTyped] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  useFocusTrap(containerRef);

  const nameMatches = typed === collectionName;

  function parseMetadata(): Record<string, unknown> | null {
    let parsed: unknown;
    try {
      parsed = JSON.parse(metadataText);
    } catch (err) {
      setError(`Invalid JSON: ${err instanceof Error ? err.message : String(err)}`);
      return null;
    }
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      setError("metadata must be a JSON object.");
      return null;
    }
    if (Object.keys(parsed as Record<string, unknown>).length === 0) {
      setError("metadata must be a non-empty object.");
      return null;
    }
    return parsed as Record<string, unknown>;
  }

  async function handleConfirm() {
    if (!nameMatches || submitting) return;
    setError(null);
    const metadata = parseMetadata();
    if (metadata === null) return;

    setSubmitting(true);
    try {
      await onConfirm(metadata);
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
        aria-label="Edit metadata for selected records"
        className="w-full max-w-md rounded border border-slate-700 bg-slate-900 p-4 shadow-xl"
      >
        <h2 className="mb-2 text-sm font-semibold text-slate-100">Edit metadata</h2>
        <p className="mb-3 text-xs text-slate-400">
          This merges the metadata below into {selectedCount} selected record
          {selectedCount === 1 ? "" : "s"} in collection "{collectionName}".
        </p>
        <p className="mb-2 text-xs text-amber-300">
          Metadata is merged, not replaced -- existing keys not listed below are
          kept, and there is no way to delete a key this way. Listed keys are
          overwritten.
        </p>
        <label className="mb-1 block text-xs text-slate-400" htmlFor="bulk-patch-metadata">
          metadata (JSON object)
        </label>
        <textarea
          id="bulk-patch-metadata"
          value={metadataText}
          onChange={(e) => setMetadataText(e.target.value)}
          rows={6}
          spellCheck={false}
          className="mb-3 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-xs text-slate-100"
        />
        <p className="mb-2 text-xs text-slate-400">
          Type{" "}
          <code className="rounded bg-slate-800 px-1 py-0.5 text-slate-200">
            {collectionName}
          </code>{" "}
          to confirm.
        </p>
        <input
          type="text"
          autoFocus
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          onKeyDown={(e) => {
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
            disabled={!nameMatches || submitting}
            onClick={handleConfirm}
            className="rounded border border-red-700 bg-red-900/60 px-2 py-1 text-xs text-red-200 hover:bg-red-900 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? "Working..." : "Apply"}
          </button>
        </div>
      </div>
    </div>
  );
}
