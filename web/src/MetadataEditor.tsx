import { forwardRef, useImperativeHandle, useState } from "react";
import type { ForwardedRef } from "react";
import { apiFetch, fetchDiff } from "./api";
import UnifiedDiffView from "./UnifiedDiffView";
import type { RecordInfo, RecordUpdateRequest } from "./types";

interface MetadataEditorProps {
  collectionName: string;
  record: RecordInfo;
  onSaved: () => void;
}

/** Imperative handle so the `e` keyboard shortcut (technical-spec §6.3) can
 * enter edit mode from App.tsx without lifting all of this component's
 * editing state up. */
export interface MetadataEditorHandle {
  startEdit: () => void;
}

type DiffKind = "added" | "removed" | "changed";

interface DiffEntry {
  key: string;
  kind: DiffKind;
  before?: unknown;
  after?: unknown;
}

/**
 * Validate that ``value`` is a flat mapping of str/int/float/bool values
 * (matching chromadb's metadata constraints, technical-spec §5.4). Throws
 * with a user-facing message on the first violation found.
 */
function validateFlatMetadata(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("metadata must be a JSON object.");
  }
  for (const [key, v] of Object.entries(value as Record<string, unknown>)) {
    const isScalar =
      typeof v === "string" || typeof v === "number" || typeof v === "boolean";
    if (!isScalar) {
      throw new Error(
        `metadata value for "${key}" must be a string, number, or boolean (got ${
          v === null ? "null" : Array.isArray(v) ? "array" : typeof v
        }).`
      );
    }
  }
  return value as Record<string, unknown>;
}

function diffMetadata(
  before: Record<string, unknown> | null,
  after: Record<string, unknown>
): DiffEntry[] {
  const beforeObj = before ?? {};
  const entries: DiffEntry[] = [];
  const keys = new Set([...Object.keys(beforeObj), ...Object.keys(after)]);
  for (const key of Array.from(keys).sort()) {
    const hasBefore = Object.prototype.hasOwnProperty.call(beforeObj, key);
    const hasAfter = Object.prototype.hasOwnProperty.call(after, key);
    if (!hasBefore && hasAfter) {
      entries.push({ key, kind: "added", after: after[key] });
    } else if (hasBefore && !hasAfter) {
      entries.push({ key, kind: "removed", before: beforeObj[key] });
    } else if (JSON.stringify(beforeObj[key]) !== JSON.stringify(after[key])) {
      entries.push({ key, kind: "changed", before: beforeObj[key], after: after[key] });
    }
  }
  return entries;
}

function diffColorClass(kind: DiffKind): string {
  switch (kind) {
    case "added":
      return "text-green-400";
    case "removed":
      return "text-red-400";
    case "changed":
      return "text-amber-400";
  }
}

function MetadataEditor(
  { collectionName, record, onSaved }: MetadataEditorProps,
  ref: ForwardedRef<MetadataEditorHandle>
) {
  const [editing, setEditing] = useState(false);
  const [metadataText, setMetadataText] = useState("");
  const [uriText, setUriText] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);

  // Confirmation step: set once validation passes, holds what will be sent.
  const [confirming, setConfirming] = useState(false);
  const [pendingMetadata, setPendingMetadata] = useState<Record<string, unknown> | null>(
    null
  );
  const [pendingUri, setPendingUri] = useState<string | null>(null);
  const [metadataChanged, setMetadataChanged] = useState(false);
  const [uriChanged, setUriChanged] = useState(false);
  // undefined = not fetched yet / in flight, null = fetch failed (fall back
  // to the key-based diff list below), string = unified diff to render.
  const [metadataDiff, setMetadataDiff] = useState<string | null | undefined>(
    undefined
  );

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  function startEdit() {
    setMetadataText(JSON.stringify(record.metadata ?? {}, null, 2));
    setUriText(record.uri ?? "");
    setValidationError(null);
    setSaveError(null);
    setEditing(true);
  }

  useImperativeHandle(ref, () => ({ startEdit }));

  function cancelEdit() {
    setEditing(false);
    setConfirming(false);
    setValidationError(null);
    setSaveError(null);
  }

  async function reviewChanges() {
    setValidationError(null);

    let parsedMetadata: unknown;
    try {
      parsedMetadata = JSON.parse(metadataText);
    } catch (err) {
      setValidationError(
        `Invalid JSON: ${err instanceof Error ? err.message : String(err)}`
      );
      return;
    }

    let flatMetadata: Record<string, unknown>;
    try {
      flatMetadata = validateFlatMetadata(parsedMetadata);
    } catch (err) {
      setValidationError(err instanceof Error ? err.message : String(err));
      return;
    }

    const newUri = uriText.trim() === "" ? "" : uriText;
    const metaChanged =
      JSON.stringify(record.metadata ?? {}) !== JSON.stringify(flatMetadata);
    const uriChangedNow = (record.uri ?? "") !== newUri;

    if (!metaChanged && !uriChangedNow) {
      setValidationError("No changes to save.");
      return;
    }

    setPendingMetadata(flatMetadata);
    setPendingUri(newUri);
    setMetadataChanged(metaChanged);
    setUriChanged(uriChangedNow);
    setMetadataDiff(undefined);
    setConfirming(true);

    if (metaChanged) {
      const result = await fetchDiff(
        JSON.stringify(record.metadata ?? {}, null, 2),
        JSON.stringify(flatMetadata, null, 2),
        "before",
        "after"
      );
      setMetadataDiff(result);
    }
  }

  async function confirmSave() {
    setSaving(true);
    setSaveError(null);

    const body: RecordUpdateRequest = {};
    if (metadataChanged) body.metadata = pendingMetadata;
    if (uriChanged) body.uri = pendingUri;

    try {
      const res = await apiFetch(
        `/api/collections/${encodeURIComponent(collectionName)}/records/${encodeURIComponent(
          record.id
        )}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      );
      if (!res.ok) {
        let detail = `request failed: ${res.status}`;
        const payload = await res.json().catch(() => null);
        if (payload && typeof payload.detail === "string") {
          detail = payload.detail;
        }
        throw new Error(detail);
      }
      setEditing(false);
      setConfirming(false);
      setToast("Record updated.");
      window.setTimeout(() => setToast(null), 3000);
      onSaved();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  const diffEntries =
    confirming && pendingMetadata ? diffMetadata(record.metadata, pendingMetadata) : [];
  const hasRemovedKeys = diffEntries.some((entry) => entry.kind === "removed");

  return (
    <div data-testid="metadata-editor">
      <div className="mb-1 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          metadata
        </h3>
        {!editing && (
          <button
            type="button"
            onClick={startEdit}
            className="rounded border border-slate-700 px-2 py-0.5 text-xs hover:bg-slate-800"
          >
            Edit
          </button>
        )}
      </div>

      {toast && (
        <p className="mb-2 rounded bg-green-900/60 px-2 py-1 text-xs text-green-300">
          {toast}
        </p>
      )}

      {!editing && (
        <pre className="max-h-64 overflow-auto rounded bg-slate-900 p-2 font-mono text-xs text-slate-300">
          {record.metadata ? JSON.stringify(record.metadata, null, 2) : "null"}
        </pre>
      )}

      {editing && !confirming && (
        <div className="space-y-2">
          <div>
            <label className="mb-1 block text-xs text-slate-400">metadata (JSON)</label>
            <p className="mb-1 text-xs text-slate-500">
              chromadb merges this metadata into the existing metadata rather than
              replacing it: keys you remove here will be dropped from the diff
              preview below but will not actually be deleted from the record.
            </p>
            <textarea
              value={metadataText}
              onChange={(e) => setMetadataText(e.target.value)}
              rows={8}
              className="w-full rounded border border-slate-700 bg-slate-900 p-2 font-mono text-xs text-slate-200"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-slate-400">uri</label>
            <input
              type="text"
              value={uriText}
              onChange={(e) => setUriText(e.target.value)}
              className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200"
            />
          </div>
          {validationError && (
            <p className="text-xs text-red-400">{validationError}</p>
          )}
          <div className="flex gap-2">
            <button
              type="button"
              onClick={reviewChanges}
              className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800"
            >
              Save
            </button>
            <button
              type="button"
              onClick={cancelEdit}
              className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {confirming && (
        <div className="space-y-2 rounded border border-slate-700 bg-slate-900 p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Confirm changes
          </p>
          {metadataChanged &&
            (metadataDiff === undefined ? (
              <p className="text-xs text-slate-500">Loading diff…</p>
            ) : metadataDiff ? (
              <UnifiedDiffView diff={metadataDiff} />
            ) : (
              <ul className="space-y-0.5 font-mono text-xs">
                {diffEntries.length === 0 && (
                  <li className="text-slate-500">(no metadata changes)</li>
                )}
                {diffEntries.map((entry) => (
                  <li key={entry.key} className={diffColorClass(entry.kind)}>
                    {entry.kind === "added" &&
                      `+ ${entry.key}: ${JSON.stringify(entry.after)}`}
                    {entry.kind === "removed" &&
                      `- ${entry.key}: ${JSON.stringify(entry.before)}`}
                    {entry.kind === "changed" &&
                      `~ ${entry.key}: ${JSON.stringify(
                        entry.before
                      )} → ${JSON.stringify(entry.after)}`}
                  </li>
                ))}
              </ul>
            ))}
          {!metadataChanged && !uriChanged && (
            <p className="font-mono text-xs text-slate-500">(no metadata changes)</p>
          )}
          {uriChanged && (
            <p className="font-mono text-xs text-amber-400">
              {`~ uri: ${JSON.stringify(record.uri ?? null)} → ${JSON.stringify(
                pendingUri
              )}`}
            </p>
          )}
          {hasRemovedKeys && (
            <p className="rounded bg-amber-900/60 px-2 py-1 text-xs text-amber-300">
              chromadb's update merges metadata rather than replacing it, so key
              deletion is not supported: keys shown as removed above will not
              actually be deleted, only overwritten values will be applied.
            </p>
          )}
          {saveError && <p className="text-xs text-red-400">{saveError}</p>}
          <div className="flex gap-2">
            <button
              type="button"
              disabled={saving}
              onClick={confirmSave}
              className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800 disabled:opacity-50"
            >
              {saving ? "Saving..." : "Confirm"}
            </button>
            <button
              type="button"
              disabled={saving}
              onClick={() => setConfirming(false)}
              className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800 disabled:opacity-50"
            >
              Back
            </button>
            <button
              type="button"
              disabled={saving}
              onClick={cancelEdit}
              className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800 disabled:opacity-50"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default forwardRef(MetadataEditor);
