import { useState } from "react";
import { apiFetch, fetchDiff } from "./api";
import { useAppConfig } from "./AppConfigContext";
import UnifiedDiffView from "./UnifiedDiffView";
import type { RecordInfo, RecordUpdateRequest } from "./types";

interface DocumentEditorProps {
  collectionName: string;
  record: RecordInfo;
  onSaved: () => void;
}

/**
 * Edit UI for a record's ``document`` (technical-spec §3.3, §5.4, §8.3,
 * roadmap M3-3).
 *
 * chromaw never recomputes embeddings implicitly, so any document edit
 * requires the user to explicitly pick ``embedding_mode``: "Re-embed"
 * (offered only when ``AppConfig.embeddingAvailable`` -- an explicit
 * ``--embedding-config`` was given at startup) computes a fresh vector
 * server-side and carries no stale warning; "Keep" leaves the vector
 * untouched and is flagged to the user as making the record's vector stale
 * relative to its (new) text before the request is sent. Mirrors
 * MetadataEditor's edit -> confirm -> save flow.
 */
export default function DocumentEditor({
  collectionName,
  record,
  onSaved,
}: DocumentEditorProps) {
  const { embeddingAvailable } = useAppConfig();

  const [editing, setEditing] = useState(false);
  const [documentText, setDocumentText] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);

  const [confirming, setConfirming] = useState(false);
  const [pendingDocument, setPendingDocument] = useState<string | null>(null);
  const [embeddingMode, setEmbeddingMode] = useState<"keep" | "reembed">(
    embeddingAvailable ? "reembed" : "keep"
  );
  // undefined = not fetched yet / in flight, null = fetch failed (fall back
  // to the char-count summary above), string = unified diff to render.
  const [diff, setDiff] = useState<string | null | undefined>(undefined);

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  function startEdit() {
    setDocumentText(record.document ?? "");
    setValidationError(null);
    setSaveError(null);
    setEmbeddingMode(embeddingAvailable ? "reembed" : "keep");
    setEditing(true);
  }

  function cancelEdit() {
    setEditing(false);
    setConfirming(false);
    setValidationError(null);
    setSaveError(null);
  }

  async function reviewChanges() {
    setValidationError(null);

    const before = record.document ?? "";
    if (documentText === before) {
      setValidationError("No changes to save.");
      return;
    }

    setPendingDocument(documentText);
    setDiff(undefined);
    setConfirming(true);

    const result = await fetchDiff(before, documentText, "before", "after");
    setDiff(result);
  }

  async function confirmSave() {
    setSaving(true);
    setSaveError(null);

    const body: RecordUpdateRequest = {
      document: pendingDocument,
      embedding_mode: embeddingMode,
    };

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

  const beforeLength = (record.document ?? "").length;
  const afterLength = (pendingDocument ?? "").length;
  const lengthDelta = afterLength - beforeLength;

  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          document
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
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded bg-slate-900 p-2 font-mono text-xs text-slate-300">
          {record.document ?? "-"}
        </pre>
      )}

      {editing && !confirming && (
        <div className="space-y-2">
          <textarea
            value={documentText}
            onChange={(e) => setDocumentText(e.target.value)}
            rows={10}
            className="w-full rounded border border-slate-700 bg-slate-900 p-2 font-mono text-xs text-slate-200"
          />
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
          <fieldset className="space-y-1">
            <legend className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Embedding
            </legend>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="radio"
                name="embedding-mode"
                value="reembed"
                disabled={!embeddingAvailable}
                checked={embeddingMode === "reembed"}
                onChange={() => setEmbeddingMode("reembed")}
              />
              Re-embed (compute a fresh vector for the new text)
              {!embeddingAvailable && (
                <span className="text-slate-500">
                  (unavailable -- restart with --embedding-config to enable)
                </span>
              )}
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="radio"
                name="embedding-mode"
                value="keep"
                checked={embeddingMode === "keep"}
                onChange={() => setEmbeddingMode("keep")}
              />
              Keep existing vector (document and embedding become inconsistent)
            </label>
          </fieldset>
          {embeddingMode === "keep" && (
            <p className="rounded bg-amber-900/60 px-2 py-1 text-xs text-amber-300">
              ⚠️ document を更新しても embedding は再計算されません。ベクトルと本文が
              不整合になります（stale とマークされます）。
            </p>
          )}
          <p className="font-mono text-xs text-slate-300">
            {beforeLength} chars → {afterLength} chars (
            {lengthDelta >= 0 ? `+${lengthDelta}` : lengthDelta})
          </p>
          {diff === undefined && (
            <p className="text-xs text-slate-500">Loading diff…</p>
          )}
          {diff === "" && <p className="text-xs text-slate-500">(no changes)</p>}
          {diff !== undefined && diff !== null && diff !== "" && (
            <UnifiedDiffView diff={diff} />
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
