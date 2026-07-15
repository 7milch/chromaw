import { useEffect, useMemo, useRef, useState } from "react";
import { apiFetch } from "./api";
import { useAppConfig } from "./AppConfigContext";
import {
  exportCollectionRecords,
  exportSelectedRecords,
  type ExportFilters,
} from "./export";
import { downloadCollectionJsonl, importCollectionJsonl } from "./jsonl";
import { useKeyboardShortcuts } from "./useKeyboardShortcuts";
import BulkPatchModal from "./BulkPatchModal";
import ConfirmNameModal from "./ConfirmNameModal";
import DocumentEditor, { type DocumentEditorHandle } from "./DocumentEditor";
import MetadataEditor, { type MetadataEditorHandle } from "./MetadataEditor";
import RenameCollectionModal from "./RenameCollectionModal";
import ShortcutsHelpModal from "./ShortcutsHelpModal";
import type {
  BulkDeleteResponse,
  BulkPatchResponse,
  CollectionInfo,
  CollectionsResponse,
  ImportResponse,
  QueryRequest,
  QueryResponse,
  RecordInfo,
  RecordMatchInfo,
  RecordsGetRequest,
  RecordsResponse,
} from "./types";

type SearchMode = "id" | "metadata" | "document" | "similarity";

const DEFAULT_N_RESULTS = 10;

interface ActiveSearch {
  ids?: string[];
  where?: Record<string, unknown>;
  where_document?: Record<string, unknown>;
}

/**
 * Parse the search bar's raw input for the given mode into the
 * ids/where/where_document fields sent to POST .../records/get
 * (technical-spec §5.5 1-3). Throws with a user-facing message on invalid
 * input.
 */
function parseSearchInput(mode: SearchMode, rawInput: string): ActiveSearch {
  const input = rawInput.trim();
  if (!input) {
    throw new Error("Enter a search value.");
  }

  if (mode === "id") {
    const ids = input
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    if (ids.length === 0) {
      throw new Error("Enter at least one id.");
    }
    return { ids };
  }

  if (mode === "metadata") {
    if (input.startsWith("{")) {
      let where: unknown;
      try {
        where = JSON.parse(input);
      } catch (err) {
        throw new Error(
          `Invalid JSON: ${err instanceof Error ? err.message : String(err)}`
        );
      }
      if (typeof where !== "object" || where === null || Array.isArray(where)) {
        throw new Error("where JSON must be an object.");
      }
      return { where: where as Record<string, unknown> };
    }

    const eqIndex = input.indexOf("=");
    if (eqIndex === -1) {
      throw new Error('Use "key=value" or a raw JSON object starting with "{".');
    }
    const key = input.slice(0, eqIndex).trim();
    const value = input.slice(eqIndex + 1).trim();
    if (!key) {
      throw new Error("Missing metadata key before \"=\".");
    }
    return { where: { [key]: value } };
  }

  // document mode
  return { where_document: { $contains: input } };
}

function formatMetadataJson(metadata: Record<string, unknown> | null): string {
  return metadata ? JSON.stringify(metadata, null, 2) : "null";
}

const PAGE_LIMIT = 50;

/**
 * Read the ``chromaw_embedding_status`` metadata flag set by document edits
 * (DocumentEditor's "keep" mode / M2-3, M3-3) so the UI can surface a stale
 * warning without the caller needing to know the metadata key.
 */
function embeddingStatus(
  metadata: Record<string, unknown> | null
): "stale" | "fresh" | null {
  const value = metadata?.chromaw_embedding_status;
  return value === "stale" || value === "fresh" ? value : null;
}

function summarizeMetadata(metadata: Record<string, unknown> | null): string {
  if (!metadata) return "-";
  const entries = Object.entries(metadata);
  if (entries.length === 0) return "-";
  return entries.map(([k, v]) => `${k}=${String(v)}`).join(", ");
}

function App() {
  const { health, error: healthError, isWriteMode, embeddingAvailable } = useAppConfig();
  const [collections, setCollections] = useState<CollectionInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);

  const [records, setRecords] = useState<RecordInfo[] | null>(null);
  const [recordsTotal, setRecordsTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [offset, setOffset] = useState(0);
  const [recordsError, setRecordsError] = useState<string | null>(null);
  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null);

  const [searchMode, setSearchMode] = useState<SearchMode>("id");
  const [searchText, setSearchText] = useState("");
  const [activeSearch, setActiveSearch] = useState<ActiveSearch | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Similarity ("query") search (technical-spec §5.6 4, §8.4, roadmap
  // M3-1): a separate mode from the id/metadata/document search above since
  // it hits POST .../query rather than .../records/get and returns
  // distance-annotated matches instead of plain records. Only one of
  // activeSearch/activeSimilarityQuery is ever set at a time.
  const [nResultsText, setNResultsText] = useState(String(DEFAULT_N_RESULTS));
  const [activeSimilarityQuery, setActiveSimilarityQuery] = useState<{
    queryText: string;
    nResults: number;
  } | null>(null);
  const [matches, setMatches] = useState<RecordMatchInfo[] | null>(null);
  const [matchesError, setMatchesError] = useState<string | null>(null);

  const [detailRecord, setDetailRecord] = useState<RecordInfo | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detailMissing, setDetailMissing] = useState(false);

  const [helpOpen, setHelpOpen] = useState(false);

  // Bumped after a collection/record delete or a collection rename to
  // force the collections list and/or the records page to refetch, same
  // ``refreshTick`` pattern as the record detail refresh above.
  const [collectionsRefreshTick, setCollectionsRefreshTick] = useState(0);
  const [recordsRefreshTick, setRecordsRefreshTick] = useState(0);

  const [deleteRecordModalOpen, setDeleteRecordModalOpen] = useState(false);
  const [deleteCollectionModalOpen, setDeleteCollectionModalOpen] = useState(false);
  const [renameCollectionModalOpen, setRenameCollectionModalOpen] = useState(false);

  // Bulk delete of the current multi-selection (M4-2, technical-spec §6.5):
  // confirmed via the collection name (mirrors DELETE .../collections/{name},
  // since a bulk operation has no single record id to type). ``bulkDeleteResult``
  // holds the last response's deleted/skipped counts so they can be surfaced
  // to the user after the modal closes.
  const [bulkDeleteModalOpen, setBulkDeleteModalOpen] = useState(false);
  const [bulkDeleteResult, setBulkDeleteResult] = useState<BulkDeleteResponse | null>(
    null
  );

  // Bulk metadata patch of the current multi-selection (M4-4, technical-spec
  // §3.3, §5.4, §6.5): same "type the collection name" confirmation as bulk
  // delete, surfaced via BulkPatchModal. ``bulkPatchResult`` holds the last
  // response's patched/skipped counts so they can be surfaced after the
  // modal closes.
  const [bulkPatchModalOpen, setBulkPatchModalOpen] = useState(false);
  const [bulkPatchResult, setBulkPatchResult] = useState<BulkPatchResponse | null>(
    null
  );

  const [exportRunning, setExportRunning] = useState(false);
  const [exportCount, setExportCount] = useState(0);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportTruncated, setExportTruncated] = useState(false);
  const exportAbortRef = useRef<AbortController | null>(null);

  // Multi-selection for "selected export" (M4-1, technical-spec §8.3), kept
  // independent of selectedRecordId (single-row detail selection). Cleared
  // whenever the collection or the active search changes.
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [selectedExportRunning, setSelectedExportRunning] = useState(false);
  const [selectedExportError, setSelectedExportError] = useState<string | null>(null);
  const [selectedExportTruncated, setSelectedExportTruncated] = useState(false);
  const selectedExportAbortRef = useRef<AbortController | null>(null);

  // JSONL import/export (M4-3, technical-spec §8). Export streams straight
  // from the server (jsonl.ts's downloadCollectionJsonl), so unlike the JSON
  // export above it has no client-side progress/cancel/truncation to track.
  const [jsonlExportError, setJsonlExportError] = useState<string | null>(null);
  const [importRunning, setImportRunning] = useState(false);
  const [importMode, setImportMode] = useState<"add" | "upsert">("add");
  const [importResult, setImportResult] = useState<ImportResponse | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const importFileInputRef = useRef<HTMLInputElement | null>(null);

  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const metadataEditorRef = useRef<MetadataEditorHandle | null>(null);
  const documentEditorRef = useRef<DocumentEditorHandle | null>(null);
  const recordRowRefs = useRef<Map<string, HTMLTableRowElement>>(new Map());

  useEffect(() => {
    apiFetch("/api/collections")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`request failed: ${res.status}`);
        }
        return res.json() as Promise<CollectionsResponse>;
      })
      .then((data) => setCollections(data.collections))
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err));
      });
  }, [collectionsRefreshTick]);

  // Reset paging/selection/search whenever the selected collection changes.
  useEffect(() => {
    setOffset(0);
    setRecordsTotal(0);
    setHasMore(false);
    setSelectedRecordId(null);
    setRecords(null);
    setRecordsError(null);
    setActiveSearch(null);
    setSearchText("");
    setSearchError(null);
    setActiveSimilarityQuery(null);
    setMatches(null);
    setMatchesError(null);
    setNResultsText(String(DEFAULT_N_RESULTS));
    setSelectedIds(new Set());
  }, [selectedName]);

  // Run a similarity search (technical-spec §5.6 4, §8.4) against
  // POST .../query whenever activeSimilarityQuery is set. Independent of
  // the id/metadata/document records fetch below -- the two search modes
  // never run at the same time since executeSearch only ever sets one of
  // activeSearch/activeSimilarityQuery.
  useEffect(() => {
    if (!selectedName || !activeSimilarityQuery) return;

    let ignore = false;

    const body: QueryRequest = {
      query_text: activeSimilarityQuery.queryText,
      n_results: activeSimilarityQuery.nResults,
      include: ["documents", "metadatas", "uris", "distances"],
    };

    apiFetch(`/api/collections/${encodeURIComponent(selectedName)}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then(async (res) => {
        if (!res.ok) {
          let detail = `request failed: ${res.status}`;
          const payload = await res.json().catch(() => null);
          if (payload && typeof payload.detail === "string") {
            detail = payload.detail;
          }
          throw new Error(detail);
        }
        return res.json() as Promise<QueryResponse>;
      })
      .then((data) => {
        if (ignore) return;
        setMatches(data.matches);
        setMatchesError(null);
      })
      .catch((err: unknown) => {
        if (ignore) return;
        setMatchesError(err instanceof Error ? err.message : String(err));
      });

    return () => {
      ignore = true;
    };
  }, [selectedName, activeSimilarityQuery]);

  useEffect(() => {
    if (!selectedName) return;
    // Similarity search is handled by the dedicated effect above; skip the
    // records/records-get fetch entirely while it's active.
    if (activeSimilarityQuery) return;

    // Guard against out-of-order responses: if selectedName/offset/search
    // change again before this request resolves, ignore its result instead
    // of clobbering state set by the newer request.
    let ignore = false;

    if (activeSearch) {
      // Search mode: POST .../records/get with ids/where/where_document,
      // keeping limit/offset so paging still works against the filtered
      // set (task §5.5, spec §8.3).
      const body: RecordsGetRequest = {
        ...activeSearch,
        limit: PAGE_LIMIT,
        offset,
        include: ["documents", "metadatas", "uris"],
      };

      apiFetch(`/api/collections/${encodeURIComponent(selectedName)}/records/get`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
        .then(async (res) => {
          if (!res.ok) {
            let detail = `request failed: ${res.status}`;
            if (res.status === 422) {
              const payload = await res.json().catch(() => null);
              if (payload && typeof payload.detail === "string") {
                detail = payload.detail;
              }
            }
            throw new Error(detail);
          }
          return res.json() as Promise<RecordsResponse>;
        })
        .then((data) => {
          if (ignore) return;
          setRecords(data.records);
          setRecordsTotal(data.total);
          setHasMore(data.has_more);
          setSearchError(null);
        })
        .catch((err: unknown) => {
          if (ignore) return;
          setSearchError(err instanceof Error ? err.message : String(err));
        });

      return () => {
        ignore = true;
      };
    }

    const params = new URLSearchParams({
      include: "documents,metadatas,uris",
      limit: String(PAGE_LIMIT),
      offset: String(offset),
    });

    apiFetch(`/api/collections/${encodeURIComponent(selectedName)}/records?${params}`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`request failed: ${res.status}`);
        }
        return res.json() as Promise<RecordsResponse>;
      })
      .then((data) => {
        if (ignore) return;
        setRecords(data.records);
        setRecordsTotal(data.total);
        setHasMore(data.has_more);
      })
      .catch((err: unknown) => {
        if (ignore) return;
        setRecordsError(err instanceof Error ? err.message : String(err));
      });

    return () => {
      ignore = true;
    };
  }, [selectedName, offset, activeSearch, activeSimilarityQuery, recordsRefreshTick]);

  // Fetch full detail (including embeddings) for the selected record.
  // Same out-of-order-response guard as the records list fetch above.
  // ``refreshTick`` is bumped after a successful metadata/uri edit so this
  // effect re-runs against the same selectedRecordId to pull the freshly
  // updated record.
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    setDetailRecord(null);
    setDetailError(null);
    setDetailMissing(false);

    if (!selectedName || !selectedRecordId) return;

    let ignore = false;
    setDetailLoading(true);

    apiFetch(`/api/collections/${encodeURIComponent(selectedName)}/records/get`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ids: [selectedRecordId],
        include: ["documents", "metadatas", "uris", "embeddings"],
      }),
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`request failed: ${res.status}`);
        }
        return res.json() as Promise<RecordsResponse>;
      })
      .then((data) => {
        if (ignore) return;
        const record = data.records[0] ?? null;
        setDetailRecord(record);
        setDetailMissing(record === null);
      })
      .catch((err: unknown) => {
        if (ignore) return;
        setDetailError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (ignore) return;
        setDetailLoading(false);
      });

    return () => {
      ignore = true;
    };
  }, [selectedName, selectedRecordId, refreshTick]);

  const selected = collections?.find((c) => c.name === selectedName) ?? null;
  const recordIds = useMemo(() => records?.map((r) => r.id) ?? [], [records]);

  // Keep the selected row visible when navigating with j/k (M1-5 carry-over).
  useEffect(() => {
    if (!selectedRecordId) return;
    recordRowRefs.current.get(selectedRecordId)?.scrollIntoView({ block: "nearest" });
  }, [selectedRecordId]);

  const modalOpen =
    helpOpen ||
    deleteRecordModalOpen ||
    deleteCollectionModalOpen ||
    renameCollectionModalOpen ||
    bulkDeleteModalOpen ||
    bulkPatchModalOpen;

  useKeyboardShortcuts({
    searchInputRef,
    recordIds,
    selectedRecordId,
    onSelectRecordId: setSelectedRecordId,
    helpOpen,
    onSetHelpOpen: setHelpOpen,
    modalOpen,
    isWriteMode,
    onEditRecord: () => metadataEditorRef.current?.startEdit(),
    onDeleteRecord: () => setDeleteRecordModalOpen(true),
  });

  function executeSearch() {
    if (searchMode === "similarity") {
      const queryText = searchText.trim();
      if (!queryText) {
        setSearchError("Enter a search value.");
        return;
      }
      const nResults = Number.parseInt(nResultsText, 10);
      if (!Number.isFinite(nResults) || nResults < 1) {
        setSearchError("n_results must be a positive integer.");
        return;
      }
      setSearchError(null);
      setActiveSearch(null);
      setActiveSimilarityQuery({ queryText, nResults });
      setSelectedIds(new Set());
      return;
    }

    try {
      const parsed = parseSearchInput(searchMode, searchText);
      setSearchError(null);
      setOffset(0);
      setActiveSimilarityQuery(null);
      setActiveSearch(parsed);
      setSelectedIds(new Set());
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : String(err));
    }
  }

  function clearSearch() {
    setSearchText("");
    setSearchError(null);
    setActiveSearch(null);
    setActiveSimilarityQuery(null);
    setMatches(null);
    setMatchesError(null);
    setOffset(0);
    setSelectedIds(new Set());
  }

  function toggleRecordSelected(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  function toggleSelectAllVisible() {
    setSelectedIds((prev) => {
      const visibleIds = records?.map((r) => r.id) ?? [];
      const allSelected =
        visibleIds.length > 0 && visibleIds.every((id) => prev.has(id));
      if (allSelected) {
        const next = new Set(prev);
        for (const id of visibleIds) next.delete(id);
        return next;
      }
      const next = new Set(prev);
      for (const id of visibleIds) next.add(id);
      return next;
    });
  }

  async function startSelectedExport() {
    if (!selectedName || selectedIds.size === 0) return;

    const controller = new AbortController();
    selectedExportAbortRef.current = controller;
    setSelectedExportRunning(true);
    setSelectedExportError(null);
    setSelectedExportTruncated(false);

    try {
      const result = await exportSelectedRecords(selectedName, Array.from(selectedIds), {
        signal: controller.signal,
      });

      const payload = {
        collection: result.collection,
        exported_at: result.exported_at,
        filters: result.filters,
        records: result.records,
        ...(result.truncated ? { truncated: true } : {}),
      };

      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${result.collection}-records-selected-${result.exported_at}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      setSelectedExportTruncated(result.truncated);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // Cancelled by the user; nothing to report.
      } else {
        setSelectedExportError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      selectedExportAbortRef.current = null;
      setSelectedExportRunning(false);
    }
  }

  async function startExport() {
    if (!selectedName) return;

    const controller = new AbortController();
    exportAbortRef.current = controller;
    setExportRunning(true);
    setExportCount(0);
    setExportError(null);
    setExportTruncated(false);

    const filters: ExportFilters | null = activeSearch
      ? {
          ...(activeSearch.ids !== undefined ? { ids: activeSearch.ids } : {}),
          ...(activeSearch.where !== undefined ? { where: activeSearch.where } : {}),
          ...(activeSearch.where_document !== undefined
            ? { where_document: activeSearch.where_document }
            : {}),
        }
      : null;

    try {
      const result = await exportCollectionRecords(selectedName, filters, {
        signal: controller.signal,
        onProgress: setExportCount,
      });

      const payload = {
        collection: result.collection,
        exported_at: result.exported_at,
        filters: result.filters,
        records: result.records,
        ...(result.truncated ? { truncated: true } : {}),
      };

      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${result.collection}-records-${result.exported_at}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      setExportTruncated(result.truncated);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // Cancelled by the user; nothing to report.
      } else {
        setExportError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      exportAbortRef.current = null;
      setExportRunning(false);
    }
  }

  function cancelExport() {
    exportAbortRef.current?.abort();
  }

  async function startJsonlExport() {
    if (!selectedName) return;
    setJsonlExportError(null);
    try {
      await downloadCollectionJsonl(selectedName);
    } catch (err) {
      setJsonlExportError(err instanceof Error ? err.message : String(err));
    }
  }

  function triggerImportFilePicker() {
    setImportError(null);
    setImportResult(null);
    importFileInputRef.current?.click();
  }

  async function handleImportFileChosen(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    // Reset so choosing the same file again still fires onChange.
    event.target.value = "";
    if (!file || !selectedName) return;

    setImportRunning(true);
    setImportError(null);
    setImportResult(null);
    try {
      const result = await importCollectionJsonl(selectedName, file, importMode);
      setImportResult(result);
      setRecordsRefreshTick((t) => t + 1);
      setCollectionsRefreshTick((t) => t + 1);
    } catch (err) {
      setImportError(err instanceof Error ? err.message : String(err));
    } finally {
      setImportRunning(false);
    }
  }

  /**
   * Shared error-extraction for the delete/rename fetches below (technical
   * -spec §3.2, §6.5, roadmap M2-7): surfaces the server's ``detail``
   * message (e.g. a 409 confirm-mismatch or duplicate-name error) so
   * ``ConfirmNameModal``/``RenameCollectionModal`` can display it.
   */
  async function throwOnError(res: Response): Promise<void> {
    if (res.ok) return;
    let detail = `request failed: ${res.status}`;
    const payload = await res.json().catch(() => null);
    if (payload && typeof payload.detail === "string") {
      detail = payload.detail;
    }
    throw new Error(detail);
  }

  async function deleteRecord() {
    if (!selectedName || !selectedRecordId) return;
    const res = await apiFetch(
      `/api/collections/${encodeURIComponent(selectedName)}/records/${encodeURIComponent(
        selectedRecordId
      )}`,
      {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm: selectedRecordId }),
      }
    );
    await throwOnError(res);
    setDeleteRecordModalOpen(false);
    setSelectedRecordId(null);
    setRecordsRefreshTick((t) => t + 1);
    setCollectionsRefreshTick((t) => t + 1);
  }

  async function deleteCollection() {
    if (!selectedName) return;
    const res = await apiFetch(`/api/collections/${encodeURIComponent(selectedName)}`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: selectedName }),
    });
    await throwOnError(res);
    setDeleteCollectionModalOpen(false);
    setSelectedName(null);
    setCollectionsRefreshTick((t) => t + 1);
  }

  async function bulkDeleteSelected() {
    if (!selectedName || selectedIds.size === 0) return;
    const res = await apiFetch(
      `/api/collections/${encodeURIComponent(selectedName)}/records/bulk-delete`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ids: Array.from(selectedIds),
          confirm: selectedName,
        }),
      }
    );
    await throwOnError(res);
    const result = (await res.json()) as BulkDeleteResponse;
    setBulkDeleteResult(result);
    setBulkDeleteModalOpen(false);
    setSelectedIds(new Set());
    setRecordsRefreshTick((t) => t + 1);
    setCollectionsRefreshTick((t) => t + 1);
  }

  async function bulkPatchSelected(metadata: Record<string, unknown>) {
    if (!selectedName || selectedIds.size === 0) return;
    const res = await apiFetch(
      `/api/collections/${encodeURIComponent(selectedName)}/records/bulk-patch`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ids: Array.from(selectedIds),
          metadata,
          confirm: selectedName,
        }),
      }
    );
    await throwOnError(res);
    const result = (await res.json()) as BulkPatchResponse;
    setBulkPatchResult(result);
    setBulkPatchModalOpen(false);
    setSelectedIds(new Set());
    setRecordsRefreshTick((t) => t + 1);
    setRefreshTick((t) => t + 1);
  }

  async function renameCollection(newName: string) {
    if (!selectedName) return;
    const res = await apiFetch(`/api/collections/${encodeURIComponent(selectedName)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: newName, confirm: selectedName }),
    });
    await throwOnError(res);
    setRenameCollectionModalOpen(false);
    setSelectedName(newName);
    setCollectionsRefreshTick((t) => t + 1);
  }

  // ids-based search returns every match in one page (chromaw ignores
  // limit/offset for it server-side), so paging controls are meaningless
  // and disabled in that mode.
  const isIdSearch = activeSearch?.ids !== undefined;
  const rangeStart = recordsTotal === 0 ? 0 : offset + 1;
  const rangeEnd = records ? offset + records.length : 0;
  const canPrev = !isIdSearch && offset > 0;
  const canNext = !isIdSearch && hasMore;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <header className="flex items-center gap-3 border-b border-slate-800 px-4 py-2 text-sm">
        <span className="font-semibold text-slate-100">chromaw</span>
        <span className="text-slate-500">|</span>
        <span className="text-slate-400 truncate">{health?.path ?? "..."}</span>
        <span className="text-slate-500">|</span>
        {health && (
          <span
            title={
              health.mode === "write"
                ? "Editing is enabled. Destructive actions require confirmation."
                : "Read-only mode. Restart with --write to enable edits."
            }
            className={`rounded-full px-2 py-0.5 text-xs font-semibold uppercase tracking-wide ${
              health.mode === "write"
                ? "bg-red-600/90 text-white ring-1 ring-red-400"
                : "bg-slate-700 text-slate-300 ring-1 ring-slate-600"
            }`}
          >
            {health.mode}
          </span>
        )}
      </header>

      {(error || healthError) && (
        <p className="px-4 py-2 text-sm text-red-400">
          Failed to reach API: {error ?? healthError}
        </p>
      )}

      <div className="flex flex-1 min-h-0">
        <aside className="w-64 shrink-0 border-r border-slate-800 overflow-y-auto">
          <h2 className="px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Collections
          </h2>
          {collections === null && !error && (
            <p className="px-3 py-2 text-sm text-slate-400">Loading...</p>
          )}
          {collections !== null && collections.length === 0 && (
            <p className="px-3 py-2 text-sm text-slate-400">No collections found.</p>
          )}
          <ul>
            {collections?.map((c) => (
              <li key={c.id}>
                <button
                  type="button"
                  onClick={() => setSelectedName(c.name)}
                  className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-slate-800 ${
                    c.name === selectedName ? "bg-slate-800 text-slate-50" : "text-slate-300"
                  }`}
                >
                  <span className="truncate">{c.name}</span>
                  <span className="ml-2 shrink-0 text-xs text-slate-500">{c.count}</span>
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <main className="flex-1 min-w-0 overflow-y-auto p-4 flex flex-col gap-3">
          {!selected && (
            <p className="text-sm text-slate-400">
              Select a collection from the left to see its records.
            </p>
          )}

          {selected && (
            <>
              <div className="rounded border border-slate-800">
                <div className="flex w-full items-center justify-between px-3 py-2 text-sm">
                  <button
                    type="button"
                    onClick={() => setDetailOpen((v) => !v)}
                    className="flex flex-1 items-center justify-between text-left"
                  >
                    <span className="font-semibold">{selected.name}</span>
                    <span className="text-xs text-slate-500">
                      {detailOpen ? "hide details ▲" : "show details ▼"}
                    </span>
                  </button>
                  <div className="ml-3 flex shrink-0 items-center gap-2">
                    {isWriteMode && (
                      <>
                        <button
                          type="button"
                          onClick={() => setRenameCollectionModalOpen(true)}
                          className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800"
                        >
                          Rename
                        </button>
                        <button
                          type="button"
                          onClick={() => setDeleteCollectionModalOpen(true)}
                          className="rounded border border-red-800 px-2 py-1 text-xs text-red-300 hover:bg-red-900/40"
                        >
                          Delete
                        </button>
                      </>
                    )}
                    {selectedIds.size > 0 && (
                      <>
                        <span className="text-xs text-slate-400">
                          {selectedIds.size} selected
                        </span>
                        <button
                          type="button"
                          disabled={selectedExportRunning}
                          onClick={startSelectedExport}
                          className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
                        >
                          {selectedExportRunning
                            ? "Exporting..."
                            : `Export selected (${selectedIds.size})`}
                        </button>
                        {isWriteMode && (
                          <button
                            type="button"
                            onClick={() => {
                              setBulkPatchResult(null);
                              setBulkPatchModalOpen(true);
                            }}
                            className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800"
                          >
                            {`Edit metadata (${selectedIds.size})`}
                          </button>
                        )}
                        {isWriteMode && (
                          <button
                            type="button"
                            onClick={() => {
                              setBulkDeleteResult(null);
                              setBulkDeleteModalOpen(true);
                            }}
                            className="rounded border border-red-800 px-2 py-1 text-xs text-red-300 hover:bg-red-900/40"
                          >
                            {`Delete selected (${selectedIds.size})`}
                          </button>
                        )}
                      </>
                    )}
                    {exportRunning ? (
                      <>
                        <span className="text-xs text-slate-400">
                          {exportCount} records...
                        </span>
                        <button
                          type="button"
                          onClick={cancelExport}
                          className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800"
                        >
                          Cancel
                        </button>
                      </>
                    ) : (
                      <button
                        type="button"
                        onClick={startExport}
                        className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800"
                      >
                        Export JSON
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={startJsonlExport}
                      className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800"
                    >
                      Export JSONL
                    </button>
                    {isWriteMode && (
                      <>
                        <select
                          value={importMode}
                          onChange={(e) => setImportMode(e.target.value as "add" | "upsert")}
                          disabled={importRunning}
                          className="rounded border border-slate-700 bg-transparent px-2 py-1 text-xs disabled:opacity-40"
                          title="How to handle ids that already exist in the collection"
                        >
                          <option value="add">add</option>
                          <option value="upsert">upsert</option>
                        </select>
                        <button
                          type="button"
                          disabled={importRunning}
                          onClick={triggerImportFilePicker}
                          className="rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
                        >
                          {importRunning ? "Importing..." : "Import JSONL"}
                        </button>
                        <input
                          ref={importFileInputRef}
                          type="file"
                          accept=".jsonl,application/x-ndjson,text/plain"
                          onChange={handleImportFileChosen}
                          className="hidden"
                        />
                      </>
                    )}
                  </div>
                </div>
                {jsonlExportError && (
                  <p className="border-t border-slate-800 px-3 py-1 text-xs text-red-400">
                    Export JSONL failed: {jsonlExportError}
                  </p>
                )}
                {importError && (
                  <p className="border-t border-slate-800 px-3 py-1 text-xs text-red-400">
                    Import failed: {importError}
                  </p>
                )}
                {importResult && (
                  <p className="border-t border-slate-800 px-3 py-1 text-xs text-slate-400">
                    Imported {importResult.imported.length} record
                    {importResult.imported.length === 1 ? "" : "s"}
                    {importResult.skipped.length > 0
                      ? ` (${importResult.skipped.length} skipped: ${importResult.skipped
                          .slice(0, 3)
                          .map((s) => `line ${s.line} - ${s.reason}`)
                          .join("; ")}${importResult.skipped.length > 3 ? "; ..." : ""})`
                      : ""}
                    .
                  </p>
                )}
                {exportError && (
                  <p className="border-t border-slate-800 px-3 py-1 text-xs text-red-400">
                    Export failed: {exportError}
                  </p>
                )}
                {!exportRunning && exportTruncated && (
                  <p className="border-t border-slate-800 px-3 py-1 text-xs text-amber-400">
                    Export truncated at 100,000 records.
                  </p>
                )}
                {selectedExportError && (
                  <p className="border-t border-slate-800 px-3 py-1 text-xs text-red-400">
                    Export selected failed: {selectedExportError}
                  </p>
                )}
                {!selectedExportRunning && selectedExportTruncated && (
                  <p className="border-t border-slate-800 px-3 py-1 text-xs text-amber-400">
                    Selected export truncated at 100,000 records.
                  </p>
                )}
                {bulkDeleteResult && (
                  <p className="border-t border-slate-800 px-3 py-1 text-xs text-slate-400">
                    Deleted {bulkDeleteResult.deleted.length} record
                    {bulkDeleteResult.deleted.length === 1 ? "" : "s"}
                    {bulkDeleteResult.skipped.length > 0
                      ? ` (${bulkDeleteResult.skipped.length} already gone, skipped)`
                      : ""}
                    .
                  </p>
                )}
                {bulkPatchResult && (
                  <p className="border-t border-slate-800 px-3 py-1 text-xs text-slate-400">
                    Patched {bulkPatchResult.patched.length} record
                    {bulkPatchResult.patched.length === 1 ? "" : "s"}
                    {bulkPatchResult.skipped.length > 0
                      ? ` (${bulkPatchResult.skipped.length} no longer present, skipped)`
                      : ""}
                    .
                  </p>
                )}
                {detailOpen && (
                  <div className="space-y-3 border-t border-slate-800 px-3 py-3">
                    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                      <dt className="text-slate-400">id</dt>
                      <dd className="break-all">{selected.id}</dd>
                      <dt className="text-slate-400">count</dt>
                      <dd>{selected.count}</dd>
                      <dt className="text-slate-400">dimension</dt>
                      <dd>{selected.dimension ?? "-"}</dd>
                    </dl>
                    <div>
                      <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                        metadata
                      </h3>
                      <pre className="overflow-x-auto rounded bg-slate-900 p-3 text-xs text-slate-300">
                        {JSON.stringify(selected.metadata, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>

              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <select
                    value={searchMode}
                    onChange={(e) => setSearchMode(e.target.value as SearchMode)}
                    className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-slate-200"
                  >
                    <option value="id">ID</option>
                    <option value="metadata">Metadata</option>
                    <option value="document">Document</option>
                    <option value="similarity">Similarity</option>
                  </select>
                  <input
                    ref={searchInputRef}
                    type="text"
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") executeSearch();
                    }}
                    placeholder={
                      searchMode === "id"
                        ? "id1, id2, ..."
                        : searchMode === "metadata"
                          ? 'key=value or {"key": "value"}'
                          : searchMode === "similarity"
                            ? "text to find similar records for"
                            : "text the document should contain"
                    }
                    className="min-w-0 flex-1 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-slate-200 placeholder:text-slate-600"
                  />
                  {searchMode === "similarity" && (
                    <input
                      type="number"
                      min={1}
                      value={nResultsText}
                      onChange={(e) => setNResultsText(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") executeSearch();
                      }}
                      title="n_results"
                      className="w-20 shrink-0 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-slate-200"
                    />
                  )}
                  <button
                    type="button"
                    onClick={executeSearch}
                    className="rounded border border-slate-700 px-2 py-1 text-sm hover:bg-slate-800"
                  >
                    Search
                  </button>
                  {(activeSearch || activeSimilarityQuery) && (
                    <button
                      type="button"
                      onClick={clearSearch}
                      className="rounded border border-slate-700 px-2 py-1 text-sm hover:bg-slate-800"
                    >
                      Clear
                    </button>
                  )}
                </div>
                {searchError && <p className="text-sm text-red-400">{searchError}</p>}
              </div>

              {activeSimilarityQuery && (
                <>
                  {matchesError && (
                    <p className="text-sm text-red-400">
                      Similarity search failed: {matchesError}
                    </p>
                  )}

                  {!matchesError && matches === null && (
                    <p className="text-sm text-slate-400">Searching...</p>
                  )}

                  {!matchesError && matches !== null && matches.length === 0 && (
                    <p className="text-sm text-slate-400">No matches found.</p>
                  )}

                  {!matchesError && matches !== null && matches.length > 0 && (
                    <div className="min-h-0 flex-1 overflow-auto rounded border border-slate-800">
                      <table className="w-full border-collapse text-sm">
                        <thead className="sticky top-0 bg-slate-900 text-xs uppercase tracking-wide text-slate-500">
                          <tr>
                            <th className="px-3 py-2 text-left font-semibold">id</th>
                            <th className="px-3 py-2 text-left font-semibold">distance</th>
                            <th className="px-3 py-2 text-left font-semibold">document</th>
                            <th className="px-3 py-2 text-left font-semibold">metadata</th>
                          </tr>
                        </thead>
                        <tbody>
                          {matches.map((m) => (
                            <tr
                              key={m.id}
                              onClick={() => setSelectedRecordId(m.id)}
                              className={`cursor-pointer border-t border-slate-800 hover:bg-slate-800/60 ${
                                m.id === selectedRecordId ? "bg-slate-800 text-slate-50" : ""
                              }`}
                            >
                              <td className="max-w-[10rem] truncate px-3 py-1.5 align-top font-mono text-xs">
                                <span className="inline-flex items-center gap-1">
                                  {embeddingStatus(m.metadata) === "stale" && (
                                    <span
                                      title="document / embedding が不整合の可能性があります (stale)"
                                      className="shrink-0 rounded bg-amber-900/60 px-1 text-[10px] font-sans font-semibold text-amber-300"
                                    >
                                      ⚠ stale
                                    </span>
                                  )}
                                  {m.id}
                                </span>
                              </td>
                              <td className="px-3 py-1.5 align-top font-mono text-xs text-slate-400">
                                {m.distance !== null ? m.distance.toFixed(4) : "-"}
                              </td>
                              <td className="max-w-xs truncate px-3 py-1.5 align-top">
                                {m.document ?? "-"}
                              </td>
                              <td className="max-w-xs truncate px-3 py-1.5 align-top text-slate-400">
                                {summarizeMetadata(m.metadata)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </>
              )}

              {!activeSimilarityQuery && recordsError && (
                <p className="text-sm text-red-400">Failed to load records: {recordsError}</p>
              )}

              {!activeSimilarityQuery && !recordsError && records === null && (
                <p className="text-sm text-slate-400">Loading records...</p>
              )}

              {!activeSimilarityQuery &&
                !recordsError &&
                records !== null &&
                records.length === 0 && (
                  <p className="text-sm text-slate-400">No records in this collection.</p>
                )}

              {!activeSimilarityQuery && !recordsError && records !== null && records.length > 0 && (
                <div className="flex flex-1 min-h-0 flex-col gap-2">
                  <div className="min-h-0 flex-1 overflow-auto rounded border border-slate-800">
                    <table className="w-full border-collapse text-sm">
                      <thead className="sticky top-0 bg-slate-900 text-xs uppercase tracking-wide text-slate-500">
                        <tr>
                          <th className="w-8 px-3 py-2 text-left font-semibold">
                            <input
                              type="checkbox"
                              aria-label="Select all visible records"
                              checked={
                                records.length > 0 &&
                                records.every((r) => selectedIds.has(r.id))
                              }
                              onChange={toggleSelectAllVisible}
                              className="cursor-pointer"
                            />
                          </th>
                          <th className="px-3 py-2 text-left font-semibold">id</th>
                          <th className="px-3 py-2 text-left font-semibold">document</th>
                          <th className="px-3 py-2 text-left font-semibold">metadata</th>
                        </tr>
                      </thead>
                      <tbody>
                        {records.map((r) => (
                          <tr
                            key={r.id}
                            ref={(el) => {
                              if (el) recordRowRefs.current.set(r.id, el);
                              else recordRowRefs.current.delete(r.id);
                            }}
                            onClick={() => setSelectedRecordId(r.id)}
                            className={`cursor-pointer border-t border-slate-800 hover:bg-slate-800/60 ${
                              r.id === selectedRecordId ? "bg-slate-800 text-slate-50" : ""
                            }`}
                          >
                            <td
                              className="px-3 py-1.5 align-top"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <input
                                type="checkbox"
                                aria-label={`Select record ${r.id}`}
                                checked={selectedIds.has(r.id)}
                                onChange={() => toggleRecordSelected(r.id)}
                                className="cursor-pointer"
                              />
                            </td>
                            <td className="max-w-[10rem] truncate px-3 py-1.5 align-top font-mono text-xs">
                              <span className="inline-flex items-center gap-1">
                                {embeddingStatus(r.metadata) === "stale" && (
                                  <span
                                    title="document / embedding が不整合の可能性があります (stale)"
                                    className="shrink-0 rounded bg-amber-900/60 px-1 text-[10px] font-sans font-semibold text-amber-300"
                                  >
                                    ⚠ stale
                                  </span>
                                )}
                                {r.id}
                              </span>
                            </td>
                            <td className="max-w-xs truncate px-3 py-1.5 align-top">
                              {r.document ?? "-"}
                            </td>
                            <td className="max-w-xs truncate px-3 py-1.5 align-top text-slate-400">
                              {summarizeMetadata(r.metadata)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>
                      {isIdSearch ? (
                        <>
                          {records?.length ?? 0} match{records?.length === 1 ? "" : "es"}
                        </>
                      ) : activeSearch ? (
                        <>
                          {rangeStart}–{rangeEnd} (total unknown while filtering)
                        </>
                      ) : (
                        <>
                          {rangeStart}–{rangeEnd} / {recordsTotal}
                        </>
                      )}
                    </span>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        disabled={!canPrev}
                        onClick={() => setOffset((o) => Math.max(0, o - PAGE_LIMIT))}
                        className="rounded border border-slate-700 px-2 py-1 disabled:cursor-not-allowed disabled:opacity-40 hover:bg-slate-800"
                      >
                        Prev
                      </button>
                      <button
                        type="button"
                        disabled={!canNext}
                        onClick={() => setOffset((o) => o + PAGE_LIMIT)}
                        className="rounded border border-slate-700 px-2 py-1 disabled:cursor-not-allowed disabled:opacity-40 hover:bg-slate-800"
                      >
                        Next
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </main>

        <aside className="w-96 shrink-0 border-l border-slate-800 overflow-y-auto p-3">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Detail
          </h2>

          {!selectedRecordId && (
            <p className="mt-2 text-sm text-slate-400">
              Select a record from the table to see its details.
            </p>
          )}

          {selectedRecordId && detailLoading && (
            <p className="mt-2 text-sm text-slate-400">Loading record...</p>
          )}

          {selectedRecordId && !detailLoading && detailError && (
            <p className="mt-2 text-sm text-red-400">
              Failed to load record: {detailError}
            </p>
          )}

          {selectedRecordId && !detailLoading && !detailError && detailMissing && (
            <p className="mt-2 text-sm text-slate-400">
              This record is no longer present in the collection.
            </p>
          )}

          {selectedRecordId && !detailLoading && !detailError && detailRecord && (
            <div className="mt-2 space-y-4 text-sm">
              <div>
                <div className="mb-1 flex items-center justify-between">
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    id
                  </h3>
                  {isWriteMode && (
                    <button
                      type="button"
                      onClick={() => setDeleteRecordModalOpen(true)}
                      className="rounded border border-red-800 px-2 py-0.5 text-xs text-red-300 hover:bg-red-900/40"
                    >
                      Delete
                    </button>
                  )}
                </div>
                <code className="block select-all break-all rounded bg-slate-900 px-2 py-1 text-xs text-slate-300">
                  {detailRecord.id}
                </code>
              </div>

              {(() => {
                const status = embeddingStatus(detailRecord.metadata);
                if (!status) return null;
                return (
                  <div>
                    {status === "stale" ? (
                      <div className="space-y-1 rounded border border-amber-800 bg-amber-900/40 px-2 py-1.5">
                        <p className="text-xs font-semibold text-amber-300">
                          ⚠ stale: document が embedding と不整合の可能性があります
                        </p>
                        <p className="text-xs text-amber-200/80">
                          Re-embed で解消できます。
                          {isWriteMode && embeddingAvailable && (
                            <>
                              {" "}
                              <button
                                type="button"
                                onClick={() => documentEditorRef.current?.startEdit()}
                                className="rounded border border-amber-700 px-1.5 py-0.5 text-xs text-amber-200 hover:bg-amber-900/60"
                              >
                                Document を編集して Re-embed
                              </button>
                            </>
                          )}
                        </p>
                      </div>
                    ) : (
                      <span className="inline-block rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-400 ring-1 ring-slate-700">
                        fresh
                      </span>
                    )}
                  </div>
                );
              })()}

              {isWriteMode ? (
                <DocumentEditor
                  ref={documentEditorRef}
                  collectionName={selectedName!}
                  record={detailRecord}
                  onSaved={() => setRefreshTick((t) => t + 1)}
                />
              ) : (
                <div>
                  <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                    document
                  </h3>
                  <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded bg-slate-900 p-2 font-mono text-xs text-slate-300">
                    {detailRecord.document ?? "-"}
                  </pre>
                </div>
              )}

              {isWriteMode ? (
                <MetadataEditor
                  ref={metadataEditorRef}
                  collectionName={selectedName!}
                  record={detailRecord}
                  onSaved={() => setRefreshTick((t) => t + 1)}
                />
              ) : (
                <div>
                  <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                    metadata
                  </h3>
                  <pre className="max-h-64 overflow-auto rounded bg-slate-900 p-2 font-mono text-xs text-slate-300">
                    {formatMetadataJson(detailRecord.metadata)}
                  </pre>
                </div>
              )}

              <div>
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  uri
                </h3>
                <p className="break-all text-xs text-slate-300">
                  {detailRecord.uri ?? "-"}
                </p>
              </div>

              <div>
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  embedding
                </h3>
                {detailRecord.embedding_dimension === null ? (
                  <p className="text-xs text-slate-400">No embedding available.</p>
                ) : (
                  <div className="space-y-1">
                    <p className="text-xs text-slate-400">
                      dimension: {detailRecord.embedding_dimension}
                    </p>
                    <pre className="overflow-x-auto rounded bg-slate-900 p-2 font-mono text-xs text-slate-300">
                      [{(detailRecord.embedding_preview ?? []).join(", ")}
                      {(detailRecord.embedding_preview?.length ?? 0) <
                      detailRecord.embedding_dimension
                        ? ", ..."
                        : ""}
                      ]
                    </pre>
                  </div>
                )}
              </div>
            </div>
          )}
        </aside>
      </div>

      {helpOpen && (
        <ShortcutsHelpModal onClose={() => setHelpOpen(false)} isWriteMode={isWriteMode} />
      )}

      {deleteRecordModalOpen && selectedRecordId && (
        <ConfirmNameModal
          title="Delete record"
          description={`This permanently deletes record ${selectedRecordId} from collection ${selectedName}.`}
          expected={selectedRecordId}
          confirmLabel="Delete"
          onConfirm={deleteRecord}
          onCancel={() => setDeleteRecordModalOpen(false)}
        />
      )}

      {deleteCollectionModalOpen && selectedName && (
        <ConfirmNameModal
          title="Delete collection"
          description={`This permanently deletes the collection "${selectedName}" and all of its records.`}
          expected={selectedName}
          confirmLabel="Delete"
          onConfirm={deleteCollection}
          onCancel={() => setDeleteCollectionModalOpen(false)}
        />
      )}

      {bulkDeleteModalOpen && selectedName && (
        <ConfirmNameModal
          title="Delete selected records"
          description={`This permanently deletes ${selectedIds.size} selected record(s) from collection "${selectedName}".`}
          expected={selectedName}
          confirmLabel="Delete"
          onConfirm={bulkDeleteSelected}
          onCancel={() => setBulkDeleteModalOpen(false)}
        />
      )}

      {bulkPatchModalOpen && selectedName && (
        <BulkPatchModal
          selectedCount={selectedIds.size}
          collectionName={selectedName}
          onConfirm={bulkPatchSelected}
          onCancel={() => setBulkPatchModalOpen(false)}
        />
      )}

      {renameCollectionModalOpen && selectedName && (
        <RenameCollectionModal
          currentName={selectedName}
          onConfirm={renameCollection}
          onCancel={() => setRenameCollectionModalOpen(false)}
        />
      )}
    </div>
  );
}

export default App;
