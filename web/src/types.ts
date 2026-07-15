export interface HealthResponse {
  ok: boolean;
  version: string;
  mode: string;
  path: string;
  embedding_available: boolean;
}

export interface CollectionInfo {
  id: string;
  name: string;
  count: number;
  metadata: Record<string, unknown> | null;
  dimension: number | null;
}

export interface CollectionsResponse {
  collections: CollectionInfo[];
}

export interface RecordInfo {
  id: string;
  document: string | null;
  metadata: Record<string, unknown> | null;
  uri: string | null;
  embedding_dimension: number | null;
  embedding_preview: number[] | null;
}

export interface RecordsResponse {
  records: RecordInfo[];
  total: number;
  has_more: boolean;
}

export interface RecordsGetRequest {
  ids?: string[];
  where?: Record<string, unknown>;
  where_document?: Record<string, unknown>;
  limit?: number;
  offset?: number;
  include?: string[];
}

export interface RecordUpdateRequest {
  metadata?: Record<string, unknown> | null;
  uri?: string | null;
  document?: string | null;
  embedding_mode?: "keep" | "reembed" | null;
}

export interface RecordDeleteRequest {
  confirm: string;
}

export interface CollectionDeleteRequest {
  confirm: string;
}

export interface BulkDeleteRequest {
  ids: string[];
  confirm: string;
}

export interface BulkDeleteResponse {
  deleted: string[];
  skipped: string[];
}

export interface CollectionUpdateRequest {
  name?: string;
  metadata?: Record<string, unknown>;
  confirm?: string;
}

export interface DeleteResponse {
  deleted: boolean;
  id: string;
}

export interface DiffRequest {
  before: string;
  after: string;
  before_label?: string;
  after_label?: string;
}

export interface DiffResponse {
  diff: string;
}

export interface QueryRequest {
  query_text?: string;
  query_embedding?: number[];
  n_results?: number;
  where?: Record<string, unknown>;
  where_document?: Record<string, unknown>;
  include?: string[];
}

export interface RecordMatchInfo {
  id: string;
  document: string | null;
  metadata: Record<string, unknown> | null;
  uri: string | null;
  distance: number | null;
  embedding_dimension: number | null;
  embedding_preview: number[] | null;
}

export interface QueryResponse {
  matches: RecordMatchInfo[];
}

export interface ImportSkip {
  line: number;
  reason: string;
}

export interface ImportResponse {
  imported: string[];
  skipped: ImportSkip[];
}
