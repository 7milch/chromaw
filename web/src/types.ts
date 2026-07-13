export interface HealthResponse {
  ok: boolean;
  version: string;
  mode: string;
  path: string;
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
}

export interface RecordsGetRequest {
  ids?: string[];
  limit?: number;
  offset?: number;
  include?: string[];
}
