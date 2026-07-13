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
