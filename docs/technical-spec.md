# chromaw 技術仕様 v0.1

## 1. 概要

`chromaw` は、ローカルの ChromaDB 永続化ディレクトリをブラウザ UI で閲覧・検索・編集するローカルファーストツールである。

想定 UX は `mo` や `difit` に近い。

```bash
chromaw ./chroma
# => http://127.0.0.1:xxxxx を起動し、ブラウザを開く
```

アプリの主目的は次の 4 つ。

1. ChromaDB の中身を安全に見える化する
2. レコード単位で documents / metadatas / embeddings / uris / ids を確認する
3. documents / metadatas / collection metadata を編集できるようにする
4. 編集によるベクトル不整合や誤削除を避けるため、diff・バックアップ・確認フローを標準搭載する

## 2. 非ゴール

MVP では以下を非ゴールにする。

- ChromaDB 以外の Vector DB 対応
- マルチユーザー SaaS 化
- インターネット公開前提の認証・認可
- 大規模クラスタ運用 UI
- 任意 embedding provider の完全自動推定
- Chroma の内部 SQLite / segment ファイルを直接編集すること

## 3. 重要な設計原則

### 3.1 ChromaDB は必ず公式クライアント経由で操作する

永続化ディレクトリ内の `chroma.sqlite3` や UUID segment ディレクトリを直接編集しない。

理由:

- 内部スキーマは Chroma の実装詳細である
- インデックスや collection metadata との整合性を壊す可能性がある
- バージョン差分に弱い

### 3.2 編集は safe-by-default

初期状態は read-only とし、ユーザーが `--write` を付けたときだけ編集を許可する。

```bash
chromaw ./chroma          # read-only
chromaw ./chroma --write  # edit enabled
```

破壊的操作は必ず確認ダイアログを出す。

- レコード削除
- collection 削除
- collection rename
- bulk update / bulk delete
- embeddings 更新
- reset 相当の操作

### 3.3 document と embedding の整合性を明示する

ChromaDB では document を更新すると、embedding の再計算が必要になるケースがある。`chromaw` は document 変更時に以下のモードを提供する。

| モード | MVP | 説明 |
|---|---:|---|
| metadata-only edit | ✅ | embedding に影響しないため安全 |
| document edit + re-embed | ✅ | 指定された embedding function / provider で再計算する |
| document edit + keep embedding | ⚠️ | document と vector が不整合になるため明示警告つき |
| embedding direct edit | v0.2 | advanced mode のみ |

MVP では、document 更新時に embedding 設定が未指定なら `keep embedding` は許可するが、レコードに `chromaw_embedding_status: "stale"` のような metadata 付与を推奨する。既存 metadata を汚したくない場合は chromaw 側の audit log に stale 状態を保存する。

## 4. 想定ユーザー

1. RAG アプリ開発者
   - ローカル ChromaDB の中身を確認したい
   - chunk / metadata / source path / score を手早く見たい

2. AI エージェント・ツール開発者
   - agent memory の中身をブラウザで点検・修正したい
   - 不要な記憶や重複 chunk を削除したい

3. データ整備担当者
   - metadata の欠損や誤りを一括修正したい
   - source ごとの削除・再投入をしたい

## 5. MVP スコープ

### 5.1 CLI

```bash
chromaw [path]
chromaw [path] --write
chromaw [path] --host 127.0.0.1 --port 0
chromaw [path] --no-open
chromaw [path] --backup-on-write
chromaw http://localhost:8000 # v0.2: remote/server mode
```

MVP では Python 実装の `chromaw` が `chromadb.PersistentClient(path=...)` でローカル ChromaDB 永続化ディレクトリに直接接続する構成を主対象にする。

```bash
chromaw ./chroma                  # chromaw API と Web UI を起動
chromaw --connect http://localhost:8000 # v0.2: 既存 Chroma server に接続
```

React frontend は開発時に Vite で build し、生成された static assets を Python wheel に同梱する。ユーザーの実行環境に Node.js / npm は要求しない。

### 5.2 コレクション一覧

表示項目:

- collection name
- collection id
- count
- metadata
- embedding dimension 推定値
- tenant / database
- 最終操作時刻。ただし Chroma から取れない場合は chromaw audit log ベース

操作:

- collection 選択
- collection metadata 編集
- collection rename
- collection delete。ただし MVP では hidden advanced action

### 5.3 レコード一覧

表示項目:

- id
- document preview
- metadata summary
- uri
- embedding dimension
- embedding preview。先頭 8 values 程度
- status flags
  - stale embedding suspected
  - metadata invalid
  - duplicate source
  - missing document

操作:

- ページング
- id 検索
- document 全文検索 / where_document
- metadata filter / where
- sort は Chroma 側で十分に対応しない可能性があるため MVP では client-side current page sort
- JSON export

### 5.4 レコード詳細・編集

詳細画面は 3 ペイン構成。

1. 左: レコード一覧 / 検索結果
2. 中央: document viewer / editor
3. 右: metadata JSON editor / embedding info / actions

編集対象:

- document text
- metadata JSON
- uri

編集フロー:

1. 編集開始
2. JSON schema validation / metadata type validation
3. before / after diff 表示
4. save confirmation
5. `collection.update(...)`
6. audit log 保存
7. toast + detail refresh

### 5.5 検索

MVP 検索モード:

1. Get by ID
2. Metadata filter
3. Document contains filter
4. Query text search

Query text search は embedding function が必要になる。MVP では以下の優先順位にする。

1. 起動時に `--embedding-config` が指定されている場合、それを使用
2. Chroma default embedding function が利用可能なら使用
3. 利用不可の場合、query embeddings を手入力または検索機能を disabled 表示

### 5.6 Bulk 操作

MVP では read UX を先に固め、bulk write は最小限にする。

- bulk export: ✅
- bulk delete by selected IDs: ✅ advanced + confirmation
- bulk metadata patch: v0.2
- bulk re-embed: v0.2
- bulk import CSV / JSONL: v0.2

## 6. UI / UX 仕様

### 6.1 方向性

- GitHub 風の読みやすい差分 UI
- ローカルツールらしい軽快な操作感
- 一画面で対象・内容・変更差分が追える
- 編集より先に閲覧・検索が気持ちよくできる

### 6.2 画面構成

```text
┌──────────────────────────────────────────────────────────────┐
│ Header: chromaw | path | mode: read-only/write | search       │
├───────────────┬──────────────────────────┬───────────────────┤
│ Collections   │ Records                  │ Detail / Editor    │
│               │                          │                   │
│ - memory      │ id / preview / metadata  │ document           │
│ - docs        │                          │ metadata JSON      │
│ - chunks      │                          │ embedding info     │
└───────────────┴──────────────────────────┴───────────────────┘
```

### 6.3 キーボードショートカット

| Key | Action |
|---|---|
| `/` | search focus |
| `j` / `k` | next / previous record |
| `e` | edit current record |
| `s` or `cmd+s` | save edit |
| `d` | show diff |
| `esc` | close modal / cancel |
| `?` | shortcut help |

### 6.4 差分表示

差分対象:

- document text
- metadata JSON prettified
- uri

表示モード:

- unified diff
- split diff

MVP では unified diff を必須、split diff は v0.2 でも可。

### 6.5 危険操作の UX

削除確認は単なる OK/Cancel ではなく、対象名の入力を要求する。

例:

```text
Delete 12 records from collection `memory`?
Type `delete memory` to confirm.
```

## 7. 技術スタック

### 7.1 推奨構成

MVP は Python backend + bundled React frontend 構成を推奨する。

理由:

- ChromaDB の local persistent path を `chromadb.PersistentClient(path=...)` で直接扱える
- `chroma run` の subprocess 起動、port 管理、終了管理を MVP で持たなくてよい
- ユーザー体験を `pipx install chromaw` / `uvx chromaw` / `pip install chromaw` に集約できる
- React は build 済み static assets として Python wheel に同梱できるため、実行時に Node.js は不要
- UI は React / TypeScript で作れるため、`mo` や `difit` のような軽快な Web UI 体験を実現しやすい

| Layer | Technology |
|---|---|
| CLI | Python + Typer |
| Backend | FastAPI or Starlette + Uvicorn |
| Chroma access | `chromadb` Python package, `PersistentClient(path=...)` |
| Frontend | Vite + React + TypeScript |
| UI | Tailwind CSS + shadcn/ui or Radix UI |
| Editor | CodeMirror 6 |
| Diff | jsdiff + custom renderer, or Monaco diff editor |
| JSON validation | zod on frontend, Pydantic on backend |
| Static serving | React build output bundled under `chromaw/static` |
| Packaging | Python wheel, `pipx install chromaw`, `uvx chromaw` |

### 7.2 React 同梱方針

React / TypeScript は開発時だけ Node.js toolchain を使う。release artifact には build 済みファイルのみを含める。

```text
web/src/*.tsx
  ↓ vite build
src/chromaw/static/index.html
src/chromaw/static/assets/*.js
src/chromaw/static/assets/*.css
  ↓ python build
chromaw-*.whl
```

ユーザー実行時に以下は行わない。

- `npm install`
- `npm build`
- frontend dependency の download

FastAPI / Starlette は package 内 static files を配信する。

```python
from importlib.resources import files
from fastapi.staticfiles import StaticFiles

static_dir = files("chromaw").joinpath("static")
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
```

### 7.3 代替案

#### Option A: Python + FastAPI + bundled React（採用）

Pros:

- local ChromaDB path 直結が最も単純
- ChromaDB の Python 利用者と導入経路が近い
- React UI を捨てずに済む
- MVP 実装が速い

Cons:

- Python runtime は必要
- `chromadb` dependency が重い可能性がある
- Python version / platform 差分は `pipx` / `uvx` 前提で緩和する必要がある

#### Option B: TypeScript CLI + managed Chroma server

Pros:

- `npx chromaw ./chroma` の体験にしやすい
- frontend/backend/API schema を TypeScript で統一できる
- Python runtime を chromaw 側で要求しない

Cons:

- local path 直結ではなく、Chroma server subprocess 起動・port 管理が必要
- ユーザー環境に `chroma` CLI がない場合の導線が必要
- Chroma server と npm client の version 整合性確認が必要

#### Option C: Rust CLI + Axum + bundled React

Pros:

- 将来的な単一バイナリ配布に向く
- 起動が速く、runtime 依存を小さくできる
- ローカルサーバー管理やファイル操作が堅牢に書ける

Cons:

- Chroma Rust client は HTTP 接続前提のため、local path には Chroma server 起動管理が必要
- MVP 実装速度は Python より遅くなりやすい
- UI 開発は結局 React / TypeScript frontend が必要

#### Option D: Tauri desktop app

Pros:

- デスクトップアプリとして自然
- ファイル選択 UI が作りやすい

Cons:

- MVP のスピードは落ちる
- Chroma Python runtime または Chroma server 同梱・更新管理が課題

## 8. Backend API 仕様

Base URL: `http://127.0.0.1:{port}/api`

### 8.1 Health

```http
GET /api/health
```

Response:

```json
{
  "ok": true,
  "version": "0.1.0",
  "mode": "read-only",
  "path": "/abs/path/to/chroma"
}
```

### 8.2 Collections

```http
GET /api/collections
```

```json
{
  "collections": [
    {
      "id": "uuid",
      "name": "memory",
      "count": 1234,
      "metadata": {},
      "dimension": 1536
    }
  ]
}
```

```http
GET /api/collections/{name}
PATCH /api/collections/{name}
DELETE /api/collections/{name}
```

`PATCH` body:

```json
{
  "name": "new_name",
  "metadata": { "key": "value" }
}
```

### 8.3 Records

```http
GET /api/collections/{name}/records?limit=50&offset=0&include=documents,metadatas,uris
```

```http
POST /api/collections/{name}/records/get
```

Body:

```json
{
  "ids": ["id1", "id2"],
  "where": { "source": "README.md" },
  "where_document": { "$contains": "hello" },
  "limit": 50,
  "offset": 0,
  "include": ["documents", "metadatas", "uris"]
}
```

```http
PATCH /api/collections/{name}/records/{id}
```

Body:

```json
{
  "document": "new text",
  "metadata": { "source": "README.md" },
  "uri": "file:///...",
  "embedding_mode": "reembed|keep|manual",
  "embedding": null
}
```

```http
DELETE /api/collections/{name}/records/{id}
```

### 8.4 Query

```http
POST /api/collections/{name}/query
```

Body:

```json
{
  "query_text": "what is chromaw?",
  "query_embedding": null,
  "n_results": 10,
  "where": {},
  "where_document": {},
  "include": ["documents", "metadatas", "distances"]
}
```

### 8.5 Diff Preview

```http
POST /api/diff
```

Body:

```json
{
  "before": { "document": "old", "metadata": {} },
  "after": { "document": "new", "metadata": {} }
}
```

Response:

```json
{
  "document_unified_diff": "...",
  "metadata_unified_diff": "..."
}
```

## 9. データ整合性・バックアップ

### 9.1 Backup strategy

`--backup-on-write` をデフォルト true にする案を推奨する。

初回 write 操作前に永続化ディレクトリを丸ごとコピーする。

```text
.chroma
.chroma.chromaw-backups/
  20260713-231500-before-first-write/
```

ただし巨大 DB ではコピーコストが高いため、将来は以下に分岐する。

- small DB: directory copy
- large DB: export changed records only + audit log
- advanced: filesystem snapshot integration

### 9.2 Audit log

chromaw 専用ディレクトリに JSONL で保存する。

```text
{chroma_path}/.chromaw/audit.jsonl
```

例:

```json
{
  "timestamp": "2026-07-13T14:15:00Z",
  "operation": "record.update",
  "collection": "memory",
  "id": "abc",
  "before_hash": "sha256:...",
  "after_hash": "sha256:...",
  "embedding_mode": "keep",
  "user_agent": "chromaw/0.1.0"
}
```

### 9.3 Concurrency

MVP では単一 `chromaw` プロセスによる操作を前提にする。

- 起動時に `.chromaw/lock` を作成
- 同一 path に対して別 chromaw が起動済みなら read-only で起動するか警告
- 他アプリが同じ ChromaDB を同時更新する可能性は検知困難なので UI に明示する

## 10. Security

`chromaw` はローカルツールだが、write API を持つためセキュリティを軽視しない。

### 10.1 Bind address

デフォルトは必ず localhost。

```text
127.0.0.1 only
```

`--host 0.0.0.0` は advanced warning を出す。

### 10.2 CSRF / local web attack 対策

ローカルサーバーでも、ブラウザからの cross-site request を避ける。

- 起動時に random token を発行
- UI の initial HTML に token を埋め込む
- API は `Authorization: Bearer <token>` 必須
- `Origin` / `Host` validation
- CORS はデフォルト disabled

### 10.3 Path access

API 経由で任意ファイルを読めるようにしない。

- Chroma path 以外への file browser は MVP では実装しない
- uri preview はクリック可能にしても、backend がファイル内容を返すのは opt-in

## 11. エラーハンドリング

### 11.1 起動時エラー

- path が存在しない
  - `--create` 指定があれば作成
  - 未指定なら確認メッセージ
- ChromaDB として読めない
  - empty directory か corrupted かを表示
- chromadb version mismatch
  - chromaw が使う chromadb version を表示
  - read-only fallback を提案

### 11.2 編集時エラー

- metadata JSON が invalid
- Chroma metadata value type が非対応
- document 更新に embedding function がない
- record id が存在しない
- concurrent update suspected

## 12. Testing strategy

### 12.1 Unit tests

- metadata validation
- diff generation
- API schema validation
- Chroma adapter abstraction

### 12.2 Integration tests

一時ディレクトリに ChromaDB を作成し、以下を検証する。

- collection list
- record get
- metadata update
- document update keep embedding
- delete record
- backup created before write
- audit log written

### 12.3 UI tests

- Playwright
- record search
- edit + diff + save
- dangerous action confirmation

## 13. ディレクトリ構成案

```text
chromaw/
  README.md
  docs/
    technical-spec.md
  pyproject.toml
  package.json                 # frontend build orchestration 用。runtime には不要
  src/chromaw/
    __init__.py
    cli.py                     # `chromaw` command
    server.py                  # FastAPI / Starlette app
    chroma_adapter.py          # chromadb PersistentClient wrapper
    models.py                  # Pydantic schemas
    backup.py
    audit.py
    security.py
    static/                    # frontend build output included in wheel
      index.html
      assets/
  web/
    package.json
    vite.config.ts
    src/
      app/
      components/
      api/
      pages/
      styles/
  tests/
    test_chroma_adapter.py
    test_api.py
    test_backup.py
    test_audit.py
```

Package build では `web` を build して `src/chromaw/static` に配置した状態で wheel を作る。sdist に frontend source を含めるかどうかは release policy で決めるが、wheel install 時に Node.js build を走らせない。

## 14. 実装マイルストーン

### Milestone 0: Skeleton

- `chromaw ./path` CLI
- `chromadb.PersistentClient(path=...)` 接続
- chromaw API server 起動
- bundled static frontend 配信
- health endpoint

### Milestone 1: Read-only viewer

- collection list
- record list
- record detail
- metadata / document rendering
- basic search by id / where / where_document

### Milestone 2: Safe editor

- `--write` mode
- metadata edit
- document edit with keep embedding warning
- diff preview
- backup before first write
- audit log

### Milestone 3: Query / re-embedding

- query text search
- embedding config
- document edit + re-embed
- stale embedding flags

### Milestone 4: Bulk / import-export

- selected export
- bulk delete
- JSONL import / export
- metadata patch

## 15. 未決定事項

1. `--write` の default を read-only にするか、初回だけ確認にするか
   - 推奨: read-only default

2. document 編集で embedding function が不明な場合の default
   - 推奨: save disabled ではなく、強い警告つき keep embedding を許可

3. audit log を Chroma path 内に置くか、ユーザー cache dir に置くか
   - 推奨: path 内 `.chromaw`。DB と一緒に履歴が残るため

4. frontend UI ライブラリ
   - 推奨: Tailwind + Radix/shadcn

5. 単一バイナリ化をいつやるか
   - 推奨: MVP は Python wheel。まずは `pipx install chromaw` / `uvx chromaw` を目指す。単一バイナリは MVP 後に検討
