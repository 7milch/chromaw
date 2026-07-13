# chromaw 実装ロードマップ

本ロードマップは技術仕様 [`docs/technical-spec.md`](../technical-spec.md) §14（実装マイルストーン）および §15（未決定事項）に基づく。

## 運用ルール

- 各タスクは Issue 単位で `issue-[number]` ブランチで作業する。
- タスク完了時はチェックボックスに `[x]` を付け、対応する GitHub Issue 番号を `(Issue #xx)` の形式で追記する。
- 実装の詳細な仕様変更・追加決定が発生した場合は、本ファイルではなく `docs/technical-spec.md` を更新し、必要に応じて `docs/planning/adr.md` に ADR として記録する。

---

## Milestone 0: Skeleton

- [ ] M0-1: `pyproject.toml` と `src/chromaw/` パッケージ構成、Typer CLI (`chromaw ./path`) の初期化
- [ ] M0-2: `chromadb.PersistentClient(path=...)` 接続と起動時エラーハンドリング（§11.1）
- [ ] M0-3: FastAPI サーバー起動（port 自動割当、`--host` / `--port` / `--no-open` オプション）
- [ ] M0-4: `GET /api/health` エンドポイント
- [ ] M0-5: `web/` の Vite + React + TypeScript + Tailwind 初期化、build 成果物を `src/chromaw/static/` へ配信
- [ ] M0-6: セキュリティ基盤（127.0.0.1 bind、ランダム token による Bearer 認証、Origin/Host 検証）（§10）
- [ ] M0-7: pytest と一時ディレクトリ ChromaDB fixture によるテスト基盤

## Milestone 1: Read-only viewer

- [ ] M1-1: collections API と一覧 UI
- [ ] M1-2: records API（paging）と一覧 UI
- [ ] M1-3: 3ペイン詳細 UI（collection / record list / detail）
- [ ] M1-4: 検索（by ID, where, where_document）
- [ ] M1-5: 読み取り系キーボードショートカット
- [ ] M1-6: JSON export

## Milestone 2: Safe editor

- [ ] M2-1: `--write` フラグとモード表示（read-only / write）
- [ ] M2-2: metadata 編集フロー（validation → diff → 確認 → update）
- [ ] M2-3: document 編集（keep embedding 警告 + stale 記録）
- [ ] M2-4: `POST /api/diff` と unified diff 表示
- [ ] M2-5: 初回 write 前バックアップ
- [ ] M2-6: audit log（`.chromaw/audit.jsonl`）
- [ ] M2-7: 削除・rename の対象名入力確認フロー
- [ ] M2-8: `.chromaw/lock` による多重起動ガード
- [ ] M2-9: 編集系キーボードショートカット

## Milestone 3: Query / re-embedding

- [ ] M3-1: query API と UI
- [ ] M3-2: `--embedding-config` と解決優先順位
- [ ] M3-3: document edit + re-embed
- [ ] M3-4: stale フラグ表示

## Milestone 4: Bulk / import-export

- [ ] M4-1: selected export
- [ ] M4-2: bulk delete（advanced + 確認フロー）
- [ ] M4-3: JSONL import / export
- [ ] M4-4: bulk metadata patch
- [ ] M4-5: Playwright UI テスト

## Release

- [ ] R-1: wheel packaging（static 同梱、Node.js 不要であることの検証）
- [ ] R-2: pipx / uvx 動作確認
- [ ] R-3: PyPI 公開

---

## 未決定事項

詳細は技術仕様 [`docs/technical-spec.md` §15](../technical-spec.md#15-未決定事項) を参照。決定が確定した際は本ロードマップではなく `docs/planning/adr.md` に ADR として記録する。

1. `--write` の default を read-only にするか、初回だけ確認にするか（推奨: read-only default）
2. document 編集で embedding function が不明な場合の default（推奨: 強い警告つき keep embedding を許可）
3. audit log の配置場所（推奨: Chroma path 内 `.chromaw`）
4. frontend UI ライブラリ（推奨: Tailwind + Radix/shadcn）
5. 単一バイナリ化の実施時期（推奨: MVP は Python wheel、単一バイナリは MVP 後に検討）
