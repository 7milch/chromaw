# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

`chromaw` は、ローカルの ChromaDB 永続化ディレクトリをブラウザ UI で閲覧・検索・編集するローカルファーストツール。`chromaw ./chroma` を実行するとローカルサーバーが起動し、ブラウザで安全に編集できる（`mo` や `difit` に近い UX）。

**現状: 実装前。** リポジトリには仕様ドキュメントのみが存在する。全体仕様・API 仕様・マイルストーンは `docs/technical-spec.md` を必ず参照すること。git リポジトリが未初期化なら、実装開始前に `git init` が必要。

## アーキテクチャ（採用方針）

- **Python backend + bundled React frontend**（technical-spec §7 の Option A を採用済み）
- CLI: Python + Typer / Backend: FastAPI + Uvicorn / Frontend: Vite + React + TypeScript + Tailwind + Radix/shadcn
- ChromaDB へは必ず `chromadb.PersistentClient(path=...)` 経由でアクセスする。`chroma.sqlite3` や segment ファイルを直接編集してはならない
- React は開発時のみ Node.js を使い、`vite build` の成果物を `src/chromaw/static/` に配置して Python wheel に同梱する。ユーザー実行時に Node.js/npm は不要
- 想定ディレクトリ構成: `src/chromaw/`（cli.py, server.py, chroma_adapter.py, models.py, backup.py, audit.py, security.py, static/）、`web/`（frontend）、`tests/`
- API は `http://127.0.0.1:{port}/api` 配下。エンドポイント仕様は technical-spec §8 に準拠する

## 重要な設計原則

- **safe-by-default**: デフォルト read-only。`--write` 指定時のみ編集可。破壊的操作（削除・rename・bulk 操作）は対象名の入力を要求する確認フロー必須
- **document と embedding の整合性を明示**: document 編集時は re-embed / keep（警告つき）/ manual のモードを持つ。keep の場合は stale 状態を audit log に記録
- **バックアップと監査ログ**: 初回 write 前にディレクトリバックアップ、全 write 操作を `{chroma_path}/.chromaw/audit.jsonl` に JSONL で記録
- **セキュリティ**: bind は 127.0.0.1 のみ、起動時 random token による Bearer 認証、Origin/Host 検証、CORS 無効。Chroma path 外へのファイルアクセス API を作らない

## 実装マイルストーン（technical-spec §14）

M0: Skeleton（CLI・PersistentClient 接続・server・static 配信・health）→ M1: Read-only viewer → M2: Safe editor（--write, diff, backup, audit）→ M3: Query / re-embedding → M4: Bulk / import-export

## Hooks

- Stop hook (`.claude/hooks/adr-check.sh`): `docs/architecture/`, `docs/detailed-design/`, `docs/workflows/`, `config/*.yaml` に変更があるのに `docs/planning/adr.md` が未更新だと完了をブロックする。設計判断を伴う変更では ADR を追記すること（採番は既存最終 ADR の次番号）

## テスト方針（technical-spec §12）

- Unit: metadata validation, diff 生成, API schema, Chroma adapter
- Integration: 一時ディレクトリに ChromaDB を作成して API・backup・audit を検証
- UI: Playwright（検索、edit + diff + save、危険操作の確認フロー）
