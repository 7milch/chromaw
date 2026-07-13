# chromaw

`chromaw` は、ローカルで動く Web UI からローカルの ChromaDB を閲覧・検索・編集するためのツールです。

コンセプトは「`chromaw /path/to/chroma` を実行すると、ローカルサーバーが起動してブラウザで ChromaDB を安全に編集できる」ことです。

## 方針

- Python backend + bundled React frontend
- `chromadb.PersistentClient(path=...)` でローカル ChromaDB に直接接続
- React は build 済み static assets として Python wheel に同梱
- ユーザー実行時に Node.js / npm は不要
- ローカルファースト
- コレクション / レコードの閲覧・検索・編集
- 変更前後の diff と確認フロー
- 破壊的操作のガード、バックアップ、監査ログ

## 想定インストール UX

```bash
pipx install chromaw
chromaw ./chroma
```

または:

```bash
uvx chromaw ./chroma
```

詳細仕様は [`docs/technical-spec.md`](docs/technical-spec.md) を参照してください。
