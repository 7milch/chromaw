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

## Installation

**注意:** `chromaw` はまだ PyPI に公開されていません。公開後は以下のコマンドがそのまま使えます。それまではローカルでビルドした wheel から実行してください。

### uvx（推奨）

インストール不要でその場実行できます。

```bash
# PyPI 公開後
uvx chromaw ./chroma

# 公開前: ローカル wheel から
uv build   # dist/chromaw-*.whl を生成（後述）
uvx --from ./dist/chromaw-0.1.0-py3-none-any.whl chromaw ./chroma
```

### pipx

グローバルにコマンドをインストールしたい場合。

```bash
# PyPI 公開後
pipx install chromaw

# 公開前: ローカル wheel から
pipx install ./dist/chromaw-0.1.0-py3-none-any.whl

chromaw ./chroma

# アンインストール
pipx uninstall chromaw
```

### pip

```bash
# PyPI 公開後
pip install chromaw

# 公開前: ローカル wheel から
pip install ./dist/chromaw-0.1.0-py3-none-any.whl
```

## Usage

```bash
chromaw ./chroma
```

デフォルトは read-only で起動します。主なオプション:

| オプション | 説明 |
| --- | --- |
| `[path]` | ChromaDB 永続化ディレクトリのパス（デフォルト: `.`） |
| `--write` | 編集を有効化します（未指定時は read-only） |
| `--host` | バインドするホスト（デフォルト: `127.0.0.1`） |
| `--port` | バインドするポート。`0` で自動割り当て（デフォルト: `0`） |
| `--no-open` | 起動時にブラウザを自動で開かない |
| `--create` | 対象ディレクトリが存在しない/空の場合に作成する |
| `--embedding-config` | クエリテキスト検索に使う embedding function を JSON ファイルで指定 |
| `--version` | バージョンを表示して終了 |

例: 編集モードで起動し、ポート `8000` を明示的に指定する場合

```bash
chromaw ./chroma --write --port 8000
```

## 開発者向け

フロントエンド（`web/`）をビルドしてから Python パッケージをビルドします。

```bash
npm --prefix web run build   # web/ から src/chromaw/static/ へ static assets を生成
uv build                     # dist/ に sdist と wheel を生成
```

生成された wheel はローカルインストール確認用です（上記 Installation の「公開前」手順を参照）。

詳細仕様は [`docs/technical-spec.md`](docs/technical-spec.md) を参照してください。
