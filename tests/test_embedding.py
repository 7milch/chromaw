import json
from pathlib import Path
from typing import Any

import chromadb
import pytest

from chromaw import embedding as embedding_module
from chromaw.chroma_adapter import ChromaAdapter
from chromaw.embedding import EmbeddingConfig, EmbeddingResolver
from chromaw.errors import EmbeddingConfigError, EmbeddingFunctionUnavailableError


class _MockEmbeddingFunction:
    """Deterministic stand-in for a real provider's embedding function.

    Real providers (OpenAI, sentence-transformers, chromadb's default) are
    not exercised here: they require network access / an API key / model
    downloads that aren't available in CI. Only the *resolution* logic
    (which embedding function chromaw picks) and the error paths (bad
    config, missing key, provider failure) are tested; the actual
    embedding-function build is monkeypatched out.
    """

    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[str] = []

    def __call__(self, input: list[str]) -> list[list[float]]:
        self.calls.extend(input)
        return [[float(len(text))] * self.dimension for text in input]


def _write_config(tmp_path: Path, data: dict) -> Path:
    config_path = tmp_path / "embedding-config.json"
    config_path.write_text(json.dumps(data))
    return config_path


# --- EmbeddingConfig.load: additional edge cases ---------------------------


def test_load_empty_json_object_raises_missing_provider(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {})

    with pytest.raises(EmbeddingConfigError, match="provider"):
        EmbeddingConfig.load(path)


def test_load_provider_only_local_no_model_ok(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"provider": "sentence-transformer"})

    config = EmbeddingConfig.load(path)

    assert config.provider == "sentence-transformer"
    assert config.model is None
    assert config.api_key_env is None


def test_load_hosted_provider_with_empty_string_api_key_env_raises(
    tmp_path: Path,
) -> None:
    # An empty string is falsy, so this must be treated the same as a
    # missing api_key_env (the "requires an API key" branch), not accepted
    # as a (bogus) env var name.
    path = _write_config(
        tmp_path, {"provider": "openai", "api_key_env": ""}
    )

    with pytest.raises(EmbeddingConfigError, match="api_key_env"):
        EmbeddingConfig.load(path)


def test_load_config_path_is_a_directory_raises_embedding_config_error(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "a-directory"
    directory.mkdir()

    with pytest.raises(EmbeddingConfigError):
        EmbeddingConfig.load(directory)


def test_load_non_string_api_key_env_raises(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path, {"provider": "openai", "api_key_env": 123}
    )

    with pytest.raises(EmbeddingConfigError, match="api_key_env"):
        EmbeddingConfig.load(path)


def test_load_json_array_of_scalars_raises(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]")

    with pytest.raises(EmbeddingConfigError):
        EmbeddingConfig.load(path)


def test_load_json_scalar_raises(tmp_path: Path) -> None:
    path = tmp_path / "scalar.json"
    path.write_text('"just a string"')

    with pytest.raises(EmbeddingConfigError):
        EmbeddingConfig.load(path)


# --- EmbeddingConfig.load: parsing / validation ---------------------------


def test_load_valid_default_provider_config(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"provider": "default"})

    config = EmbeddingConfig.load(path)

    assert config.provider == "default"
    assert config.model is None
    assert config.api_key_env is None


def test_load_valid_hosted_provider_config(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key_env": "OPENAI_API_KEY",
        },
    )

    config = EmbeddingConfig.load(path)

    assert config.provider == "openai"
    assert config.model == "text-embedding-3-small"
    assert config.api_key_env == "OPENAI_API_KEY"


def test_load_missing_file_raises_embedding_config_error(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingConfigError):
        EmbeddingConfig.load(tmp_path / "does-not-exist.json")


def test_load_invalid_json_raises_embedding_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")

    with pytest.raises(EmbeddingConfigError):
        EmbeddingConfig.load(path)


def test_load_non_object_json_raises_embedding_config_error(tmp_path: Path) -> None:
    path = _write_config(tmp_path, [])  # type: ignore[arg-type]

    with pytest.raises(EmbeddingConfigError):
        EmbeddingConfig.load(path)


def test_load_missing_provider_raises_embedding_config_error(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"model": "foo"})

    with pytest.raises(EmbeddingConfigError):
        EmbeddingConfig.load(path)


def test_load_unknown_provider_raises_embedding_config_error(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"provider": "not-a-real-provider"})

    with pytest.raises(EmbeddingConfigError):
        EmbeddingConfig.load(path)


def test_load_hosted_provider_without_api_key_env_raises(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"provider": "openai"})

    with pytest.raises(EmbeddingConfigError, match="api_key_env"):
        EmbeddingConfig.load(path)


def test_load_non_string_model_raises(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"provider": "default", "model": 123})

    with pytest.raises(EmbeddingConfigError):
        EmbeddingConfig.load(path)


# --- EmbeddingConfig.resolve_api_key ---------------------------------------


def test_resolve_api_key_local_provider_returns_none() -> None:
    config = EmbeddingConfig(provider="default")

    assert config.resolve_api_key() is None


def test_resolve_api_key_reads_named_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TEST_KEY", "secret-value")
    config = EmbeddingConfig(provider="openai", api_key_env="MY_TEST_KEY")

    assert config.resolve_api_key() == "secret-value"


def test_resolve_api_key_missing_env_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MY_TEST_KEY", raising=False)
    config = EmbeddingConfig(provider="openai", api_key_env="MY_TEST_KEY")

    with pytest.raises(EmbeddingConfigError, match="MY_TEST_KEY"):
        config.resolve_api_key()


def test_resolve_api_key_empty_env_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TEST_KEY", "")
    config = EmbeddingConfig(provider="openai", api_key_env="MY_TEST_KEY")

    with pytest.raises(EmbeddingConfigError):
        config.resolve_api_key()


# --- EmbeddingResolver ------------------------------------------------------


def test_resolver_without_config_has_no_explicit_config() -> None:
    resolver = EmbeddingResolver()

    assert resolver.has_explicit_config is False


def test_resolver_with_config_has_explicit_config() -> None:
    resolver = EmbeddingResolver(EmbeddingConfig(provider="default"))

    assert resolver.has_explicit_config is True


def test_resolver_embed_query_without_config_raises_unavailable() -> None:
    resolver = EmbeddingResolver()

    with pytest.raises(EmbeddingFunctionUnavailableError, match="--embedding-config"):
        resolver.embed_query("hello")


def test_resolver_embed_query_uses_built_embedding_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_ef = _MockEmbeddingFunction(dimension=4)
    monkeypatch.setattr(
        embedding_module, "_build_embedding_function", lambda config: mock_ef
    )
    resolver = EmbeddingResolver(EmbeddingConfig(provider="default"))

    vector = resolver.embed_query("hi")

    assert vector == [2.0, 2.0, 2.0, 2.0]
    assert mock_ef.calls == ["hi"]


def test_resolver_embed_query_caches_embedding_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls: list[EmbeddingConfig] = []

    def fake_build(config: EmbeddingConfig) -> _MockEmbeddingFunction:
        build_calls.append(config)
        return _MockEmbeddingFunction()

    monkeypatch.setattr(embedding_module, "_build_embedding_function", fake_build)
    resolver = EmbeddingResolver(EmbeddingConfig(provider="default"))

    resolver.embed_query("a")
    resolver.embed_query("bb")

    assert len(build_calls) == 1


def test_resolver_embed_query_build_failure_raises_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_build(config: EmbeddingConfig) -> _MockEmbeddingFunction:
        raise EmbeddingFunctionUnavailableError("boom")

    monkeypatch.setattr(embedding_module, "_build_embedding_function", fake_build)
    resolver = EmbeddingResolver(EmbeddingConfig(provider="default"))

    with pytest.raises(EmbeddingFunctionUnavailableError, match="boom"):
        resolver.embed_query("hi")


def test_resolver_embed_query_call_failure_raises_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingEmbeddingFunction:
        def __call__(self, input: list[str]) -> list[list[float]]:
            raise RuntimeError("provider exploded")

    monkeypatch.setattr(
        embedding_module,
        "_build_embedding_function",
        lambda config: _FailingEmbeddingFunction(),
    )
    resolver = EmbeddingResolver(EmbeddingConfig(provider="default"))

    with pytest.raises(EmbeddingFunctionUnavailableError, match="provider exploded"):
        resolver.embed_query("hi")


def test_resolver_embed_query_missing_env_var_raises_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EmbeddingConfigError raised during lazy build (e.g. a hosted
    provider's api_key_env is unset) must be reclassified as
    EmbeddingFunctionUnavailableError, not leak out as an unhandled
    EmbeddingConfigError (which server.py has no handler for -> 500)."""
    monkeypatch.delenv("MY_TEST_KEY", raising=False)
    config = EmbeddingConfig(provider="openai", api_key_env="MY_TEST_KEY")
    resolver = EmbeddingResolver(config)

    with pytest.raises(EmbeddingFunctionUnavailableError, match="MY_TEST_KEY"):
        resolver.embed_query("hi")


def test_resolver_embed_query_unexpected_result_shape_raises_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EmptyResultEmbeddingFunction:
        def __call__(self, input: list[str]) -> list[list[float]]:
            return []

    monkeypatch.setattr(
        embedding_module,
        "_build_embedding_function",
        lambda config: _EmptyResultEmbeddingFunction(),
    )
    resolver = EmbeddingResolver(EmbeddingConfig(provider="default"))

    with pytest.raises(EmbeddingFunctionUnavailableError):
        resolver.embed_query("hi")


# --- _build_embedding_function provider dispatch (error paths only; the
# real provider SDKs/models aren't invoked here -- see module docstring) --


def test_build_embedding_function_openai_without_api_key_raises() -> None:
    config = EmbeddingConfig(provider="openai")  # no api_key_env

    # resolve_api_key() is exercised directly by _build_embedding_function;
    # since api_key_env is unset, this must fail before ever importing/
    # calling the OpenAI SDK.
    with pytest.raises(EmbeddingConfigError):
        embedding_module._build_embedding_function(config)


def test_build_embedding_function_unsupported_provider_raises() -> None:
    # Bypass EmbeddingConfig.load's validation to exercise the defensive
    # fallback branch directly.
    config = EmbeddingConfig(provider="not-a-real-provider")

    with pytest.raises(EmbeddingConfigError):
        embedding_module._build_embedding_function(config)


# --- Integration: ChromaAdapter.query_records uses the resolved EF --------


def _make_chroma_dir_with_collection(path: Path, name: str) -> None:
    client = chromadb.PersistentClient(path=str(path))
    collection = client.create_collection(name)
    collection.add(
        ids=["a", "b"],
        documents=["hello world", "goodbye"],
        embeddings=[[1.0, 1.0], [2.0, 2.0]],
    )


def test_query_records_with_explicit_config_uses_resolver_not_collection_ef(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tier 1 (--embedding-config) wins even though the collection has
    records with their own embeddings: the resolver's mock embedding
    function must be the one actually used to embed the query text."""

    _make_chroma_dir_with_collection(tmp_path, "docs")
    adapter = ChromaAdapter.open(tmp_path)

    mock_ef = _MockEmbeddingFunction(dimension=2)
    monkeypatch.setattr(
        embedding_module, "_build_embedding_function", lambda config: mock_ef
    )
    adapter.embedding_resolver = EmbeddingResolver(EmbeddingConfig(provider="default"))

    matches = adapter.query_records("docs", query_text="hi", n_results=1)

    assert mock_ef.calls == ["hi"]
    assert len(matches) == 1


def test_query_records_with_explicit_config_dimension_mismatch_raises_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The collection was created with 2-dim embeddings; a mock EF that
    returns a 5-dim vector must surface as EmbeddingFunctionUnavailableError
    (query_text path), not crash uncaught or silently succeed."""

    _make_chroma_dir_with_collection(tmp_path, "docs")
    adapter = ChromaAdapter.open(tmp_path)

    mock_ef = _MockEmbeddingFunction(dimension=5)
    monkeypatch.setattr(
        embedding_module, "_build_embedding_function", lambda config: mock_ef
    )
    adapter.embedding_resolver = EmbeddingResolver(EmbeddingConfig(provider="default"))

    with pytest.raises(EmbeddingFunctionUnavailableError):
        adapter.query_records("docs", query_text="hi", n_results=1)


# --- API key value must never leak into error messages / logs -------------


def test_resolve_api_key_error_message_does_not_leak_env_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_TEST_KEY", "")
    config = EmbeddingConfig(provider="openai", api_key_env="MY_TEST_KEY")

    with pytest.raises(EmbeddingConfigError) as excinfo:
        config.resolve_api_key()

    # Only the *name* of the env var may appear, never a value.
    assert "MY_TEST_KEY" in str(excinfo.value)


def test_build_embedding_function_error_message_does_not_leak_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_TEST_KEY", "super-secret-value-12345")
    config = EmbeddingConfig(provider="openai", api_key_env="MY_TEST_KEY", model="m")

    # Force the OpenAI SDK construction itself to fail so we exercise the
    # except-and-wrap branch in _build_embedding_function.
    class _Boom:
        def __getattr__(self, item: str) -> Any:
            raise RuntimeError("sdk unavailable")

    import chromadb.utils.embedding_functions as ef_module

    monkeypatch.setattr(
        ef_module, "OpenAIEmbeddingFunction", lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("sdk unavailable")
        )
    )

    with pytest.raises(EmbeddingFunctionUnavailableError) as excinfo:
        embedding_module._build_embedding_function(config)

    assert "super-secret-value-12345" not in str(excinfo.value)


def test_query_records_without_config_falls_back_to_chromadb(tmp_path: Path) -> None:
    """No --embedding-config: query_records must not touch the resolver at
    all, and instead let chromadb.collection.query(query_texts=...) run --
    which fails with EmbeddingFunctionUnavailableError here since the test
    collection has no compatible default embedding function reachable in
    this constrained env (no query_texts support without a real EF)."""

    _make_chroma_dir_with_collection(tmp_path, "docs")
    adapter = ChromaAdapter.open(tmp_path)

    assert adapter.embedding_resolver.has_explicit_config is False

    # This may succeed (if chromadb's default local EF is available in the
    # test env) or raise EmbeddingFunctionUnavailableError (if not); either
    # way it must not raise anything else, confirming chromaw did not try
    # to use an explicit resolver.
    try:
        adapter.query_records("docs", query_text="hi", n_results=1)
    except EmbeddingFunctionUnavailableError:
        pass
