"""Embedding function resolution for ``query_text`` searches
(technical-spec §5.6 4, M3-2 / GitHub issue #24).

Resolution priority, used by ``ChromaAdapter.query_records``:

1. ``--embedding-config`` (an explicit :class:`EmbeddingConfig`), if given --
   always wins, even when the collection has its own embedding function.
2. The collection's own embedding function, as configured by chromadb when
   the collection was created. This is exactly what
   ``collection.query(query_texts=...)`` already uses internally, so tier 2
   requires no code here: ``ChromaAdapter.query_records`` just passes
   ``query_texts`` straight through to chromadb when there is no explicit
   config.
3. chromadb's default embedding function, likewise handled by chromadb
   itself inside ``collection.query()`` when a collection has none of its
   own configured.
4. If none of the above works, ``collection.query()`` raises and
   ``ChromaAdapter.query_records`` reclassifies it as
   ``EmbeddingFunctionUnavailableError`` (see chroma_adapter.py), whose
   message points the caller at ``--embedding-config``.

Only tier 1 is implemented by this module (:class:`EmbeddingResolver`):
tiers 2-4 are chromadb's own responsibility.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chromaw.errors import EmbeddingConfigError, EmbeddingFunctionUnavailableError

# Providers chromaw knows how to build via chromadb.utils.embedding_functions.
# "default" and "sentence-transformer" run locally and need no API key;
# "openai" and "cohere" are hosted APIs and require api_key_env.
_SUPPORTED_PROVIDERS = {"default", "sentence-transformer", "openai", "cohere"}
_LOCAL_PROVIDERS = {"default", "sentence-transformer"}


@dataclass(frozen=True)
class EmbeddingConfig:
    """Parsed ``--embedding-config`` JSON file.

    Minimal schema chosen for chromaw (technical-spec §5.6 4 mandates the
    *priority order* but leaves the file format to us)::

        {
          "provider": "openai" | "sentence-transformer" | "cohere" | "default",
          "model": "text-embedding-3-small",   // optional, provider-dependent
          "api_key_env": "OPENAI_API_KEY"      // name of an env var holding
                                                // the key; the key itself is
                                                // never written to this file
        }

    ``api_key_env`` is a variable *name*, not a secret -- chromaw refuses to
    read API keys directly out of the config file so they don't end up
    committed alongside a chromaw config.
    """

    provider: str
    model: str | None = None
    api_key_env: str | None = None

    @classmethod
    def load(cls, path: Path) -> "EmbeddingConfig":
        """Load and validate an ``--embedding-config`` file.

        Raises:
            EmbeddingConfigError: the file is unreadable, not valid JSON,
                not a JSON object, missing/has a non-string ``provider``,
                names an unsupported ``provider``, or has a non-string
                ``model``/``api_key_env``.
        """
        path = Path(path)
        try:
            raw_text = path.read_text()
        except OSError as exc:
            raise EmbeddingConfigError(
                f"could not read --embedding-config file: {path} ({exc})"
            ) from exc

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise EmbeddingConfigError(
                f"--embedding-config is not valid JSON: {path} ({exc})"
            ) from exc

        if not isinstance(data, dict):
            raise EmbeddingConfigError(
                f"--embedding-config must contain a JSON object with a "
                f"'provider' field, got {type(data).__name__}: {path}"
            )

        provider = data.get("provider")
        if not isinstance(provider, str) or not provider:
            raise EmbeddingConfigError(
                f"--embedding-config is missing a string 'provider' field: {path}"
            )
        if provider not in _SUPPORTED_PROVIDERS:
            raise EmbeddingConfigError(
                f"--embedding-config has unsupported provider {provider!r} "
                f"(supported: {', '.join(sorted(_SUPPORTED_PROVIDERS))}): {path}"
            )

        model = data.get("model")
        if model is not None and not isinstance(model, str):
            raise EmbeddingConfigError(
                f"--embedding-config 'model' must be a string: {path}"
            )

        api_key_env = data.get("api_key_env")
        if api_key_env is not None and not isinstance(api_key_env, str):
            raise EmbeddingConfigError(
                f"--embedding-config 'api_key_env' must be a string: {path}"
            )
        if provider not in _LOCAL_PROVIDERS and not api_key_env:
            raise EmbeddingConfigError(
                f"--embedding-config provider {provider!r} requires an API "
                f"key: add an 'api_key_env' field naming the environment "
                f"variable that holds it (chromaw never reads keys "
                f"directly from the config file): {path}"
            )

        return cls(provider=provider, model=model, api_key_env=api_key_env)

    def resolve_api_key(self) -> str | None:
        """Return the API key for hosted providers, or None for local ones.

        Raises:
            EmbeddingConfigError: the provider needs a key but
                ``api_key_env`` names an environment variable that is unset
                or empty. Callers should surface this eagerly (at startup,
                or on first use) rather than let chromadb fail obscurely.
        """
        if self.provider in _LOCAL_PROVIDERS:
            return None
        # api_key_env is guaranteed set by EmbeddingConfig.load's validation
        # for non-local providers.
        api_key = os.environ.get(self.api_key_env or "")
        if not api_key:
            raise EmbeddingConfigError(
                f"environment variable {self.api_key_env!r} (named by "
                f"--embedding-config's 'api_key_env' for provider "
                f"{self.provider!r}) is not set or empty"
            )
        return api_key


def _build_embedding_function(config: EmbeddingConfig) -> Any:
    """Instantiate the chromadb embedding function described by ``config``.

    Provider SDK imports are deferred to this function (via
    ``chromadb.utils.embedding_functions``) so that chromaw starts up, and
    the read-only viewer works, without pulling in optional embedding
    provider dependencies when no ``--embedding-config`` is given.
    """
    from chromadb.utils import embedding_functions

    if config.provider == "default":
        return embedding_functions.DefaultEmbeddingFunction()

    if config.provider == "sentence-transformer":
        kwargs: dict[str, Any] = {}
        if config.model:
            kwargs["model_name"] = config.model
        try:
            return embedding_functions.SentenceTransformerEmbeddingFunction(**kwargs)
        except Exception as exc:
            raise EmbeddingFunctionUnavailableError(
                f"failed to load sentence-transformer embedding function "
                f"(model={config.model!r}): {exc}"
            ) from exc

    if config.provider == "openai":
        kwargs = {"api_key": config.resolve_api_key()}
        if config.model:
            kwargs["model_name"] = config.model
        try:
            return embedding_functions.OpenAIEmbeddingFunction(**kwargs)
        except Exception as exc:
            raise EmbeddingFunctionUnavailableError(
                f"failed to create OpenAI embedding function: {exc}"
            ) from exc

    if config.provider == "cohere":
        kwargs = {"api_key": config.resolve_api_key()}
        if config.model:
            kwargs["model_name"] = config.model
        try:
            return embedding_functions.CohereEmbeddingFunction(**kwargs)
        except Exception as exc:
            raise EmbeddingFunctionUnavailableError(
                f"failed to create Cohere embedding function: {exc}"
            ) from exc

    # Unreachable: EmbeddingConfig.load already rejects unknown providers.
    raise EmbeddingConfigError(f"unsupported provider: {config.provider}")


class EmbeddingResolver:
    """Resolves the embedding function used to embed ``query_text`` for
    ``ChromaAdapter.query_records`` (technical-spec §5.6 4).

    Holds at most one explicit :class:`EmbeddingConfig` (tier 1 of the
    priority order); tiers 2-4 are left to chromadb itself, see the module
    docstring.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self._config = config
        self._embedding_function: Any | None = None

    @property
    def has_explicit_config(self) -> bool:
        """True if an ``--embedding-config`` was given (tier 1 applies)."""
        return self._config is not None

    def embed_query(self, text: str) -> list[float]:
        """Embed ``text`` using the explicit ``--embedding-config``.

        Only meaningful (and should only be called) when
        ``has_explicit_config`` is True; ``ChromaAdapter.query_records``
        otherwise leaves embedding to chromadb's own tiers 2/3.

        The embedding function is built lazily on first use (not at CLI
        startup) so that e.g. a sentence-transformer model download only
        happens once a text query is actually run, and is cached afterwards.

        Raises:
            EmbeddingFunctionUnavailableError: no explicit config is set, or
                the configured embedding function fails to build or to
                embed ``text``.
        """
        if self._config is None:
            raise EmbeddingFunctionUnavailableError(
                "no embedding function available: no --embedding-config was "
                "given, no embedding function is configured on the "
                "collection, and chromadb's default embedding function is "
                "unavailable. Pass --embedding-config pointing at a JSON "
                "file such as "
                '{"provider": "openai", "model": "text-embedding-3-small", '
                '"api_key_env": "OPENAI_API_KEY"} to enable text search.'
            )

        if self._embedding_function is None:
            try:
                self._embedding_function = _build_embedding_function(self._config)
            except EmbeddingConfigError as exc:
                raise EmbeddingFunctionUnavailableError(str(exc)) from exc

        try:
            result = self._embedding_function([text])
        except Exception as exc:
            raise EmbeddingFunctionUnavailableError(
                f"embedding function (provider={self._config.provider!r}) "
                f"failed to embed the query text: {exc}"
            ) from exc

        try:
            vector = list(result[0])
        except (IndexError, TypeError, KeyError) as exc:
            raise EmbeddingFunctionUnavailableError(
                f"embedding function (provider={self._config.provider!r}) "
                f"returned an unexpected result shape: {result!r}"
            ) from exc

        return [float(v) for v in vector]
