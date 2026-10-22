"""semantic_cache: a tiny, dependency-free semantic cache for LLM calls."""

from .cache import Embedding, SemanticCache, CacheStats

__all__ = ["Embedding", "SemanticCache", "CacheStats"]
__version__ = "0.2.0"
