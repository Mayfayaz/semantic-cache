"""A small, dependency-free semantic cache for LLM calls.

The idea: LLM calls are slow and expensive, and users often ask the *same
question in slightly different words*. A semantic cache stores past
(prompt, response) pairs and, on a cache miss, compares the new prompt to
everything stored using cosine similarity over a simple character-trigram
embedding. If a stored prompt is similar enough (above a threshold), its
response is returned without calling the model at all.

Everything here is stdlib-only: the "embedding" is a hashed character
trigram vector, which needs no model downloads or API keys. Swap in a real
embedding model by passing any ``embed(text) -> list[float]`` callable.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def trigram_embed(text: str, dim: int = 512) -> List[float]:
    """Deterministic character-trigram embedding, L2-normalized.

    Lowercases the text, strips punctuation, then hashes every overlapping
    3-character window into one of ``dim`` buckets. Cheap, stateless, and
    good at near-duplicate detection (same question with small wording
    changes, typos, or extra filler words).

    Limitation: this is not a neural embedding. It measures
    surface similarity, so heavily reworded paraphrases may score low.
    Real-world deployments swap in a model embedding via the ``embed``
    parameter; the cache works with any ``embed(text) -> list[float]``.
    """
    vec = [0.0] * dim
    cleaned = re.sub(r"[^a-z0-9 ]", "", text.lower())
    padded = " " + cleaned + " "
    for i in range(len(padded) - 2):
        trigram = padded[i : i + 3]
        idx = int(hashlib.md5(trigram.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two (assumed normalized) vectors."""
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

# Public type for embedding vectors.
Embedding = List[float]


@dataclass
class CacheEntry:
    prompt: str
    response: str
    embedding: List[float]
    created_at: float = field(default_factory=time.time)


@dataclass
class CacheStats:
    calls: int = 0          # total lookups
    hits: int = 0           # answered from cache (exact or semantic)
    semantic_hits: int = 0  # subset of hits that were paraphrase matches
    misses: int = 0         # forwarded to the model

    @property
    def hit_rate(self) -> float:
        return self.hits / self.calls if self.calls else 0.0

    def __str__(self) -> str:
        return (
            f"calls={self.calls} hits={self.hits} "
            f"(semantic={self.semantic_hits}) misses={self.misses} "
            f"hit_rate={self.hit_rate:.1%}"
        )


class SemanticCache:
    """In-memory semantic cache for LLM completions.

    Args:
        similarity_threshold: cosine similarity (0-1) above which a stored
            prompt counts as a semantic hit for a new prompt. Higher =
            stricter (fewer false positives, more model calls). The 0.6
            default is calibrated for the built-in trigram embedding:
            mild paraphrases score ~0.65-0.85, unrelated prompts ~0-0.15.
            With a real embedding model, 0.85-0.95 is more typical.
        max_size: maximum entries; evicts least-recently-used on overflow.
        ttl_seconds: entries older than this are treated as expired.
        embed: ``embed(text) -> list[float]``; defaults to :func:`trigram_embed`.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.60,
        max_size: int = 1000,
        ttl_seconds: Optional[float] = None,
        embed: Callable[[str], List[float]] = trigram_embed,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.embed = embed
        # OrderedDict gives us free LRU: most-recently-used moves to the end.
        self._entries: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._entries)

    # -- internals ----------------------------------------------------------

    def _expired(self, entry: CacheEntry, now: float) -> bool:
        return self.ttl_seconds is not None and (now - entry.created_at) > self.ttl_seconds

    def _find_best(self, embedding: List[float], now: float) -> Tuple[Optional[CacheEntry], float]:
        best, best_score = None, 0.0
        expired_keys = []
        for key, entry in self._entries.items():
            if self._expired(entry, now):
                expired_keys.append(key)
                continue
            score = cosine_similarity(embedding, entry.embedding)
            if score > best_score:
                best, best_score = entry, score
        for key in expired_keys:
            del self._entries[key]
        return best, best_score

    # -- public API ---------------------------------------------------------

    def get(self, prompt: str) -> Optional[Tuple[str, float]]:
        """Return ``(response, similarity)`` for a cached prompt, else None."""
        now = time.time()
        embedding = self.embed(prompt)
        entry, score = self._find_best(embedding, now)
        self.stats.calls += 1
        if entry is not None and score >= self.threshold:
            self._entries.move_to_end(entry.prompt)  # mark as recently used
            self.stats.hits += 1
            if score < 1.0:
                self.stats.semantic_hits += 1
            return entry.response, score
        self.stats.misses += 1
        return None

    def put(self, prompt: str, response: str) -> None:
        """Store a (prompt, response) pair, evicting LRU entries as needed."""
        if prompt in self._entries:
            del self._entries[prompt]
        self._entries[prompt] = CacheEntry(
            prompt=prompt, response=response, embedding=self.embed(prompt)
        )
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)  # evict least-recently-used

    def call(self, prompt: str, llm_fn: Callable[[str], str]) -> str:
        """Run ``llm_fn`` through the cache: hit → return, miss → call & store."""
        cached = self.get(prompt)
        if cached is not None:
            return cached[0]
        response = llm_fn(prompt)
        self.put(prompt, response)
        return response

    def clear(self) -> None:
        self._entries.clear()
