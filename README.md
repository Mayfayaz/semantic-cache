# semantic-cache

A dependency-free semantic cache for LLM API calls, implemented in pure Python (standard library only).

Repeated model calls increase cost and latency. Identical prompts are rare in practice; users rephrase the same question. A semantic cache stores prompt-response pairs and returns the stored response when a new prompt is sufficiently similar to a cached one, avoiding a redundant model call.

## What it does

Standard caches match on exact prompt equality. This cache additionally matches on paraphrase:

- Each prompt is embedded as a vector.
- A new prompt is compared against stored prompts using cosine similarity.
- If the highest similarity exceeds a configurable threshold, the stored response is returned without calling the model.

```
"What is the capital of France?"              → miss: model called, response stored
"What is the capital of France?"              → exact hit: no model call
"Please tell me the capital city of France."  → semantic hit: no model call
"How do I bake sourdough?"                    → miss: model called
```

## Why it matters

Production LLM applications (chatbots, RAG pipelines, agents) exhibit high prompt repetition: retries, shared templates, and common questions. Semantic caching reduces inference cost and p99 latency. Systems such as GPTCache report 30–60% hit rates on real traffic. This project implements the core mechanism from scratch in ~150 lines of readable Python.

## How it works

- **Embedding** — deterministic character-trigram hash vectors (standard library only; no downloads, no API keys). Scope: this detects near-duplicates (minor wording changes, filler words, typos), not deep paraphrase. Any `embed(text) -> list[float]` function can be substituted — e.g. a sentence-transformer model — for true semantic matching.
- **Similarity** — cosine similarity against stored prompts. Matches above the threshold (default 0.60, calibrated for the trigram embedding) return the stored response.
- **Eviction and expiry** — LRU eviction at `max_size`; optional per-entry TTL.
- **Statistics** — `CacheStats` records total calls, exact hits, semantic hits, misses, and hit rate.

## Run

Python 3.9+, no dependencies, no API keys.

```bash
python demo.py        # 5 prompts, 3 model calls: 2 served from cache
python test_cache.py  # 19 tests: embeddings, hits/misses, eviction, TTL
```

## Usage

```python
from semantic_cache import SemanticCache

cache = SemanticCache(similarity_threshold=0.60, max_size=1000)

def my_llm(prompt: str) -> str:
    ...  # expensive model call

answer = cache.call("What is the capital of France?", my_llm)
print(cache.stats)
```

## Project structure

```
semantic-cache/
├── semantic_cache/
│   ├── __init__.py      # public API: SemanticCache, CacheStats, Embedding
│   └── cache.py         # trigram embedding and cache implementation
├── demo.py              # runnable demo with a stubbed LLM (no API keys)
├── test_cache.py        # 19 standard-library-only tests
└── LICENSE              # MIT
```

## Limitations

- The built-in embedding measures surface similarity. Heavily reworded paraphrases may fall below the threshold; supply a production embedding model via `embed=` for those cases.
- In-memory only. A Redis or SQLite backend would persist entries across restarts.
- No streaming support. Cached streaming responses would require chunk replay.

## License

MIT. See LICENSE.
