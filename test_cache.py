"""Tests for semantic_cache. Stdlib only: `python test_cache.py`."""

import time

from semantic_cache import SemanticCache
from semantic_cache.cache import cosine_similarity, trigram_embed

PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def fake_llm_factory():
    calls = {"n": 0}

    def llm(prompt):
        calls["n"] += 1
        return f"answer:{prompt}"

    return llm, calls


print("embedding")
emb = trigram_embed("hello world")
check("normalized (unit length)", abs(sum(v * v for v in emb) - 1.0) < 1e-9)
check("deterministic", trigram_embed("hello world") == trigram_embed("hello world"))
check("identical texts score 1.0", cosine_similarity(emb, emb) == 1.0)
check(
    "unrelated texts score low",
    cosine_similarity(trigram_embed("the capital of france"), trigram_embed("quantum chromodynamics primer")) < 0.3,
)
check(
    "mild paraphrases score above threshold",
    cosine_similarity(
        trigram_embed("what is the capital of france"),
        trigram_embed("tell me the capital city of france"),
    ) > 0.6,
)

print("cache basics")
cache = SemanticCache()  # default threshold 0.60
llm, calls = fake_llm_factory()
check("miss returns None", cache.get("brand new prompt xyz") is None)
cache.put("the sky is blue", "yes it is")
hit = cache.get("the sky is blue")
check("exact hit returns response", hit is not None and hit[0] == "yes it is")
check("exact hit scores 1.0", hit is not None and abs(hit[1] - 1.0) < 1e-9)

print("semantic behaviour")
hit = cache.get("please tell me the sky is blue")
check("paraphrase is a semantic hit", hit is not None and hit[0] == "yes it is")
check("semantic hit scores below 1.0", hit is not None and hit[1] < 1.0)
miss = cache.get("how do I bake sourdough bread at home")
check("unrelated prompt misses", miss is None)

print("call() wrapper")
cache2 = SemanticCache()
llm2, calls2 = fake_llm_factory()
r1 = cache2.call("at what temperature does water boil", llm2)
r2 = cache2.call("at what temperature does water boil", llm2)
r3 = cache2.call("what temp does water boil at", llm2)
check("same answer for repeats", r1 == r2)
check("model called once for exact repeat", calls2["n"] == 1)
check("paraphrase served from cache", r3 == r1 and calls2["n"] == 1)
check("stats consistent", cache2.stats.hits == 2 and cache2.stats.misses == 1)

print("eviction")
small = SemanticCache(max_size=2, similarity_threshold=0.99)
for i in range(3):
    small.put(f"unique prompt number {i}", f"r{i}")
check("max size respected", len(small) == 2)
check("oldest evicted", small.get("unique prompt number 0") is None)

print("ttl")
ttl_cache = SemanticCache(ttl_seconds=0.05, similarity_threshold=0.99)
ttl_cache.put("ephemeral prompt", "gone soon")
check("present before expiry", ttl_cache.get("ephemeral prompt") is not None)
time.sleep(0.08)
check("expired after ttl", ttl_cache.get("ephemeral prompt") is None)

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
