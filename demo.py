"""Runnable demo: a semantic cache in front of a (fake) LLM.

No API keys needed — FakeLLM simulates an expensive model call by counting
invocations, so you can see exactly how many model calls the cache saves.
"""

from semantic_cache import SemanticCache


class FakeLLM:
    """Pretends to be an expensive LLM: counts calls, returns canned answers."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        lowered = prompt.lower()
        if "france" in lowered and "capital" in lowered:
            return "The capital of France is Paris."
        if "boil" in lowered:
            return "Water boils at 100°C (212°F) at sea level."
        if "photosynthesis" in lowered:
            return "Photosynthesis converts light, water and CO₂ into glucose and oxygen."
        return f"[model] I don't have a canned answer for: {prompt!r}"


def main() -> None:
    llm = FakeLLM()
    cache = SemanticCache()  # default threshold 0.60, calibrated for trigram embeddings

    prompts = [
        # Exact repeat → cache hit.
        "What is the capital of France?",
        "What is the capital of France?",
        # Mild paraphrase → semantic hit (no model call).
        "Please tell me the capital city of France.",
        # Different question → miss, model called.
        "At what temperature does water boil?",
        # Paraphrase of that → semantic hit.
        "What temp does water boil at?",
    ]

    for p in prompts:
        answer = cache.call(p, llm)
        print(f"Q: {p}\nA: {answer}\n")

    print("-" * 60)
    print(f"Model calls made : {llm.calls} (out of {len(prompts)} questions)")
    print(f"Cache stats      : {cache.stats}")
    print(f"Calls saved      : {len(prompts) - llm.calls}")


if __name__ == "__main__":
    main()
