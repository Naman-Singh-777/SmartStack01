from app.cache import SemanticCache


def make_cache(**kwargs):
    cache = SemanticCache(**kwargs)
    cache.store("What is the capital of France?", {"answer": "Paris"})
    return cache


def test_same_question_with_different_style_hits():
    hit = make_cache().lookup("what is the capital of france")
    assert hit is not None
    assert hit.entry.payload["answer"] == "Paris"


def test_filler_words_and_punctuation_do_not_matter():
    assert make_cache().lookup("Whats the capital of France") is not None


def test_misspelled_key_word_is_a_miss_on_purpose():
    # The word check is strict: a wrong cached answer is worse than a cache miss.
    assert make_cache().lookup("Whats the capitol of France?") is None


def test_different_country_misses():
    # Looks almost the same letter by letter, but the answer would be wrong.
    assert make_cache().lookup("What is the capital of Spain?") is None


def test_unrelated_question_misses():
    assert make_cache().lookup("How do spot instances save money?") is None


def test_old_entries_expire():
    cache = make_cache(ttl_seconds=-1)  # everything is already too old
    assert cache.lookup("What is the capital of France?") is None


def test_storing_same_question_twice_keeps_one_entry():
    cache = make_cache()
    cache.store("what is the capital of france?", {"answer": "Paris again"})
    assert len(cache) == 1


def test_oldest_entry_is_dropped_when_full():
    cache = SemanticCache(max_entries=2)
    for text in ["alpha question one", "bravo question two", "charlie question three"]:
        cache.store(text, {})
    assert len(cache) == 2
    assert cache.lookup("alpha question one") is None
