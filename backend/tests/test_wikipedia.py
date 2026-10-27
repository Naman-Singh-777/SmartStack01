from app.wikipedia import is_usable, parse_pages, rank_pages, search_variants


def page(title, extract, votes=1, position=1):
    return {"title": title, "extract": extract, "url": f"https://en.wikipedia.org/wiki/{title}",
            "position": position, "votes": votes}


def test_search_variants_strip_question_words():
    variants = search_variants("What does CPU stand for?")
    assert variants[0] == "cpu"
    assert "What does CPU stand for" in variants


def test_parse_pages_sorts_by_wikipedia_position():
    data = {"query": {"pages": {
        "2": {"title": "B", "extract": "Second one.", "fullurl": "u2", "index": 2},
        "1": {"title": "A", "extract": "First  one.", "fullurl": "u1", "index": 1},
    }}}
    parsed = parse_pages(data)
    assert [p["title"] for p in parsed] == ["A", "B"]
    assert parsed[0]["extract"] == "First one."  # whitespace tidied


def test_parse_pages_handles_no_results():
    assert parse_pages({"batchcomplete": ""}) == []


def test_country_page_beats_capital_punishment_page():
    pages = [
        page("Capital punishment in France", "Capital punishment in France is banned by Article 66-1.", position=1),
        page("France", "France is a country in Western Europe. Its capital and largest city is Paris.", position=3),
    ]
    best = rank_pages("What is the capital of France?", pages, variant_count=1)[0]
    assert best.title == "Wikipedia: France"


def test_irrelevant_pages_are_dropped():
    pages = [page("Cloud computing", "Cloud computing is defined by the ISO as a paradigm.")]
    assert rank_pages("How do I bake sourdough bread?", pages, variant_count=1) == []


def test_disambiguation_pages_are_skipped():
    assert not is_usable(page("Capital", "Capital and its variations may refer to:"))
    assert not is_usable(page("Mercury (disambiguation)", "Mercury is a name."))


def test_plural_question_matches_singular_page():
    pages = [page("Vaccine", "A vaccine is a biological preparation that provides immunity.")]
    assert rank_pages("How do vaccines work?", pages, variant_count=1)


def candidate(title, extract, votes=2):
    return {"title": title, "extract": extract, "url": "u", "position": 1, "votes": votes}


def test_a_band_does_not_beat_the_real_topic():
    pages = [
        candidate("The Vaccines", "The Vaccines are an English indie rock band formed in London. Vaccine work shown."),
        candidate("Vaccine", "A vaccine is a biological preparation that provides immunity. Vaccines work by training the body."),
    ]
    hits = rank_pages("How do vaccines work?", pages, 2)
    assert hits[0].title == "Wikipedia: Vaccine"


def test_a_weak_runner_up_is_dropped():
    pages = [
        candidate("France", "France is a country in Europe. Its capital is Paris."),
        candidate("Capital punishment in France", "Capital punishment in France was abolished in 1981.", votes=1),
    ]
    hits = rank_pages("What is the capital of France?", pages, 2)
    assert [h.title for h in hits] == ["Wikipedia: France"]
