from hybrid.research import _rank_tone3000_metadata


def test_tone3000_metadata_score_prioritizes_title_matches():
    query = "Marshall JCM800 high gain amp"
    title_match = {"title": "Marshall JCM800", "description": "High gain capture."}
    description_match = {"title": "British Head", "description": "Marshall JCM800 high gain capture."}

    assert _rank_tone3000_metadata(query, title_match)[0] == 75
    assert _rank_tone3000_metadata(query, description_match)[0] == 50


def test_tone3000_metadata_score_ignores_generic_query_words():
    result = {"title": "Clean Combo", "description": "A bright clean capture."}

    assert _rank_tone3000_metadata("I need a guitar amp with this tone", result)[0] == 0
