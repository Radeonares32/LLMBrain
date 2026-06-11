from llmbrain.services.token_budget import compare_context_size, estimate_tokens


def test_compare_context_size_reports_savings():
    result = compare_context_size('{"a": 1, "b": 2, "c": 3}', "a | b | c\n1 | 2 | 3\n")

    assert estimate_tokens("abcd") == 1
    assert result["json_chars"] > result["brainframe_chars"]
    assert result["saved_tokens"] >= 0
    assert result["saved_percent"] >= 0
