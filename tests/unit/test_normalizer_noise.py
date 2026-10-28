from quantguard.results.normalizer import normalize_lm_eval


def _normalize(raw):
    return normalize_lm_eval(raw, model_id="m", model_version="1", evaluation_config_hash="h", backend_version="x")


def test_stderr_and_sample_count_are_captured() -> None:
    result = _normalize(
        {
            "results": {"arc_easy": {"acc,none": 0.5, "acc_stderr,none": 0.012}},
            "higher_is_better": {"arc_easy": {"acc": True}},
            "n-samples": {"arc_easy": {"original": 2376, "effective": 100}},
        }
    )
    item = result.metrics[0]
    assert item.stderr == 0.012
    assert item.n_samples == 100


def test_na_stderr_is_none_for_perplexity() -> None:
    result = _normalize(
        {
            "results": {"wikitext": {"word_perplexity,none": 12.3, "word_perplexity_stderr,none": "N/A"}},
            "higher_is_better": {"wikitext": {"word_perplexity": False}},
        }
    )
    assert result.metrics[0].stderr is None
    assert result.metrics[0].higher_is_better is False


def test_per_sample_values_attach_to_matching_metric() -> None:
    result = _normalize(
        {
            "results": {"arc_easy": {"acc,none": 0.5}},
            "higher_is_better": {"arc_easy": {"acc": True}},
            "per_sample": {"arc_easy": {"acc": {"0": 1.0, "1": 0.0}}},
        }
    )
    assert result.metrics[0].samples == {"0": 1.0, "1": 0.0}
