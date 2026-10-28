import pytest

from quantguard.results.normalizer import normalize_lm_eval


def _norm(raw):
    return normalize_lm_eval(raw, model_id="m", model_version="1", evaluation_config_hash="h", backend_version="0.4")


def test_stderr_is_attached_to_its_metric() -> None:
    raw = {
        "results": {"arc_easy": {"acc,none": 0.8, "acc_stderr,none": 0.01, "acc_norm,none": 0.7, "acc_norm_stderr,none": 0.02}},
        "higher_is_better": {"arc_easy": {"acc": True, "acc_norm": True}},
    }
    by_name = {m.name: m for m in _norm(raw).metrics}
    assert by_name["acc"].stderr == pytest.approx(0.01)
    assert by_name["acc_norm"].stderr == pytest.approx(0.02)


def test_na_stderr_becomes_none() -> None:
    raw = {
        "results": {"wikitext": {"word_perplexity,none": 12.3, "word_perplexity_stderr,none": "N/A"}},
        "higher_is_better": {"wikitext": {"word_perplexity": False}},
    }
    metric = _norm(raw).metrics[0]
    assert metric.stderr is None and metric.higher_is_better is False


def test_filters_are_distinct_metrics() -> None:
    raw = {
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 0.3,
                "exact_match_stderr,strict-match": 0.01,
                "exact_match,flexible-extract": 0.4,
                "exact_match_stderr,flexible-extract": 0.02,
            }
        },
        "higher_is_better": {"gsm8k": {"exact_match": True}},
    }
    by_name = {m.name: m for m in _norm(raw).metrics}
    assert set(by_name) == {"exact_match,strict-match", "exact_match,flexible-extract"}
    assert by_name["exact_match,flexible-extract"].stderr == pytest.approx(0.02)


def test_per_sample_scores_are_kept() -> None:
    raw = {
        "results": {"t": {"acc,none": 0.5, "acc_stderr,none": 0.1}},
        "higher_is_better": {"t": {"acc": True}},
        "per_sample": {"t": {"acc": {"0": 1, "1": 0}}},
    }
    assert _norm(raw).per_sample == {"t": {"acc": {"0": 1.0, "1": 0.0}}}
