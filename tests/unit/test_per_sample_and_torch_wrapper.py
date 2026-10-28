import pytest

from quantguard.evaluators.lm_eval import _per_sample_scores


def test_per_sample_scores_keep_scalar_metrics_only() -> None:
    samples = {
        "arc_easy": [
            {"doc_id": 0, "doc": {"q": "x"}, "target": "a", "resps": [[("-1.0", False)]], "acc": 1.0, "acc_norm": 0.0, "filter": "none"},
            {"doc_id": 1, "doc": {}, "target": "b", "acc": 0.0, "acc_norm": 1.0, "filter": "none"},
        ],
        "wikitext": [{"doc_id": 0, "word_perplexity": (-12.0, 30), "filter": "none"}],
        "gsm8k": [{"doc_id": 5, "exact_match": 1.0, "filter": "strict-match"}],
    }
    reduced = _per_sample_scores(samples)
    assert reduced["arc_easy"]["acc"] == {"0": 1.0, "1": 0.0}
    assert reduced["arc_easy"]["acc_norm"] == {"0": 0.0, "1": 1.0}
    assert "wikitext" not in reduced  # tuple-valued metric: falls back to harness stderr
    assert reduced["gsm8k"]["exact_match,strict-match"] == {"5": 1.0}


def test_per_sample_scores_tolerate_missing_samples() -> None:
    assert _per_sample_scores(None) == {}
    assert _per_sample_scores({"t": [{"no_doc_id": 1}]}) == {}


def test_apply_fake_rtn_quantizes_linear_layers_in_a_torch_model() -> None:
    torch = pytest.importorskip("torch")
    from quantguard.models.fakequant import apply_fake_rtn

    model = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8))
    before = model[0].weight.detach().clone()
    apply_fake_rtn(model, {"bits": 2, "group_size": 32})
    after = model[0].weight.detach()
    assert after.shape == before.shape and not torch.equal(after, before)
    assert len(torch.unique(after[0, :32])) <= 4
