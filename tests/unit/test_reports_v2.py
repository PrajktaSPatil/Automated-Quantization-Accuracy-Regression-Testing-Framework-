from quantguard.reporting.pr_comment import MARKER, render_pr_comment
from quantguard.reporting.trend import TrendPoint, collect_points, render_trend_svg
from tests.unit.test_reporting_and_storage import _payload


def test_pr_comment_has_marker_status_and_table() -> None:
    comment = render_pr_comment(_payload(), report_url="https://example.test/run/1")
    assert comment.startswith(MARKER)
    assert "QuantGuard: FAIL" in comment
    assert "| base | arc_easy | acc |" in comment
    assert "https://example.test/run/1" in comment


def test_pr_comment_shows_standard_errors_and_noise() -> None:
    payload = _payload()
    payload["comparisons"][0].update(baseline_stderr=0.012, candidate_stderr=0.013)
    payload["decision"]["checks"][0]["noise_display"] = "noise band 3 pp"
    assert "± 0.012" in render_pr_comment(payload)


def test_trend_svg_has_points_and_handles_empty() -> None:
    svg = render_trend_svg(
        [TrendPoint("r1", 0.70, 0.01), TrendPoint("r2", 0.69, 0.01), TrendPoint("r3", 0.71, None)],
        task="arc_easy",
        metric="acc",
    )
    assert svg.startswith("<svg") and svg.count("<circle") == 3 and "<polyline" in svg
    assert "No stored releases" in render_trend_svg([], task="arc_easy", metric="acc")
    assert collect_points([], "arc_easy", "acc") == []
