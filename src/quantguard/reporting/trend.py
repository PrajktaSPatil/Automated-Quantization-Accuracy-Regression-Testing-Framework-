"""Trend chart over stored releases. Standalone HTML with inline SVG; no dependencies."""

from __future__ import annotations

from html import escape

from quantguard.baselines.store import BaselineRecord

_W, _H, _PAD_L, _PAD_R, _PAD_T, _PAD_B = 560, 220, 60, 20, 24, 54


def _series(releases: list[BaselineRecord]) -> dict[tuple[str, str], list[tuple[str, float, float | None]]]:
    series: dict[tuple[str, str], list[tuple[str, float, float | None]]] = {}
    for record in releases:
        label = str(record.model.get("version"))
        for metric in record.evaluation.metrics:
            series.setdefault((metric.task, metric.name), []).append((label, metric.value, metric.stderr))
    return series


def _chart(title: str, points: list[tuple[str, float, float | None]], baseline: float | None) -> str:
    values = [v for _, v, _ in points] + ([baseline] if baseline is not None else [])
    errs = [e or 0.0 for _, _, e in points]
    low = min(v - e for (_, v, _), e in zip(points, errs)) if points else 0.0
    high = max(v + e for (_, v, _), e in zip(points, errs)) if points else 1.0
    if baseline is not None:
        low, high = min(low, baseline), max(high, baseline)
    if high == low:
        high, low = high + 1e-9, low - 1e-9
    margin = (high - low) * 0.1
    low, high = low - margin, high + margin
    plot_w, plot_h = _W - _PAD_L - _PAD_R, _H - _PAD_T - _PAD_B

    def x(i: int) -> float:
        return _PAD_L + (plot_w / 2 if len(points) == 1 else plot_w * i / (len(points) - 1))

    def y(v: float) -> float:
        return _PAD_T + plot_h * (1 - (v - low) / (high - low))

    parts = [f'<svg viewBox="0 0 {_W} {_H}" role="img" aria-label="{escape(title)}">']
    parts.append(f'<text x="{_PAD_L}" y="14" class="t">{escape(title)}</text>')
    for tick in range(5):
        v = low + (high - low) * tick / 4
        parts.append(f'<line x1="{_PAD_L}" x2="{_W - _PAD_R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="g"/>')
        parts.append(f'<text x="{_PAD_L - 6}" y="{y(v) + 4:.1f}" text-anchor="end" class="a">{v:.4g}</text>')
    if baseline is not None:
        parts.append(
            f'<line x1="{_PAD_L}" x2="{_W - _PAD_R}" y1="{y(baseline):.1f}" y2="{y(baseline):.1f}" class="b"/>'
            f'<text x="{_W - _PAD_R}" y="{y(baseline) - 4:.1f}" text-anchor="end" class="a">FP16 baseline</text>'
        )
    coords = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, (_, v, _) in enumerate(points))
    parts.append(f'<polyline points="{coords}" class="l"/>')
    for i, (label, v, e) in enumerate(points):
        if e:
            parts.append(f'<line x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{y(v - e):.1f}" y2="{y(v + e):.1f}" class="e"/>')
        parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3.5" class="p"/>')
        parts.append(
            f'<text x="{x(i):.1f}" y="{_H - 30}" text-anchor="end" class="a" '
            f'transform="rotate(-30 {x(i):.1f} {_H - 30})">{escape(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_trend_html(baseline: BaselineRecord | None, releases: list[BaselineRecord]) -> str:
    base_values = {(m.task, m.name): m.value for m in baseline.evaluation.metrics} if baseline else {}
    charts = [
        _chart(f"{task} / {name}", points, base_values.get((task, name)))
        for (task, name), points in sorted(_series(releases).items())
    ]
    body = "\n".join(f"<section>{chart}</section>" for chart in charts) or "<p>No stored releases yet.</p>"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>QuantGuard trend</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 64rem; color: #1b1b1b; }}
section {{ margin-bottom: 1.5rem; }} svg {{ width: 100%; max-width: 560px; }}
.t {{ font-size: 13px; font-weight: 600; fill: #1b1b1b; }} .a {{ font-size: 10px; fill: #555; }}
.g {{ stroke: #e3e3e3; }} .b {{ stroke: #888; stroke-dasharray: 5 4; }}
.l {{ fill: none; stroke: #1f6feb; stroke-width: 2; }} .e {{ stroke: #1f6feb; stroke-opacity: .45; stroke-width: 2; }}
.p {{ fill: #1f6feb; }}
</style></head><body>
<h1>QuantGuard release trend</h1>
<p>Points are stored release scores; vertical bars are one standard error; the dashed line is the FP16 baseline.</p>
{body}
</body></html>
"""


from dataclasses import dataclass

@dataclass(frozen=True)
class TrendPoint:
    label: str
    value: float
    stderr: float | None = None

def collect_points(records, task: str, metric: str):
    points = []
    for record in records:
        for item in getattr(getattr(record, "evaluation", None), "metrics", []):
            if item.task == task and item.name == metric:
                points.append(TrendPoint(str(getattr(record, "model", {}).get("version", record.created_at)), item.value, item.stderr))
    return points

def render_trend_svg(points, *, task: str, metric: str, z: float = 1.96) -> str:
    if not points:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360"><text>No stored releases contain {task}.{metric}</text></svg>'
    values = [p.value for p in points]
    low, high = min(values), max(values)
    if low == high:
        low, high = low - .01, high + .01
    w, h, pad = 720, 360, 50
    x = lambda i: pad + (w - 2 * pad) * (i / max(1, len(points) - 1))
    y = lambda value: h - pad - (h - 2 * pad) * (value - low) / (high - low)
    coords = " ".join(f"{x(i):.1f},{y(p.value):.1f}" for i, p in enumerate(points))
    circles = "".join(f'<circle cx="{x(i):.1f}" cy="{y(p.value):.1f}" r="4" />' for i, p in enumerate(points))
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><polyline points="{coords}" />{circles}</svg>'
