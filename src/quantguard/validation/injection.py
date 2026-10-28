"""Run a set of deliberately damaged candidates through the real release pipeline.

Each scenario overrides the candidate's `fake-rtn` settings. Scenarios labelled `bad`
must be caught (FAIL); scenarios labelled `equivalent` must not be (no FAIL). The
suite reports the two rates the project is judged on: regression detection rate and
false-alarm rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from quantguard.config.schema import QuantGuardConfig
from quantguard.evaluators.base import EvaluationBackend
from quantguard.pipeline import ReleasePipeline


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: Literal["bad", "equivalent", "borderline"]
    quantization_config: dict[str, Any]
    description: str = ""


def default_scenarios(equivalent_seeds: int = 5) -> list[Scenario]:
    scenarios = [
        Scenario("w2-g128", "bad", {"bits": 2, "group_size": 128}, "2-bit weights"),
        Scenario("w3-per-channel", "bad", {"bits": 3, "group_size": -1}, "3-bit, group size far too large"),
        Scenario(
            "w4-bad-calibration",
            "bad",
            {"bits": 4, "group_size": 128, "clip_ratio": 0.5, "scale_jitter": 0.3},
            "4-bit with under-estimated, noisy ranges",
        ),
        Scenario("w4-g128", "borderline", {"bits": 4, "group_size": 128}, "ordinary 4-bit; reported, not scored"),
    ]
    for seed in range(equivalent_seeds):
        scenarios.append(
            Scenario(
                f"w8-g128-jitter-seed{seed}",
                "equivalent",
                {"bits": 8, "group_size": 128, "scale_jitter": 0.001, "seed": seed},
                "8-bit with tiny scale noise; should be indistinguishable from baseline",
            )
        )
    return scenarios


@dataclass
class ScenarioResult:
    scenario: Scenario
    status: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class SuiteResult:
    results: list[ScenarioResult]

    def _rate(self, kind: str, statuses: set[str]) -> tuple[int, int]:
        group = [r for r in self.results if r.scenario.kind == kind]
        return sum(r.status in statuses for r in group), len(group)

    @property
    def detection(self) -> tuple[int, int]:
        return self._rate("bad", {"FAIL"})

    @property
    def flagged(self) -> tuple[int, int]:
        return self._rate("bad", {"FAIL", "WARNING"})

    @property
    def false_alarms(self) -> tuple[int, int]:
        return self._rate("equivalent", {"FAIL"})

    @property
    def errors(self) -> int:
        return sum(r.status == "ERROR" for r in self.results)

    def to_markdown(self, title: str = "Injected-regression validation") -> str:
        det, det_n = self.detection
        flg, _ = self.flagged
        fa, fa_n = self.false_alarms
        lines = [
            f"# {title}",
            "",
            f"- Regression detection rate (FAIL on bad candidates): {det}/{det_n}",
            f"- Flagged (FAIL or WARNING) on bad candidates: {flg}/{det_n}",
            f"- False-alarm rate (FAIL on equivalent candidates): {fa}/{fa_n}",
            f"- Infrastructure errors: {self.errors}",
            "",
            "| Scenario | Kind | Gate | First reason |",
            "| --- | --- | --- | --- |",
        ]
        for r in self.results:
            reason = (r.reasons[0] if r.reasons else "").replace("|", "/")
            lines.append(f"| {r.scenario.name} | {r.scenario.kind} | {r.status} | {reason} |")
        return "\n".join(lines) + "\n"


def run_suite(
    config: QuantGuardConfig,
    evaluator: EvaluationBackend,
    repo_root: Path,
    scenarios: list[Scenario] | None = None,
    *,
    profile: bool = False,
) -> SuiteResult:
    """Run each scenario as an independent release run (nothing is recorded as a release)."""

    results: list[ScenarioResult] = []
    for scenario in scenarios or default_scenarios():
        trial = config.model_copy(deep=True)
        trial.candidate.quantization_method = "fake-rtn"
        trial.candidate.quantization_config = dict(scenario.quantization_config)
        trial.candidate.version = f"{config.candidate.version}-inject-{scenario.name}"
        trial.report_dir = str(Path(config.report_dir) / "injection")
        trial.database_path = str(Path(config.report_dir) / "injection" / "injection.sqlite")
        # Injected candidates are compared with the baseline only; they must not be judged
        # against, or become, a stored release.
        if trial.baseline_store is not None:
            trial.baseline_store = trial.baseline_store.model_copy(update={"compare_previous_release": False})
        _, decision = ReleasePipeline(evaluator, repo_root).run(trial, profile=profile, record_release=False)
        results.append(ScenarioResult(scenario, decision.status, list(decision.reasons)))
    return SuiteResult(results)


# Compatibility API for earlier test and CLI clients. The suite API above is canonical.
def _scenario_compat_init(self, name, kind=None, quantization_config=None, description="", expect=None, candidate=None):
    object.__setattr__(self, "name", name)
    object.__setattr__(self, "kind", {"detect": "bad", "clean": "equivalent"}.get(expect, kind or "equivalent"))
    object.__setattr__(self, "quantization_config", dict((candidate or {}).get("quantization_config", quantization_config or {})))
    object.__setattr__(self, "description", description)

Scenario.__init__ = _scenario_compat_init
Scenario.expect = property(lambda self: {"bad": "detect", "equivalent": "clean", "borderline": "borderline"}[self.kind])
Scenario.candidate = property(lambda self: {"quantization_config": dict(self.quantization_config)})

@dataclass
class ScenarioOutcome:
    name: str
    expect: str
    status: str
    seconds: float
    failing_checks: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    experiment_id: str | None = None

@dataclass
class ValidationSummary:
    outcomes: list[ScenarioOutcome]
    detection_rate: float | None
    false_alarm_rate: float | None
    warn_on_regression_rate: float | None
    error_count: int
    mean_seconds: float
    def model_dump(self):
        return {"outcomes": [vars(x) for x in self.outcomes], "detection_rate": self.detection_rate,
                "false_alarm_rate": self.false_alarm_rate, "warn_on_regression_rate": self.warn_on_regression_rate,
                "error_count": self.error_count, "mean_seconds": self.mean_seconds}

def summarize(outcomes):
    detect = [x for x in outcomes if x.expect in ("detect", "bad") and x.status != "ERROR"]
    clean = [x for x in outcomes if x.expect in ("clean", "equivalent") and x.status != "ERROR"]
    return ValidationSummary(outcomes,
        sum(x.status == "FAIL" for x in detect) / len(detect) if detect else None,
        sum(x.status == "FAIL" for x in clean) / len(clean) if clean else None,
        sum(x.status == "WARNING" for x in detect) / len(detect) if detect else None,
        sum(x.status == "ERROR" for x in outcomes),
        sum(x.seconds for x in outcomes) / len(outcomes) if outcomes else 0.0)

def scenario_config(base, scenario, report_dir):
    config = base.model_copy(deep=True)
    config.candidate.quantization_method = "fake-rtn"
    config.candidate.quantization_config = dict(scenario.candidate.get("quantization_config", scenario.quantization_config))
    config.candidate.version = f"{base.candidate.version}-{scenario.name}"
    config.report_dir = report_dir
    return config

def load_scenarios(path):
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [Scenario(name=x["name"], expect=x.get("expect"), candidate=x.get("candidate", {}), description=x.get("description", "")) for x in raw.get("scenarios", [])]

def run_scenarios(evaluator, base, scenarios, output_dir, repo_root):
    import time
    outcomes = []
    for scenario in scenarios:
        started = time.perf_counter()
        config = scenario_config(base, scenario, str(Path(output_dir) / scenario.name))
        _, decision = ReleasePipeline(evaluator, repo_root).run(config, profile=False, record_release=False)
        outcomes.append(ScenarioOutcome(scenario.name, scenario.expect, decision.status, time.perf_counter() - started, reasons=list(decision.reasons)))
    return summarize(outcomes)

def write_summary(summary, output_dir):
    import json
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    json_path, md_path = folder / "validation_results.json", folder / "validation_results.md"
    json_path.write_text(json.dumps(summary.model_dump(), indent=2), encoding="utf-8")
    md_path.write_text(summary_markdown(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}

def summary_markdown(summary):
    converted = [ScenarioResult(Scenario(x.name, {"detect": "bad", "clean": "equivalent"}.get(x.expect, "borderline"), {}), x.status, x.reasons) for x in summary.outcomes]
    return SuiteResult(converted).to_markdown()
