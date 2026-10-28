"""QuantGuard command line."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from quantguard import __version__
from quantguard.config.loader import config_hash, load_config
from quantguard.exceptions import ConfigurationError, QuantGuardError
from quantguard.gate.release import decide
from quantguard.logging import configure_logging, get_logger
from quantguard.metadata.environment import collect_environment
from quantguard.policy.engine import evaluate_quality_policy
from quantguard.regression.engine import compare_evaluations
from quantguard.reporting.engine import write_reports
from quantguard.results.normalizer import normalize_lm_eval
from quantguard.results.schema import NormalizedEvaluation
from quantguard.storage.sqlite import ExperimentStore

app = typer.Typer(no_args_is_help=True, help="Quantization quality regression and release gating.")
logger = get_logger("cli")


def _repo_root() -> Path:
    return Path.cwd()


def _fail_config(exc: ConfigurationError) -> None:
    typer.echo(f"ERROR {exc}", err=True)
    raise typer.Exit(code=2)


@app.callback()
def main(
    log_level: str = typer.Option("INFO", help="Log level."),
    structured_logs: bool = typer.Option(False, help="Emit JSON log lines."),
) -> None:
    configure_logging(log_level, structured=structured_logs)


@app.command()
def version() -> None:
    """Print the QuantGuard version."""
    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """Print the Python, package, and GPU environment."""
    info = collect_environment(_repo_root())
    typer.echo(info.model_dump_json(indent=2))
    if info.torch_version and "+cpu" in info.torch_version:
        raise typer.Exit(code=2)
    if info.gpu_name is None:
        typer.echo("ERROR CUDA is not available to this interpreter.", err=True)
        raise typer.Exit(code=2)


@app.command()
def dashboard(
    port: int = typer.Option(8765, min=1, max=65535, help="Local HTTP port."),
    database: Path = typer.Option(Path("quantguard.sqlite"), help="SQLite run database, relative to the project root."),
) -> None:
    """Serve the local dashboard for stored evaluation results."""
    from quantguard.dashboard import serve_dashboard

    try:
        serve_dashboard(_repo_root(), port=port, database=database)
    except ValueError as exc:
        typer.echo(f"ERROR {exc}", err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def validate(config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config.")) -> None:
    """Validate a configuration file and print its hash."""
    try:
        loaded = load_config(config)
    except ConfigurationError as exc:
        _fail_config(exc)
    typer.echo(json.dumps({"name": loaded.name, "config_hash": config_hash(loaded)}, indent=2))


@app.command()
def evaluate(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    role: str = typer.Option("baseline", help="baseline or candidate"),
) -> None:
    """Evaluate one model and write normalized JSON beside the config."""
    try:
        loaded = load_config(config)
    except ConfigurationError as exc:
        _fail_config(exc)
    if role not in {"baseline", "candidate"}:
        typer.echo("ERROR role must be baseline or candidate", err=True)
        raise typer.Exit(code=2)
    from quantguard.evaluators.lm_eval import LMEvalEvaluator

    spec = loaded.baseline if role == "baseline" else loaded.candidate
    output = _repo_root() / loaded.report_dir / f"{role}_raw.json"
    try:
        raw = LMEvalEvaluator().evaluate(spec, loaded.evaluation, collect_environment(_repo_root()), output)
        normalized = normalize_lm_eval(
            raw,
            model_id=spec.model_id,
            model_version=spec.version,
            evaluation_config_hash=config_hash(loaded),
            backend_version=str((raw.get("quantguard") or {}).get("backend_version")),
        )
    except QuantGuardError as exc:
        typer.echo(f"ERROR {exc}", err=True)
        raise typer.Exit(code=2) from exc
    normalized_path = output.with_name(f"{role}_normalized.json")
    normalized_path.write_text(normalized.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(str(normalized_path))


@app.command()
def compare(
    baseline: Path = typer.Option(..., exists=True, dir_okay=False),
    candidate: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Compare two normalized evaluation files and print regression JSON."""
    try:
        loaded = load_config(config)
        left = NormalizedEvaluation.model_validate_json(baseline.read_text(encoding="utf-8"))
        right = NormalizedEvaluation.model_validate_json(candidate.read_text(encoding="utf-8"))
    except ConfigurationError as exc:
        _fail_config(exc)
    except Exception as exc:
        typer.echo(f"ERROR {exc}", err=True)
        raise typer.Exit(code=2) from exc
    proportion = {
        (metric.task, metric.name)
        for metric in left.metrics + right.metrics
        if _policy_scale(loaded, metric.task, metric.name) == "proportion"
    }
    comparisons = compare_evaluations(left, right, proportion_metrics=proportion)
    typer.echo(json.dumps([item.model_dump() for item in comparisons], indent=2))


@app.command()
def gate(
    baseline: Path = typer.Option(..., exists=True, dir_okay=False),
    candidate: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    """Apply the quality policy to two normalized evaluations."""
    try:
        loaded = load_config(config)
        left = NormalizedEvaluation.model_validate_json(baseline.read_text(encoding="utf-8"))
        right = NormalizedEvaluation.model_validate_json(candidate.read_text(encoding="utf-8"))
    except ConfigurationError as exc:
        _fail_config(exc)
    except Exception as exc:
        typer.echo(f"ERROR {exc}", err=True)
        raise typer.Exit(code=2) from exc
    from quantguard.policy.engine import resolve_metric_policy

    proportion = {
        (metric.task, metric.name)
        for metric in left.metrics + right.metrics
        if resolve_metric_policy(loaded.quality_policy, metric.task, metric.name).scale == "proportion"
    }
    comparisons = compare_evaluations(left, right, proportion_metrics=proportion)
    try:
        checks = evaluate_quality_policy(comparisons, loaded.quality_policy)
        decision = decide(checks)
    except QuantGuardError as exc:
        decision = decide([], infrastructure_error=str(exc))
    typer.echo(decision.model_dump_json(indent=2))
    raise typer.Exit(code=decision.exit_code)


@app.command()
def report(
    experiment: Path = typer.Option(..., exists=True, dir_okay=False, help="Experiment JSON payload."),
    output_dir: Path = typer.Option(Path("reports/manual"), help="Directory for generated reports."),
) -> None:
    """Render JSON, Markdown, and HTML from an experiment payload."""
    payload = json.loads(experiment.read_text(encoding="utf-8"))
    paths = write_reports(payload, output_dir)
    typer.echo(json.dumps(paths, indent=2))


@app.command()
def history(
    database: Path = typer.Option(Path("quantguard.sqlite"), help="SQLite database."),
    limit: int = typer.Option(20, min=1),
) -> None:
    """List persisted experiments."""
    if not database.exists():
        typer.echo("[]")
        return
    typer.echo(json.dumps(ExperimentStore(database).list_recent(limit), indent=2))


baseline_app = typer.Typer(help="Manage stored baseline and release scores.")
app.add_typer(baseline_app, name="baseline")


@baseline_app.command("create")
def baseline_create(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    with_performance: bool = typer.Option(False, help="Also store baseline latency and memory."),
    limit: int = typer.Option(0, help="Override evaluation.limit (0 keeps the config value)."),
) -> None:
    """Evaluate the approved baseline once and write baselines/<name>/baseline.json."""
    try:
        loaded = load_config(config)
    except ConfigurationError as exc:
        _fail_config(exc)
    _apply_limit(loaded, limit)
    from quantguard.evaluators.lm_eval import LMEvalEvaluator
    from quantguard.pipeline import create_baseline

    try:
        path = create_baseline(loaded, LMEvalEvaluator(), _repo_root(), profile=with_performance)
    except QuantGuardError as exc:
        typer.echo(f"ERROR {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(str(path))


@app.command("pr-comment")
def pr_comment(
    report: Path = typer.Option(..., exists=True, dir_okay=False, help="report.json from a run."),
    output: Path = typer.Option(Path("pr_comment.md")),
    run_url: str = typer.Option("", help="Link to the CI run or artifacts."),
) -> None:
    """Write the compact Markdown used for the pull-request comment."""
    from quantguard.reporting.engine import render_pr_comment

    payload = json.loads(report.read_text(encoding="utf-8"))
    output.write_text(render_pr_comment(payload, run_url=run_url or None), encoding="utf-8")
    typer.echo(str(output))


@app.command()
def trend(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(Path("reports/trend.html")),
) -> None:
    """Render a score-over-releases chart from stored baselines and releases."""
    try:
        loaded = load_config(config)
    except ConfigurationError as exc:
        _fail_config(exc)
    from quantguard.baselines.store import BaselineStore
    from quantguard.reporting.trend import render_trend_html

    if loaded.baseline_store is None:
        typer.echo("ERROR config has no baseline_store section", err=True)
        raise typer.Exit(code=2)
    store = BaselineStore(_repo_root() / loaded.baseline_store.directory, loaded.baseline_store.name or loaded.name)
    base = store.load_baseline() if store.has_baseline() else None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_trend_html(base, store.all_releases()), encoding="utf-8")
    typer.echo(str(output))


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    skip_performance: bool = typer.Option(False, help="Skip latency and memory measurement."),
    record_release: bool = typer.Option(False, help="Store candidate scores as a release when the gate passes."),
    limit: int = typer.Option(0, help="Override evaluation.limit (0 keeps the config value)."),
) -> None:
    """Evaluate baseline and candidate, then apply the release gate."""
    try:
        loaded = load_config(config)
    except ConfigurationError as exc:
        _fail_config(exc)
    _apply_limit(loaded, limit)
    from quantguard.evaluators.lm_eval import LMEvalEvaluator
    from quantguard.pipeline import ReleasePipeline

    pipeline = ReleasePipeline(LMEvalEvaluator(), _repo_root())
    try:
        payload, decision = pipeline.run(loaded, profile=not skip_performance, record_release=record_release)
    except QuantGuardError as exc:
        typer.echo(f"ERROR {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        json.dumps(
            {
                "experiment_id": payload["experiment_id"],
                "status": decision.status,
                "report_dir": str(Path(loaded.report_dir) / payload["experiment_id"]),
            },
            indent=2,
        )
    )
    raise typer.Exit(code=decision.exit_code)


def _apply_limit(loaded, limit: int) -> None:
    if limit:
        loaded.evaluation.limit = limit


@app.command("inject-suite")
def inject_suite(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    backend: str = typer.Option("lm_eval", help="lm_eval (real model) or toy (numpy stand-in, no GPU)."),
    output: Path = typer.Option(Path("reports/injection.md")),
    equivalent_seeds: int = typer.Option(5, min=1, help="How many equivalent (should-pass) candidates to score."),
    limit: int = typer.Option(0, help="Override evaluation.limit (0 keeps the config value)."),
    min_detection: float = typer.Option(1.0, min=0, max=1, help="Fail below this detection rate."),
    max_false_alarms: float = typer.Option(0.0, min=0, max=1, help="Fail above this false-alarm rate."),
) -> None:
    """Inject known-bad quantization and check the gate catches it (and spares equivalent models)."""
    try:
        loaded = load_config(config)
    except ConfigurationError as exc:
        _fail_config(exc)
    _apply_limit(loaded, limit)
    from quantguard.validation.injection import default_scenarios, run_suite

    if backend == "toy":
        from quantguard.validation.toy import ToyEvaluator

        evaluator = ToyEvaluator()
    elif backend == "lm_eval":
        from quantguard.evaluators.lm_eval import LMEvalEvaluator

        evaluator = LMEvalEvaluator()
    else:
        typer.echo("ERROR backend must be lm_eval or toy", err=True)
        raise typer.Exit(code=2)
    result = run_suite(loaded, evaluator, _repo_root(), default_scenarios(equivalent_seeds))
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown = result.to_markdown()
    output.write_text(markdown, encoding="utf-8")
    typer.echo(markdown)
    detected, bad = result.detection
    alarms, equivalent = result.false_alarms
    if result.errors:
        raise typer.Exit(code=2)
    if (bad and detected / bad < min_detection) or (equivalent and alarms / equivalent > max_false_alarms):
        raise typer.Exit(code=1)


def _policy_scale(loaded, task: str, metric: str) -> str:
    from quantguard.policy.engine import resolve_metric_policy

    return resolve_metric_policy(loaded.quality_policy, task, metric).scale
