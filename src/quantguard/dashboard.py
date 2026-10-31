"""Local dashboard for stored QuantGuard evaluation records and safe smoke runs."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from quantguard.config.loader import load_config
from quantguard.exceptions import ConfigurationError
from quantguard.storage.sqlite import ExperimentStore


_PAGE = Path(__file__).with_name("dashboard_ui.html").read_text(encoding="utf-8")


class _DashboardState:
    def __init__(
        self,
        root: Path,
        database: Path = Path("quantguard.sqlite"),
    ) -> None:
        self.root = root.resolve()
        self.database = (self.root / database).resolve()

        if self.database != self.root and self.root not in self.database.parents:
            raise ValueError(
                "The dashboard database must be inside the project root."
            )

        self.lock = threading.Lock()
        self.active: dict[str, Any] | None = None

    def start(self, config: str) -> None:
        if config not in self.configs():
            raise ValueError(
                "Choose a configuration inside the project's configs directory."
            )

        with self.lock:
            if self.active and self.active.get("running"):
                raise ValueError("An evaluation is already running.")

            self.active = {
                "running": True,
                "output": "",
                "exit_code": None,
                "config": config,
            }

            threading.Thread(
                target=self._run,
                args=(config,),
                daemon=True,
            ).start()

    def _run(self, config: str) -> None:
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "quantguard.cli.main",
                    "run",
                    "--config",
                    config,
                    "--skip-performance",
                ],
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=os.environ.copy(),
            )

            output = ""
            assert process.stdout is not None

            for line in process.stdout:
                output = (output + line)[-24_000:]
                with self.lock:
                    if self.active:
                        self.active["output"] = output

            exit_code = process.wait()

            with self.lock:
                if self.active:
                    self.active.update(
                        running=False,
                        exit_code=exit_code,
                        output=output,
                    )

        except Exception as exc:
            with self.lock:
                if self.active:
                    self.active.update(
                        running=False,
                        exit_code=2,
                        output=str(exc),
                    )

    def configs(self) -> list[str]:
        folder = (self.root / "configs").resolve()
        valid: list[str] = []

        for path in folder.glob("*.yaml"):
            if not path.is_file():
                continue

            try:
                load_config(path)
            except ConfigurationError:
                continue

            valid.append(
                str(path.relative_to(self.root)).replace("\\", "/")
            )

        return sorted(valid)

    def data(self) -> dict[str, Any]:
        database = self.database
        runs: list[dict[str, Any]] = []
        latest: dict[str, Any] | None = None
        errors: list[str] = []
        total_evaluations = 0

        if database.exists():
            try:
                store = ExperimentStore(database)
                recent_rows = store.list_recent(100)

                with sqlite3.connect(database) as connection:
                    total_evaluations = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM experiments"
                        ).fetchone()[0]
                    )

                report_urls: dict[str, str] = {}

                for report_json in (
                    self.root / "reports"
                ).glob("**/report.json"):
                    try:
                        report = json.loads(
                            report_json.read_text(encoding="utf-8")
                        )
                        experiment_id = report.get("experiment_id")
                        report_html = report_json.with_name("report.html")

                        if experiment_id and report_html.is_file():
                            report_urls[str(experiment_id)] = (
                                f"/api/reports/{experiment_id}"
                            )

                    except (OSError, json.JSONDecodeError):
                        continue

                for row in recent_rows:
                    experiment_id = row.get("experiment_id")

                    try:
                        payload = store.get(experiment_id)
                        if payload is None:
                            continue

                        payload["gate_status"] = row.get("gate_status")
                        payload["report_url"] = report_urls.get(experiment_id)
                        runs.append(payload)

                        if latest is None:
                            latest = payload

                    except (json.JSONDecodeError, sqlite3.Error, OSError) as exc:
                        errors.append(
                            f"Skipped unreadable run {experiment_id}: {exc}"
                        )

            except (sqlite3.Error, OSError) as exc:
                errors.append(f"Could not read run history: {exc}")

        injections: list[dict[str, Any]] = []

        for path in sorted(
            (self.root / "reports").glob("**/validation_results.json")
        )[-12:]:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                relative_path = str(path.relative_to(self.root)).replace(
                    "\\", "/"
                )

                if isinstance(record, dict):
                    injections.append({
                        **record,
                        "file": relative_path,
                    })

                elif isinstance(record, list):
                    injections.extend(
                        {
                            **item,
                            "file": relative_path,
                        }
                        for item in record
                        if isinstance(item, dict)
                    )

            except (OSError, json.JSONDecodeError):
                continue

        determinism: list[dict[str, Any]] = []

        for path in sorted(
            (self.root / "reports").glob("**/determinism.json")
        )[-12:]:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))

                if isinstance(record, dict):
                    determinism.append({
                        **record,
                        "file": str(path.relative_to(self.root)).replace(
                            "\\", "/"
                        ),
                    })

            except (OSError, json.JSONDecodeError):
                continue

        config_sources: dict[str, str] = {}

        for relative_path in self.configs():
            config_path = (self.root / relative_path).resolve()

            if self.root not in config_path.parents or not config_path.is_file():
                continue

            try:
                config_sources[relative_path] = config_path.read_text(
                    encoding="utf-8"
                )[:60_000]
            except OSError as exc:
                errors.append(f"Could not read {relative_path}: {exc}")

        with self.lock:
            active = dict(self.active) if self.active else None

        return {
            "configs": self.configs(),
            "config_sources": config_sources,
            "runs": runs,
            "latest": latest,
            "total_evaluations": total_evaluations,
            "injections": injections,
            "determinism": determinism,
            "active": active,
            "errors": errors,
        }


def serve_dashboard(
    root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    database: Path = Path("quantguard.sqlite"),
) -> None:
    """Serve the dashboard on localhost only."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The dashboard may only bind to localhost.")

    state = _DashboardState(root, database)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, data: Any, status: int = 200) -> None:
            payload = json.dumps(data, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8",
            )
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            route = urlparse(self.path).path

            if route == "/":
                payload = _PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "text/html; charset=utf-8",
                )
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return

            if route == "/api/data":
                self._json(state.data())
                return

            if route == "/api/run":
                with state.lock:
                    run_state = state.active or {
                        "running": False,
                        "exit_code": None,
                        "output": "",
                    }
                self._json(run_state)
                return

            if route.startswith("/api/reports/"):
                experiment_id = route.removeprefix("/api/reports/")

                if not re.fullmatch(r"exp_[A-Za-z0-9_]+", experiment_id):
                    self._json({"error": "Report not found"}, 404)
                    return

                report_root = (state.root / "reports").resolve()
                report_path: Path | None = None

                for candidate in report_root.glob(
                    f"**/{experiment_id}/report.html"
                ):
                    resolved = candidate.resolve()

                    if report_root in resolved.parents and resolved.is_file():
                        report_path = resolved
                        break

                if report_path is None:
                    self._json({"error": "Report not found"}, 404)
                    return

                payload = report_path.read_bytes()
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "text/html; charset=utf-8",
                )
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return

            self._json({"error": "Not found"}, 404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/run":
                self._json({"error": "Not found"}, 404)
                return

            try:
                request_size = int(self.headers.get("Content-Length", "0"))

                if not 0 < request_size <= 2048:
                    raise ValueError("Invalid request size.")

                request_data = json.loads(self.rfile.read(request_size))

                if (
                    not isinstance(request_data, dict)
                    or not isinstance(request_data.get("config"), str)
                ):
                    raise ValueError("A configuration name is required.")

                state.start(request_data["config"])
                self._json({"started": True}, 202)

            except (ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)

        def log_message(self, fmt: str, *args: Any) -> None:
            print("quantguard-dashboard:", fmt % args)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"QuantGuard dashboard: http://{host}:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Dashboard stopped.")
    finally:
        server.server_close()
