"""Small localhost dashboard for existing QuantGuard records and safe smoke runs."""

from __future__ import annotations

import json
import os
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


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantGuard · Local evaluation dashboard</title><style>
:root{color-scheme:dark;--bg:#0b1118;--panel:#121c26;--line:#243444;--muted:#9aabba;--text:#edf4f8;--teal:#5ce1cf;--green:#58d68d;--red:#ff7272;--amber:#ffca67}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 85% -20%,#17353a 0,transparent 48%),var(--bg);color:var(--text);font:15px/1.5 Inter,Segoe UI,sans-serif}main{max-width:1250px;margin:auto;padding:38px 24px 80px}header{display:flex;justify-content:space-between;align-items:center;margin-bottom:32px}.brand{display:flex;gap:13px;align-items:center}.mark{width:38px;height:38px;border:1px solid #377e7b;border-radius:12px;display:grid;place-items:center;color:var(--teal);font-weight:800;font-size:20px}.eyebrow{text-transform:uppercase;letter-spacing:.15em;color:var(--teal);font-size:11px;font-weight:700}.subtitle{color:var(--muted);font-size:13px}.badge{border:1px solid var(--line);border-radius:99px;padding:7px 12px;color:var(--muted);font-size:12px}.hero{display:flex;align-items:end;justify-content:space-between;margin:30px 0 22px}h1{font-size:30px;letter-spacing:-.04em;margin:2px 0 3px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}.card{background:linear-gradient(145deg,#14212c,#101923);border:1px solid var(--line);border-radius:16px;padding:20px;min-width:0}.wide{grid-column:span 8}.side{grid-column:span 4}.half{grid-column:span 6}.full{grid-column:span 12}.card h2{font-size:14px;margin:0 0 16px;letter-spacing:.01em}.muted{color:var(--muted)}.status{font-size:25px;font-weight:800;letter-spacing:.04em}.PASS{color:var(--green)}.FAIL,.ERROR{color:var(--red)}.WARNING,.BLOCKED,.SKIPPED{color:var(--amber)}.metric{display:grid;grid-template-columns:1.2fr 1fr .85fr .9fr .7fr;gap:9px;padding:11px 0;border-top:1px solid #22313d;font-size:12px}.metric.head{color:var(--muted);font-size:10px;text-transform:uppercase}.pill{border-radius:99px;padding:3px 8px;background:#26343f;font-size:11px}.meta{display:grid;grid-template-columns:1fr 1fr;gap:13px}.meta span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.meta b{display:block;margin-top:3px;overflow-wrap:anywhere}.row{display:flex;justify-content:space-between;gap:12px;border-top:1px solid #22313d;padding:10px 0;font-size:13px}.empty{padding:22px;border:1px dashed #35505f;border-radius:10px;color:var(--muted);text-align:center}.table-wrap{overflow:auto}button,select{background:#182733;color:var(--text);border:1px solid #35505f;border-radius:8px;padding:10px 12px;font:inherit}button{cursor:pointer;background:#174a4a;border-color:#347e78;font-weight:700}button:disabled{opacity:.55;cursor:wait}button:hover{background:#1c5b59}pre{max-height:220px;overflow:auto;background:#0a1118;border-radius:9px;padding:12px;color:#bfd0dc;font-size:12px;white-space:pre-wrap}svg{width:100%;height:210px}.legend{font-size:11px;color:var(--muted)}.toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap}.small{font-size:12px}.runbox{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.foot{margin:18px 0;color:var(--muted);font-size:12px}@media(max-width:800px){.wide,.side,.half{grid-column:span 12}.hero{display:block}.badge{display:inline-block;margin-top:14px}.metric{grid-template-columns:1.2fr 1fr .8fr}.metric>*:nth-child(4),.metric>*:last-child{display:none}}
</style></head><body><main><header><div class="brand"><div class="mark">Q</div><div><b>QuantGuard</b><div class="subtitle">Accuracy regression testing</div></div></div><span class="badge">LOCAL WORKSPACE · READ ONLY REPORTS</span></header>
<section class="hero"><div><div class="eyebrow">Evaluation overview</div><h1>Model quality, at a glance.</h1><div class="subtitle">Live view of stored runs, gate decisions and measured evidence.</div></div><div class="toolbar"><button onclick="refresh()">↻ Refresh data</button></div></section>
<div id="alert" class="foot"></div><div class="grid"><section class="card wide"><h2>Latest evaluation <span id="when" class="muted small"></span></h2><div id="latest" class="empty">Loading records…</div></section>
<section class="card side"><h2>Run a smoke evaluation</h2><p class="muted small">Runs the project's configured candidate comparison. Model downloads may be required.</p><div class="runbox"><select id="config"></select><button id="run" onclick="startRun()">Start run</button></div><div id="runstate" class="foot">No dashboard run started.</div><pre id="logs" hidden></pre></section>
<section class="card half"><h2>Metric comparison</h2><div id="metrics" class="empty">No stored evaluation yet.</div></section>
<section class="card half"><h2>Run history</h2><div id="chart" class="empty">History appears after evaluations are recorded.</div><div id="history" class="foot"></div></section>
<section class="card half"><h2>Experiment evidence</h2><div id="evidence" class="empty">No experiment records found. Simulated validation is not a real model measurement.</div></section>
<section class="card half"><h2>Model and runtime</h2><div id="metadata" class="empty">No run metadata available.</div></section></div>
<div class="foot">Data comes from the local SQLite experiment store and report files. Opening this dashboard does not load a model. Not measured values are shown as unavailable.</div></main>
<script>
const $=x=>document.getElementById(x);let runTimer=null;
function esc(s){return String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function n(v,d=4){return v==null?'—':Number(v).toLocaleString(undefined,{maximumFractionDigits:d})}
async function api(url,opt){let r=await fetch(url,opt);let x=await r.json();if(!r.ok)throw Error(x.error||r.statusText);return x}
function status(s){return `<b class="status ${esc(s)}">${esc(s||'UNKNOWN')}</b>`}
function drawChart(runs){if(!runs.length){$('chart').className='empty';$('chart').textContent='History appears after evaluations are recorded.';return}let vals=[];for(const r of [...runs].reverse()){let c=(r.comparisons||[]).find(x=>x.absolute_difference!=null);if(c)vals.push({label:r.created_at?.slice(0,10)||r.experiment_id,value:c.candidate_value,base:c.baseline_value,status:r.decision?.status})}if(!vals.length){$('chart').className='empty';$('chart').textContent='No comparable numeric metric found in stored records.';return}let w=650,h=190,p=34,all=vals.flatMap(x=>[x.value,x.base]),lo=Math.min(...all),hi=Math.max(...all);if(lo===hi){lo-=.01;hi+=.01}let x=i=>p+(w-2*p)*(vals.length===1?.5:i/(vals.length-1)),y=v=>h-p-(h-2*p)*(v-lo)/(hi-lo),line=a=>a.map((q,i)=>`${x(i)},${y(q)}`).join(' ');$('chart').className='';$('chart').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Candidate and baseline trend"><polyline points="${line(vals.map(v=>v.base))}" fill="none" stroke="#667887" stroke-dasharray="5 5"/><polyline points="${line(vals.map(v=>v.value))}" fill="none" stroke="#5ce1cf" stroke-width="3"/>${vals.map((v,i)=>`<circle cx="${x(i)}" cy="${y(v.value)}" r="4" fill="#5ce1cf"/>`).join('')}<text x="${p}" y="14" fill="#9aabba" font-size="11">${esc(vals[0].label)} → ${esc(vals.at(-1).label)}</text><text x="${w-p}" y="14" text-anchor="end" fill="#9aabba" font-size="11">${esc(n(hi))}</text></svg><div class="legend">● candidate score &nbsp; · · · baseline score · First comparable metric from each actual run record.</div>`}
async function refresh(){try{let d=await api('/api/data');$('config').innerHTML=d.configs.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');$('alert').textContent=(d.errors||[]).join(' · ');if(d.error){$('latest').className='empty';$('latest').textContent=d.error;return}let r=d.latest;if(!r){$('latest').className='empty';$('latest').textContent='No runs stored yet. Run a local evaluation from the panel to begin.';$('metrics').className='empty';$('metrics').textContent='No stored evaluation yet.';return}$('latest').className='';$('when').textContent=r.created_at?`· ${esc(r.created_at)}`:'';let checks=r.decision?.checks||[];let comps=r.comparisons||[];$('latest').innerHTML=`<div style="display:flex;justify-content:space-between;align-items:center;gap:10px">${status(r.decision?.status)}<span class="pill">${esc(r.name||'QuantGuard run')}</span></div><p class="muted">${esc(r.decision?.reasons?.join(' · ')||r.decision?.summary||'Stored evaluation record')}</p><div class="meta"><div><span>Experiment</span><b>${esc(r.experiment_id)}</b></div><div><span>Gate checks</span><b>${checks.filter(x=>x.status==='fail').length} failed · ${checks.filter(x=>x.status==='inconclusive').length} inconclusive</b></div></div>`;
let rows=comps.filter(c=>c.baseline_value!=null&&c.candidate_value!=null);$('metrics').className='';$('metrics').innerHTML=rows.length?`<div class="metric head"><b>Task / metric</b><b>Baseline → candidate</b><b>Δ / SE</b><b>Threshold</b><b>Decision</b></div>`+rows.map(c=>{let policy=checks.find(x=>x.task===c.task&&x.metric===c.metric&&x.scope==='quality');return `<div class="metric"><b>${esc(c.task)} / ${esc(c.metric)}</b><span>${n(c.baseline_value)} → ${n(c.candidate_value)}</span><span>${n(c.absolute_difference)}${c.regression_pp!=null?' ('+n(c.absolute_difference*100,2)+' pp)':''} / SE ${n(c.difference_stderr)}</span><span>${esc(policy?.threshold_display||'—')}</span><span class="pill">${esc(policy?.status||c.status)}${c.stderr_method?' · '+esc(c.stderr_method):''}</span></div>`}).join(''):'<div class="empty">Stored record has no comparable metric rows.</div>';
let model=r.candidate||{},perf=r.candidate_performance||{};$('metadata').className='';$('metadata').innerHTML=`<div class="meta"><div><span>Model</span><b>${esc(model.model_id||r.candidate_fingerprint_inputs?.model_id)}</b></div><div><span>Revision</span><b>${esc(model.revision||r.candidate_fingerprint_inputs?.revision)}</b></div><div><span>Quantization</span><b>${esc(model.quantization_method)}</b></div><div><span>Runtime</span><b>${r.runtime_seconds==null?'Unavailable':n(r.runtime_seconds,2)+' sec'}</b></div><div><span>Performance</span><b>${Object.keys(perf).length?esc(JSON.stringify(perf)):'Unavailable'}</b></div><div><span>Tasks / samples</span><b>${esc((r.evaluation?.tasks||[]).join(', '))} / ${esc(r.evaluation?.limit??'full')}</b></div></div>`;
drawChart(d.runs);$('history').innerHTML=d.runs.slice(0,8).map(x=>`<div class="row"><span>${esc(x.created_at?.slice(0,16)||x.experiment_id)}</span><b class="${esc(x.gate_status)}">${esc(x.gate_status)}</b></div>`).join('');let inj=d.injections||[],det=d.determinism||[];$('evidence').className=inj.length||det.length?'':'empty';$('evidence').innerHTML=(inj.length?inj.map(x=>`<div><div class="row"><b>Injection validation</b><span>${esc(x.file)}</span></div><div class="meta"><div><span>Detection rate</span><b>${x.detection_rate==null?'Unavailable':n(x.detection_rate*100,1)+'%'}</b></div><div><span>False-alarm rate</span><b>${x.false_alarm_rate==null?'Unavailable':n(x.false_alarm_rate*100,1)+'%'}</b></div><div><span>Scenarios / errors</span><b>${esc(x.outcomes?.length??'—')} / ${esc(x.error_count??'—')}</b></div><div><span>Mean runtime</span><b>${x.mean_seconds==null?'Unavailable':n(x.mean_seconds,2)+' sec'}</b></div></div></div>`).join(''):'')+(det.length?det.map(x=>`<div><div class="row"><b>Run-to-run consistency</b><span>${esc(x.file)} · ${esc(x.runs)} runs</span></div>${(x.metrics||[]).map(m=>`<div class="row"><span>${esc(m.task)} / ${esc(m.metric)} · SD ${n(m.stdev)}</span><b>${esc(m.identical?'identical':'varied')}</b></div>`).join('')}</div>`).join(''):'')||'No real experiment result files found. Simulation results are not shown as model measurements.';
if(d.active){$('runstate').textContent=d.active.running?'Evaluation running…':`Finished with exit code ${d.active.exit_code}.`;$('logs').hidden=false;$('logs').textContent=d.active.output||'';if(d.active.running&&!runTimer)runTimer=setInterval(pollRun,1200);if(!d.active.running&&runTimer){clearInterval(runTimer);runTimer=null}}}catch(e){$('latest').className='empty';$('latest').textContent='Dashboard data error: '+e.message}}
async function startRun(){let b=$('run');b.disabled=true;$('runstate').textContent='Starting configured comparison…';try{await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({config:$('config').value})});await pollRun();if(!runTimer)runTimer=setInterval(pollRun,1200)}catch(e){$('runstate').textContent=e.message;b.disabled=false}}
async function pollRun(){try{let d=await api('/api/run');$('runstate').textContent=d.running?'Evaluation running…':`Finished with exit code ${d.exit_code}.`;$('logs').hidden=false;$('logs').textContent=d.output||'';$('run').disabled=d.running;if(!d.running){clearInterval(runTimer);runTimer=null;await refresh()}}catch(e){$('runstate').textContent=e.message;$('run').disabled=false}}
refresh();
</script></body></html>"""


class _DashboardState:
    def __init__(self, root: Path, database: Path = Path("quantguard.sqlite")):
        self.root = root.resolve()
        self.database = (self.root / database).resolve()
        if self.database != self.root and self.root not in self.database.parents:
            raise ValueError("The dashboard database must be inside the project root.")
        self.lock = threading.Lock()
        self.active: dict[str, Any] | None = None

    def start(self, config: str) -> None:
        configs = self.configs()
        if config not in configs:
            raise ValueError("Choose a configuration inside the project's configs directory.")
        with self.lock:
            if self.active and self.active.get("running"):
                raise ValueError("An evaluation is already running.")
            self.active = {"running": True, "output": "", "exit_code": None, "config": config}
            threading.Thread(target=self._run, args=(config,), daemon=True).start()

    def _run(self, config: str) -> None:
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "quantguard.cli.main", "run", "--config", config, "--skip-performance"],
                cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", env=os.environ.copy(),
            )
            output = ""
            assert proc.stdout is not None
            for line in proc.stdout:
                output = (output + line)[-24000:]
                with self.lock:
                    if self.active:
                        self.active["output"] = output
            code = proc.wait()
            with self.lock:
                if self.active:
                    self.active.update(running=False, exit_code=code, output=output)
        except Exception as exc:
            with self.lock:
                if self.active:
                    self.active.update(running=False, exit_code=2, output=str(exc))

    def configs(self) -> list[str]:
        folder = (self.root / "configs").resolve()
        valid = []
        for path in folder.glob("*.yaml"):
            if not path.is_file():
                continue
            try:
                load_config(path)
            except ConfigurationError:
                continue
            valid.append(str(path.relative_to(self.root)).replace("\\", "/"))
        return sorted(valid)

    def data(self) -> dict[str, Any]:
        db = self.database
        runs: list[dict[str, Any]] = []
        latest = None
        errors: list[str] = []
        if db.exists():
            try:
                store = ExperimentStore(db)
                runs = store.list_recent(100)
                for row in runs:
                    try:
                        latest = store.get(row["experiment_id"])
                        if latest is not None:
                            break
                    except (json.JSONDecodeError, sqlite3.Error, OSError) as exc:
                        errors.append(f"Skipped unreadable run {row.get('experiment_id')}: {exc}")
            except (sqlite3.Error, OSError) as exc:
                errors.append(f"Could not read run history: {exc}")
        injections = []
        for path in sorted((self.root / "reports").glob("**/validation_results.json"))[-12:]:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(record, dict):
                    injections.append({**record, "file": str(path.relative_to(self.root)).replace("\\", "/")})
                elif isinstance(record, list):
                    injections.extend({**x, "file": str(path.relative_to(self.root)).replace("\\", "/")} for x in record if isinstance(x, dict))
            except (OSError, json.JSONDecodeError):
                continue
        determinism = []
        for path in sorted((self.root / "reports").glob("**/determinism.json"))[-12:]:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(record, dict):
                    determinism.append({**record, "file": str(path.relative_to(self.root)).replace("\\", "/")})
            except (OSError, json.JSONDecodeError):
                continue
        with self.lock:
            active = dict(self.active) if self.active else None
        return {"configs": self.configs(), "runs": runs, "latest": latest, "injections": injections, "determinism": determinism, "active": active, "errors": errors}


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
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/":
                payload = _PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif route == "/api/data":
                self._json(state.data())
            elif route == "/api/run":
                with state.lock:
                    self._json(state.active or {"running": False, "exit_code": None, "output": ""})
            else:
                self._json({"error": "Not found"}, 404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/run":
                self._json({"error": "Not found"}, 404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2048:
                    raise ValueError("Invalid request size.")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict) or not isinstance(data.get("config"), str):
                    raise ValueError("A configuration name is required.")
                state.start(data["config"])
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
