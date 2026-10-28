# Local dashboard

Start from the repository root after installing the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
quantguard dashboard
```

Open `http://127.0.0.1:8765`. Stop the server with `Ctrl+C`. Change the port with
`quantguard dashboard --port 8766` if needed. The dashboard binds to loopback only.

It reads `quantguard.sqlite`, the latest stored experiment payload and
`reports/**/validation_results.json`. Viewing existing records does not load a model. The
configuration picker lists YAML files directly inside `configs/`; the run button starts the normal
`quantguard run --skip-performance --config <selected-config>` pipeline. The server accepts no
arbitrary command or path, rejects other config locations, and retains only the latest 24 KB of
process output in memory. A run may still need network access to fetch model weights or datasets.

The history chart plots the first comparable metric found in each stored release record and labels
it as such. Runtime is the full elapsed pipeline time recorded by QuantGuard. Size, latency,
throughput, and memory appear only if the run recorded them; otherwise the dashboard says they are
unavailable. The injection panel reads actual validation summary files and remains empty when none
exist. It does not generate sample data. `rtn-fakequant` validation is a synthetic weight
perturbation experiment, not evidence that a production quantizer changes model storage or runtime.

The server is intended for a trusted local machine. Do not expose its port to a network interface or
reverse proxy.
