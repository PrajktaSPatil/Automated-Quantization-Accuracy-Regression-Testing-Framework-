<!-- GitHub file: src/quantguard/dashboard_ui.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>QUANTGUARD · Model Quality Control Center</title>
  <style>
    :root {
      color-scheme: dark;
      --bg:#080d16; --panel:#101827; --panel2:#131f30; --line:#223149;
      --text:#edf3fc; --muted:#91a0b7; --blue:#5b9cff; --cyan:#4de0df;
      --green:#45d483; --amber:#f3b94f; --red:#ff6477; --radius:14px;
      font-family:Inter,"Segoe UI",system-ui,sans-serif;
    }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-size:14px}
    button,input,select{font:inherit;color:inherit}
    button,select,input{background:#101a2a;border:1px solid var(--line);border-radius:8px;padding:9px 11px}
    button{cursor:pointer}button:hover{border-color:var(--blue);background:#16253a}
    a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}
    .layout{min-height:100vh;display:grid;grid-template-columns:240px minmax(0,1fr)}
    aside{position:sticky;top:0;height:100vh;border-right:1px solid var(--line);padding:24px 16px;background:#0b111d}
    .brand{display:flex;gap:11px;align-items:center;padding:4px 8px 26px}
    .logo{width:36px;height:36px;border-radius:10px;background:#122b47;border:1px solid #244b76;display:grid;place-items:center;color:var(--cyan);font-weight:900}
    .brand strong{letter-spacing:.12em;font-size:13px}.brand small{display:block;color:var(--muted);font-size:10px;margin-top:4px}
    .nav-label{font-size:10px;color:#63738b;text-transform:uppercase;letter-spacing:.14em;padding:14px 10px 7px}
    nav{display:grid;gap:5px}
    nav button{text-align:left;border:0;background:transparent;color:var(--muted);padding:10px;border-radius:8px}
    nav button.active,nav button:hover{background:#13233a;color:var(--text)}
    nav button.active{box-shadow:inset 2px 0 var(--cyan)}
    .side-foot{position:absolute;bottom:18px;left:24px;color:var(--muted);font-size:11px}
    main{min-width:0;padding:28px clamp(18px,3vw,42px) 50px;max-width:1700px;width:100%;margin:auto}
    .topbar{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:26px}
    .crumb{color:var(--muted);font-size:12px}.crumb b{color:var(--text)}
    .live{font-size:11px;color:var(--green);border:1px solid #244b3b;border-radius:99px;padding:6px 10px}
    h1{font-size:25px;letter-spacing:-.03em;margin:0 0 5px}
    h2{font-size:14px;margin:0 0 15px}
    h3{font-size:12px;margin:0 0 12px}
    .sub{color:var(--muted);margin:0 0 22px;font-size:13px}
    .page{display:none}.page.active{display:block}
    .grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:14px}
    .card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:18px;min-width:0}
    .span-3{grid-column:span 3}.span-4{grid-column:span 4}.span-5{grid-column:span 5}
    .span-6{grid-column:span 6}.span-7{grid-column:span 7}.span-8{grid-column:span 8}
    .span-12{grid-column:span 12}
    .kpi-label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}
    .kpi-value{font-size:27px;font-weight:750;margin:9px 0 3px;letter-spacing:-.03em}
    .kpi-note{color:var(--muted);font-size:11px}
    .status{font-weight:800;letter-spacing:.06em}
    .PASS,.pass{color:var(--green)}.FAIL,.ERROR,.fail{color:var(--red)}
    .WARNING,.BLOCKED,.SKIPPED,.uncertain{color:var(--amber)}
    .pill{border:1px solid var(--line);border-radius:99px;padding:4px 8px;font-size:10px;color:var(--muted);white-space:nowrap}
    .toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
    .toolbar label{color:var(--muted);font-size:11px}
    .filters{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
    .filters select,.filters input{font-size:12px;min-width:130px}
    .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}
    table{border-collapse:collapse;width:100%;min-width:760px;font-size:12px}
    th,td{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}
    th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.07em;background:#0d1522;position:sticky;top:0}
    tr:last-child td{border-bottom:0}
    .muted{color:var(--muted)}.small{font-size:11px}
    .empty{padding:22px;text-align:center;color:var(--muted);border:1px dashed #35445b;border-radius:10px}
    .decision{display:flex;justify-content:space-between;align-items:center;gap:12px}
    .decision-mark{font-size:24px;font-weight:850}
    .metric-list{display:grid;gap:8px}
    .metric-row{display:grid;grid-template-columns:minmax(120px,1fr) repeat(3,minmax(90px,.7fr));gap:10px;padding:10px 0;border-top:1px solid var(--line);align-items:center}
    .bar-row{display:grid;grid-template-columns:130px 1fr 110px;align-items:center;gap:10px;margin:13px 0;font-size:11px}
    .bar-track{height:9px;background:#202c40;border-radius:8px;overflow:hidden;position:relative}
    .bar{height:100%;border-radius:8px;background:var(--blue)}
    .bar.candidate{background:var(--cyan)}
    .bar.threshold{background:var(--amber)}
    .chart{width:100%;height:250px;display:block}
    .chart text{fill:var(--muted);font:10px Inter,"Segoe UI",sans-serif}
    .legend{display:flex;gap:15px;flex-wrap:wrap;color:var(--muted);font-size:11px;margin-top:8px}
    .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
    pre{white-space:pre-wrap;overflow:auto;max-height:420px;background:#080e18;border:1px solid var(--line);padding:14px;border-radius:9px;color:#b8c9df;font-size:11px}
    details{border-top:1px solid var(--line);padding:11px 0}
    summary{cursor:pointer}
    .notice{color:var(--muted);font-size:11px;line-height:1.55}
    .error{color:var(--red);white-space:pre-wrap}
    @media(max-width:950px){.layout{grid-template-columns:190px minmax(0,1fr)}.span-3{grid-column:span 6}.span-4,.span-5,.span-7,.span-8{grid-column:span 12}}
    @media(max-width:650px){.layout{display:block}aside{position:relative;height:auto;border-right:0;border-bottom:1px solid var(--line);padding:12px}.brand{padding:4px 8px 12px}.nav-label,.side-foot{display:none}nav{display:flex;overflow:auto}nav button{white-space:nowrap;padding:8px}main{padding:20px 13px}.span-3,.span-6{grid-column:span 12}.metric-row{grid-template-columns:1fr 1fr}.bar-row{grid-template-columns:90px 1fr 80px}}
  </style>
</head>
<body>
<div class="layout">
  <aside>
    <div class="brand"><div class="logo">Q</div><div><strong>QUANTGUARD</strong><small>Quantization Accuracy Regression Testing Framework</small></div></div>
    <div class="nav-label">Workspace</div>
    <nav id="nav">
      <button class="active" data-page="overview">◈ &nbsp; Command Center</button>
      <button data-page="accuracy">⌁ &nbsp; Regression Monitor</button>
      <button data-page="quantization">▤ &nbsp; Quantization Analytics</button>
      <button data-page="experiments">⌗ &nbsp; Experiment Analytics</button>
      <button data-page="lab">◎ &nbsp; Regression Testing Lab</button>
      <button data-page="history">☷ &nbsp; Evaluation History</button>
      <button data-page="config">⚙ &nbsp; Configuration Center</button>
    </nav>
    <div class="side-foot">LOCAL WORKSPACE · READ ONLY REPORTS</div>
  </aside>
  <main>
    <div class="topbar"><div class="crumb">QuantGuard <span> / </span><b id="crumb">Command Center</b></div><div class="toolbar"><span class="live">● LOCAL DATA</span><button id="refresh">↻ Refresh</button></div></div>
    <div id="alert" class="small error"></div>

    <section id="overview" class="page active">
      <h1>Model quality control center</h1>
      <p class="sub">Release decisions and measured evidence from your local QuantGuard records.</p>
      <div class="grid">
        <article class="card span-3"><div class="kpi-label">Total evaluations</div><div id="k-total" class="kpi-value">—</div><div class="kpi-note">Stored in local experiment history</div></article>
        <article class="card span-3"><div class="kpi-label">Passed</div><div id="k-pass" class="kpi-value pass">—</div><div class="kpi-note">Gate status PASS</div></article>
        <article class="card span-3"><div class="kpi-label">Failed</div><div id="k-fail" class="kpi-value fail">—</div><div class="kpi-note">Gate status FAIL or ERROR</div></article>
        <article class="card span-3"><div class="kpi-label">Warnings / inconclusive</div><div id="k-warn" class="kpi-value uncertain">—</div><div class="kpi-note">Warning, blocked, skipped, or inconclusive</div></article>
        <article class="card span-7"><h2>Latest regression-gate decision</h2><div id="latest" class="empty">Loading stored evaluations…</div></article>
        <article class="card span-5"><h2>Tracked models</h2><div id="models" class="empty">Loading…</div></article>
        <article class="card span-12"><h2>Recent evaluation activity</h2><div id="activity" class="empty">Loading…</div></article>
      </div>
    </section>

    <section id="accuracy" class="page">
      <h1>Accuracy regression monitor</h1>
      <p class="sub">Baseline versus candidate values, gate thresholds, and uncertainty from stored comparisons.</p>
      <div class="grid">
        <article class="card span-3"><div class="kpi-label">Baseline accuracy</div><div id="a-base" class="kpi-value">—</div><div id="a-base-note" class="kpi-note">Latest available accuracy metric</div></article>
        <article class="card span-3"><div class="kpi-label">Candidate accuracy</div><div id="a-candidate" class="kpi-value">—</div><div class="kpi-note">Latest available accuracy metric</div></article>
        <article class="card span-3"><div class="kpi-label">Difference</div><div id="a-delta" class="kpi-value">—</div><div class="kpi-note">Candidate minus baseline</div></article>
        <article class="card span-3"><div class="kpi-label">Gate decision</div><div id="a-status" class="kpi-value">—</div><div class="kpi-note">From latest stored run</div></article>
        <article class="card span-12"><h2>Per-task comparison</h2><div id="accuracy-chart" class="empty">No comparable accuracy metrics found.</div><div id="accuracy-table"></div></article>
      </div>
    </section>

    <section id="quantization" class="page">
      <h1>Quantization analytics</h1>
      <p class="sub">Only measured performance fields are shown. Missing measurements remain unavailable.</p>
      <div class="filters"><select id="q-model"><option value="">All models</option></select><select id="q-method"><option value="">All methods</option></select></div>
      <div id="quant-table" class="empty">Loading measured quantization metrics…</div>
    </section>

    <section id="experiments" class="page">
      <h1>Experiment analytics</h1>
      <p class="sub">Filter stored runs and compare measured accuracy and runtime trends.</p>
      <div class="filters">
        <select id="f-model"><option value="">All models</option></select>
        <select id="f-method"><option value="">All quantization methods</option></select>
        <select id="f-task"><option value="">All tasks</option></select>
        <label class="small muted">From <input id="f-from" type="date"></label>
        <label class="small muted">To <input id="f-to" type="date"></label>
      </div>
      <div class="grid">
        <article class="card span-6"><h2>Accuracy trend</h2><div id="trend-accuracy" class="empty">No measured accuracy trend.</div></article>
        <article class="card span-6"><h2>Runtime trend</h2><div id="trend-runtime" class="empty">Runtime not measured.</div></article>
        <article class="card span-12"><h2>Filtered runs</h2><div id="experiment-table" class="empty">No runs match these filters.</div></article>
      </div>
    </section>

    <section id="lab" class="page">
      <h1>Regression testing lab</h1>
      <p class="sub">Synthetic injection validation is reported separately from real-model evaluation runs.</p>
      <div class="grid">
        <article class="card span-6"><h2>Injected-regression scenarios</h2><div id="injection-data" class="empty">No validation result files found.</div></article>
        <article class="card span-6"><h2>Determinism evidence</h2><div id="determinism-data" class="empty">No determinism result files found.</div></article>
        <article class="card span-12"><h2>Real-model evaluations</h2><div id="real-runs" class="empty">No real-model runs found.</div></article>
      </div>
    </section>

    <section id="history" class="page">
      <h1>Evaluation history</h1>
      <p class="sub">Search stored runs, inspect their details, and open generated reports.</p>
      <div class="filters"><input id="history-search" placeholder="Search run, model, task, or method"><select id="history-status"><option value="">All gate statuses</option><option>PASS</option><option>WARNING</option><option>FAIL</option><option>ERROR</option></select></div>
      <div id="history-table" class="empty">No stored evaluation history.</div>
    </section>

    <section id="config" class="page">
      <h1>Configuration center</h1>
      <p class="sub">Validated project YAML files and baseline metadata. Evaluation launch uses an existing validated config.</p>
      <div class="grid">
        <article class="card span-5"><h2>Validated configurations</h2><div class="toolbar"><select id="config-select"></select><button id="run-button">Start configured evaluation</button></div><div id="run-state" class="notice">No dashboard run started.</div><pre id="run-log" hidden></pre><div id="config-list"></div></article>
        <article class="card span-7"><h2>Configuration contents</h2><pre id="config-content">Select a validated YAML configuration.</pre></article>
        <article class="card span-12"><h2>Latest baseline and model metadata</h2><div id="baseline-meta" class="empty">No baseline metadata available.</div></article>
      </div>
    </section>
  </main>
</div>
<script>
const $=id=>document.getElementById(id);
let DATA=null, runTimer=null;
const esc=v=>String(v??"—").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
const fmt=(v,d=4)=>v==null||!Number.isFinite(Number(v))?"Unavailable":Number(v).toLocaleString(undefined,{maximumFractionDigits:d});
const titleCase=s=>String(s||"UNKNOWN").replaceAll("_"," ");
async function api(url,options){const r=await fetch(url,options);const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d}
function statusClass(s){return ["PASS","WARNING","FAIL","ERROR","BLOCKED","SKIPPED"].includes(s)?s:"uncertain"}
function records(){return DATA?.runs||[]}
function modelOf(r){return r?.candidate?.model_id||r?.candidate_fingerprint_inputs?.model_id||"Unknown model"}
function methodOf(r){return r?.candidate?.quantization_method||r?.candidate?.precision||"Unknown method"}
function tasksOf(r){return r?.evaluation?.tasks||[]}
function allComparisons(r){return r?.comparisons||[]}
function renderTable(rows,detail=true){
  if(!rows.length)return '<div class="empty">No matching stored evaluations.</div>';
  return `<div class="table-wrap"><table><thead><tr><th>Run / timestamp</th><th>Model</th><th>Quantization</th><th>Task</th><th>Accuracy / metric</th><th>Gate</th><th>Runtime</th><th>Report</th></tr></thead><tbody>${rows.map(r=>{
    const cs=allComparisons(r),c=cs.find(x=>x.metric==="acc")||cs[0],s=r.decision?.status||r.gate_status;
    const report=r.report_url?`<a href="${esc(r.report_url)}" target="_blank" rel="noopener">Open report</a>`:"Unavailable";
    const cells=`<td><b>${esc(r.experiment_id)}</b><br><span class="muted">${esc(r.created_at)}</span></td><td>${esc(modelOf(r))}</td><td>${esc(methodOf(r))}</td><td>${esc(tasksOf(r).join(", ")||c?.task||"Unavailable")}</td><td>${c?`${fmt(c.baseline_value)} → ${fmt(c.candidate_value)}<br><span class="muted">Δ ${fmt(c.absolute_difference)}${c.difference_stderr!=null?` · SE ${fmt(c.difference_stderr)}`:""}</span>`:"Unavailable"}</td><td class="${statusClass(s)}">${esc(s)}</td><td>${r.runtime_seconds==null?"Unavailable":`${fmt(r.runtime_seconds,2)} s`}</td><td>${report}</td>`;
    return `<tr>${cells}</tr>${detail?`<tr><td colspan="8"><details><summary>Run details · ${esc(r.name||r.config_hash||"configuration")}</summary><div class="grid"><div class="span-6"><p class="small muted">Baseline</p><pre>${esc(JSON.stringify(r.baseline||{},null,2))}</pre></div><div class="span-6"><p class="small muted">Candidate and gate checks</p><pre>${esc(JSON.stringify({candidate:r.candidate,checks:r.decision?.checks,notes:r.notes},null,2))}</pre></div></div></details></td></tr>`:""}`;
  }).join("")}</tbody></table></div>`;
}
function metricRows(r){
  const checks=r?.decision?.checks||[];
  return allComparisons(r).map(c=>{
    const check=checks.find(x=>x.scope==="quality"&&x.task===c.task&&x.metric===c.metric&&x.reference!=="previous_release");
    return {...c,check};
  }).filter(c=>c.baseline_value!=null&&c.candidate_value!=null);
}
function renderAccuracy(){
  const latest=DATA?.latest, rows=metricRows(latest).filter(c=>c.metric==="acc");
  const first=rows[0];
  $("a-base").textContent=first?fmt(first.baseline_value):"Unavailable";
  $("a-candidate").textContent=first?fmt(first.candidate_value):"Unavailable";
  $("a-delta").textContent=first?`${fmt(first.absolute_difference*100,2)} pp`:"Unavailable";
  $("a-status").innerHTML=latest?`<span class="${statusClass(latest.decision?.status)}">${esc(latest.decision?.status)}</span>`:"Unavailable";
  $("a-base-note").textContent=first?`${esc(first.task)} · ${esc(first.metric)}`:"No stored accuracy result";
  $("accuracy-table").innerHTML=renderMetricTable(metricRows(latest));
  if(!rows.length){$("accuracy-chart").className="empty";$("accuracy-chart").textContent="No comparable accuracy metrics in the latest run.";return}
  $("accuracy-chart").className="";
  const max=Math.max(...rows.flatMap(r=>[r.baseline_value,r.candidate_value]));
  const min=Math.min(0,...rows.flatMap(r=>[r.baseline_value,r.candidate_value]));
  const range=Math.max(max-min,1e-9);
  $("accuracy-chart").innerHTML=rows.map(r=>{
    const bp=Math.max(2,100*(r.baseline_value-min)/range),cp=Math.max(2,100*(r.candidate_value-min)/range);
    const decision=r.check?.status||r.status||"Unavailable";
    return `<div class="bar-row"><b>${esc(r.task)}</b><div><div class="bar-track"><div class="bar" style="width:${bp}%"></div></div><div class="bar-track" style="margin-top:5px"><div class="bar candidate" style="width:${cp}%"></div></div></div><span class="${statusClass(decision)}">${fmt(r.candidate_value)} · ${esc(decision)}</span></div>`;
  }).join("")+`<div class="legend"><span><i class="dot" style="background:var(--blue)"></i>Baseline</span><span><i class="dot" style="background:var(--cyan)"></i>Candidate</span><span class="muted">Amber threshold decision comes from the stored gate check.</span></div>`;
}
function renderMetricTable(rows){
  if(!rows.length)return '<div class="empty">No stored baseline/candidate comparisons.</div>';
  return `<div class="table-wrap"><table><thead><tr><th>Task / metric</th><th>Baseline</th><th>Candidate</th><th>Difference</th><th>Threshold</th><th>Uncertainty</th><th>Gate</th></tr></thead><tbody>${rows.map(c=>`<tr><td>${esc(c.task)} / ${esc(c.metric)}</td><td>${fmt(c.baseline_value)}</td><td>${fmt(c.candidate_value)}</td><td>${fmt(c.absolute_difference)}${c.metric==="acc"?` (${fmt(c.absolute_difference*100,2)} pp)`:""}</td><td>${esc(c.check?.threshold_display||"Unavailable")}</td><td>${esc(c.check?.noise_display||c.stderr_method||"Unavailable")}</td><td class="${statusClass(c.check?.status||c.status)}">${esc(c.check?.status||c.status||"Unavailable")}</td></tr>`).join("")}</tbody></table></div>`;
}
function populateFilters(){
  const sets={model:new Set(),method:new Set(),task:new Set()};
  records().forEach(r=>{sets.model.add(modelOf(r));sets.method.add(methodOf(r));tasksOf(r).forEach(t=>sets.task.add(t))});
  [["f-model",sets.model],["q-model",sets.model]].forEach(([id,set])=>fillSelect($(id),set,"All models"));
  [["f-method",sets.method],["q-method",sets.method]].forEach(([id,set])=>fillSelect($(id),set,"All quantization methods"));
  fillSelect($("f-task"),sets.task,"All tasks");
}
function fillSelect(el,set,label){
  if(!el)return;const current=el.value;el.innerHTML=`<option value="">${esc(label)}</option>`+[...set].sort().map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join("");el.value=current;
}
function filteredRuns(){
  const model=$("f-model").value,method=$("f-method").value,task=$("f-task").value,from=$("f-from").value,to=$("f-to").value;
  return records().filter(r=>(!model||modelOf(r)===model)&&(!method||methodOf(r)===method)&&(!task||tasksOf(r).includes(task))&&(!from||String(r.created_at||"").slice(0,10)>=from)&&(!to||String(r.created_at||"").slice(0,10)<=to));
}
function renderSvgTrend(el,values,label,color){
  if(!values.length){el.className="empty";el.textContent=`No measured ${label} values for these filters.`;return}
  const w=700,h=230,p=40,nums=values.map(x=>Number(x.value)).filter(Number.isFinite);
  if(!nums.length){el.className="empty";el.textContent=`${label} is unavailable in these runs.`;return}
  let lo=Math.min(...nums),hi=Math.max(...nums);if(lo===hi){lo-=Math.abs(lo*.05)||1;hi+=Math.abs(hi*.05)||1}
  const x=i=>p+(w-2*p)*(values.length===1?.5:i/(values.length-1));
  const y=v=>h-p-(h-2*p)*(v-lo)/(hi-lo);
  const points=values.map((v,i)=>`${x(i)},${y(Number(v.value))}`).join(" ");
  el.className="";
  el.innerHTML=`<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)} trend"><line x1="${p}" y1="${h-p}" x2="${w-p}" y2="${h-p}" stroke="#314159"/><polyline points="${points}" fill="none" stroke="${color}" stroke-width="3"/>${values.map((v,i)=>`<circle cx="${x(i)}" cy="${y(Number(v.value))}" r="4" fill="${color}"><title>${esc(v.label)}: ${fmt(v.value)}</title></circle>`).join("")}<text x="${p}" y="18">${fmt(hi)}</text><text x="${p}" y="${h-8}">${fmt(lo)}</text><text x="${p}" y="${h-2}">${esc(values[0].label)}</text><text x="${w-p}" y="${h-2}" text-anchor="end">${esc(values.at(-1).label)}</text></svg><div class="legend">${esc(label)} · values are from stored runs</div>`;
}
function renderExperiments(){
  const runs=filteredRuns();
  $("experiment-table").className="";
  $("experiment-table").innerHTML=renderTable(runs,false);
  const task=$("f-task").value;
  const accuracy=[];
  [...runs].reverse().forEach(r=>{
    const c=allComparisons(r).find(c=>c.metric==="acc"&&(!task||c.task===task));
    if(c)accuracy.push({label:String(r.created_at||r.experiment_id).slice(0,10),value:c.candidate_value});
  });
  renderSvgTrend($("trend-accuracy"),accuracy,"Candidate accuracy","var(--cyan)");
  const runtime=[...runs].reverse().filter(r=>r.runtime_seconds!=null).map(r=>({label:String(r.created_at||r.experiment_id).slice(0,10),value:r.runtime_seconds}));
  renderSvgTrend($("trend-runtime"),runtime,"Runtime (seconds)","var(--blue)");
}
function renderQuantization(){
  const model=$("q-model").value,method=$("q-method").value;
  const rows=records().filter(r=>(!model||modelOf(r)===model)&&(!method||methodOf(r)===method));
  if(!rows.length){$("quant-table").className="empty";$("quant-table").textContent="No measured quantization runs match these filters.";return}
  $("quant-table").className="";
  $("quant-table").innerHTML=`<div class="table-wrap"><table><thead><tr><th>Model / method</th><th>Size before</th><th>Size after</th><th>Reduction</th><th>Latency</th><th>Throughput</th><th>Peak memory</th><th>Accuracy</th></tr></thead><tbody>${rows.map(r=>{
    const b=r.baseline_performance||{},c=r.candidate_performance||{},acc=allComparisons(r).find(x=>x.metric==="acc");
    const before=b.model_size_bytes,after=c.model_size_bytes;
    const reduction=before>0&&after!=null?`${fmt((before-after)/before*100,2)}%`:"Unavailable";
    return `<tr><td>${esc(modelOf(r))}<br><span class="muted">${esc(methodOf(r))}</span></td><td>${before==null?"Unavailable":`${fmt(before/1048576,2)} MiB`}</td><td>${after==null?"Unavailable":`${fmt(after/1048576,2)} MiB`}</td><td>${reduction}</td><td>${c.latency_ms_median==null?"Unavailable":`${fmt(c.latency_ms_median,2)} ms`}</td><td>${c.throughput_seq_per_s==null?"Unavailable":`${fmt(c.throughput_seq_per_s,2)} seq/s`}</td><td>${c.gpu_memory_mib==null?"Unavailable":`${fmt(c.gpu_memory_mib,1)} MiB`}</td><td>${acc?`${fmt(acc.baseline_value)} → ${fmt(acc.candidate_value)}`:"Unavailable"}</td></tr>`;
  }).join("")}</tbody></table></div><p class="notice">Size and performance fields are shown only when measured and stored in the report.</p>`;
}
function renderLab(){
  const injections=DATA?.injections||[],det=DATA?.determinism||[];
  $("injection-data").className=injections.length?"":"empty";
  $("injection-data").innerHTML=injections.length?injections.map(x=>`<details><summary>${esc(x.file)} · detection ${x.detection_rate==null?"Unavailable":fmt(x.detection_rate*100,1)+"%"} · false alarm ${x.false_alarm_rate==null?"Unavailable":fmt(x.false_alarm_rate*100,1)+"%"}</summary><pre>${esc(JSON.stringify(x,null,2))}</pre></details>`).join(""):"No validation result files found.";
  $("determinism-data").className=det.length?"":"empty";
  $("determinism-data").innerHTML=det.length?det.map(x=>`<details><summary>${esc(x.file)} · ${esc(x.runs)} runs</summary><pre>${esc(JSON.stringify(x,null,2))}</pre></details>`).join(""):"No determinism result files found.";
  const real=records().filter(r=>!String(modelOf(r)).startsWith("toy/"));
  $("real-runs").className=real.length?"":"empty";
  $("real-runs").innerHTML=real.length?renderTable(real.slice(0,20),false):"No real-model runs found. Synthetic scenarios above are shown separately.";
}
function renderHistory(){
  const q=$("history-search").value.toLowerCase(),s=$("history-status").value;
  const rows=records().filter(r=>{
    const hay=[r.experiment_id,r.name,modelOf(r),methodOf(r),tasksOf(r).join(" ")].join(" ").toLowerCase();
    return (!q||hay.includes(q))&&(!s||(r.decision?.status||r.gate_status)===s);
  });
  $("history-table").className=rows.length?"":"empty";
  $("history-table").innerHTML=rows.length?renderTable(rows,true):"No stored runs match the search and status filters.";
}
function renderOverview(){
  const runs=records(),statuses=runs.map(r=>r.decision?.status||r.gate_status||"UNKNOWN");
  $("k-total").textContent=DATA?.total_evaluations??runs.length;
  $("k-pass").textContent=statuses.filter(s=>s==="PASS").length;
  $("k-fail").textContent=statuses.filter(s=>s==="FAIL"||s==="ERROR").length;
  $("k-warn").textContent=statuses.filter(s=>["WARNING","BLOCKED","SKIPPED","INCONCLUSIVE"].includes(s)).length;
  const latest=DATA?.latest;
  $("latest").className=latest?"":"empty";
  $("latest").innerHTML=latest?`<div class="decision"><span class="decision-mark ${statusClass(latest.decision?.status)}">${esc(latest.decision?.status||"UNKNOWN")}</span><span class="pill">${esc(latest.created_at)}</span></div><p>${esc(latest.name||latest.experiment_id)}</p><p class="muted">${esc(modelOf(latest))} · ${esc(methodOf(latest))} · ${esc(tasksOf(latest).join(", ")||"No tasks recorded")}</p><p class="notice">${esc((latest.decision?.reasons||[]).join(" · ")||"No gate reason was stored.")}</p>${latest.report_url?`<a href="${esc(latest.report_url)}" target="_blank" rel="noopener">Open latest report →</a>`:""}`
    :"No evaluations are stored yet.";
  const models=[...new Set(runs.map(modelOf))];
  $("models").className=models.length?"":"empty";
  $("models").innerHTML=models.length?models.map(m=>`<div class="metric-row"><b>${esc(m)}</b><span class="muted">${runs.filter(r=>modelOf(r)===m).length} recent runs</span></div>`).join(""):"No model records.";
  $("activity").className=runs.length?"":"empty";
  $("activity").innerHTML=runs.length?renderTable(runs.slice(0,8),false):"No evaluation activity yet.";
}
function renderBaseline(){
  const r=DATA?.latest;
  $("baseline-meta").className=r?"":"empty";
  $("baseline-meta").innerHTML=r?`<div class="grid"><div class="card span-6"><h3>Approved baseline</h3><pre>${esc(JSON.stringify(r.baseline||{},null,2))}</pre></div><div class="card span-6"><h3>Candidate</h3><pre>${esc(JSON.stringify(r.candidate||{},null,2))}</pre></div></div>`:"No baseline metadata is stored.";
}
function renderConfigs(){
  const configs=DATA?.configs||[];
  $("config-select").innerHTML=configs.map(c=>`<option value="${esc(c)}">${esc(c)}</option>`).join("");
  $("config-list").innerHTML=configs.length?configs.map(c=>`<div class="metric-row"><b>${esc(c)}</b><span class="muted">Validated</span></div>`).join(""):"No validated YAML configs found.";
  const show=()=>{$("config-content").textContent=DATA?.config_sources?.[$("config-select").value]||"Configuration contents unavailable."};
  $("config-select").onchange=show;show();
}
async function refresh(){
  $("alert").textContent="";
  try{DATA=await api("/api/data");if(DATA.error){$("alert").textContent=DATA.error}
    if(DATA.errors?.length)$("alert").textContent=DATA.errors.join(" · ");
    populateFilters();renderOverview();renderAccuracy();renderQuantization();renderExperiments();renderLab();renderHistory();renderBaseline();renderConfigs();
    updateRun();
  }catch(e){$("alert").textContent=`Could not load dashboard data: ${e.message}`}
}
function updateRun(){
  const run=DATA?.active;if(!run)return;
  $("run-state").textContent=run.running?"Evaluation is running…":`Finished with exit code ${run.exit_code}.`;
  $("run-log").hidden=false;$("run-log").textContent=run.output||"";
  $("run-button").disabled=run.running;
  if(run.running&&!runTimer)runTimer=setInterval(pollRun,1200);
  if(!run.running&&runTimer){clearInterval(runTimer);runTimer=null}
}
async function pollRun(){
  try{const d=await api("/api/run");$("run-state").textContent=d.running?"Evaluation is running…":`Finished with exit code ${d.exit_code}.`;$("run-log").hidden=false;$("run-log").textContent=d.output||"";$("run-button").disabled=!!d.running;if(!d.running){clearInterval(runTimer);runTimer=null;await refresh()}}
  catch(e){$("run-state").textContent=e.message;$("run-button").disabled=false}
}
$("nav").addEventListener("click",e=>{
  const b=e.target.closest("button[data-page]");if(!b)return;
  document.querySelectorAll("nav button").forEach(x=>x.classList.toggle("active",x===b));
  document.querySelectorAll(".page").forEach(x=>x.classList.toggle("active",x.id===b.dataset.page));
  $("crumb").textContent=b.textContent.trim().replace(/^[◈⌁▤⌗◎☷⚙]\s*/,"");
});
$("refresh").onclick=refresh;
["f-model","f-method","f-task","f-from","f-to"].forEach(id=>$(id).addEventListener("change",renderExperiments));
["q-model","q-method"].forEach(id=>$(id).addEventListener("change",renderQuantization));
$("history-search").addEventListener("input",renderHistory);
$("history-status").addEventListener("change",renderHistory);
$("run-button").onclick=async()=>{
  const config=$("config-select").value;if(!config){$("run-state").textContent="No validated config is available.";return}
  $("run-button").disabled=true;$("run-state").textContent="Starting configured evaluation…";
  try{await api("/api/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({config})});await pollRun()}
  catch(e){$("run-state").textContent=e.message;$("run-button").disabled=false}
};
refresh();
</script>
</body>
</html>
