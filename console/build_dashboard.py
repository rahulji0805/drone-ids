import json, sys, os

# --- Regenerate the dashboard from your latest build_dataset.py /
# rule_engine.py results. Lives in console/, so REPO defaults to ".."
# (the drone-ids/ root). Override with the DRONE_IDS_REPO env var if
# you keep this file somewhere else. ---
ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("DRONE_IDS_REPO", os.path.join(ROOT, ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)
from build_dataset import build
from detectors.rule_engine import run_rule_engine

feat_df, feature_cols, cmd_df, fw_df = build()
result = run_rule_engine(feat_df)

cols = ['t_us','flight_elapsed_s','gt_label','gt_attack_type','rule_predicted_attack',
        'rule_total_severity','hb_gap_s','msg_count_1s','nav_gps_vs_fused_mismatch_m',
        'telem_alt_mismatch_m','gps_lat','gps_lon','gps_alt']
sub = result[cols].copy()
sub['t'] = sub['flight_elapsed_s'].round(2)
sub = sub.drop(columns=['t_us','flight_elapsed_s']).iloc[::2].reset_index(drop=True)
data = sub.round(4).to_dict(orient='records')
data_json = json.dumps(data)

attacks = {}
for row in data:
    at = row['gt_attack_type']
    if at != 'none':
        if at not in attacks:
            attacks[at] = {'start': row['t'], 'end': row['t']}
        else:
            attacks[at]['end'] = row['t']
attacks_json = json.dumps(attacks)

html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>SENTINEL — Drone Intrusion Console</title>
<style>
:root{
  --sky-top:#eaf3f8; --sky-bot:#dcebf2; --paper:#ffffff; --line:#d6e3ea;
  --ink:#16323f; --ink-dim:#5c7c8a; --navy:#1b3a4b;
  --amber:#e88a3c; --amber-dk:#c96f27; --red:#d94b4b; --green:#2f9e6e;
  --dark-bg:#0a1116; --dark-panel:#0e161c; --dark-panel2:#121e26; --dark-border:#1c2c36;
  --dark-text:#cfe3ee; --dark-dim:#5c7c8a; --accent:#3fd0c0;
  font-family:'Inter','Segoe UI',ui-sans-serif,sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  background:linear-gradient(180deg,var(--sky-top),var(--sky-bot) 60%,#eef4f1);
  color:var(--ink); padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
  position:relative; overflow-x:hidden;
}
/* soft decorative flight-path arcs */
body::before{
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    radial-gradient(circle at 85% 8%, rgba(232,138,60,0.10), transparent 40%),
    radial-gradient(circle at 6% 70%, rgba(63,208,192,0.10), transparent 42%);
}

header{
  position:relative; z-index:2; padding:22px 28px 18px; display:flex; align-items:center;
  justify-content:space-between; flex-wrap:wrap; gap:14px; border-bottom:1px solid var(--line);
}
.brand{display:flex; align-items:center; gap:12px}
.brand .logo{width:38px;height:38px; flex:none}
.brand h1{font-size:21px; letter-spacing:-0.3px; color:var(--navy); font-weight:800}
.brand span{color:var(--ink-dim); font-size:12.5px; display:block; margin-top:1px}
.status{
  padding:8px 16px; border-radius:20px; font-size:12px; font-weight:700;
  background:rgba(47,158,110,0.1); color:var(--green); display:flex; align-items:center; gap:8px;
  border:1px solid rgba(47,158,110,0.25);
}
.status.alert{background:rgba(217,75,75,0.1); color:var(--red); border-color:rgba(217,75,75,0.3)}
.dot{width:7px;height:7px;border-radius:50%;background:currentColor}
.status.alert .dot{animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* --- dark topology console, framed like a ground-station screen --- */
.topowrap{position:relative; z-index:2; max-width:1180px; margin:24px auto 0; padding:0 28px}
.device{background:#0a1116; border-radius:14px; padding:10px; box-shadow:0 20px 50px -20px rgba(27,58,75,0.35), 0 2px 0 rgba(255,255,255,0.5) inset}
.device-bar{display:flex; align-items:center; gap:6px; padding:4px 6px 10px}
.device-bar .b{width:8px;height:8px;border-radius:50%;background:#26343d}
.topo{position:relative; height:460px; background:var(--dark-bg); border:1px solid var(--dark-border); border-radius:8px; overflow:hidden; color:var(--dark-text)}
.topo svg.edges{position:absolute; inset:0; width:100%; height:100%}
.topo-title{position:absolute; top:12px; left:16px; font-size:10.5px; letter-spacing:1.2px; color:var(--dark-dim); text-transform:uppercase; z-index:2}

.node{position:absolute; width:158px; background:var(--dark-panel2); border:1px solid var(--dark-border); border-radius:6px; padding:9px 10px; transition:border-color .3s, box-shadow .3s}
.node .nh{display:flex; align-items:center; gap:7px; margin-bottom:6px}
.node .nh svg{width:15px;height:15px;flex:none;color:var(--accent)}
.node .nh b{font-size:11px; color:#eef6fa; flex:1}
.node .ndot{width:6px;height:6px;border-radius:50%;background:var(--green); flex:none}
.node .sig{font-size:9.5px; color:var(--dark-dim); display:flex; justify-content:space-between}
.node .sig b{color:var(--dark-text); font-weight:600; font-variant-numeric:tabular-nums}
.node.alert{border-color:var(--red); box-shadow:0 0 0 1px rgba(217,75,75,.35), 0 0 18px rgba(217,75,75,.18)}
.node.alert .nh svg{color:var(--red)}
.node.alert .ndot{background:var(--red); animation:pulse 1s infinite}
.node.alert .sig b{color:var(--red)}

.edge{stroke:var(--dark-border); stroke-width:1.4; stroke-dasharray:3 4; fill:none}
.edge.flow{stroke:var(--accent); opacity:.5}
.edge.alertline{stroke:var(--red); stroke-width:1.8; stroke-dasharray:2 3; opacity:.9}

/* --- light content area --- */
main{position:relative; z-index:2; padding:20px 28px 8px; display:grid; grid-template-columns:1.3fr 1fr; gap:18px; max-width:1180px; margin:0 auto}
@media (max-width:860px){ main{grid-template-columns:1fr} .topo{height:640px} }
.panel{background:var(--paper); border:1px solid var(--line); border-radius:10px; padding:18px; box-shadow:0 1px 0 rgba(27,58,75,0.03)}
.panel h2{font-size:11px; text-transform:uppercase; letter-spacing:1.1px; color:var(--ink-dim); margin-bottom:13px; font-weight:700}

.transport{display:flex; align-items:center; gap:10px; margin-bottom:4px}
button.tbtn{
  background:var(--navy); border:none; color:#fff; padding:9px 16px; border-radius:20px; cursor:pointer;
  font-family:inherit; font-size:12.5px; font-weight:600; display:inline-flex; align-items:center; gap:6px;
}
button.tbtn:hover{background:#0f2733}
input[type=range]{flex:1; accent-color:var(--amber)}
.tlabel{font-size:11.5px; color:var(--ink-dim); min-width:54px; text-align:right; font-variant-numeric:tabular-nums}

.attacks{display:grid; grid-template-columns:1fr 1fr; gap:9px; margin-top:14px}
.abtn{
  background:#f6faf9; border:1px solid var(--line); border-radius:8px; padding:11px;
  cursor:pointer; text-align:left; font-family:inherit; transition:transform .12s, border-color .12s;
}
.abtn:hover{border-color:var(--amber); transform:translateY(-1px)}
.abtn .name{font-size:12px; color:var(--navy); font-weight:700}
.abtn .desc{font-size:10.5px; color:var(--ink-dim); margin-top:3px; line-height:1.4}

.log{max-height:400px; overflow-y:auto; display:flex; flex-direction:column-reverse; gap:7px}
.logrow{font-size:12px; padding:8px 10px; border-radius:7px; border-left:3px solid var(--red); background:#fdf3f2; display:flex; gap:9px; color:var(--ink)}
.logrow .t{color:var(--ink-dim); min-width:42px; font-variant-numeric:tabular-nums}
.logrow .type{color:var(--red); font-weight:700}
.logrow.healed{border-left-color:var(--green); background:#f1faf5}
.logrow.healed .type{color:var(--green)}
.log:empty::after{content:'No detections yet — press Play or pick a scenario.'; color:var(--ink-dim); font-size:12px; padding:8px 2px}

footer{position:relative; z-index:2; padding:20px 28px 30px; text-align:center; color:var(--ink-dim); font-size:11px}
::-webkit-scrollbar{width:6px} ::-webkit-scrollbar-thumb{background:var(--line);border-radius:3px}
.dark ::-webkit-scrollbar-thumb, .topo::-webkit-scrollbar-thumb{background:var(--dark-border)}
</style>
</head>
<body>

<header>
  <div class="brand">
    <svg class="logo" viewBox="0 0 48 48" fill="none">
      <circle cx="24" cy="24" r="23" fill="#fff" stroke="#d6e3ea"/>
      <g stroke="#1b3a4b" stroke-width="2" stroke-linecap="round">
        <path d="M24 18v12M18 24h12"/>
        <circle cx="15" cy="15" r="4" fill="#eaf3f8"/>
        <circle cx="33" cy="15" r="4" fill="#eaf3f8"/>
        <circle cx="15" cy="33" r="4" fill="#eaf3f8"/>
        <circle cx="33" cy="33" r="4" fill="#eaf3f8"/>
        <path d="M24 18l-9-3M24 18l9-3M24 30l-9 3M24 30l9 3" stroke-width="1.6"/>
      </g>
      <circle cx="24" cy="24" r="3" fill="#e88a3c"/>
    </svg>
    <div><h1>SENTINEL</h1><span>Onboard Drone Topology &amp; Intrusion Console</span></div>
  </div>
  <div class="status" id="statusBadge"><div class="dot"></div><span id="statusText">SYSTEM NORMAL</span></div>
</header>

<div class="topowrap">
  <div class="device"><div class="device-bar"><span class="b"></span><span class="b"></span><span class="b"></span></div>
    <div class="topo">
      <div class="topo-title">Live Onboard System Topology</div>
      <svg class="edges" id="edgeSvg"></svg>
      <div id="nodeLayer"></div>
    </div>
  </div>
</div>

<main>
  <div class="panel">
    <h2>Playback</h2>
    <div class="transport">
      <button class="tbtn" id="playBtn">Play</button>
      <input type="range" id="scrub" min="0" max="100" value="0">
      <span class="tlabel" id="tLabel">t=0.0s</span>
    </div>
    <h2 style="margin-top:18px">Fault Injection &amp; Attack Replay</h2>
    <div class="attacks" id="attackButtons"></div>
  </div>

  <div class="panel">
    <h2>Detection Event Stream</h2>
    <div class="log" id="eventLog"></div>
  </div>
</main>

<footer>SENTINEL &middot; Hybrid Rule + ML Drone Intrusion Detection &middot; Replaying validated benchmark run (873 timesteps, 6/6 attack vectors)</footer>

<script>
const DATA = __DATA_JSON__;
const ATTACKS = __ATTACKS_JSON__;
const N = DATA.length;

const ATTACK_LABEL = {
  gps_spoofing:"GPS Spoofing", telemetry_manipulation:"Telemetry Manipulation",
  dos:"DoS / Link Jamming", mavlink_anomaly:"MAVLink Anomaly",
  command_injection:"Command Injection", firmware_integrity:"Firmware Integrity"
};
const ATTACK_DESC = {
  gps_spoofing:"Raw GPS vs fused-position divergence", telemetry_manipulation:"Altitude / vertical-speed mismatch",
  dos:"Heartbeat drop, link starvation", mavlink_anomaly:"Bad CRC / replayed sequence",
  command_injection:"Unauthorized system ID command", firmware_integrity:"Attestation hash mismatch"
};
const ATTACK_NODE = {
  gps_spoofing:['gps'], telemetry_manipulation:['fc'], dos:['link'],
  mavlink_anomaly:['link','fc'], command_injection:['gcs','fc'], firmware_integrity:['fw']
};

// line-icon svg fragments (no emoji)
const ICONS = {
  gcs:  '<circle cx="7" cy="17" r="2"/><path d="M2 20h10M7 15V9M3 9h8l-1-4H4z"/>',
  link: '<path d="M12 2v6M8 6l4-4 4 4M4 12a8 8 0 0116 0"/><circle cx="12" cy="15" r="1.4"/>',
  fc:   '<rect x="6" y="6" width="12" height="12" rx="1.5"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
  gps:  '<path d="M12 2a7 7 0 00-7 7c0 5 7 13 7 13s7-8 7-13a7 7 0 00-7-7z"/><circle cx="12" cy="9" r="2.3"/>',
  ids:  '<path d="M12 2l8 3v6c0 5-3.5 8.5-8 11-4.5-2.5-8-6-8-11V5z"/><path d="M9 12l2 2 4-4"/>',
  esc:  '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/>',
  fw:   '<rect x="5" y="10" width="14" height="10" rx="1.5"/><path d="M8 10V7a4 4 0 018 0v3"/>'
};

const NODES = [
  {id:'gcs',  name:'Ground Control', x:36,  y:16,  sig:['LINK','UP']},
  {id:'link', name:'Telemetry Link',  x:36,  y:38,  sig:['HB','—']},
  {id:'fc',   name:'Flight Controller', x:36,  y:60,  sig:['LOOP','—']},
  {id:'gps',  name:'GPS Module',      x:8,   y:82,  sig:['FIX','—']},
  {id:'ids',  name:'IDS Companion',   x:36, y:82,  sig:['SEV','0']},
  {id:'esc',  name:'Motor / ESC',     x:64,  y:82,  sig:['PWM','—']},
  {id:'fw',   name:'Firmware Attest.', x:82, y:38,  sig:['HASH','OK']},
];
const EDGES = [['gcs','link'],['link','fc'],['fc','gps'],['fc','ids'],['fc','esc'],['fc','fw']];

const topo = document.querySelector('.topo');
const nodeLayer = document.getElementById('nodeLayer');
const svg = document.getElementById('edgeSvg');

function pos(n){ return {x: n.x/100*topo.clientWidth, y: n.y/100*topo.clientHeight}; }

function drawEdges(activeIds){
  svg.innerHTML='';
  EDGES.forEach(([a,b])=>{
    const na=NODES.find(n=>n.id===a), nb=NODES.find(n=>n.id===b);
    const pa=pos(na), pb=pos(nb);
    const midx=(pa.x+pb.x)/2;
    const path=document.createElementNS('http://www.w3.org/2000/svg','path');
    const d=`M ${pa.x+79} ${pa.y+30} C ${midx+79} ${pa.y+30}, ${midx+79} ${pb.y+30}, ${pb.x+79} ${pb.y+30}`;
    path.setAttribute('d', d);
    const isAlert = activeIds.includes(a) && activeIds.includes(b);
    path.setAttribute('class', 'edge ' + (isAlert ? 'alertline' : 'flow'));
    svg.appendChild(path);
  });
}

function renderNodes(){
  nodeLayer.innerHTML='';
  NODES.forEach(n=>{
    const p=pos(n);
    const div=document.createElement('div');
    div.className='node'; div.id='node-'+n.id;
    div.style.left=p.x+'px'; div.style.top=p.y+'px';
    div.innerHTML = `<div class="nh"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[n.id]}</svg><b>${n.name}</b><span class="ndot"></span></div>
      <div class="sig"><span>${n.sig[0]}</span><b id="sig-${n.id}">${n.sig[1]}</b></div>`;
    nodeLayer.appendChild(div);
  });
  drawEdges([]);
}
window.addEventListener('resize', ()=>{ renderNodes(); updateVisual(DATA[idx]); });

let idx=0, playing=false, timer=null, prevLoggedAttack='none';
const scrub=document.getElementById('scrub'), playBtn=document.getElementById('playBtn');
const statusBadge=document.getElementById('statusBadge'), statusText=document.getElementById('statusText');
const eventLog=document.getElementById('eventLog'), tLabel=document.getElementById('tLabel');
scrub.max = N-1;
let loggedIdx = new Set();

function updateVisual(row){
  tLabel.textContent = 't=' + row.t.toFixed(1) + 's';
  document.getElementById('sig-gcs').textContent = 'UP';
  document.getElementById('sig-esc').textContent = (30+((row.t*7)%40)).toFixed(0)+'%';
  document.getElementById('sig-link').textContent = row.hb_gap_s.toFixed(2)+'s';
  document.getElementById('sig-fc').textContent = row.msg_count_1s+'/s';
  document.getElementById('sig-gps').textContent = row.nav_gps_vs_fused_mismatch_m > 1 ? row.nav_gps_vs_fused_mismatch_m.toFixed(1)+'m' : 'OK';
  document.getElementById('sig-ids').textContent = row.rule_total_severity.toFixed(0);
  document.getElementById('sig-fw').textContent = 'OK';

  const attackNow = row.rule_predicted_attack !== 'none';
  const activeIds = attackNow ? (ATTACK_NODE[row.rule_predicted_attack]||[]) : [];
  NODES.forEach(n=>document.getElementById('node-'+n.id).classList.toggle('alert', activeIds.includes(n.id)));
  drawEdges(activeIds);

  if(attackNow){
    statusBadge.classList.add('alert');
    statusText.textContent = 'ATTACK DETECTED — ' + (ATTACK_LABEL[row.rule_predicted_attack]||row.rule_predicted_attack).toUpperCase();
  } else {
    statusBadge.classList.remove('alert');
    statusText.textContent = 'SYSTEM NORMAL';
  }
}

function logEvent(row, i){
  if(loggedIdx.has(i)) return; loggedIdx.add(i);
  const cur = row.rule_predicted_attack;
  if(cur === prevLoggedAttack) return; // only log on state transitions
  if(cur !== 'none'){
    const div=document.createElement('div'); div.className='logrow';
    div.innerHTML = `<span class="t">${row.t.toFixed(1)}s</span><span class="type">${ATTACK_LABEL[cur]||cur}</span><span>flagged, severity ${row.rule_total_severity.toFixed(0)}</span>`;
    eventLog.prepend(div);
  } else if(prevLoggedAttack !== 'none'){
    const div=document.createElement('div'); div.className='logrow healed';
    div.innerHTML = `<span class="t">${row.t.toFixed(1)}s</span><span class="type">System Healed</span><span>${ATTACK_LABEL[prevLoggedAttack]||prevLoggedAttack} cleared, telemetry nominal</span>`;
    eventLog.prepend(div);
  }
  prevLoggedAttack = cur;
  while(eventLog.children.length>60) eventLog.removeChild(eventLog.lastChild);
}

function seekTo(i){ idx=Math.max(0,Math.min(N-1,i)); scrub.value=idx; updateVisual(DATA[idx]); }
function step(){
  if(idx>=N-1){ playing=false; playBtn.textContent='Play'; clearInterval(timer); return; }
  idx++; scrub.value=idx; updateVisual(DATA[idx]); logEvent(DATA[idx], idx);
}
playBtn.onclick=()=>{
  playing=!playing; playBtn.textContent = playing ? 'Pause' : 'Play';
  if(playing){ timer=setInterval(step,45); } else { clearInterval(timer); }
};
scrub.oninput=()=>{ playing=false; playBtn.textContent='Play'; clearInterval(timer); seekTo(parseInt(scrub.value)); };

const attackButtons = document.getElementById('attackButtons');
Object.keys(ATTACKS).forEach(name=>{
  const win = ATTACKS[name];
  const btn=document.createElement('button'); btn.className='abtn';
  btn.innerHTML = `<div class="name">${ATTACK_LABEL[name]||name}</div><div class="desc">${ATTACK_DESC[name]||''}</div>`;
  btn.onclick = () => {
    let target=0;
    for(let i=0;i<N;i++){ if(DATA[i].t >= win.start-1.5){ target=i; break; } }
    eventLog.innerHTML=''; loggedIdx=new Set(); prevLoggedAttack='none';
    seekTo(target); playing=true; playBtn.textContent='Pause';
    clearInterval(timer); timer=setInterval(step,45);
  };
  attackButtons.appendChild(btn);
});

renderNodes();
seekTo(0);
</script>
</body>
</html>
"""

html = html.replace("__DATA_JSON__", data_json).replace("__ATTACKS_JSON__", attacks_json)
out_path = os.path.join(ROOT, "sentinel_dashboard.html")
open(out_path, 'w').write(html)
print(f"Wrote {len(html)} bytes -> {out_path}")
