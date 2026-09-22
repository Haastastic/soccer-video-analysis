"""Local browser UI for placing pitch-calibration anchor points and saving pitch_anchors.local.json.

Click landmarks on a frame from the run's clip, pair each with a pitch coordinate in meters (quick-pick chips
for the standard Law-of-the-Game points, or type your own), and save. Writes the same anchor format
pitch_calibrate.py reads, so `pitch_calibrate.py apply` (or the Save & Apply button here) can run right after.

Runs a local-only HTTP server on 127.0.0.1 with no external network calls (no CDN scripts, no fonts) because
the frames it serves show minors. Nothing here is written outside the run folder or the anchors file.

Usage:
  python pitch_anchor_ui.py --run data\\clipB
  python pitch_anchor_ui.py --run data\\clipB --anchors-out data\\clipB\\pitch_anchors.local.json --port 8765
"""

import argparse
import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from pitch_calibrate import ANCHORS_FILE, apply_calibration, read_frame, solve_anchor
from sv_common import require_under_data

FRAME_RE = re.compile(r"^anchor_frame_-?[0-9.]+s\.png$")


def list_frames(run: Path) -> list:
    out = []
    for p in sorted(run.glob("anchor_frame_*s.png")):
        m = re.match(r"^anchor_frame_(-?[0-9.]+)s\.png$", p.name)
        if m:
            out.append({"filename": p.name, "time_s": float(m.group(1))})
    return out


def load_existing_anchors(path: Path) -> list:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("anchors", [])
    except (json.JSONDecodeError, OSError):
        return []


def orientation_warnings(anchors: list) -> list:
    """Flag anchors whose homography has the opposite orientation from the majority.

    A fixed sideline camera should map pitch to image with the same handedness in every anchor. A lone anchor
    with the opposite sign usually means a landmark was measured from the wrong goal (e.g. the far one instead
    of the one used as the shared X=0 reference) or a near/far touchline mixup, not a camera-motion problem.
    """
    signs = []
    for a in anchors:
        h, _ = solve_anchor(a["points"])
        signs.append(1 if np.linalg.det(h[:2, :2]) >= 0 else -1)
    if len(set(signs)) < 2:
        return []
    majority = 1 if sum(signs) >= 0 else -1
    return [
        f"anchor at {a['time_s']}s has the opposite homography orientation from the other anchors - "
        "check it uses the same reference goal and near/far touchline as the rest"
        for a, s in zip(anchors, signs, strict=True)
        if s != majority
    ]


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Pitch anchor calibration</title>
<style>
  body { font: 14px/1.4 system-ui, sans-serif; margin: 0; padding: 16px; background: #1b1b1f; color: #e8e8ea; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color: #9a9aa2; margin-bottom: 14px; }
  .layout { display: flex; gap: 16px; align-items: flex-start; }
  .imgcol { flex: 1 1 auto; min-width: 0; }
  .sidecol { flex: 0 0 420px; }
  #wrap { position: relative; overflow: auto; max-height: 78vh; max-width: 100%; border: 1px solid #3a3a42;
    background: #000; display: inline-block; }
  #wrap img { display: block; image-rendering: pixelated; }
  #wrap canvas { display: block; position: absolute; top: 0; left: 0; image-rendering: pixelated;
    cursor: crosshair; }
  .row { display: flex; gap: 6px; align-items: center; margin: 6px 0; flex-wrap: wrap; }
  select, input[type=text], input[type=number] { background: #26262c; color: #e8e8ea; border: 1px solid #444;
    border-radius: 4px; padding: 4px 6px; }
  input[type=number] { width: 72px; }
  button { background: #3a3a44; color: #e8e8ea; border: 1px solid #555; border-radius: 4px; padding: 5px 10px;
    cursor: pointer; }
  button:hover { background: #46464f; }
  button.primary { background: #2f5fb8; border-color: #2f5fb8; }
  button.primary:hover { background: #3a6fce; }
  button.armed { background: #b8752f; border-color: #b8752f; }
  button.chip { font-size: 12px; padding: 4px 8px; }
  table { border-collapse: collapse; width: 100%; margin: 6px 0; }
  th, td { border-bottom: 1px solid #333; padding: 3px 5px; text-align: left; font-size: 12px; }
  fieldset { border: 1px solid #3a3a42; border-radius: 6px; margin: 10px 0; }
  legend { color: #9a9aa2; padding: 0 6px; }
  .rms-ok { color: #6fce6a; }
  .rms-warn { color: #e0c34a; }
  .rms-bad { color: #e06a6a; }
  .muted { color: #9a9aa2; font-size: 12px; }
  .warn-box { background: #3a2020; border: 1px solid #e06a6a; color: #f0a0a0; border-radius: 6px; padding: 8px;
    margin: 8px 0; font-size: 12px; }
  pre { background: #111; padding: 8px; border-radius: 6px; overflow: auto; max-height: 260px; font-size: 12px; }
</style></head>
<body>
<h1>Pitch anchor calibration</h1>
<div class="sub">run: <b id="runPath"></b> &middot; anchors will be saved to: <b id="anchorsOut"></b> &middot;
  local only &mdash; nothing here leaves this machine</div>

<div class="layout">
  <div class="imgcol">
    <div class="row">
      <label>Frame: <select id="frameSelect"></select></label>
      <span class="muted">or extract a new one at</span>
      <input type="number" id="newTime" step="0.1" placeholder="seconds">
      <button id="extractBtn">Extract</button>
      <span class="muted">Zoom:</span>
      <button data-zoom="0.5">50%</button>
      <button data-zoom="1">100%</button>
      <button data-zoom="1.5">150%</button>
      <button data-zoom="2">200%</button>
      <button data-zoom="3">300%</button>
    </div>
    <div id="wrap"><img id="frameImg"><canvas id="overlay"></canvas></div>
    <div class="muted" id="cursorReadout">click the image to place the armed landmark, or a free point</div>
  </div>

  <div class="sidecol">
    <fieldset>
      <legend>Standard landmarks (Laws of the Game, meters)</legend>
      <div class="row" id="chips"></div>
      <div class="muted">Arm one, then click its spot on the image. X = distance from goal line, Y = across the
        field. "Near" means the touchline closer to the camera in THIS shot, "far" the one farther away - since
        the camera stays on one side of the pitch all game, that's the same physical side in every frame, so you
        never have to remember an abstract left/right or +/- convention.</div>
    </fieldset>

    <fieldset>
      <legend>Points on this frame</legend>
      <table id="pointsTable"><thead><tr>
        <th>name</th><th>img x</th><th>img y</th><th>pitch X</th><th>pitch Y</th><th></th>
      </tr></thead><tbody></tbody></table>
      <div id="rmsReadout" class="muted">need at least 4 points</div>
    </fieldset>

    <fieldset>
      <legend>Anchor frames</legend>
      <table id="anchorsTable"><thead><tr><th>time (s)</th><th>points</th><th>RMS (m)</th><th></th></tr></thead>
      <tbody></tbody></table>
    </fieldset>

    <div class="row">
      <button class="primary" id="saveBtn">Save anchors</button>
      <button class="primary" id="saveApplyBtn">Save &amp; run apply</button>
    </div>
    <div id="saveWarnings"></div>
    <pre id="result" style="display:none"></pre>
  </div>
</div>

<script>
// Y > 0 is the touchline nearer the camera, Y < 0 the far one. The camera sits on one fixed sideline all game,
// so "near" and "far" are the same physical side in every frame - unlike +Y/-Y, which needs a memorized sign.
const LANDMARKS = [
  {name: "goalpost, near touchline", pitch: [0, 3.66]},
  {name: "goalpost, far touchline", pitch: [0, -3.66]},
  {name: "six-yard corner, near touchline", pitch: [5.5, 9.16]},
  {name: "six-yard corner, far touchline", pitch: [5.5, -9.16]},
  {name: "penalty spot", pitch: [11, 0]},
  {name: "eighteen-yard corner, near touchline", pitch: [16.5, 20.16]},
  {name: "eighteen-yard corner, far touchline", pitch: [16.5, -20.16]},
  {name: "arc tangent, near touchline", pitch: [16.5, 7.31]},
  {name: "arc tangent, far touchline", pitch: [16.5, -7.31]},
  {name: "arc apex", pitch: [20.15, 0]},
];

let state = { frames: {} };  // time_s (string key) -> {points: [{name, img:[x,y], pitch:[x,y]}]}
let currentTime = null;
let armed = null;
let zoom = 1;
let natW = 0, natH = 0;
let rmsTimer = null;

const img = document.getElementById("frameImg");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const wrap = document.getElementById("wrap");

function api(path, body) {
  const opts = body
    ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)}
    : {};
  return fetch(path, opts).then(r => r.json().then(j => ({ok: r.ok, data: j})));
}

function curFrame() {
  if (currentTime === null) return null;
  const key = String(currentTime);
  if (!state.frames[key]) state.frames[key] = {points: []};
  return state.frames[key];
}

function renderChips() {
  const wrapEl = document.getElementById("chips");
  wrapEl.innerHTML = "";
  LANDMARKS.forEach((lm, i) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = lm.name + " (" + lm.pitch[0] + ", " + lm.pitch[1] + ")";
    b.onclick = () => {
      armed = (armed === i) ? null : i;
      renderChips();
    };
    if (armed === i) b.classList.add("armed");
    wrapEl.appendChild(b);
  });
}

function setImageSize() {
  const w = Math.round(natW * zoom), h = Math.round(natH * zoom);
  img.style.width = w + "px"; img.style.height = h + "px";
  canvas.style.width = w + "px"; canvas.style.height = h + "px";
}

function loadFrameImage(filename, time_s) {
  currentTime = time_s;
  img.onload = () => {
    natW = img.naturalWidth; natH = img.naturalHeight;
    canvas.width = natW; canvas.height = natH;
    setImageSize();
    redraw();
  };
  img.src = "/image/" + filename + "?t=" + Date.now();
}

function toImagePos(e) {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
  return [Math.round((e.clientX - rect.left) * sx), Math.round((e.clientY - rect.top) * sy)];
}

function redraw(cursor) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const f = curFrame();
  if (f) {
    f.points.forEach(p => {
      ctx.strokeStyle = "#ff5050"; ctx.fillStyle = "#ff5050"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(p.img[0], p.img[1], 5, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(p.img[0] - 8, p.img[1]); ctx.lineTo(p.img[0] + 8, p.img[1]); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(p.img[0], p.img[1] - 8); ctx.lineTo(p.img[0], p.img[1] + 8); ctx.stroke();
      ctx.font = "13px monospace";
      ctx.fillText(p.name, p.img[0] + 8, p.img[1] - 8);
    });
  }
  if (cursor) {
    ctx.strokeStyle = "rgba(255,255,0,0.6)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cursor[0], 0); ctx.lineTo(cursor[0], canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, cursor[1]); ctx.lineTo(canvas.width, cursor[1]); ctx.stroke();
  }
}

canvas.addEventListener("mousemove", e => {
  const pos = toImagePos(e);
  document.getElementById("cursorReadout").textContent = "image (" + pos[0] + ", " + pos[1] + ")";
  redraw(pos);
});

canvas.addEventListener("click", e => {
  const pos = toImagePos(e);
  const f = curFrame();
  if (!f) return;
  let name, pitch;
  if (armed !== null) {
    name = LANDMARKS[armed].name; pitch = LANDMARKS[armed].pitch.slice();
    armed = null; renderChips();
  } else {
    name = "point " + (f.points.length + 1); pitch = [null, null];
  }
  f.points.push({name, img: pos, pitch});
  renderPoints();
  redraw(pos);
});

function renderPoints() {
  const f = curFrame();
  const tbody = document.querySelector("#pointsTable tbody");
  tbody.innerHTML = "";
  if (!f) return;
  f.points.forEach((p, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><input type="text" value="${p.name}" data-k="name"></td>
      <td><input type="number" value="${p.img[0]}" data-k="ix"></td>
      <td><input type="number" value="${p.img[1]}" data-k="iy"></td>
      <td><input type="number" step="0.01" value="${p.pitch[0] ?? ''}" data-k="px"></td>
      <td><input type="number" step="0.01" value="${p.pitch[1] ?? ''}" data-k="py"></td>
      <td><button data-k="del">x</button></td>`;
    tr.querySelectorAll("input").forEach(inp => inp.addEventListener("input", () => {
      const k = inp.dataset.k, v = inp.value;
      if (k === "name") p.name = v;
      else if (k === "ix") p.img[0] = parseFloat(v);
      else if (k === "iy") p.img[1] = parseFloat(v);
      else if (k === "px") p.pitch[0] = v === "" ? null : parseFloat(v);
      else if (k === "py") p.pitch[1] = v === "" ? null : parseFloat(v);
      redraw(); scheduleValidate();
    }));
    tr.querySelector('[data-k="del"]').addEventListener("click", () => {
      f.points.splice(i, 1); renderPoints(); redraw(); scheduleValidate();
    });
    tbody.appendChild(tr);
  });
  scheduleValidate();
  renderAnchorsTable();
}

function scheduleValidate() {
  clearTimeout(rmsTimer);
  rmsTimer = setTimeout(validateCurrent, 300);
}

function usablePoints(f) {
  return f.points.filter(p => Number.isFinite(p.img[0]) && Number.isFinite(p.img[1])
    && Number.isFinite(p.pitch[0]) && Number.isFinite(p.pitch[1]));
}

async function validateCurrent() {
  const f = curFrame();
  const el = document.getElementById("rmsReadout");
  if (!f) return;
  const pts = usablePoints(f);
  if (pts.length < 4) {
    el.textContent = `need at least 4 usable points (have ${pts.length})`; el.className = "muted"; return;
  }
  const {ok, data} = await api("/api/validate", {points: pts.map(p => ({img: p.img, pitch: p.pitch}))});
  if (ok) {
    el.textContent = `reprojection RMS: ${data.rms.toFixed(3)} m`;
    el.className = data.rms < 0.5 ? "rms-ok" : (data.rms < 1.5 ? "rms-warn" : "rms-bad");
  } else {
    el.textContent = "error: " + data.error; el.className = "rms-bad";
  }
}

function renderAnchorsTable() {
  const tbody = document.querySelector("#anchorsTable tbody");
  tbody.innerHTML = "";
  Object.keys(state.frames).sort((a, b) => parseFloat(a) - parseFloat(b)).forEach(key => {
    const f = state.frames[key];
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${key}</td><td>${f.points.length}</td><td>-</td><td></td>`;
    const btn = document.createElement("button");
    btn.textContent = "select"; btn.onclick = () => selectFrameByTime(parseFloat(key));
    const del = document.createElement("button");
    del.textContent = "remove"; del.onclick = () => { delete state.frames[key]; renderAnchorsTable(); };
    tr.lastElementChild.appendChild(btn); tr.lastElementChild.appendChild(del);
    tbody.appendChild(tr);
  });
}

function selectFrameByTime(time_s) {
  const opts = [...document.getElementById("frameSelect").options];
  const opt = opts.find(o => Math.abs(parseFloat(o.dataset.time) - time_s) < 1e-6);
  if (opt) {
    document.getElementById("frameSelect").value = opt.value;
    loadFrameImage(opt.value, time_s);
    renderPoints();
  }
}

document.querySelectorAll("button[data-zoom]").forEach(b => b.addEventListener("click", () => {
  zoom = parseFloat(b.dataset.zoom); setImageSize();
}));

document.getElementById("frameSelect").addEventListener("change", e => {
  const opt = e.target.selectedOptions[0];
  loadFrameImage(opt.value, parseFloat(opt.dataset.time));
  renderPoints();
});

document.getElementById("extractBtn").addEventListener("click", async () => {
  const t = parseFloat(document.getElementById("newTime").value);
  if (!Number.isFinite(t)) { alert("enter a time in seconds"); return; }
  const {ok, data} = await api("/api/extract", {time_s: t});
  if (!ok) { alert("extract failed: " + data.error); return; }
  addFrameOption(data.filename, data.time_s, true);
});

function addFrameOption(filename, time_s, select) {
  const sel = document.getElementById("frameSelect");
  if (![...sel.options].some(o => o.value === filename)) {
    const opt = document.createElement("option");
    opt.value = filename; opt.dataset.time = time_s; opt.textContent = `${time_s}s (${filename})`;
    sel.appendChild(opt);
  }
  if (select) { sel.value = filename; loadFrameImage(filename, time_s); renderPoints(); }
}

async function doSave() {
  const anchors = Object.keys(state.frames).map(key => ({
    time_s: parseFloat(key),
    points: usablePoints(state.frames[key]).map(p => ({name: p.name, img: p.img, pitch: p.pitch})),
  })).filter(a => a.points.length >= 4);
  if (!anchors.length) { alert("no anchor frame has 4 or more usable points yet"); return null; }
  const {ok, data} = await api("/api/save", {anchors});
  const pre = document.getElementById("result");
  pre.style.display = "block";
  pre.textContent = ok ? JSON.stringify(data, null, 2) : "save failed: " + data.error;
  const warnBox = document.getElementById("saveWarnings");
  warnBox.innerHTML = "";
  if (ok && data.warnings && data.warnings.length) {
    const box = document.createElement("div");
    box.className = "warn-box";
    box.textContent = "⚠ " + data.warnings.join(" — ");
    warnBox.appendChild(box);
  }
  if (ok) renderAnchorsTableWithRms(data.anchor_fit_rms_m, anchors);
  return ok ? data : null;
}

function renderAnchorsTableWithRms(rmsList, anchors) {
  const tbody = document.querySelector("#anchorsTable tbody");
  [...tbody.children].forEach((tr, i) => {
    const a = anchors[i];
    if (a && rmsList[i] !== undefined) tr.children[2].textContent = rmsList[i].toFixed(3);
  });
}

document.getElementById("saveBtn").addEventListener("click", doSave);
document.getElementById("saveApplyBtn").addEventListener("click", async () => {
  const saved = await doSave();
  if (!saved) return;
  const {ok, data} = await api("/api/apply", {});
  const pre = document.getElementById("result");
  pre.style.display = "block";
  pre.textContent = ok ? JSON.stringify(data, null, 2) : "apply failed: " + data.error;
});

async function init() {
  const {data} = await api("/api/state");
  document.getElementById("runPath").textContent = data.run;
  document.getElementById("anchorsOut").textContent = data.anchors_out;
  renderChips();
  const sel = document.getElementById("frameSelect");
  data.frames.forEach(f => addFrameOption(f.filename, f.time_s, false));
  data.existing_anchors.forEach(a => {
    const pts = a.points.map(p => ({name: p.name, img: p.img.slice(), pitch: p.pitch.slice()}));
    state.frames[String(a.time_s)] = {points: pts};
    if (!data.frames.some(f => Math.abs(f.time_s - a.time_s) < 1e-6)) {
      const opt = document.createElement("option");
      opt.value = "__missing__"; opt.dataset.time = a.time_s;
      opt.textContent = `${a.time_s}s (no frame file - extract it)`;
      sel.appendChild(opt);
    }
  });
  renderAnchorsTable();
  if (sel.options.length && sel.options[0].value !== "__missing__") {
    sel.value = sel.options[0].value;
    loadFrameImage(sel.options[0].value, parseFloat(sel.options[0].dataset.time));
  } else if (sel.options.length) {
    currentTime = parseFloat(sel.options[0].dataset.time);
  }
  renderPoints();
}
init();
</script>
</body></html>
"""


def make_handler(run: Path, anchors_out: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass  # quiet; this is a local tool, not a service

        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/state":
                self._send_json(
                    {
                        "run": str(run),
                        "anchors_out": str(anchors_out),
                        "frames": list_frames(run),
                        "existing_anchors": load_existing_anchors(anchors_out),
                    }
                )
            elif path.startswith("/image/"):
                name = path[len("/image/") :]
                if not FRAME_RE.match(name):
                    self._send_json({"error": "bad filename"}, 400)
                    return
                fp = run / name
                if not fp.exists():
                    self._send_json({"error": "not found"}, 404)
                    return
                data = fp.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                if path == "/api/extract":
                    body = self._read_json()
                    time_s = float(body["time_s"])
                    img = read_frame(run, time_s)
                    dest = run / f"anchor_frame_{time_s:g}s.png"
                    cv2.imwrite(str(dest), img)
                    self._send_json({"filename": dest.name, "time_s": time_s})
                elif path == "/api/validate":
                    body = self._read_json()
                    _, rms = solve_anchor(body["points"])
                    self._send_json({"rms": rms})
                elif path == "/api/save":
                    body = self._read_json()
                    anchors = body["anchors"]
                    rms_list = []
                    for a in anchors:
                        _, rms = solve_anchor(a["points"])
                        rms_list.append(round(rms, 4))
                    anchors_out.write_text(json.dumps({"anchors": anchors}, indent=2))
                    from pitch_calibrate import Calibration
                    from sv_common import Cache

                    cross_check = []
                    if len(anchors) >= 2:
                        cal = Calibration(Cache(run / "cache"), anchors)
                        cross_check = cal.cross_check(anchors)
                    self._send_json(
                        {
                            "saved_to": str(anchors_out),
                            "anchor_fit_rms_m": rms_list,
                            "cross_check": cross_check,
                            "warnings": orientation_warnings(anchors),
                        }
                    )
                elif path == "/api/apply":
                    anchors = json.loads(anchors_out.read_text())["anchors"]
                    report = apply_calibration(run, anchors)
                    self._send_json(report)
                else:
                    self._send_json({"error": "not found"}, 404)
            except (ValueError, KeyError, OSError, FileNotFoundError) as e:
                self._send_json({"error": str(e)}, 400)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=require_under_data)
    ap.add_argument("--anchors-out", type=Path, default=ANCHORS_FILE, help="where to write the anchors JSON")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    args = ap.parse_args()

    if not (args.run / "clip.mp4").exists():
        raise SystemExit(f"{args.run / 'clip.mp4'} not found - run detect_cache.py on this run first.")
    if not (args.run / "cache" / "meta.json").exists():
        raise SystemExit(f"{args.run / 'cache'} has no cache - run detect_cache.py on this run first.")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.run, args.anchors_out))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving at {url} (local only). Frames show minors - do not port-forward or expose this.")
    print(f"Run: {args.run}  Anchors will be saved to: {args.anchors_out}")
    print("Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
