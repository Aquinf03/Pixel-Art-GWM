"""Serve the world model so you can play it in a browser.

Keys go in, imagined frames come out. After the first 8 context frames nothing
real is ever shown again -- every pixel is the model's own prediction, fed back
into itself.
"""

from __future__ import annotations

from pathlib import Path

import modal

_TOK = Path(__file__).resolve().parent / "tokenizer"

app = modal.App("pixel-world-live")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "numpy==2.*", "pillow", "fastapi[standard]")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(_TOK), "/root/tok")
)
data_vol = modal.Volume.from_name("pixel-world-data", create_if_missing=True)
run_vol = modal.Volume.from_name("d4-runs", create_if_missing=True)
DATA, RUNS = "/data", "/runs"

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Pixel World -- live</title><!--v2--><style>
 body{background:#0d1219;color:#dfe7ee;font:14px ui-monospace,Menlo,monospace;
      margin:0;display:flex;flex-direction:column;align-items:center;gap:12px;padding:24px}
 h1{font-size:15px;letter-spacing:.14em;text-transform:uppercase;color:#e8843e;margin:0}
 canvas{image-rendering:pixelated;width:512px;height:512px;background:#1a2634;
        border:1px solid #223040}
 .row{display:flex;gap:20px;align-items:center;color:#8fa4b4;font-size:12px}
 kbd{background:#1b2733;border:1px solid #2b3b4a;border-radius:3px;padding:1px 6px;color:#dfe7ee}
 #act{color:#e8843e;min-width:74px;display:inline-block}
 #busy{color:#7d8fa0}
 button{background:#1b2733;color:#dfe7ee;border:1px solid #2b3b4a;border-radius:3px;
        padding:5px 11px;font:inherit;cursor:pointer}
 button.on{background:#e8843e;color:#101820;border-color:#e8843e}
 .pad{display:grid;grid-template-columns:repeat(3,56px);gap:6px;justify-content:center}
 .pad button{padding:9px 0}
</style></head><body>
<h1>Pixel World &mdash; imagined live</h1>
<canvas id=c width=128 height=128></canvas>
<div class=row><span>action <b id=act>none</b></span><span>frame <b id=n>0</b></span>
<span id=busy></span><button onclick="reset()">reset</button></div>
<div class=row id=score>
 <span>score <b id=pts>0</b></span><span>coins <b id=coins>4</b>/4</span>
 <span>crate <b id=crate>-</b></span><span>door <b id=door>-</b></span></div>
<div class=row>speed <input id=fps type=range min=1 max=20 value=3 style="width:150px">
 <b id=fpsv>3</b> fps</div>
<div class=pad>
 <div></div><button data-a=jump>UP</button><div></div>
 <button data-a=left>LEFT</button><button data-a=use>USE</button><button data-a=right>RIGHT</button>
 <div></div><button data-a=crouch>DOWN</button><div></div>
</div>
<div class=row><kbd>left</kbd><kbd>right</kbd> move &nbsp; <kbd>up</kbd> jump &nbsp;
<kbd>down</kbd> crouch &nbsp; <kbd>space</kbd> use</div>
<div class=row style="max-width:530px;text-align:center;line-height:1.55;color:#7d8fa0">
The world only advances while you hold a key or a button. Release and it freezes.
Every frame after the first is the model's own prediction fed back into itself.</div>
<script>
const c=document.getElementById('c'),x=c.getContext('2d');
let sid=Math.random().toString(36).slice(2), held=null, n=0, busy=false;
const MAP={ArrowLeft:'left',ArrowRight:'right',ArrowUp:'jump',ArrowDown:'crouch',' ':'use'};
function setHeld(a){held=a;document.getElementById('act').textContent=a||'none';
  document.querySelectorAll('.pad button').forEach(b=>b.classList.toggle('on',b.dataset.a===a));}
addEventListener('keydown',e=>{if(MAP[e.key]){setHeld(MAP[e.key]);e.preventDefault();}});
addEventListener('keyup',e=>{if(MAP[e.key]===held)setHeld(null);});
document.querySelectorAll('.pad button').forEach(b=>{
  const a=b.dataset.a;
  b.addEventListener('mousedown',()=>setHeld(a));
  b.addEventListener('touchstart',e=>{e.preventDefault();setHeld(a);});
  ['mouseup','mouseleave','touchend'].forEach(ev=>b.addEventListener(ev,()=>{if(held===a)setHeld(null);}));
});
function draw(b64){const i=new Image();i.onload=()=>x.drawImage(i,0,0);i.src='data:image/png;base64,'+b64;}
let queue=[], sq=[], playFps=3, playTimer=null, best=0;
function showScore(s){ if(!s) return;
  best=Math.max(best,s.points);
  document.getElementById('pts').textContent=s.points+(best>s.points?'  (best '+best+')':'');
  document.getElementById('coins').textContent=s.coins_left;
  document.getElementById('crate').textContent=s.crate_broken?'broken':'intact';
  document.getElementById('door').textContent=s.door_open?'OPEN':'shut'; }
function setFps(f){ playFps=f;
  document.getElementById('fpsv').textContent=f;
  if(playTimer) clearInterval(playTimer);
  playTimer=setInterval(()=>{ if(queue.length){ draw(queue.shift());
    showScore(sq.shift());
    document.getElementById('n').textContent=++n; } }, 1000/playFps); }
document.getElementById('fps').addEventListener('input',e=>setFps(+e.target.value));
setFps(3);   // slow by default -- easier to see what the model is doing
async function reset(){setHeld(null);const r=await fetch('reset?sid='+sid,{method:'POST'});
  const j=await r.json();n=0;queue=[];sq=[];best=0;
  document.getElementById('n').textContent=0;draw(j.frame);showScore(j.score);}
async function tick(){
  if(busy||!held||queue.length>Math.max(4,playFps)) return;   // keep the queue shallow so input stays responsive
  busy=true; document.getElementById('busy').textContent='imagining...';
  try{const r=await fetch('step?sid='+sid+'&a='+held+'&n='+Math.max(3,Math.round(playFps)),{method:'POST'});
      const j=await r.json(); queue.push(...j.frames);
      if(j.scores) sq.push(...j.scores);}catch(e){}
  document.getElementById('busy').textContent=''; busy=false; }
reset(); setInterval(tick,50);   // burst HTTP: ~22 fps, above the env's native 15
</script></body></html>"""


# min_containers=0: a warm A10G costs ~$26/day whether or not anyone plays.
# Cold start is ~30 s on the first request; after that it stays warm while in use.
@app.cls(image=image, volumes={DATA: data_vol, RUNS: run_vol}, gpu="A10G",
         scaledown_window=300, timeout=1800)
class Live:
    @modal.enter()
    def load(self):
        import glob, sys
        import numpy as np, torch
        sys.path.insert(0, "/root/tok")
        from model import PaletteTokenizer, palette_tensor
        from dynamics import FlowDynamics

        self.torch, self.np = torch, np
        dev = torch.device("cuda")
        self.dev = dev
        tck = torch.load(f"{RUNS}/pal_v2/latest.pt", map_location=dev, weights_only=False)
        ta = tck["args"]
        self.tok = PaletteTokenizer(ta["width"], ta["z_ch"], ta["downs"]).to(dev)
        self.tok.load_state_dict(tck["model"]); self.tok.eval()
        dck = torch.load(f"{RUNS}/dyn_v2/latest.pt", map_location=dev, weights_only=False)
        da = dck["args"]
        self.dyn = FlowDynamics(ta["z_ch"], da["ctx"], 16, da["width"], da["depth"]).to(dev)
        self.dyn.load_state_dict(dck["model"]); self.dyn.eval()
        self.pal = palette_tensor(dev)
        self.ctx = da["ctx"]
        d = np.load(sorted(glob.glob(f"{DATA}/val/shard_*.npz"))[0])
        self.fr, self.ep, self.term = d["frames"], d["ep_id"], d["terminal"]
        self.sessions = {}

    # The env's own reward layout. Score is read out of the imagined frame by
    # checking whether each coin is still drawn -- no reward head needed, and
    # it scores what the MODEL believes happened, not the real environment.
    COIN_SPOTS = [(12, 106), (75, 78), (32, 50), (102, 22)]
    COIN_RGB = [(246, 206, 84), (255, 244, 190)]
    CRATE_SPOT, CRATE_RGB = (96, 104), [(146, 100, 58), (98, 66, 38)]
    DOOR_SPOT, DOOR_OPEN_RGB = (118, 102), (206, 190, 236)

    def _score(self, chw01):
        """(points, coins_left, crate_broken, door_open) from one frame."""
        import numpy as np
        a = (chw01.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

        def near(cx, cy, rgbs, r=4):
            y0, y1 = max(0, cy - r), min(a.shape[0], cy + r + 1)
            x0, x1 = max(0, cx - r), min(a.shape[1], cx + r + 1)
            patch = a[y0:y1, x0:x1].reshape(-1, 3).astype(np.int16)
            for rgb in rgbs:
                if (np.abs(patch - np.array(rgb, np.int16)).sum(1) < 24).any():
                    return True
            return False

        coins_left = sum(near(cx, cy, self.COIN_RGB) for cx, cy in self.COIN_SPOTS)
        crate = near(*self.CRATE_SPOT, self.CRATE_RGB)
        door_open = near(*self.DOOR_SPOT, [self.DOOR_OPEN_RGB])
        pts = (len(self.COIN_SPOTS) - coins_left) * 1 + (0 if crate else 2) + (5 if door_open else 0)
        return {"points": int(pts), "coins_left": int(coins_left),
                "crate_broken": bool(not crate), "door_open": bool(door_open)}

    def _png(self, chw01):
        import base64, io
        from PIL import Image
        a = (chw01.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        b = io.BytesIO(); Image.fromarray(a).save(b, "PNG")
        return base64.b64encode(b.getvalue()).decode()

    @modal.asgi_app()
    def web(self):
        # ONE function serves the page and the API. Separate endpoints are
        # separate Modal functions and do not share the container holding
        # session state, which is why the first attempt 500'd on /step.
        from fastapi import FastAPI, WebSocket
        from fastapi.responses import HTMLResponse
        api = FastAPI()

        @api.get("/")
        def page():
            return HTMLResponse(PAGE)

        @api.post("/reset")
        def _reset(sid: str):
            return self.do_reset(sid)

        @api.websocket("/ws")
        async def _ws(ws: WebSocket):
            """Persistent connection: ~1.1s of per-request HTTP overhead was
            94% of the cost. Over a socket the client streams actions and the
            server streams frames, so latency is one frame, not one request."""
            import asyncio
            await ws.accept()
            sid = f"ws{id(ws)}"
            self.do_reset(sid)
            action = {"a": "idle"}

            async def reader():
                try:
                    while True:
                        m = await ws.receive_json()
                        action["a"] = m.get("a", "idle")
                except Exception:
                    action["a"] = None            # closed

            task = asyncio.create_task(reader())
            try:
                while action["a"] is not None:
                    a = action["a"]
                    if a == "idle":               # do not advance without input
                        await asyncio.sleep(0.03)
                        continue
                    out = self.do_step(sid, a, 1)
                    await ws.send_json({"frame": out["frames"][0]})
                    await asyncio.sleep(0)
            except Exception:
                pass
            finally:
                task.cancel()
                self.sessions.pop(sid, None)

        @api.post("/step")
        def _step(sid: str, a: str = "idle", n: int = 8):
            # A burst, not a single frame. The model runs at ~93 fps on-GPU but
            # a per-frame HTTP round trip capped the browser at 0.93 fps -- a
            # 100x gap that was entirely transport, not compute.
            return self.do_step(sid, a, n)

        return api

    def do_reset(self, sid: str):
        import sys
        sys.path.insert(0, "/root/tok")
        from model import to_rgb
        torch, np = self.torch, self.np
        ctx = self.ctx
        s = int(np.random.default_rng().integers(ctx, len(self.fr) - 4))
        while not (self.ep[s] == self.ep[s - ctx] and not self.term[s - ctx:s].any()):
            s = int(np.random.default_rng().integers(ctx, len(self.fr) - 4))
        x = torch.from_numpy(self.fr[np.arange(s - ctx, s)]).to(self.dev)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            past = self.tok.encoder(x).unsqueeze(0)
            frame = to_rgb(self.tok.decoder(past[:, -1]).argmax(1), self.pal)[0]
        self.sessions[sid] = past
        return {"frame": self._png(frame), "score": self._score(frame)}

    def do_step(self, sid: str, a: str = "idle", n: int = 8):
        import sys
        sys.path.insert(0, "/root/tok")
        from model import to_rgb
        from dynamics import sample_next
        torch = self.torch
        if sid not in self.sessions:
            self.do_reset(sid)
        past = self.sessions[sid]
        v = {"idle": (0, 0, 0), "left": (-1, 0, 0), "right": (1, 0, 0),
             "jump": (0, 1, 0), "crouch": (0, -1, 0), "use": (0, 0, 1)}.get(a, (0, 0, 0))
        act = torch.zeros(1, 16, device=self.dev)
        act[0, 0], act[0, 1], act[0, 2] = v
        import time
        n = max(1, min(int(n), 32))
        t0 = time.time()
        imgs = []
        with torch.no_grad():
            for _ in range(n):
                z = sample_next(self.dyn, past, act, steps=4)
                past = torch.cat([past[:, 1:], z.unsqueeze(1)], 1)
                imgs.append(to_rgb(self.tok.decoder(z).argmax(1), self.pal)[0])
        torch.cuda.synchronize()
        t_gpu = time.time() - t0
        t1 = time.time()
        frames = [self._png(i) for i in imgs]
        scores = [self._score(i) for i in imgs]
        t_enc = time.time() - t1
        self.sessions[sid] = past
        return {"frames": frames, "scores": scores,
                "ms_gpu": round(t_gpu * 1000, 1),
                "ms_encode": round(t_enc * 1000, 1),
                "n": n}
