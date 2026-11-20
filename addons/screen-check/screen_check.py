"""Anthias screen check: flag broken screens from what the screen shows.

Takes screenshots from the viewer's cage compositor with grim (the
composited output, video included), runs the screen-state model, and
confirms a problem over time before alerting. Anthias itself is not
touched: the compositor's socket is reached through /proc.

Status: GET / (page), /status (JSON), /frame.jpg (last model input).
Environment: SCREEN_CHECK_FPS (2), SCREEN_CHECK_THREADS (1),
SCREEN_CHECK_WEBHOOK (POST on problem/cleared), SCREEN_CHECK_PORT
(8090), SCREEN_CHECK_SOURCE (testing: replay a folder of frames).

The model is not shipped with the code. On first start it is downloaded
from the private Hugging Face repo in settings.json (pinned revision,
SHA-256 checked) with HF_TOKEN, and kept in the data volume. Or point
SCREEN_CHECK_MODEL at a copy you already have.
"""

import glob
import hashlib
import io
import json
import logging
import math
import os
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).parent
CFG = json.loads((HERE / 'settings.json').read_text())
OUTCOMES = CFG['outcomes']
FPS = float(os.environ.get('SCREEN_CHECK_FPS', '2'))
THREADS = int(os.environ.get('SCREEN_CHECK_THREADS', '1'))
PORT = int(os.environ.get('SCREEN_CHECK_PORT', '8090'))
WEBHOOK = os.environ.get('SCREEN_CHECK_WEBHOOK')
SOURCE = os.environ.get('SCREEN_CHECK_SOURCE')
DATA = Path(os.environ.get('SCREEN_CHECK_DATA', '/data'))
ALERTS = DATA / 'alerts'
STILL = 2.0  # grey levels a 16x16 block may change and count as still
RERUN_S = 10.0  # run the model at least this often on a still screen

log = logging.getLogger('screen-check')
ort.set_default_logger_severity(3)  # no GPU-discovery warning on a Pi


# Capture


def wayland_socket() -> str:
    """The viewer's Wayland socket, via its /proc root (needs pid: host)."""
    for environ in glob.glob('/proc/[0-9]*/environ'):
        try:
            pairs = Path(environ).read_bytes().split(b'\0')
        except OSError:
            continue
        env = dict(
            p.decode(errors='ignore').split('=', 1) for p in pairs if b'=' in p
        )
        if 'WAYLAND_DISPLAY' in env and 'XDG_RUNTIME_DIR' in env:
            sock = (
                f'{environ[:-7]}root{env["XDG_RUNTIME_DIR"]}/'
                f'{env["WAYLAND_DISPLAY"]}'
            )
            if os.path.exists(sock):
                return sock
    raise RuntimeError('no Wayland compositor (is the viewer on cage?)')


def screenshots() -> Iterator[Image.Image | None]:
    """grim screenshots forever; None after a failed capture."""
    sock = None
    while True:
        try:
            if not (sock and os.path.exists(sock)):
                sock = wayland_socket()
                log.info('capturing from %s', sock)
            raw = subprocess.run(
                ['grim', '-t', 'ppm', '-'],
                capture_output=True,
                check=True,
                timeout=10,
                env={
                    **os.environ,
                    'WAYLAND_DISPLAY': sock,
                    'XDG_RUNTIME_DIR': '/tmp',
                },
            )
            w, h = map(int, raw.stdout[:32].split()[1:3])
            # P6 PPM: a short text header, then w*h*3 bytes of RGB
            pixels = raw.stdout[-w * h * 3 :]
            yield Image.frombuffer('RGB', (w, h), pixels, 'raw', 'RGB', 0, 1)
        except (OSError, RuntimeError, subprocess.SubprocessError) as e:
            log.warning('capture failed: %s', e)
            sock = None
            yield None


def replay(folder: str) -> Iterator[Image.Image]:
    for f in sorted(glob.glob(f'{folder}/*.[jp][pn]g')):
        yield Image.open(f).convert('RGB')


def shrink(img: Image.Image) -> np.ndarray:
    """Area-average to ~pixel_budget pixels, sides multiples of 32."""
    s = math.sqrt(CFG['pixel_budget'] / (img.width * img.height))
    size = (round(img.width * s / 32) * 32, round(img.height * s / 32) * 32)
    while img.width >= 2 * size[0] and img.height >= 2 * size[1]:
        img = img.reduce(2)  # fast exact halving first
    return np.asarray(img.resize(size, Image.Resampling.BOX))


# Model and frame combining


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def model_file() -> Path:
    """The model, checked against settings.json; downloaded if missing."""
    want = CFG['model_sha256']
    given = os.environ.get('SCREEN_CHECK_MODEL')
    if given:
        path = Path(given)
        if sha256(path) != want:
            raise RuntimeError(f'{path} is not the model in settings.json')
        return path
    rev = CFG['model_revision']
    path = DATA / 'model' / rev[:12] / CFG['model']
    if path.exists() and sha256(path) == want:
        return path
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError(
            f'no model yet: set HF_TOKEN (read access to {CFG["model_repo"]})'
        )
    url = (
        f'https://huggingface.co/{CFG["model_repo"]}/resolve/{rev}/'
        f'{CFG["model"]}'
    )
    log.info('downloading %s', url)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix('.part')
    req = urllib.request.Request(
        url, headers={'Authorization': f'Bearer {token}'}
    )
    try:
        with (
            urllib.request.urlopen(req, timeout=60) as r,
            part.open('wb') as f,
        ):
            while block := r.read(1 << 20):
                f.write(block)
    except OSError as e:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f'model download failed ({e}): check HF_TOKEN'
        ) from e
    if sha256(part) != want:
        part.unlink()
        raise RuntimeError('downloaded model does not match settings.json')
    part.replace(path)
    return path


class Model:
    def __init__(self, path: Path) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = THREADS, 1
        opts.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(path), opts)
        std = np.array(CFG['std'], np.float32)[:, None, None]
        self.scale = 1 / (255 * std)
        self.shift = np.array(CFG['mean'], np.float32)[:, None, None] / std

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        x = rgb.transpose(2, 0, 1).astype(np.float32)
        x *= self.scale
        x -= self.shift
        (probs,) = self.session.run(['probs'], {'image': x[None]})
        return np.asarray(probs[0])


def signature(luma: np.ndarray) -> np.ndarray:
    """32x18 points (2x2 means): the demo's screen-change measure."""
    h, w = luma.shape
    ys = np.minimum(h - 2, ((np.arange(18) + 0.5) * h / 18).astype(int))
    xs = np.minimum(w - 2, ((np.arange(32) + 0.5) * w / 32).astype(int))
    corners = (luma[ys + dy][:, xs + dx] for dy in (0, 1) for dx in (0, 1))
    return np.asarray(sum(corners) / 4)


def blocks(luma: np.ndarray) -> np.ndarray:
    """Mean of each 16x16 block, to spot any change at all."""
    h, w = (n // 16 * 16 for n in luma.shape)
    return np.asarray(
        luma[:h, :w].reshape(h // 16, 16, w // 16, 16).mean((1, 3))
    )


class Harness:
    """The demo's frame combining (tempoStep), with tuned settings."""

    def __init__(self, s: dict[str, Any]) -> None:
        self.s = s
        self.avg: np.ndarray | None = None
        self.cand: str | None = None
        self.outcome: str | None = None
        self.held = self.below = self.stuck = 0.0
        self.state = 'ok'

    def problem(self, p: np.ndarray) -> str | None:
        top = OUTCOMES[2 + int(p[2:].argmax())]
        thr = self.s['thr_by_type'].get(top, self.s['thr'])
        return top if 1 - p[0] - p[1] >= thr else None

    def step(self, p: np.ndarray, change: float, dt: float) -> None:
        s = self.s
        if self.avg is None:
            self.avg = p
        elif change >= s['cut'] or np.abs(p - self.avg).sum() >= s['jump']:
            self.avg, now = p, self.problem(p)  # a cut: start afresh
            if now != self.cand:
                self.held = 0.0
            if not now:
                self.cand = None
            if self.state == 'issue' and now != self.outcome:
                self.state, self.outcome = 'ok', None
        else:
            self.avg = self.avg + (1 - math.exp(-dt / s['tau'])) * (
                p - self.avg
            )
        loading = (
            s['stuck_p'] > 0
            and self.avg[OUTCOMES.index('loading')] >= s['stuck_p']
        )
        self.stuck = self.stuck + dt if loading else 0.0
        stuck = s['stuck_p'] > 0 and self.stuck >= s['stuck_s']
        prob = self.problem(self.avg) or ('loading' if stuck else None)
        if prob:
            settling = (
                p[OUTCOMES.index('in_transition')] >= 0.5
                and change >= s['moving']
            )
            if not self.cand:
                self.held = 0.0
            elif not settling or prob == 'video_glitch':
                self.held += dt
            self.cand, self.below = prob, 0.0
        else:
            self.below += dt
            if self.state != 'issue' and self.below > s['grace']:
                self.cand, self.held = None, 0.0
        need = (
            0.0
            if self.cand == 'loading' and stuck
            else {
                'video_glitch': s['confirm_glitch'],
                'loading': s['confirm_loading'],
            }.get(self.cand or '', s['confirm_other'])
        )
        if self.state == 'issue':
            if prob:
                self.outcome = prob
            elif self.below >= s['clear']:
                self.state, self.outcome = 'ok', None
                self.cand, self.held = None, 0.0
        elif self.cand:
            self.state = 'issue' if self.held >= need else 'checking'
            self.outcome = self.cand if self.state == 'issue' else None
        else:
            self.state = 'ok'


# Status, alerts and the status page

STATUS: dict[str, Any] = {
    'state': 'starting',
    'problem': None,
    'since': None,
    'checks': 0,
    'model_runs': 0,
    'model': CFG['model_id'],
}

PAGE = b"""<!doctype html><meta charset=utf-8><title>Screen check</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>body{font:16px system-ui;margin:24px;max-width:960px}
img{width:100%;border-radius:8px}#s{font-size:28px}.ok{color:#127a3a}
.checking{color:#b76e00}.issue,.error{color:#c62828}</style>
<h1>Screen check</h1><p id=s></p><img id=f><pre id=j></pre><script>
const say={ok:'Screen looks normal',checking:'Checking',starting:'Starting'}
setInterval(async()=>{const d=await(await fetch('status')).json()
s.className=d.state;f.src='frame.jpg?'+Date.now()
s.textContent=say[d.state]||(d.problem||d.error||'').replaceAll('_',' ')
j.textContent=JSON.stringify(d,null,1)},1000)</script>"""


class LastFrame:
    """The last model input, JPEG-encoded only when someone asks."""

    def __init__(self) -> None:
        self.rgb: np.ndarray | None = None
        self._of: np.ndarray | None = None
        self._jpeg = b''

    def jpeg(self) -> bytes:
        if self.rgb is not None and self._of is not self.rgb:
            buf = io.BytesIO()
            Image.fromarray(self.rgb).save(buf, 'JPEG', quality=80)
            self._jpeg, self._of = buf.getvalue(), self.rgb
        return self._jpeg


LAST = LastFrame()


def alert(event: str, problem: str | None) -> str:
    at = datetime.now().astimezone().isoformat(timespec='seconds')
    body = {'event': event, 'problem': problem, 'at': at}
    log.info('ALERT %s', json.dumps(body))
    if event == 'problem':
        ALERTS.mkdir(parents=True, exist_ok=True)
        (ALERTS / f'{datetime.now():%Y%m%d-%H%M%S}-{problem}.jpg').write_bytes(
            LAST.jpeg()
        )
        for old in sorted(ALERTS.glob('*.jpg'))[:-50]:
            old.unlink()
    if WEBHOOK:
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    WEBHOOK,
                    json.dumps(body).encode(),
                    {'Content-Type': 'application/json'},
                ),
                timeout=5,
            ).close()
        except OSError as e:
            log.warning('webhook failed: %s', e)
    return at


def status() -> bytes:
    return json.dumps(STATUS).encode()


ROUTES: dict[str, tuple[str, Callable[[], bytes]]] = {
    '/': ('text/html', lambda: PAGE),
    '/status': ('application/json', status),
    '/frame.jpg': ('image/jpeg', LAST.jpeg),
}


class Web(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        route = ROUTES.get(self.path.split('?')[0])
        if not route:
            return self.send_error(404)
        body = route[1]()
        self.send_response(200)
        self.send_header('Content-Type', route[0])
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


class Watcher:
    """Model plus frame combining over a stream of screens."""

    def __init__(self, model: Path) -> None:
        self.model, self.harness = Model(model), Harness(CFG['harness'])
        self.p: np.ndarray | None = None
        self.sig: np.ndarray | None = None
        self.still: np.ndarray | None = None
        self.last_run = 0.0

    def see(self, rgb: np.ndarray, now: float, dt: float) -> np.ndarray:
        """Step on one screen; returns the model's probabilities for it."""
        luma = rgb @ np.array([77, 150, 29], np.float32) / 256
        sig, still = signature(luma), blocks(luma)
        if (
            self.p is None
            or self.still is None
            or now - self.last_run >= RERUN_S
            or still.shape != self.still.shape
            or np.abs(still - self.still).max() >= STILL
        ):  # same picture, same answer: otherwise reuse the last one
            self.p, self.still, self.last_run = self.model(rgb), still, now
            STATUS['model_runs'] += 1
        change = 0.0 if self.sig is None else np.abs(sig - self.sig).mean()
        self.sig = sig
        self.harness.step(self.p, float(change) / 255, dt)
        return self.p


# Main loop


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s'
    )
    os.nice(10)  # the viewer always wins the CPU
    server = ThreadingHTTPServer(('', PORT), Web)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        model = model_file()
    except (OSError, RuntimeError) as e:
        log.error('%s', e)
        STATUS.update(state='error', error=str(e))
        time.sleep(60)  # keep the status page up; the restart policy retries
        raise SystemExit(1) from e
    watcher = Watcher(model)
    harness = watcher.harness
    log.info(
        'started: %s, %.1f checks/s, status on :%d', CFG['model_id'], FPS, PORT
    )
    period = 1 / FPS
    last = time.monotonic()
    for img in replay(SOURCE) if SOURCE else screenshots():
        start = time.monotonic()
        if img is not None:
            was = harness.state, harness.outcome
            LAST.rgb = shrink(img)
            dt = period if SOURCE else start - last
            p = watcher.see(LAST.rgb, start, dt)
            last = start
            if harness.state == 'issue' != was[0]:
                STATUS['since'] = alert('problem', harness.outcome)
            elif was[0] == 'issue' != harness.state:
                alert('cleared', was[1])
                STATUS['since'] = None
            STATUS.update(
                state=harness.state,
                problem=harness.outcome,
                issue_probability=round(float(1 - p[0] - p[1]), 3),
                checks=STATUS['checks'] + 1,
                error=None,
                ms_per_check=round((time.monotonic() - start) * 1000),
            )
            if SOURCE:
                print(
                    json.dumps({k: STATUS[k] for k in ('state', 'problem')}),
                    flush=True,
                )
        else:
            STATUS.update(state='error', error='capture failed')
        if not SOURCE:
            time.sleep(max(0.0, period - (time.monotonic() - start)))


if __name__ == '__main__':
    main()
