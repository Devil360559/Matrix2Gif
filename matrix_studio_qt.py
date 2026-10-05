#!/usr/bin/env python3
"""
matrix_studio_qt.py -- AniMe Matrix Studio, PyQt6 edition (self-contained).
Assemble PARTS 1-4 in order into ONE file. Requires: pip install PyQt6 numpy pillow
"""
from __future__ import annotations
import sys, os, io, re, json, time, copy, hashlib, shutil, zipfile, threading, traceback
import platform, collections, logging
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

APP_VERSION = "4.0.0-qt"
MS = 2  # filmstrip base scale


# ---------------------------------------------------------------- admin check
def is_admin() -> bool:
    try:
        if os.name == "nt":
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        return os.geteuid() == 0
    except Exception:
        return False


# ---------------------------------------------------------------- logging core
FMT = logging.Formatter("%(asctime)s [%(threadName)s] %(levelname)-7s %(name)s | %(message)s")
TRACEBACKS: list = []


class RingHandler(logging.Handler):
    def __init__(self, maxlen=60000):
        super().__init__(level=logging.DEBUG)
        self.buffer = collections.deque(maxlen=maxlen)

    def emit(self, rec):
        self.buffer.append(rec)


class BridgeHandler(logging.Handler):
    """Queues formatted records; the Qt log view drains this queue."""
    def __init__(self, maxlen=20000):
        super().__init__(level=logging.INFO)
        self.queue = collections.deque(maxlen=maxlen)

    def emit(self, rec):
        self.queue.append(self.format(rec))


RING = RingHandler()
BRIDGE = BridgeHandler()
BRIDGE.setFormatter(FMT); RING.setFormatter(FMT)
LOG = logging.getLogger("mstudio")
LOG.setLevel(logging.DEBUG)
if not LOG.handlers:
    LOG.addHandler(RING); LOG.addHandler(BRIDGE)
    _e = logging.StreamHandler(sys.stderr); _e.setLevel(logging.INFO); _e.setFormatter(FMT)
    LOG.addHandler(_e)
    LOG.propagate = False

_THROTTLE = {}


def logt(key, interval, level, msg, *a):
    now = time.monotonic()
    if now - _THROTTLE.get(key, 0.0) >= interval:
        _THROTTLE[key] = now
        getattr(LOG, level)(msg, *a)


@contextmanager
def stage(name, **kw):
    t0 = time.perf_counter()
    LOG.debug("stage enter: %s %s", name, kw or "")
    try:
        yield
    except Exception:
        LOG.exception("stage FAILED: %s", name); raise
    finally:
        LOG.debug("stage exit: %s (%.1f ms)", name, (time.perf_counter() - t0) * 1000.0)


def _capture_tb(t, e, tb):
    text = "".join(traceback.format_exception(t, e, tb))
    TRACEBACKS.append(text)
    if len(TRACEBACKS) > 20: TRACEBACKS.pop(0)
    return text


def install_excepthooks():
    def eh(t, e, tb):
        _capture_tb(t, e, tb); LOG.critical("UNCAUGHT EXCEPTION", exc_info=(t, e, tb))
    def teh(args):
        _capture_tb(args.exc_type, args.exc_value, args.exc_traceback)
        LOG.critical("UNCAUGHT THREAD EXCEPTION",
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    sys.excepthook = eh
    threading.excepthook = teh


# ---------------------------------------------------------------- paths / utils
def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    cand = (Path(sys.executable).resolve().parent / "data") if getattr(sys, "frozen", False) \
        else app_dir() / "data"
    try:
        cand.mkdir(parents=True, exist_ok=True); return cand
    except OSError:
        fb = Path(os.environ.get("APPDATA", Path.home())) / "MatrixStudio"
        fb.mkdir(parents=True, exist_ok=True); return fb


def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def hexdump(buf, limit=64, width=16):
    out = []
    for b in range(0, min(limit, len(buf)), width):
        c = buf[b:b + width]
        out.append(f"{b:08x}  {' '.join(f'{x:02x}' for x in c):<{width*3}}  "
                   f"|{''.join(chr(x) if 32 <= x < 127 else '.' for x in c)}|")
    return "\n".join(out)


class DecodeCache:
    def __init__(self, enabled=True, cap_mb=300):
        self.dir = data_dir() / "cache"; self.enabled = enabled
        self.cap = cap_mb * 1024 * 1024

    def key(self, fh, member, mode, bg): return f"{fh}_{sha16((member + mode + str(bg)).encode())}"

    def get(self, key):
        if not self.enabled: return None
        p = self.dir / (key + ".npz")
        if not p.exists():
            LOG.debug("cache MISS %s", key); return None
        t0 = time.perf_counter()
        try:
            with np.load(p) as z:
                p.touch()
                LOG.debug("cache HIT %s (%.1f ms)", key, (time.perf_counter() - t0) * 1000)
                return z["frames"], z["delays"]
        except Exception:
            LOG.exception("cache read failed %s", key); return None

    def put(self, key, frames, delays):
        if not self.enabled: return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.dir / (key + ".npz"), frames=frames, delays=delays)
            LOG.debug("cache PUT %s shape=%s", key, frames.shape)
            self._evict()
        except Exception:
            LOG.exception("cache write failed %s", key)

    def _evict(self):
        files = sorted(self.dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        while total > self.cap and files:
            p = files.pop(0); total -= p.stat().st_size
            try: p.unlink()
            except OSError: pass

    def clear(self):
        LOG.info("cache cleared by user"); shutil.rmtree(self.dir, ignore_errors=True)


CACHE = DecodeCache()


# ---------------------------------------------------------------- model / guards
class StudioError(Exception): pass
class UnsupportedFormat(StudioError): pass
KNOWN_HEADER = "matrix_led"
SUPPORTED_MAJOR = (3, 4, 5)


@dataclass
class Entry:
    key: int; name: str; speed: float; repeat: int
    start: int; length: int; z: int; image_path: str = ""


@dataclass
class Manifest:
    text: str = ""; header: str = ""; version: str = ""
    period: float = 0.0; model: str = ""; rows: int = 0; cols: int = 0
    entries: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    salvaged: bool = False


@dataclass
class Meta:
    n: int; w: int; h: int; delays: np.ndarray
    interlaced: bool; has_transparency: bool
    size_warnings: list = field(default_factory=list)


@dataclass(eq=False)
class Clip:
    pack_hash: str; name: str; member: str; data: bytes
    meta: Meta; entry: Entry | None
    warnings: list = field(default_factory=list)
    _lazy: dict = field(default_factory=dict)

    def frames_for(self, mode, bg):
        if (mode, bg) not in self._lazy:
            self._lazy[(mode, bg)] = LazyFrames(self.data, mode, bg, (self.meta.h, self.meta.w))
        return self._lazy[(mode, bg)]

    def materialize(self, mode, bg, cache=CACHE):
        key = cache.key(self.pack_hash, self.member, mode, bg) if cache else None
        if cache:
            hit = cache.get(key)
            if hit: return hit
        fr = self.frames_for(mode, bg).materialize()
        if cache: cache.put(key, fr, self.meta.delays)
        return fr, self.meta.delays


@dataclass
class Pack:
    path: Path; file_hash: str; manifest: Manifest | None
    clips: list; warnings: list


_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _decode_xml(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "utf-8"):
        try: return b.decode(enc)
        except Exception: continue
    return b.decode("latin-1", "replace")


def _txt(el, tag):
    c = el.find(tag)
    return c.text.strip() if c is not None and c.text else None


def guard_raw(raw: bytes, path: Path):
    if raw[:4] != b"PK\x03\x04":
        LOG.error("guard_raw REJECT %s magic=%s", path.name, raw[:4].hex(" "))
        raise UnsupportedFormat(f"{path.name}: not a ZIP container (magic {raw[:4].hex(' ')}).\n"
                                f"{hexdump(raw, 64)}")
    LOG.debug("guard_raw OK %s", path.name)


def guard_manifest(man, allow_unknown: bool):
    if man.header and man.header.strip().lower() != KNOWN_HEADER:
        raise UnsupportedFormat(f"manifest header is '{man.header}', expected 'Matrix_LED'.")
    m = re.match(r"\s*(\d+)", man.version or "")
    if m and int(m.group(1)) not in SUPPORTED_MAJOR and not allow_unknown:
        LOG.warning("guard_manifest: unverified version=%s (tolerant parse)", man.version)
        man.warnings.append(f"unverified manifest version {man.version} (tolerant parse)")
    LOG.debug("guard_manifest OK header=%r version=%s", man.header, man.version)


def parse_manifest_text(text: str) -> Manifest:
    with stage("parse_manifest", bytes=len(text)):
        m = Manifest(text=text); root = None
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            LOG.warning("manifest XML ParseError; stripping control chars")
            fixed = _CTRL.sub("", text)
            try:
                root = ET.fromstring(fixed)
                m.warnings.append("manifest XML contained invalid control characters (repaired)")
            except ET.ParseError:
                blocks = re.findall(r"<layerItem\b.*?</layerItem>", text, re.S)
                if not blocks: raise StudioError("manifest XML unparseable and unsalvageable")
                root = ET.fromstring("<root><layer>" + "".join(blocks) + "</layer></root>")
                m.warnings.append("manifest XML broken; salvaged layerItem blocks only"); m.salvaged = True
        m.header = _txt(root, "header") or ""; m.version = _txt(root, "version") or ""
        m.period = float(_txt(root, "period") or 0); m.model = _txt(root, "modelName") or ""
        m.rows = int(_txt(root, "rowCount") or 0); m.cols = int(_txt(root, "columnCount") or 0)
        seen = {}
        for i, li in enumerate(root.iter("layerItem")):
            raw_name = _txt(li, "name") or f"clip{i}"; name = raw_name
            if name in seen:
                seen[name] += 1; name = f"{raw_name} ({seen[name]})"
                m.warnings.append(f"duplicate animation name '{raw_name}' renamed")
            else:
                seen[name] = 0
            try: key = int(li.get("key", i))
            except ValueError:
                key = i; m.warnings.append(f"layerItem '{name}' has no valid key")
            m.entries.append(Entry(key=key, name=name, speed=float(_txt(li, "speed") or 2),
                                   repeat=max(1, int(_txt(li, "repeat") or 1)),
                                   start=int(_txt(li, "start") or 0), length=int(_txt(li, "length") or 0),
                                   z=int(_txt(li, "layer") or 0), image_path=_txt(li, "imagePath") or ""))
        m.entries.sort(key=lambda e: e.key)
        if not m.entries:
            m.warnings.append("manifest has no layerItem entries; member order used")
        LOG.debug("manifest parsed model=%s entries=%d salvaged=%s", m.model, len(m.entries), m.salvaged)
        return m


def frame_luma(im, mode: str, bg_lum: float) -> np.ndarray:
    if mode == "ignore" or "transparency" not in im.info:
        return np.asarray(im.convert("L"), np.uint8)
    rgba = np.asarray(im.convert("RGBA"), np.uint8).astype(np.float32)
    a = rgba[..., 3:4] / 255.0
    lum = (rgba[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    return np.clip((lum * a + bg_lum * (1 - a))[..., 0], 0, 255).astype(np.uint8)


def _gif_descriptor_offsets(data: bytes) -> list:
    offs = []; i = 13
    if len(data) > 13 and data[10] & 0x80:
        i += 3 * (2 << (data[10] & 7))
    while i + 9 < len(data):
        c = data[i]
        if c == 0x3B: break
        if c == 0x21:
            i += 2
            while i < len(data) and data[i]: i += 1 + data[i]
            i += 1
        elif c == 0x2C:
            offs.append(i + 9); i += 10
            if data[i - 1] & 0x80: i += 3 * (2 << (data[i - 1] & 7))
            i += 1
            while i < len(data) and data[i]: i += 1 + data[i]
            i += 1
        else:
            i += 1
    return offs


def _gif_interlaced(data: bytes) -> bool:
    return any(data[o] & 0x40 for o in _gif_descriptor_offsets(data))


def _gif_force_interlace(data: bytes, on: bool) -> bytes:
    b = bytearray(data)
    for o in _gif_descriptor_offsets(data):
        b[o] = (b[o] | 0x40) if on else (b[o] & ~0x40)
    return bytes(b)


class LazyFrames:
    def __init__(self, data: bytes, mode: str, bg_lum: float, target=None):
        self.data, self.mode, self.bg = data, mode, bg_lum
        self._im = None; self._cache = {}; self._lock = threading.Lock(); self._target = target

    def _open(self):
        if self._im is None: self._im = Image.open(io.BytesIO(self.data))
        return self._im

    @property
    def n_frames(self): return getattr(self._open(), "n_frames", 1)

    def _norm(self, arr):
        if self._target is None: return arr
        h, w = self._target
        if arr.shape == (h, w): return arr
        out = np.zeros((h, w), np.uint8)
        sh, sw = min(h, arr.shape[0]), min(w, arr.shape[1])
        out[:sh, :sw] = arr[:sh, :sw]; return out

    def get(self, i: int) -> np.ndarray:
        with self._lock:
            if i in self._cache: return self._cache[i]
            im = self._open(); im.seek(i)
            arr = self._norm(frame_luma(im, self.mode, self.bg))
            if len(self._cache) > 12: self._cache.pop(next(iter(self._cache)))
            self._cache[i] = arr; return arr

    def materialize(self) -> np.ndarray:
        t0 = time.perf_counter()
        out = np.stack([self.get(i) for i in range(self.n_frames)])
        LOG.debug("materialize %d frames shape=%s (%.1f ms)", out.shape[0], out.shape,
                  (time.perf_counter() - t0) * 1000)
        return out

    def close(self):
        with self._lock:
            if self._im is not None: self._im.close(); self._im = None
            self._cache.clear()


def probe_lazy(lf: LazyFrames) -> Meta:
    im = lf._open(); n = getattr(im, "n_frames", 1)
    delays, sizes = [], []; trans = False
    inter = _gif_interlaced(lf.data)
    for i in range(n):
        im.seek(i)
        delays.append(float(im.info.get("duration") or 0))
        if im.info.get("interlace"): inter = True
        if "transparency" in im.info: trans = True
        sizes.append(im.size)
    d = np.array(delays, float); d[d <= 0] = 100.0
    w, h = sizes[0]; warn = []
    if inter: warn.append("interlaced GIF frames detected; decoded and validated")
    if len(set(sizes)) > 1:
        warn.append(f"inconsistent frame sizes {sorted(set(sizes))}; normalising to {w}x{h}")
    lf._target = (h, w)
    return Meta(n, w, h, d, inter, trans, warn)


# ==== PART 1 END ====
# ============================================================================
# PACK / CLIP LOADING
# ============================================================================
def load_pack(path: Path, transparency="black", bg_lum=0.0, allow_unknown=False) -> Pack:
    with stage("load_pack", path=str(path), transparency=transparency, bg_lum=bg_lum):
        raw = path.read_bytes()
        LOG.debug("read %s: %d bytes", path.name, len(raw))
        guard_raw(raw, path)
        fh = sha16(raw)
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as e:
            LOG.error("corrupt ZIP in %s: %s", path.name, e)
            raise UnsupportedFormat(f"{path.name}: ZIP magic present but archive corrupt ({e})") from None
        names = zf.namelist()
        LOG.debug("zip members (%d): %s", len(names), names)
        mains = {Path(n).stem: n for n in names
                 if n.lower().endswith(".default") and "_thumbnail" not in n.lower()}
        warnings: list = []
        manifest = None
        xmls = [n for n in names if n.lower().endswith(".xml")]
        if xmls:
            try:
                manifest = parse_manifest_text(_decode_xml(zf.read(xmls[0])))
                warnings.extend(manifest.warnings)
                guard_manifest(manifest, allow_unknown)
            except UnsupportedFormat:
                raise
            except StudioError as e:
                LOG.warning("manifest unusable: %s", e)
                warnings.append(f"manifest unusable ({e}); member-only mode")
        else:
            warnings.append("no XML manifest found; member-only mode")
        order = [e.name for e in manifest.entries] if manifest and manifest.entries else sorted(mains)
        entry_by_name = {e.name: e for e in (manifest.entries if manifest else [])}
        clips = []
        for name in order:
            e = entry_by_name.get(name)
            stem = Path(e.image_path).stem if (e and e.image_path) else name
            member = mains.get(stem) or mains.get(name)
            if member is None:
                LOG.warning("missing member for '%s' (tried %s)", name, stem)
                warnings.append(f"manifest references missing member '{name}' -- skipped")
                continue
            data = zf.read(member)
            if data[:6] not in (b"GIF87a", b"GIF89a"):
                LOG.warning("member '%s' is not a GIF (starts %s)", name, data[:4].hex(" "))
                warnings.append(f"'{name}' is not a GIF (starts {data[:4].hex(' ')}) -- skipped")
                continue
            lf = LazyFrames(data, transparency, bg_lum)
            meta = probe_lazy(lf)
            cw = list(meta.size_warnings)
            if meta.has_transparency and transparency != "ignore":
                cw.append(f"transparency present; composited with mode '{transparency}'")
            LOG.info("clip %-28s %dx%d x%d ~%.0fms interlaced=%s transp=%s",
                     name, meta.w, meta.h, meta.n, meta.delays.mean(),
                     meta.interlaced, meta.has_transparency)
            clips.append(Clip(fh, name, member, data, meta, e, cw))
            lf.close()
        if not clips:
            LOG.error("no decodable clips in %s", path.name)
            raise StudioError(f"{path.name}: no decodable animation members found")
        LOG.info("pack %s loaded: %d clips", path.name, len(clips))
        return Pack(path, fh, manifest, clips, warnings)


def load_loose_gif(path: Path, transparency="black", bg_lum=0.0) -> Pack:
    with stage("load_loose_gif", path=str(path)):
        raw = path.read_bytes()
        if raw[:6] not in (b"GIF87a", b"GIF89a"):
            raise UnsupportedFormat(f"{path.name}: not a GIF (starts {raw[:4].hex(' ')})")
        lf = LazyFrames(raw, transparency, bg_lum)
        meta = probe_lazy(lf)
        lf.close()
        clip = Clip(sha16(raw), path.stem, path.name, raw, meta, None,
                    ["standalone GIF imported as clip"])
        LOG.info("imported GIF %s (%d frames %dx%d)", path.name, meta.n, meta.w, meta.h)
        return Pack(path, sha16(raw), None, [clip], [])


def load_png_sequence(dirpath: Path, delay_ms=40.0) -> Pack:
    files = sorted(dirpath.glob("*.png"))
    if not files:
        raise StudioError(f"no .png files in {dirpath}")
    frames = [np.asarray(Image.open(f).convert("L"), np.uint8) for f in files]
    h = min(f.shape[0] for f in frames); w = min(f.shape[1] for f in frames)
    arr = np.stack([f[:h, :w] for f in frames])
    meta = Meta(n=arr.shape[0], w=w, h=h, delays=np.full(arr.shape[0], delay_ms),
                interlaced=False, has_transparency=False)
    clip = Clip(sha16(dirpath.name.encode()), dirpath.name, f"pngseq:{dirpath.name}",
                b"", meta, None, ["PNG sequence imported as clip"])
    clip._pre = arr                      # pre-materialised frames (no GIF bytes behind it)
    LOG.info("imported PNG sequence %s (%d frames %dx%d)", dirpath.name, meta.n, w, h)
    return Pack(dirpath, sha16(dirpath.name.encode()), None, [clip], [])


def frames_of(clip: Clip, mode="black", bg_lum=0.0, cache=CACHE):
    """Uniform frame access: pre-materialised clips (_pre) bypass the GIF decoder."""
    pre = getattr(clip, "_pre", None)
    if pre is not None:
        return pre, clip.meta.delays
    return clip.materialize(mode, bg_lum, cache)


# ============================================================================
# PER-CLIP EDIT TRANSFORMS
# ============================================================================
@dataclass
class EditState:
    reverse: bool = False
    pingpong: bool = False
    crop: list | None = None          # [x, y, w, h] or None
    dx: int = 0
    dy: int = 0
    scale: float = 1.0
    brightness: int = 0               # -128..128 additive
    contrast: float = 1.0             # about mid-grey
    gamma: float = 1.0


def seq_index(e: EditState, i: int, n: int) -> int:
    if n <= 0: return 0
    if e.pingpong:
        cyc = max(1, 2 * n - 2); j = i % cyc
        return cyc - j if j >= n else j
    if e.reverse:
        return (n - 1) - (i % n)
    return i % n


def seq_len(e: EditState, n: int) -> int:
    if e.pingpong: return max(1, 2 * n - 2)
    return max(1, n)


def xframe(frames: np.ndarray, e: EditState, i: int) -> np.ndarray:
    n = frames.shape[0]
    arr = frames[seq_index(e, i, n)]
    if e.crop:
        x, y, w, h = (max(0, int(v)) for v in e.crop)
        if w > 0 and h > 0:
            cut = arr[y:min(y + h, arr.shape[0]), x:min(x + w, arr.shape[1])]
            if cut.size: arr = cut
    if e.scale != 1.0:
        arr = np.asarray(Image.fromarray(arr).resize(
            (max(1, int(arr.shape[1] * e.scale)), max(1, int(arr.shape[0] * e.scale))),
            Image.NEAREST), np.uint8)
    if e.dx or e.dy:
        out = np.zeros(arr.shape, np.uint8)
        sy, dy = max(0, e.dy), max(0, -e.dy)
        sx, dx = max(0, e.dx), max(0, -e.dx)
        hh = min(arr.shape[0] - sy, arr.shape[0] - dy)
        ww = min(arr.shape[1] - sx, arr.shape[1] - dx)
        if hh > 0 and ww > 0:
            out[dy:dy + hh, dx:dx + ww] = arr[sy:sy + hh, sx:sx + ww]
        arr = out
    a = arr.astype(np.float32)
    if e.brightness: a = a + e.brightness
    if e.contrast != 1.0: a = (a - 127.5) * e.contrast + 127.5
    if e.gamma != 1.0: a = 255.0 * np.power(np.clip(a, 0, 255) / 255.0, 1.0 / e.gamma)
    return np.clip(a, 0, 255).astype(np.uint8)


def clip_sequence(frames: np.ndarray, delays, e: EditState):
    n = frames.shape[0]
    L = seq_len(e, n)
    fr = [xframe(frames, e, i) for i in range(L)]
    dl = [float(delays[seq_index(e, i, n)]) for i in range(L)]
    return fr, dl


# ============================================================================
# EXPORT ENGINE
# ============================================================================
def ramp_palette(bg, fg) -> bytes:
    r = np.linspace(0, 1, 256)[:, None]
    return np.rint(np.array(bg, float) + (np.array(fg, float) - np.array(bg, float)) * r) \
        .astype(np.uint8).tobytes()


def cs(ms, min_ms): return int(max(1, round(max(ms, min_ms) / 10)) * 10)


def export_gif(sequences, out, scale=1, min_delay=20.0, loop=True,
               fg=(255, 255, 255), bg=(0, 0, 0), progress=None, cancel=None):
    """sequences: list of (frames_list, delays_list). Returns (n_frames, seconds) or None if cancelled."""
    with stage("export_gif", out=str(out), scale=scale):
        pal = ramp_palette(bg, fg)
        imgs, durs = [], []
        done = 0
        total = sum(len(f) for f, _ in sequences)
        for frames, delays in sequences:
            for i in range(len(frames)):
                if cancel is not None and cancel.is_set():
                    LOG.warning("export cancelled at frame %d/%d", done, total)
                    return None
                a = frames[i]
                if scale > 1:
                    a = np.repeat(np.repeat(a, scale, 0), scale, 1)
                p = Image.frombytes("P", (a.shape[1], a.shape[0]), a.tobytes())
                p.putpalette(pal)
                d = cs(float(delays[i]), min_delay)
                p.info["duration"] = d
                imgs.append(p); durs.append(d)
                done += 1
                if progress and done % 5 == 0:
                    progress(done, total)
        if not imgs:
            raise StudioError("nothing to export")
        imgs[0].save(out, save_all=True, append_images=imgs[1:],
                     loop=0 if loop else None, disposal=2, duration=durs)
        if progress: progress(done, total)
        LOG.info("exported %s: %dx%d %d frames %.2fs %d KiB", Path(out).name,
                 imgs[0].size[0], imgs[0].size[1], len(imgs), sum(durs) / 1000,
                 Path(out).stat().st_size // 1024)
        return len(imgs), sum(durs) / 1000


# ============================================================================
# GENERATORS (synthetic clips)
# ============================================================================
FONT3x5 = {
    ' ': ["...", "...", "...", "...", "..."],
    'A': [".#.", "#.#", "###", "#.#", "#.#"], 'B': ["##.", "#.#", "##.", "#.#", "##."],
    'C': [".##", "#..", "#..", "#..", ".##"], 'D': ["##.", "#.#", "#.#", "#.#", "##."],
    'E': ["###", "#..", "##.", "#..", "###"], 'F': ["###", "#..", "##.", "#..", "#.."],
    'G': [".##", "#..", "#.#", "#.#", ".##"], 'H': ["#.#", "#.#", "###", "#.#", "#.#"],
    'I': ["###", ".#.", ".#.", ".#.", "###"], 'J': ["..#", "..#", "..#", "#.#", ".#."],
    'K': ["#.#", "#.#", "##.", "#.#", "#.#"], 'L': ["#..", "#..", "#..", "#..", "###"],
    'M': ["#.#", "###", "###", "#.#", "#.#"], 'N': ["#.#", "###", "###", "###", "#.#"],
    'O': [".#.", "#.#", "#.#", "#.#", ".#."], 'P': ["##.", "#.#", "##.", "#..", "#.."],
    'Q': [".#.", "#.#", "#.#", "##.", ".##"], 'R': ["##.", "#.#", "##.", "#.#", "#.#"],
    'S': [".##", "#..", ".#.", "..#", "##."], 'T': ["###", ".#.", ".#.", ".#.", ".#."],
    'U': ["#.#", "#.#", "#.#", "#.#", "###"], 'V': ["#.#", "#.#", "#.#", "#.#", ".#."],
    'W': ["#.#", "#.#", "###", "###", "#.#"], 'X': ["#.#", "#.#", ".#.", "#.#", "#.#"],
    'Y': ["#.#", "#.#", ".#.", ".#.", ".#."], 'Z': ["###", "..#", ".#.", "#..", "###"],
    '0': [".#.", "#.#", "#.#", "#.#", ".#."], '1': [".#.", "##.", ".#.", ".#.", "###"],
    '2': ["##.", "..#", ".#.", "#..", "###"], '3': ["##.", "..#", ".##", "..#", "##."],
    '4': ["#.#", "#.#", "###", "..#", "..#"], '5': ["###", "#..", "##.", "..#", "##."],
    '6': [".##", "#..", "##.", "#.#", ".#."], '7': ["###", "..#", "..#", ".#.", ".#."],
    '8': [".#.", "#.#", ".#.", "#.#", ".#."], '9': [".#.", "#.#", ".##", "..#", "##."],
    '!': [".#.", ".#.", ".#.", "...", ".#."], '.': ["...", "...", "...", "...", ".#."],
    '-': ["...", "...", "###", "...", "..."], '_': ["...", "...", "...", "...", "###"],
}


def text_bitmap(s: str) -> np.ndarray:
    cols = []
    for ch in s.upper():
        g = FONT3x5.get(ch, FONT3x5[' '])
        cols.append([[1 if c == '#' else 0 for c in row] for row in g])
        cols.append([[0, 0, 0]])
    if not cols: cols = [[[0, 0, 0]]]
    h = 5; w = sum(len(c[0]) for c in cols)
    out = np.zeros((h, w), np.uint8); x = 0
    for c in cols:
        a = np.array(c, np.uint8); out[:, x:x + a.shape[1]] = a; x += a.shape[1]
    return out * 255


def make_marquee(text="ROG", w=74, h=36, step=2, delay=40.0):
    bmp = text_bitmap(text or "ROG")
    strip = np.concatenate([bmp, np.zeros((5, 8), np.uint8)], axis=1)
    frames = []
    for off in range(0, strip.shape[1] + w, max(1, step)):
        canvas = np.zeros((h, w), np.uint8)
        y0 = (h - 5) // 2
        for x in range(w):
            sx = x + off - w
            if 0 <= sx < strip.shape[1]:
                canvas[y0:y0 + 5, x] = strip[:, sx]
        frames.append(canvas)
    return np.stack(frames), delay


def make_plasma(w=74, h=36, n=60, delay=50.0):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    frames = []
    for t in range(n):
        v = np.sin(xs / 6 + t / 5) + np.sin(ys / 5 - t / 7) + np.sin((xs + ys) / 9 + t / 9)
        frames.append(np.clip((v + 3) / 6 * 255, 0, 255).astype(np.uint8))
    return np.stack(frames), delay


def make_life(w=74, h=36, n=80, delay=100.0, seed=None):
    rng = np.random.default_rng(seed)
    g = (rng.random((h, w)) > 0.75).astype(np.uint8)
    frames = []
    for _ in range(n):
        nb = sum(np.roll(np.roll(g, dy, 0), dx, 1)
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
        g = ((g == 1) & ((nb == 2) | (nb == 3))) | ((g == 0) & (nb == 3))
        frames.append(g.astype(np.uint8) * 255)
    return np.stack(frames), delay


def make_bounce(w=74, h=36, n=90, delay=40.0):
    frames = []
    px, py, vx, vy = 2.0, 2.0, 1.3, 0.9
    for _ in range(n):
        canvas = np.zeros((h, w), np.uint8)
        x, y = int(px), int(py)
        canvas[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = 255
        px += vx; py += vy
        if px <= 0 or px >= w - 1: vx = -vx
        if py <= 0 or py >= h - 1: vy = -vy
        frames.append(canvas)
    return np.stack(frames), delay


def make_sweep(w=74, h=36, delay=30.0):
    frames = []
    for t in range(w):
        d = np.abs(np.arange(w) - t)
        row = np.clip(255 - d * 40, 0, 255).astype(np.uint8)
        frames.append(np.repeat(row[None, :], h, axis=0))
    return np.stack(frames), delay


GENERATORS = ("marquee", "plasma", "life", "bounce", "sweep")


def generated_clip(kind: str, **kw) -> Clip:
    fn = {"marquee": make_marquee, "plasma": make_plasma, "life": make_life,
          "bounce": make_bounce, "sweep": make_sweep}[kind]
    frames, delay = fn(**kw)
    meta = Meta(n=frames.shape[0], w=frames.shape[2], h=frames.shape[1],
                delays=np.full(frames.shape[0], delay),
                interlaced=False, has_transparency=False)
    clip = Clip("generated", kind, f"gen:{kind}", b"", meta, None, ["generated clip"])
    clip._pre = frames
    LOG.info("generated clip '%s' (%d frames)", kind, frames.shape[0])
    return clip


# ============================================================================
# SELF-TESTS / GOLDENS
# ============================================================================
def _synth_gif(delays, size=(4, 4), interlace=False, transparency=None):
    buf = io.BytesIO()
    vals = (40, 120, 220)
    frames = [Image.fromarray(np.full(size, vals[i % len(vals)], np.uint8), "L")
              for i in range(max(1, len(delays)))]
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=list(delays), loop=0, interlace=interlace,
                   transparency=transparency, disposal=2)
    return buf.getvalue()


def builtin_tests():
    res = []
    def t(name, fn):
        try:
            fn(); res.append((name, True, "")); LOG.debug("selftest PASS %s", name)
        except Exception as e:
            res.append((name, False, str(e))); LOG.exception("selftest FAIL %s", name)
    def roundtrip():
        lf = LazyFrames(_synth_gif([40, 60, 80]), "black", 0); m = probe_lazy(lf)
        assert m.n == 3 and list(m.delays) == [40, 60, 80]
        assert lf.materialize().shape == (3, 4, 4)
    def interlace_flag():
        raw = _synth_gif([50, 50])
        assert not probe_lazy(LazyFrames(raw, "black", 0)).interlaced, "false positive"
        forced = _gif_force_interlace(raw, True)
        assert probe_lazy(LazyFrames(forced, "black", 0)).interlaced, "probe missed interlace bit"
        assert not probe_lazy(LazyFrames(_gif_force_interlace(forced, False), "black", 0)).interlaced
    def transparency_modes():
        data = _synth_gif([30, 30], transparency=0)
        a = LazyFrames(data, "black", 0).get(0); b = LazyFrames(data, "ignore", 0).get(0)
        assert a.shape == b.shape
    def guard_raises():
        try: guard_raw(b"NOTAZIP......", Path("x.matrix"))
        except UnsupportedFormat: return
        raise AssertionError("guard did not raise on non-ZIP")
    def manifest_salvage():
        bad = ("<root><header>Matrix_LED</header><layer><layerItem key='0'>"
               "<name>a\x01b</name><speed>2</speed></layerItem></layer></root>")
        assert parse_manifest_text(bad).entries
    def edit_transforms():
        fr = np.zeros((4, 4, 4), np.uint8)
        for i in range(4): fr[i] = i + 1
        assert xframe(fr, EditState(reverse=True), 0)[0, 0] == 4
        e = EditState(pingpong=True)
        assert seq_len(e, 4) == 6 and seq_index(e, 4, 4) == 2
        assert xframe(fr, EditState(brightness=10), 1)[0, 0] == 12
        assert xframe(fr, EditState(dx=1), 0)[0, 0] == 0 and xframe(fr, EditState(dx=1), 0)[0, 1] == 1
    t("gif roundtrip + delays", roundtrip)
    t("interlace detection", interlace_flag)
    t("transparency modes", transparency_modes)
    t("format guard", guard_raises)
    t("manifest salvage", manifest_salvage)
    t("edit transforms", edit_transforms)
    return res


def golden_record(pack, mode, bg_lum):
    rec = {"file_sha": pack.file_hash,
           "model": pack.manifest.model if pack.manifest else "",
           "period": pack.manifest.period if pack.manifest else 0,
           "order": [c.name for c in pack.clips], "clips": {}}
    for c in pack.clips:
        fr, dl = frames_of(c, mode, bg_lum, cache=None)
        rec["clips"][c.name] = {"w": c.meta.w, "h": c.meta.h, "n": c.meta.n,
                                "delays_sha": sha16(np.asarray(dl).tobytes()),
                                "frames_sha": sha16(fr.tobytes()),
                                "interlaced": c.meta.interlaced,
                                "transparency": c.meta.has_transparency}
    return rec


def _golden_name(pack) -> str:
    """Windows-safe golden filename for any pack (loose imports, png-seq dirs...)."""
    stem = re.sub(r'[<>:"/\\|?*]', "_", pack.path.stem).strip() or "pack"
    return stem + ".golden.json"


def snapshot_goldens(packs, mode, bg_lum):
    gdir = data_dir() / "goldens"; gdir.mkdir(parents=True, exist_ok=True)
    for p in packs:
        if p.file_hash == "generated":
            LOG.info("skipping golden snapshot for generated pack '%s' (session-only)", p.path.stem)
            continue
        (gdir / _golden_name(p)).write_text(
            json.dumps(golden_record(p, mode, bg_lum), indent=1))
        LOG.info("golden snapshot written: %s", p.path.stem)
    return gdir


def run_golden_tests(packs, mode, bg_lum):
    gdir = data_dir() / "goldens"; res = []
    for p in packs:
        if p.file_hash == "generated":
            continue
        gp = gdir / _golden_name(p)
        if not gp.exists():
            res.append((f"golden:{p.path.stem}", None, "no snapshot yet (run snapshot once)"))
            continue
        old = json.loads(gp.read_text()); new = golden_record(p, mode, bg_lum); diffs = []
        if old["file_sha"] != new["file_sha"]:
            diffs.append("source file changed (re-snapshot expected)")
        if old["order"] != new["order"]:
            diffs.append(f"order {old['order']} -> {new['order']}")
        for name, oc in old["clips"].items():
            nc = new["clips"].get(name)
            if nc is None:
                diffs.append(f"clip '{name}' missing"); continue
            for k in ("w", "h", "n", "delays_sha", "frames_sha"):
                if oc[k] != nc[k]:
                    diffs.append(f"clip '{name}' field {k}: {oc[k]} -> {nc[k]}")
        res.append((f"golden:{p.path.stem}", not diffs, "; ".join(diffs)))
    return res


# ============================================================================
# DIAGNOSTIC TEXT (bug-button export)
# ============================================================================
def env_block() -> str:
    import PIL
    lines = [f"app_version   : {APP_VERSION}",
             f"timestamp     : {datetime.now().isoformat()}",
             f"platform      : {platform.platform()}",
             f"machine       : {platform.machine()}",
             f"python        : {sys.version.replace(chr(10), ' ')}",
             f"executable    : {sys.executable}",
             f"frozen        : {getattr(sys, 'frozen', False)}",
             f"cwd           : {Path.cwd()}",
             f"argv          : {sys.argv}",
             f"numpy         : {np.__version__}",
             f"pillow        : {PIL.__version__}",
             f"admin         : {is_admin()}",
             f"data_dir      : {data_dir()}",
             f"cache_dir     : {CACHE.dir} enabled={CACHE.enabled}"]
    try:
        from PyQt6.QtCore import QT_VERSION_STR
        lines.append(f"qt            : {QT_VERSION_STR}")
    except Exception:
        lines.append("qt            : not imported")
    return "\n".join(lines)


def state_snapshot(app=None) -> str:
    L = [f"app object      : {'present' if app else 'none'}"]
    if app is not None:
        snap = getattr(app, "diag_state", None)
        if callable(snap):
            L.extend(snap())
    try:
        files = sorted(CACHE.dir.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
        total = sum(f.stat().st_size for f in files)
        L.append(f"cache entries   : {len(files)} total={total / 1e6:.1f}MB")
        for f in files[:30]:
            L.append(f"  {f.name} {f.stat().st_size:,}B")
    except Exception as e:
        L.append(f"cache listing failed: {e}")
    try:
        g = sorted((data_dir() / "goldens").glob("*.json"))
        L.append(f"golden files    : {[p.name for p in g]}")
    except Exception as e:
        L.append(f"goldens listing failed: {e}")
    return "\n".join(L)


def build_diag_text(app=None) -> str:
    out = ["#" * 100, "MATRIX STUDIO QT DIAGNOSTIC LOG (verbose, DEBUG level)",
           env_block(), "#" * 100,
           "---- FULL SESSION LOG (oldest -> newest) ----"]
    out.extend(FMT.format(rec) for rec in list(RING.buffer))
    out += ["#" * 100, "---- SESSION STATE SNAPSHOT ----", state_snapshot(app)]
    if TRACEBACKS:
        out += ["#" * 100, "---- CAPTURED TRACEBACKS ----"] + TRACEBACKS[-20:]
    out.append("#" * 100)
    return "\n".join(out)


# ==== PART 2 END ====
# ============================================================================
# PyQt6 LAYER
# ============================================================================
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QScrollArea, QFrame, QSplitter, QTabWidget, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QEvent
from PyQt6.QtGui import (QPixmap, QImage, QKeySequence, QShortcut,
                         QDragEnterEvent, QDropEvent, QColor, QPalette)

def gray_to_pixmap(arr, scale=1):
    """2-D grayscale uint8 array -> QPixmap (safe copy out of numpy memory)."""
    if arr is None or arr.size == 0:
        return QPixmap()
    a = np.ascontiguousarray(arr)
    if scale > 1:
        a = np.repeat(np.repeat(a, scale, 0), scale, 1)
    h, w = a.shape
    img = QImage(a.data, w, h, w, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(img.copy())


def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    pal = app.palette()
    dark = QColor(27, 27, 31); mid = QColor(42, 42, 48); light = QColor(232, 232, 234)
    pal.setColor(QPalette.ColorRole.Window, dark)
    pal.setColor(QPalette.ColorRole.WindowText, light)
    pal.setColor(QPalette.ColorRole.Base, QColor(20, 20, 20))
    pal.setColor(QPalette.ColorRole.AlternateBase, mid)
    pal.setColor(QPalette.ColorRole.ToolTipBase, mid)
    pal.setColor(QPalette.ColorRole.ToolTipText, light)
    pal.setColor(QPalette.ColorRole.Text, light)
    pal.setColor(QPalette.ColorRole.Button, mid)
    pal.setColor(QPalette.ColorRole.ButtonText, light)
    pal.setColor(QPalette.ColorRole.Highlight, QColor(228, 68, 76))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)
    LOG.debug("dark theme applied (Fusion palette)")


class AdminOverlay(QWidget):
    """Full-window input-swallowing block shown when running elevated."""
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet("background-color: rgba(10, 10, 10, 225);")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box = QFrame()
        box.setStyleSheet("background-color: #2a0000; border: 3px solid #ff3333;"
                          "border-radius: 18px;")
        bl = QVBoxLayout(box)
        bl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = QLabel("🛑"); icon.setStyleSheet("font-size: 64px; background: transparent;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("ADMINISTRATOR MODE DETECTED")
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #ff5555; background: transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg = QLabel("Running this app as Administrator breaks Windows Drag & Drop\n"
                     "and other shell integrations (UIPI blocks drop messages).\n\n"
                     "The application is locked.\n"
                     "Close it and relaunch as a STANDARD user.")
        msg.setStyleSheet("font-size: 16px; color: #eeeeee; background: transparent;")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(icon); bl.addWidget(title); bl.addWidget(msg)
        box.setMaximumWidth(640)
        lay.addWidget(box)

    def eventFilter(self, obj, event):
        if obj is self:
            return False
        t = event.type()
        if t in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
                 QEvent.Type.MouseButtonDblClick, QEvent.Type.MouseMove,
                 QEvent.Type.KeyPress, QEvent.Type.KeyRelease, QEvent.Type.Wheel,
                 QEvent.Type.Shortcut, QEvent.Type.ShortcutOverride,
                 QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop,
                 QEvent.Type.FocusIn, QEvent.Type.FocusOut, QEvent.Type.ContextMenu):
            return True                      # swallow: app is locked
        return False


class MiniCell(QFrame):
    clicked = pyqtSignal(int)

    def __init__(self, index: int, clip, parent=None):
        super().__init__(parent)
        self.index, self.clip, self.fi = index, clip, 0
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.img = QLabel()
        self.img.setFixedSize(148, 72)
        self.img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img.setStyleSheet("background-color: #000;")
        self.name = QLabel(clip.name)
        self.name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.img); lay.addWidget(self.name)
        self.set_selected(False)

    def set_selected(self, on: bool):
        col = "#e4444c" if on else "#3a3a42"
        self.setStyleSheet(f"MiniCell {{ background-color: #232329; border: 2px solid {col};"
                           f"border-radius: 6px; }}")

    def mousePressEvent(self, e):
        self.clicked.emit(self.index)
        super().mousePressEvent(e)

    def advance(self, frames):
        if frames is None or frames.shape[0] == 0:
            return
        self.fi = (self.fi + 1) % frames.shape[0]
        pm = gray_to_pixmap(frames[self.fi])
        self.img.setPixmap(pm.scaled(148, 72, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.FastTransformation))


class MainWindow(QMainWindow):
    def __init__(self, opts=None):
        super().__init__()
        if opts is None:
            import types
            opts = types.SimpleNamespace(transparency="black", bg_lum=0.0, allow_unknown=False)
        self.opts = opts
        self.setWindowTitle(f"AniMe Matrix Studio Qt v{APP_VERSION}")
        self.resize(1240, 800)

        # ---- state
        self.packs, self.order, self.mat = [], [], {}
        self._packs_by_hash = {}
        self.edits, self.undo_stack, self.redo_stack = {}, [], []
        self.items, self.idx, self.sel_idx = [], 0, 0
        self.playing, self.preview_all = False, False
        self.mode, self.bg_lum, self.zoom = opts.transparency, opts.bg_lum, 4
        self.minis = []
        self._setting_scrub = False

        # ---- timers
        self.play_timer = QTimer(self); self.play_timer.setSingleShot(True)
        self.play_timer.timeout.connect(self._tick_play)
        self.mini_timer = QTimer(self); self.mini_timer.setInterval(90)
        self.mini_timer.timeout.connect(self._tick_minis); self.mini_timer.start()
        self.log_timer = QTimer(self); self.log_timer.setInterval(150)
        self.log_timer.timeout.connect(self._drain_log); self.log_timer.start()

        self._build_ui()

        # ---- admin lock (app opens, but is unusable)
        self.admin_overlay = None
        if is_admin():
            LOG.warning("ADMIN PRIVILEGES DETECTED -> locking UI")
            self.admin_overlay = AdminOverlay(self)
            self.admin_overlay.setGeometry(self.rect())
            self.admin_overlay.show(); self.admin_overlay.raise_()
            QApplication.instance().installEventFilter(self.admin_overlay)

        if hasattr(type(self), "build_panels"):      # wired by PART 4
            self.build_panels()
        self.rebuild()
        LOG.info("Qt GUI initialised (admin=%s)", is_admin())

    # ------------------------------------------------------------ UI skeleton
    def _build_ui(self):
        self.setAcceptDrops(True)
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)

        bar = QHBoxLayout()
        self.btn_add = QPushButton("Add…"); self.btn_add.clicked.connect(self.add_files)
        self.btn_export = QPushButton("Export GIF…"); self.btn_export.clicked.connect(self.on_export)
        self.btn_play = QPushButton("▶"); self.btn_play.setFixedWidth(36)
        self.btn_play.clicked.connect(self.toggle_play)
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setRange(0, 0)
        self.scrub.valueChanged.connect(self._on_scrub)
        self.lbl_pos = QLabel("0/0"); self.lbl_pos.setMinimumWidth(220)
        self.chk_all = QCheckBox("Preview all")
        self.chk_all.toggled.connect(self._on_preview_all)
        self.sp_zoom = QSpinBox(); self.sp_zoom.setRange(1, 16); self.sp_zoom.setValue(self.zoom)
        self.sp_zoom.valueChanged.connect(self._on_zoom)
        for txt, fn, w in (("←", lambda: self.move_sel(-1), 30),
                           ("→", lambda: self.move_sel(1), 30),
                           ("✕", self.remove_sel, 30),
                           ("Reset", self.reset_order, 60),
                           ("Clear all", self.clear_all, 70)):
            b = QPushButton(txt); b.setFixedWidth(w); b.clicked.connect(fn)
            bar.addWidget(b)
        self.btn_log = QPushButton("📜"); self.btn_log.setFixedWidth(36)
        self.btn_log.clicked.connect(self.open_log_window)
        self.btn_bug = QPushButton("🐛"); self.btn_bug.setFixedWidth(36)
        self.btn_bug.clicked.connect(self.export_diag)
        for w in (self.btn_add, self.btn_export, self.btn_play):
            bar.addWidget(w)
        bar.addWidget(self.scrub, 1)
        bar.addWidget(self.lbl_pos)
        bar.addWidget(self.chk_all)
        bar.addWidget(QLabel("zoom"))
        bar.addWidget(self.sp_zoom)
        bar.addWidget(self.btn_log); bar.addWidget(self.btn_bug)
        root.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        prev_wrap = QWidget(); pl = QVBoxLayout(prev_wrap); pl.setContentsMargins(0, 0, 0, 0)
        self.prev_scroll = QScrollArea(); self.prev_scroll.setWidgetResizable(True)
        self.prev_label = QLabel("No clips loaded.\nUse Add… or drag files here.")
        self.prev_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prev_scroll.setWidget(self.prev_label)
        pl.addWidget(self.prev_scroll)
        self.tabs = QTabWidget()                      # populated by PART 4
        self.edit_tab = QWidget(); self.tabs.addTab(self.edit_tab, "Edits")
        self.adv_tab = QWidget(); self.tabs.addTab(self.adv_tab, "Advanced")
        self.tabs.setMinimumWidth(340)
        split.addWidget(prev_wrap); split.addWidget(self.tabs)
        split.setSizes([760, 420])
        root.addWidget(split, 1)

        strip_scroll = QScrollArea(); strip_scroll.setFixedHeight(128)
        strip_scroll.setWidgetResizable(True)
        strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.strip_host = QWidget()
        self.strip_layout = QHBoxLayout(self.strip_host)
        self.strip_layout.setContentsMargins(4, 4, 4, 4)
        strip_scroll.setWidget(self.strip_host)
        root.addWidget(strip_scroll)

        # ---- shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self._step(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self._step(1))
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self.remove_sel)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.add_files)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self.on_export)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.redo)
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self.open_log_window)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.admin_overlay is not None:
            self.admin_overlay.setGeometry(self.rect())
            self.admin_overlay.raise_()

    def closeEvent(self, e):
        self.play_timer.stop(); self.mini_timer.stop(); self.log_timer.stop()
        LOG.info("Qt GUI closing")
        super().closeEvent(e)

    # ------------------------------------------------------------ drag & drop (native Qt)
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()]
        LOG.info("gui: drop -> %s", paths)
        for p in paths:
            self.load_path(p)
        self.rebuild()
        e.acceptProposedAction()

    # ------------------------------------------------------------ loading / order
    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add packs / GIFs", "", "Matrix / GIF (*.matrix *.zip *.gif);;All files (*)")
        LOG.info("gui: add dialog -> %s", paths or "cancelled")
        for p in paths:
            self.load_path(Path(p))
        self.rebuild()

    def load_path(self, p: Path):
        try:
            if p.is_dir():
                pk = load_png_sequence(p)
            elif p.suffix.lower() == ".gif":
                pk = load_loose_gif(p, self.mode, self.bg_lum)
            else:
                pk = load_pack(p, self.mode, self.bg_lum, self.opts.allow_unknown)
        except UnsupportedFormat as e:
            LOG.error("guard reject %s: %s", p.name, e)
            QMessageBox.critical(self, "Unsupported format", str(e)); return
        except StudioError as e:
            LOG.exception("load failed %s", p.name)
            QMessageBox.critical(self, "Load failed", str(e)); return
        except Exception as e:
            LOG.exception("unexpected load failure %s", p.name)
            QMessageBox.critical(self, "Load failed", f"{p.name}: {e}"); return
        if pk.file_hash in self._packs_by_hash:
            self.statusBar().showMessage(f"{p.name} already loaded", 4000); return
        self.packs.append(pk); self._packs_by_hash[pk.file_hash] = pk
        for c in pk.clips:
            if c not in self.order:
                self.order.append(c)
        for w in pk.warnings:
            LOG.warning("%s: %s", p.name, w)
        self.statusBar().showMessage(f"loaded {p.name}: {len(pk.clips)} clip(s)", 5000)

    def move_sel(self, d):
        i = self.sel_idx; j = i + d
        if i < 0 or j < 0 or j >= len(self.order): return
        self.order[i], self.order[j] = self.order[j], self.order[i]
        self.sel_idx = j
        LOG.info("gui: moved clip %d -> %d", i, j)
        self.rebuild()

    def remove_sel(self):
        if not self.order: return
        c = self.order.pop(self.sel_idx)
        LOG.info("gui: removed clip '%s' (remaining %d)", c.name, len(self.order))
        self.sel_idx = max(0, min(self.sel_idx, len(self.order) - 1)) if self.order else 0
        self.rebuild()

    def reset_order(self):
        self.order = [c for pk in self.packs for c in pk.clips]
        self.sel_idx = 0
        LOG.info("gui: order reset (%d clips)", len(self.order))
        self.rebuild()

    def clear_all(self):
        if not self.packs and not self.order: return
        if QMessageBox.question(self, "Clear all", "Remove all loaded packs and clips?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.packs.clear(); self.order.clear(); self._packs_by_hash.clear()
        self.mat.clear(); self.sel_idx = 0
        LOG.info("gui: cleared all packs/clips")
        self.rebuild()

    # ------------------------------------------------------------ edits / undo
    def edit_of(self, clip) -> EditState:
        key = (clip.pack_hash, clip.member)
        if key not in self.edits:
            self.edits[key] = EditState()
        return self.edits[key]

    def commit_edits(self):
        self.undo_stack.append(copy.deepcopy(self.edits))
        if len(self.undo_stack) > 60: self.undo_stack.pop(0)
        self.redo_stack.clear()
        LOG.debug("edit committed (undo depth %d)", len(self.undo_stack))
        self.rebuild()

    def undo(self):
        if not self.undo_stack: return
        self.redo_stack.append(copy.deepcopy(self.edits))
        self.edits = self.undo_stack.pop()
        LOG.debug("undo (depth %d)", len(self.undo_stack))
        self.rebuild()

    def redo(self):
        if not self.redo_stack: return
        self.undo_stack.append(copy.deepcopy(self.edits))
        self.edits = self.redo_stack.pop()
        self.rebuild()

    # ------------------------------------------------------------ pipeline
    def rebuild(self):
        for c in list(self.order):
            if c not in self.mat:
                try:
                    self.mat[c] = frames_of(c, self.mode, self.bg_lum)[0]
                except Exception:
                    LOG.exception("decode fail %s", c.name)
                    self.mat[c] = None
        self.sel_idx = max(0, min(self.sel_idx, len(self.order) - 1)) if self.order else 0
        self.items = []
        use = self.order if (self.preview_all or not self.order) else \
            ([self.order[self.sel_idx]] if 0 <= self.sel_idx < len(self.order) else [])
        for c in use:
            fr = self.mat.get(c)
            if fr is None: continue
            e = self.edit_of(c); n = fr.shape[0]; L = seq_len(e, n)
            for i in range(L):
                self.items.append((c, i, float(c.meta.delays[seq_index(e, i, n)])))
        self.idx = min(self.idx, max(0, len(self.items) - 1))
        self._setting_scrub = True
        self.scrub.setRange(0, max(0, len(self.items) - 1))
        self.scrub.setValue(self.idx)
        self._setting_scrub = False
        self._build_filmstrip()
        if hasattr(self, "sync_edit_panel"):
            self.sync_edit_panel()
        self.redraw()
        if self.playing: self._restart_play()

    def _build_filmstrip(self):
        while self.strip_layout.count():
            it = self.strip_layout.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self.minis = []
        for i, c in enumerate(self.order):
            cell = MiniCell(i, c, self)
            cell.clicked.connect(self.select_clip)
            self.strip_layout.addWidget(cell)
            self.minis.append(cell)
        self.strip_layout.addStretch(1)
        self._highlight()
        LOG.debug("filmstrip rebuilt: %d minis", len(self.minis))

    def _highlight(self):
        for k, cell in enumerate(self.minis):
            cell.set_selected(not self.preview_all and k == self.sel_idx)

    def select_clip(self, i):
        if i == self.sel_idx and not self.preview_all: return
        LOG.debug("gui: select clip %d", i)
        self.sel_idx = i; self.idx = 0
        if self.preview_all:
            self.preview_all = False
            self.chk_all.blockSignals(True); self.chk_all.setChecked(False); self.chk_all.blockSignals(False)
        self.rebuild()

    def _on_preview_all(self, on):
        self.preview_all = bool(on)
        LOG.debug("gui: preview_all=%s", self.preview_all)
        self.rebuild()

    def _on_zoom(self, v):
        self.zoom = int(v)
        logt("zoom", 0.5, "debug", "gui: zoom=%d", self.zoom)
        self.redraw()

    def on_decode_change(self):
        LOG.info("decode opts changed: transparency=%s bg_lum=%s", self.mode, self.bg_lum)
        self.mat.clear()
        self.rebuild()

    # ------------------------------------------------------------ playback / render
    def redraw(self):
        if not self.items:
            self.prev_label.setPixmap(QPixmap())
            self.prev_label.setText("No clips loaded.\nUse Add… or drag files here.")
            self.lbl_pos.setText("0/0")
            return
        c, i, delay = self.items[self.idx]
        arr = xframe(self.mat[c], self.edit_of(c), i)
        self.prev_label.setPixmap(gray_to_pixmap(arr, self.zoom))
        tot = sum(d for _, _, d in self.items) / 1000
        cur = sum(d for _, _, d in self.items[:self.idx]) / 1000
        self.lbl_pos.setText(f"{self.idx + 1}/{len(self.items)}  {cur:.1f}s/{tot:.1f}s "
                             f"({delay:.0f}ms)  [{c.name}]")

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.setText("❚❚" if self.playing else "▶")
        LOG.info("gui: playback %s", "started" if self.playing else "paused")
        if self.playing: self._restart_play()
        else: self.play_timer.stop()

    def _restart_play(self):
        if not self.items: return
        self.play_timer.start(int(max(15, min(self.items[self.idx][2], 2000))))

    def _tick_play(self):
        if not self.items: return
        self.idx = (self.idx + 1) % len(self.items)
        self._setting_scrub = True
        self.scrub.setValue(self.idx)
        self._setting_scrub = False
        self.redraw()
        self._restart_play()

    def _step(self, d):
        if not self.items: return
        if self.playing: self.toggle_play()
        self.idx = max(0, min(len(self.items) - 1, self.idx + d))
        self._setting_scrub = True
        self.scrub.setValue(self.idx)
        self._setting_scrub = False
        self.redraw()

    def _on_scrub(self, val):
        if self._setting_scrub or not self.items: return
        if self.playing: self.toggle_play()
        self.idx = int(max(0, min(len(self.items) - 1, val)))
        logt("scrub", 0.5, "debug", "gui: scrub -> frame %d", self.idx)
        self.redraw()

    def _tick_minis(self):
        for cell in self.minis:
            cell.advance(self.mat.get(cell.clip))

    def _drain_log(self):
        view = getattr(self, "log_view", None)
        if view is None: return
        q = BRIDGE.queue
        if not q: return
        view.moveCursor(__import__("PyQt6.QtGui", fromlist=["QTextCursor"]).QTextCursor.MoveOperation.End)
        while q:
            view.appendPlainText(q.popleft())
        view.ensureCursorVisible()

    # ------------------------------------------------------------ diagnostics
    def export_diag(self):
        LOG.info("bug button pressed -> diagnostic export dialog")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export diagnostic log",
            f"matrix_studio_log_{datetime.now():%Y%m%d_%H%M%S}.txt",
            "Text (*.txt);;Log (*.log)")
        if not path:
            LOG.info("diagnostic export cancelled"); return
        try:
            Path(path).write_text(build_diag_text(self), encoding="utf-8")
        except Exception:
            LOG.exception("diagnostic export FAILED")
            QMessageBox.critical(self, "Export failed", "Could not write the log file.")
            return
        LOG.info("diagnostic log exported: %s", path)
        QMessageBox.information(self, "Diagnostic log saved",
                                f"{path}\nShare this file when reporting issues.")

    def diag_state(self):
        L = [f"packs loaded    : {len(self.packs)}"]
        for pk in self.packs:
            L.append(f"  pack {pk.path.name} sha={pk.file_hash} clips={len(pk.clips)}")
            for c in pk.clips:
                L.append(f"    clip {c.name}: {c.meta.w}x{c.meta.h} n={c.meta.n} "
                         f"mean={c.meta.delays.mean():.1f}ms")
        L.append(f"order           : {[c.name for c in self.order]}")
        L.append(f"selection       : {self.sel_idx}  preview_all={self.preview_all}")
        L.append(f"items/idx       : {len(self.items)}/{self.idx}  playing={self.playing}")
        L.append(f"transparency    : {self.mode}  bg_lum={self.bg_lum}  zoom={self.zoom}")
        L.append(f"admin           : {is_admin()}")
        return L

    # ------------------------------------------------------------ export (sync fallback;
    # threaded version is installed by PART 4)
    def on_export(self):
        if not self.order:
            QMessageBox.warning(self, "Nothing to export", "Add packs first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Export GIF", "combined.gif", "GIF (*.gif)")
        if not path:
            LOG.info("gui: export dialog cancelled"); return
        seqs = []
        for c in self.order:
            fr = self.mat.get(c)
            if fr is None: continue
            seqs.append(clip_sequence(fr, c.meta.delays, self.edit_of(c)))
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            n, secs = export_gif(seqs, Path(path), scale=self.zoom)
            self.statusBar().showMessage(f"wrote {Path(path).name}: {n} frames, {secs:.2f}s", 8000)
            LOG.info("gui: exported %s (%d frames, %.2fs)", path, n, secs)
        except Exception as e:
            LOG.exception("export failed")
            QMessageBox.critical(self, "Export failed", str(e))
        finally:
            QApplication.restoreOverrideCursor()


# ==== PART 3 END ====
# ============================================================================
# PART 4: PANELS, THREADED EXPORT, PROJECTS/PRESETS, LOG WINDOW, ENTRY POINT
# ============================================================================
from dataclasses import asdict
from PyQt6.QtWidgets import (QDialog, QProgressDialog, QFormLayout, QGridLayout,
                             QInputDialog, QPlainTextEdit)
from PyQt6.QtCore import QThread
from PyQt6.QtGui import QTextCursor


class ExportThread(QThread):
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(object)

    def __init__(self, sequences, path, scale, parent=None):
        super().__init__(parent)
        self.sequences, self.path, self.scale = sequences, path, scale
        self.cancel = threading.Event()

    def run(self):
        try:
            res = export_gif(self.sequences, Path(self.path), scale=self.scale,
                             progress=lambda d, t: self.progress.emit(d, t),
                             cancel=self.cancel)
            self.finished_ok.emit(res)
        except Exception as e:
            LOG.exception("export failed")
            self.finished_ok.emit(("error", str(e)))


class TestThread(QThread):
    line = pyqtSignal(str)

    def __init__(self, packs, mode, bg_lum, parent=None):
        super().__init__(parent)
        self.packs, self.mode, self.bg_lum = packs, mode, bg_lum

    def run(self):
        for name, passed, detail in builtin_tests():
            self.line.emit(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")
        for name, passed, detail in run_golden_tests(self.packs, self.mode, self.bg_lum):
            tag = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
            self.line.emit(f"[{tag}] {name} {detail}")
        self.line.emit("self-test finished")


class SnapThread(QThread):
    done = pyqtSignal(str)

    def __init__(self, packs, mode, bg_lum, parent=None):
        super().__init__(parent)
        self.packs, self.mode, self.bg_lum = packs, mode, bg_lum

    def run(self):
        self.done.emit(str(snapshot_goldens(self.packs, self.mode, self.bg_lum)))


# ---------------------------------------------------------------- Edits/Advanced panels
def _build_panels(self):
    # ---------------- Edits tab ----------------
    lay = QGridLayout(self.edit_tab)
    self._syncing = False
    self.lbl_edit_info = QLabel("no clip selected")
    lay.addWidget(self.lbl_edit_info, 0, 0, 1, 4)
    self.chk_rev = QCheckBox("Reverse"); self.chk_pp = QCheckBox("Ping-pong")
    lay.addWidget(self.chk_rev, 1, 0); lay.addWidget(self.chk_pp, 1, 1)
    lay.addWidget(QLabel("dx"), 2, 0)
    self.sp_dx = QSpinBox(); self.sp_dx.setRange(-200, 200); lay.addWidget(self.sp_dx, 2, 1)
    lay.addWidget(QLabel("dy"), 2, 2)
    self.sp_dy = QSpinBox(); self.sp_dy.setRange(-200, 200); lay.addWidget(self.sp_dy, 2, 3)
    lay.addWidget(QLabel("scale"), 3, 0)
    self.sp_esc = QDoubleSpinBox(); self.sp_esc.setRange(0.1, 10.0); self.sp_esc.setSingleStep(0.25)
    lay.addWidget(self.sp_esc, 3, 1)
    lay.addWidget(QLabel("bright"), 3, 2)
    self.sp_br = QSpinBox(); self.sp_br.setRange(-128, 128); lay.addWidget(self.sp_br, 3, 3)
    lay.addWidget(QLabel("contrast"), 4, 0)
    self.sp_co = QDoubleSpinBox(); self.sp_co.setRange(0.1, 5.0); self.sp_co.setSingleStep(0.1)
    lay.addWidget(self.sp_co, 4, 1)
    lay.addWidget(QLabel("gamma"), 4, 2)
    self.sp_ga = QDoubleSpinBox(); self.sp_ga.setRange(0.1, 5.0); self.sp_ga.setSingleStep(0.1)
    lay.addWidget(self.sp_ga, 4, 3)
    btn_crop = QPushButton("Crop…"); btn_reset = QPushButton("Reset edits")
    lay.addWidget(btn_crop, 5, 0, 1, 2); lay.addWidget(btn_reset, 5, 2, 1, 2)
    btn_undo = QPushButton("↺ Undo"); btn_redo = QPushButton("↻ Redo")
    lay.addWidget(btn_undo, 6, 0, 1, 2); lay.addWidget(btn_redo, 6, 2, 1, 2)
    btn_crop.clicked.connect(self._crop_dialog)
    btn_reset.clicked.connect(self._reset_edit)
    btn_undo.clicked.connect(self.undo); btn_redo.clicked.connect(self.redo)
    for w in (self.chk_rev, self.chk_pp, self.sp_dx, self.sp_dy,
              self.sp_esc, self.sp_br, self.sp_co, self.sp_ga):
        sig = w.stateChanged if isinstance(w, QCheckBox) else w.valueChanged
        sig.connect(self._on_edit_change)

    # ---------------- Advanced tab ----------------
    alay = QVBoxLayout(self.adv_tab)
    form = QFormLayout()
    self.cmb_trans = QComboBox(); self.cmb_trans.addItems(["black", "bg", "ignore"])
    self.cmb_trans.setCurrentText(self.mode)
    self.sp_bglum = QDoubleSpinBox(); self.sp_bglum.setRange(0, 255); self.sp_bglum.setValue(self.bg_lum)
    form.addRow("Transparency", self.cmb_trans)
    form.addRow("BG luminance", self.sp_bglum)
    alay.addLayout(form)
    self.cmb_trans.currentTextChanged.connect(self._on_trans)
    self.sp_bglum.valueChanged.connect(self._on_bglum)
    self.chk_cache = QCheckBox("Decode cache (disk)"); self.chk_cache.setChecked(CACHE.enabled)
    self.chk_cache.toggled.connect(lambda on: setattr(CACHE, "enabled", bool(on)))
    alay.addWidget(self.chk_cache)
    btn_clear = QPushButton("Clear cache"); btn_clear.clicked.connect(lambda: CACHE.clear())
    alay.addWidget(btn_clear)
    self.chk_allow = QCheckBox("Allow unknown format versions")
    self.chk_allow.setChecked(bool(getattr(self.opts, "allow_unknown", False)))
    self.chk_allow.toggled.connect(lambda on: setattr(self.opts, "allow_unknown", bool(on)))
    alay.addWidget(self.chk_allow)
    self.chk_verbose = QCheckBox("verbose console (DEBUG)")
    self.chk_verbose.toggled.connect(
        lambda on: BRIDGE.setLevel(logging.DEBUG if on else logging.INFO))
    alay.addWidget(self.chk_verbose)
    grid2 = QGridLayout()
    b_test = QPushButton("Run self-test"); b_snap = QPushButton("Snapshot goldens")
    b_reload = QPushButton("Reload packs"); b_gen = QPushButton("＋ Generate")
    b_imp = QPushButton("📥 Import GIF/PNG"); b_psav = QPushButton("💾 Project")
    b_pload = QPushButton("📂 Project"); b_prsav = QPushButton("★ Save preset")
    b_prapp = QPushButton("★ Apply preset")
    for i, b in enumerate((b_test, b_snap, b_reload, b_gen, b_imp,
                           b_psav, b_pload, b_prsav, b_prapp)):
        grid2.addWidget(b, i // 2, i % 2)
    alay.addLayout(grid2)
    b_test.clicked.connect(self.run_selftest)
    b_snap.clicked.connect(self.run_snapshot)
    b_reload.clicked.connect(self.reload_packs)
    b_gen.clicked.connect(self.generate)
    b_imp.clicked.connect(self.import_loose)
    b_psav.clicked.connect(self.save_project)
    b_pload.clicked.connect(self.load_project)
    b_prsav.clicked.connect(self.save_preset)
    b_prapp.clicked.connect(self.apply_preset)
    alay.addWidget(QLabel("Session log:"))
    self.log_view = QPlainTextEdit(); self.log_view.setReadOnly(True)
    self.log_view.setMaximumBlockCount(4000)
    alay.addWidget(self.log_view, 1)
    self.log_dialog_view = None
    self._log_dialog = None
    self.log_view.setPlainText("\n".join(FMT.format(r) for r in list(RING.buffer)))


def _sel_clip(self):
    if self.order and 0 <= self.sel_idx < len(self.order):
        return self.order[self.sel_idx]
    return None


def _sync_edit_panel(self):
    self._syncing = True
    try:
        c = self.sel_clip()
        if c is None:
            self.lbl_edit_info.setText("no clip selected")
            return
        e = self.edit_of(c)
        self.lbl_edit_info.setText(
            f"editing: {c.name}  ({c.meta.w}×{c.meta.h}, {c.meta.n}f)")
        self.chk_rev.setChecked(e.reverse); self.chk_pp.setChecked(e.pingpong)
        self.sp_dx.setValue(e.dx); self.sp_dy.setValue(e.dy)
        self.sp_esc.setValue(e.scale); self.sp_br.setValue(e.brightness)
        self.sp_co.setValue(e.contrast); self.sp_ga.setValue(e.gamma)
    finally:
        self._syncing = False


def _on_edit_change(self, *a):
    if self._syncing: return
    c = self.sel_clip()
    if c is None: return
    e = self.edit_of(c)
    e.reverse = self.chk_rev.isChecked(); e.pingpong = self.chk_pp.isChecked()
    e.dx = int(self.sp_dx.value()); e.dy = int(self.sp_dy.value())
    e.scale = float(self.sp_esc.value()); e.brightness = int(self.sp_br.value())
    e.contrast = float(self.sp_co.value()); e.gamma = float(self.sp_ga.value())
    self.commit_edits()


def _crop_dialog(self):
    c = self.sel_clip()
    if c is None: return
    e = self.edit_of(c)
    cur = ",".join(map(str, e.crop)) if e.crop else ""
    s, ok = QInputDialog.getText(self, "Crop", "x,y,w,h  (empty = no crop):", text=cur)
    if not ok: return
    try:
        vals = [int(v) for v in s.split(",")] if s.strip() else None
        e.crop = vals if (vals and len(vals) == 4) else None
    except ValueError:
        e.crop = None
    self.commit_edits()


def _reset_edit(self):
    c = self.sel_clip()
    if c is None: return
    self.edits[(c.pack_hash, c.member)] = EditState()
    self.commit_edits()


def _on_trans(self, text):
    self.mode = text
    self.on_decode_change()


def _on_bglum(self, v):
    self.bg_lum = float(v)
    self.on_decode_change()


# ---------------------------------------------------------------- advanced actions
def _reload_packs(self):
    paths = [pk.path for pk in self.packs]
    self.packs.clear(); self.order.clear(); self._packs_by_hash.clear(); self.mat.clear()
    for p in paths:
        self.load_path(p)
    self.rebuild()


def _run_selftest(self):
    self.tabs.setCurrentWidget(self.adv_tab)
    self._test_thread = TestThread(self.packs, self.mode, self.bg_lum, self)
    self._test_thread.line.connect(self.log_view.appendPlainText)
    self._test_thread.start()


def _run_snapshot(self):
    self._snap_thread = SnapThread(self.packs, self.mode, self.bg_lum, self)
    self._snap_thread.done.connect(
        lambda d: (self.log_view.appendPlainText(f"goldens written to {d}"),
                   self.statusBar().showMessage(f"goldens written to {d}", 6000)))
    self._snap_thread.start()


def _generate(self):
    kind, ok = QInputDialog.getItem(self, "Generate", "kind:", list(GENERATORS), 0, False)
    if not ok: return
    kw = {}
    if kind == "marquee":
        txt, ok2 = QInputDialog.getText(self, "Marquee", "Text:", text="ROG")
        if not ok2: return
        kw["text"] = txt or "ROG"
    clip = generated_clip(kind, **kw)
    clip.member = f"{clip.member}_{len(self.order)}"     # keep edits keys unique
    self.packs.append(Pack(Path("<generated>"), clip.pack_hash, None, [clip], []))
    self.order.append(clip)
    self.rebuild()


def _import_loose(self):
    p, _ = QFileDialog.getOpenFileName(self, "Import GIF", "", "GIF (*.gif);;All files (*)")
    if not p:
        d = QFileDialog.getExistingDirectory(self, "…or pick a PNG-sequence folder")
        if not d: return
        self.load_path(Path(d)); self.rebuild(); return
    self.load_path(Path(p)); self.rebuild()


# ---------------------------------------------------------------- projects & presets
def _presets_file(self):
    return data_dir() / "presets_qt.json"


def _save_project(self):
    p, _ = QFileDialog.getSaveFileName(self, "Save project", "project.m2gproj",
                                       "Matrix project (*.m2gproj)")
    if not p: return
    data = {"packs": [str(pk.path) for pk in self.packs if pk.path.exists()],
            "order": [[c.pack_hash, c.member] for c in self.order],
            "edits": {f"{k[0]}|{k[1]}": asdict(v) for k, v in self.edits.items()},
            "settings": {"mode": self.mode, "bg_lum": self.bg_lum, "zoom": self.zoom,
                         "preview_all": self.preview_all}}
    Path(p).write_text(json.dumps(data, indent=1), encoding="utf-8")
    LOG.info("project saved -> %s", p)
    self.statusBar().showMessage(f"project saved: {Path(p).name}", 5000)


def _load_project(self):
    p, _ = QFileDialog.getOpenFileName(self, "Load project", "", "Matrix project (*.m2gproj)")
    if not p: return
    try:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as e:
        QMessageBox.critical(self, "Project load failed", str(e)); return
    self.packs.clear(); self.order.clear(); self._packs_by_hash.clear(); self.mat.clear()
    for ps in d.get("packs", []):
        if Path(ps).exists():
            self.load_path(Path(ps))
    bykey = {(c.pack_hash, c.member): c for c in self.order}
    wanted = {tuple(k) for k in d.get("order", [])}
    new_order = [bykey[tuple(k)] for k in d.get("order", []) if tuple(k) in bykey]
    new_order += [c for c in self.order if (c.pack_hash, c.member) not in wanted]
    self.order = new_order
    self.edits = {tuple(k.split("|", 1)): EditState(**v)
                  for k, v in d.get("edits", {}).items() if "|" in k}
    s = d.get("settings", {})
    self.mode = s.get("mode", self.mode); self.bg_lum = s.get("bg_lum", self.bg_lum)
    self.zoom = int(s.get("zoom", self.zoom)); self.sp_zoom.setValue(self.zoom)
    self.preview_all = bool(s.get("preview_all", False))
    self.chk_all.setChecked(self.preview_all)
    self.rebuild()
    LOG.info("project loaded -> %s", p)


def _save_preset(self):
    c = self.sel_clip()
    if c is None: return
    name, ok = QInputDialog.getText(self, "Preset", "Preset name:")
    if not ok or not name: return
    allp = json.loads(self._presets_file().read_text()) if self._presets_file().exists() else {}
    allp[name] = asdict(self.edit_of(c))
    self._presets_file().write_text(json.dumps(allp, indent=1), encoding="utf-8")
    LOG.info("preset saved: %s", name)


def _apply_preset(self):
    c = self.sel_clip()
    if c is None: return
    if not self._presets_file().exists():
        QMessageBox.information(self, "Presets", "No presets saved yet."); return
    allp = json.loads(self._presets_file().read_text())
    name, ok = QInputDialog.getItem(self, "Preset", "Apply preset:", sorted(allp), 0, False)
    if not ok or name not in allp: return
    self.edits[(c.pack_hash, c.member)] = EditState(**allp[name])
    self.commit_edits()


# ---------------------------------------------------------------- threaded export
def _export_threaded(self):
    if not self.order:
        QMessageBox.warning(self, "Nothing to export", "Add packs first."); return
    p, _ = QFileDialog.getSaveFileName(self, "Export GIF", "combined.gif", "GIF (*.gif)")
    if not p:
        LOG.info("gui: export dialog cancelled"); return
    seqs = []
    for c in self.order:
        fr = self.mat.get(c)
        if fr is None: continue
        seqs.append(clip_sequence(fr, c.meta.delays, self.edit_of(c)))
    total = sum(len(f) for f, _ in seqs)
    dlg = QProgressDialog("Exporting…", "Cancel", 0, max(1, total), self)
    dlg.setWindowModality(Qt.WindowModality.WindowModal)
    dlg.setMinimumDuration(0); dlg.setAutoClose(False); dlg.setAutoReset(False)
    self._export_thread = ExportThread(seqs, p, self.zoom, self)
    self._export_thread.progress.connect(lambda d, t: (dlg.setMaximum(t), dlg.setValue(d)))
    dlg.canceled.connect(self._export_thread.cancel.set)

    def _fin(res):
        dlg.close()
        if isinstance(res, tuple) and res and isinstance(res[0], str):
            QMessageBox.critical(self, "Export failed", res[1]); return
        if res is None:
            self.statusBar().showMessage("export cancelled", 5000)
            LOG.info("gui: export cancelled"); return
        n, secs = res
        self.statusBar().showMessage(f"wrote {Path(p).name}: {n} frames, {secs:.2f}s", 8000)
        LOG.info("gui: exported %s (%d frames, %.2fs)", p, n, secs)

    self._export_thread.finished_ok.connect(_fin)
    self._export_thread.start()


# ---------------------------------------------------------------- log window
def _open_log_window(self):
    if getattr(self, "_log_dialog", None) is not None:
        self._log_dialog.raise_(); self._log_dialog.activateWindow(); return
    dlg = QDialog(self)
    dlg.setWindowTitle("Session log (live)")
    dlg.resize(900, 480)
    vl = QVBoxLayout(dlg)
    view = QPlainTextEdit(); view.setReadOnly(True); view.setMaximumBlockCount(20000)
    view.setPlainText("\n".join(FMT.format(r) for r in list(RING.buffer)))
    vl.addWidget(view)
    hb = QHBoxLayout()
    b_copy = QPushButton("Copy full log"); b_save = QPushButton("Save diagnostic…")
    b_close = QPushButton("Close")
    hb.addWidget(b_copy); hb.addWidget(b_save); hb.addStretch(1); hb.addWidget(b_close)
    vl.addLayout(hb)
    b_copy.clicked.connect(lambda: (
        QApplication.clipboard().setText("\n".join(FMT.format(r) for r in list(RING.buffer))),
        LOG.info("log copied to clipboard (%d records)", len(RING.buffer))))
    b_save.clicked.connect(self.export_diag)
    b_close.clicked.connect(dlg.close)
    dlg.finished.connect(self._on_log_dialog_closed)
    self._log_dialog = dlg
    self.log_dialog_view = view
    dlg.show()
    LOG.info("gui: log window opened")


def _on_log_dialog_closed(self):
    self._log_dialog = None
    self.log_dialog_view = None


def _drain_log(self):
    q = BRIDGE.queue
    if not q: return
    lines = []
    while q: lines.append(q.popleft())
    for view in (getattr(self, "log_view", None), getattr(self, "log_dialog_view", None)):
        if view is not None:
            view.moveCursor(QTextCursor.MoveOperation.End)
            for ln in lines: view.appendPlainText(ln)
            view.ensureCursorVisible()


# ---------------------------------------------------------------- wire into MainWindow
MainWindow.build_panels = _build_panels
MainWindow.sel_clip = _sel_clip
MainWindow.sync_edit_panel = _sync_edit_panel
MainWindow._on_edit_change = _on_edit_change
MainWindow._crop_dialog = _crop_dialog
MainWindow._reset_edit = _reset_edit
MainWindow._on_trans = _on_trans
MainWindow._on_bglum = _on_bglum
MainWindow.reload_packs = _reload_packs
MainWindow.run_selftest = _run_selftest
MainWindow.run_snapshot = _run_snapshot
MainWindow.generate = _generate
MainWindow.import_loose = _import_loose
MainWindow.save_project = _save_project
MainWindow.load_project = _load_project
MainWindow._presets_file = _presets_file
MainWindow.save_preset = _save_preset
MainWindow.apply_preset = _apply_preset
MainWindow.on_export = _export_threaded          # replaces the sync fallback
MainWindow.open_log_window = _open_log_window
MainWindow._on_log_dialog_closed = _on_log_dialog_closed
MainWindow._drain_log = _drain_log               # feeds inline + floating log views


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    install_excepthooks()
    LOG.info("=" * 40 + " PROCESS START (Qt) " + "=" * 40)
    argv = sys.argv[1:]
    if any(a in ("--selftest", "--snapshot-goldens") for a in argv):
        import argparse as _ap
        ap = _ap.ArgumentParser(prog="matrix_studio_qt")
        ap.add_argument("inputs", nargs="*")
        ap.add_argument("--selftest", action="store_true")
        ap.add_argument("--snapshot-goldens", action="store_true")
        ap.add_argument("--transparency", default="black")
        ap.add_argument("--bg-lum", type=float, default=0.0)
        o = ap.parse_args(argv)
        paths = [Path(x) for x in o.inputs if Path(x).exists()]
        packs = [load_pack(p, o.transparency, o.bg_lum) for p in paths]
        ok = True
        if o.selftest:
            for name, passed, detail in builtin_tests():
                print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}"); ok &= bool(passed)
            for name, passed, detail in run_golden_tests(packs, o.transparency, o.bg_lum):
                tag = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
                print(f"[{tag}] {name} {detail}"); ok &= passed is not False
        if o.snapshot_goldens and packs:
            print(f"[ok] goldens -> {snapshot_goldens(packs, o.transparency, o.bg_lum)}")
        return 0 if ok else 1

    app = QApplication(sys.argv)
    app.setApplicationName("AniMe Matrix Studio Qt")
    apply_dark_theme(app)
    import types
    opts = types.SimpleNamespace(transparency="black", bg_lum=0.0, allow_unknown=False)
    win = MainWindow(opts)
    win.show()
    if is_admin():
        LOG.warning("running elevated: UI locked by AdminOverlay")
    sys.exit(app.exec())


if __name__ == "__main__":
    sys.exit(main())

# ==== PART 4 END ====
