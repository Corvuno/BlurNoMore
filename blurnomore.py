#!/usr/bin/env python3
"""
BlurNoMore - turn a folder of small, blurry drawings into one sharp, print-ready A4 PDF for tracing,
or make bigger, AI-restored copies of any pictures ("Upscale images").

Start it by double-clicking BlurNoMore.bat (or run:  python blurnomore.py).
That opens a window where you choose the folder, how many images and all settings.

Command line (optional):
    python blurnomore.py "D:\\drawings" --count 6 --pick spread --thicken 70

Per image it: crops empty paper, upscales with Real-ESRGAN (GPU) when the image is
too small for A4 at the chosen dpi, strengthens very thin lines, fits it on A4 and
writes one multi-page PDF with the date and time in its name. Originals are never
touched. Work files and logs go in <save folder>\\_blurnomore_work.

Upscale mode (--mode upscale) only enlarges: 2x, 3x or 4x, into a new folder
BlurNoMore_upscaled_<date>_<time> in the save folder. --preview shows up to 4 images
(original next to result, plus close-ups) as picture sheets before a real run.
"""
import argparse
import hashlib
import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import img2pdf
except ImportError:
    sys.exit("Missing packages. Run:  pip install pillow numpy img2pdf")

Image.MAX_IMAGE_PIXELS = None

VERSION = "4.0"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
A4_MM = (210.0, 297.0)
DEFAULT_FOLDER = str(Path.home() / "Pictures")
ESRGAN_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
              "realesrgan-ncnn-vulkan-20220424-windows.zip")
MODEL_GENERAL = "realesrgan-x4plus"
MODEL_ANIME = "realesrgan-x4plus-anime"
AI_MIN_SCALE = 1.25  # below this much enlargement, plain Lanczos is enough
APP_DIR = Path(__file__).parent
SETTINGS_FILE = APP_DIR / "blurnomore_settings.json"
ESRGAN_DIR = APP_DIR / "realesrgan"   # downloaded once, next to the program
MODELS_DIR = APP_DIR / "models"       # extra models: NAME.param + NAME.bin pairs
BIG_RESULT_PIXELS = 100_000_000


# ===================================================================== options
@dataclass
class Options:
    folder: str = DEFAULT_FOLDER
    count: int = 0               # 0 = all images
    pick: str = "spread"         # "spread" (evenly over the folder) or "first"
    repeat_last: bool = False    # run again on exactly the images used last time
    margin_mm: float = 10.0      # white border on the A4 page
    fit: str = "whole"           # "whole" picture on the page, or "fill" the page (crops edges)
    dpi: int = 300
    autocrop: bool = True        # cut away empty paper around the drawing
    thicken: int = 70            # 0-100, strengthen very thin dark lines
    fade: int = 0                # 0-100, lighten the whole print
    keep_color: bool = True      # False = print everything in grey
    color_boost: int = 0         # 0-100, stronger colours + contrast
    use_ai: bool = True          # Real-ESRGAN upscaling
    color_model: str = "general"  # "general" or "anime" (coloured art only)
    redo: bool = False           # ignore pages made in earlier runs
    mode: str = "pdf"            # "pdf" = Tracing PDF, "upscale" = Upscale images
    save_to: str = ""            # where results and _blurnomore_work go; empty = the images folder
    scale: int = 2               # upscale mode: 2, 3 or 4 times bigger
    up_model: str = "auto"       # upscale mode: "auto", "general", "anime" or an extra model's name
    file_type: str = "png"       # upscale mode: "png" or "jpg"
    open_when_done: bool = True  # GUI only


# ================================================================ image basics
def load_rgb(path):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    return im.convert("RGB")


def classify(im):
    """Return (class, stats). Classes: lineart, shaded (black & white), flatcolor, photo (colour)."""
    small = im.copy()
    small.thumbnail((800, 800))
    hsv = np.asarray(small.convert("HSV")).astype(np.float32) / 255.0
    color_frac = float((hsv[..., 1] > 0.2).mean())
    g = np.asarray(small.convert("L"))
    mid_frac = float(((g > 70) & (g < 185)).mean())
    stats = {"color_frac": round(color_frac, 3), "mid_frac": round(mid_frac, 3)}
    if color_frac < 0.03:  # effectively monochrome
        return ("lineart" if mid_frac < 0.12 else "shaded"), stats
    q = np.asarray(small).astype(np.uint32) >> 3
    packed = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    n_colors = int(len(np.unique(packed)))
    stats["colors"] = n_colors
    return ("flatcolor" if n_colors < 1500 else "photo"), stats


def find_content_box(im, pad_frac=0.02, min_gain=0.04):
    """Bounding box (in pixels of im) of the drawing, without the blank paper around it.
    Only acts when the image has a plain, light, neutral paper border (a scan or photo of
    a drawing). Pictures that run to the edge, such as full-colour art or art with a
    black or coloured frame, are left alone. Returns None when there is nothing to crop."""
    w, h = im.size
    small = im.copy()
    small.thumbnail((1000, 1000))
    sw, sh = small.size
    a = np.asarray(small).astype(np.int16)
    b = max(2, int(0.015 * max(sw, sh)))
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                             a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    paper = np.median(border, axis=0)
    if paper.mean() < 150 or (paper.max() - paper.min()) > 60:
        return None                       # dark or coloured edge: it is part of the picture
    if (np.abs(border - paper).max(axis=1) < 25).mean() < 0.8:
        return None                       # edge is not plain paper
    ink = np.abs(a - paper).max(axis=2) > 45
    tot = float(ink.sum())
    if tot < 0.002 * sw * sh:
        return None
    # Trim 0.5% of the ink mass from each side: ignores dust, scan edges and stray
    # annotations, while thin frame lines (lots of ink) stay inside.
    q = 0.005
    cc = np.cumsum(ink.sum(axis=0)) / tot
    rc = np.cumsum(ink.sum(axis=1)) / tot
    xs = (int(np.searchsorted(cc, q)), int(np.searchsorted(cc, 1 - q)) + 1)
    ys = (int(np.searchsorted(rc, q)), int(np.searchsorted(rc, 1 - q)) + 1)
    pad = pad_frac * max(sw, sh)
    x0, x1 = max(0, xs[0] - pad), min(sw, xs[1] + pad)
    y0, y1 = max(0, ys[0] - pad), min(sh, ys[1] + pad)
    if ((x1 - x0) * (y1 - y0)) / float(sw * sh) > 1 - min_gain:
        return None
    k = w / float(sw)
    return (int(x0 * k), int(y0 * k), min(w, int(x1 * k)), min(h, int(y1 * k)))


def page_geometry(w, h, dpi, margin_mm, fill=False):
    landscape = (w / h) > 1.15
    pw_mm, ph_mm = (A4_MM[1], A4_MM[0]) if landscape else A4_MM
    px = lambda mm: int(round(mm / 25.4 * dpi))
    page = (px(pw_mm), px(ph_mm))
    box = (px(pw_mm - 2 * margin_mm), px(ph_mm - 2 * margin_mm))
    s = max(box[0] / w, box[1] / h) if fill else min(box[0] / w, box[1] / h)
    target = (max(1, round(w * s)), max(1, round(h * s)))
    return page, target, s, box


# ================================================================ Real-ESRGAN
def _find_exe(base):
    if not base.exists():
        return None
    for p in base.rglob("realesrgan-ncnn-vulkan*"):
        if p.is_file() and p.suffix.lower() in (".exe", ""):
            return p
    return None


def extra_models():
    """Names of the extra models in the models folder next to the program (sorted)."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(p.name[:-6] for p in MODELS_DIR.glob("*.param") if (MODELS_DIR / (p.name[:-6] + ".bin")).is_file())


def model_scale(name):
    """Native enlargement of an extra model, read from its name (2x / x2, 3x / x3, else 4)."""
    m = re.search(r"(?<![0-9])(?:([23])x|x([23]))(?![0-9])", name.lower())
    return int(m.group(1) or m.group(2)) if m else 4


def ensure_esrgan(work, log):
    # Next to the program is the home of Real-ESRGAN; the old place (in the work folder) still counts.
    for base in (ESRGAN_DIR, work / "realesrgan"):
        exe = _find_exe(base)
        if exe:
            return exe
    d = ESRGAN_DIR
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_test"
        probe.write_text("x")
        probe.unlink()
    except OSError:
        d = work / "realesrgan"           # the program folder is not writable
    try:
        log("Downloading Real-ESRGAN (about 45 MB) ...")
        d.mkdir(parents=True, exist_ok=True)
        z = d / "realesrgan.zip"
        urllib.request.urlretrieve(ESRGAN_URL, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
        return _find_exe(d)
    except Exception as e:  # noqa: BLE001
        log(f"  Could not download Real-ESRGAN: {e}")
        log(f"  Download it manually from {ESRGAN_URL}")
        log(f"  and unzip it into {d}. Continuing with plain resizing only.")
        return None


def run_esrgan(img, model, exe, tmp, log, model_dir=None, native=4):
    tin, tout = tmp / "in.png", tmp / "out.png"
    if tout.exists():
        tout.unlink()
    img.save(tin)
    cmd = [str(exe), "-i", str(tin), "-o", str(tout), "-n", model,
           "-s", str(native), "-m", str(model_dir or exe.parent / "models"), "-f", "png"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tout.exists():
        log("  Real-ESRGAN failed. Output:")
        log("  " + (r.stderr or r.stdout or "(no output)")[-600:].strip().replace("\n", "\n  "))
        return None
    out = Image.open(tout).convert("RGB")
    out.load()
    return out


# ============================================================ tone & line work
def strengthen_thin(im, amount, k=7):
    """Darken/widen only dark lines narrower than ~k px; thick lines and fills are left alone."""
    if amount <= 0:
        return im
    g = ImageOps.grayscale(im)
    closed = g.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
    diff = np.asarray(closed).astype(np.float32) - np.asarray(g).astype(np.float32)
    mask = np.clip(diff / 50.0, 0, 1)
    mask = Image.fromarray((mask * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(1))
    m = (np.asarray(mask).astype(np.float32) / 255.0 * amount)[..., None]
    widened = np.asarray(im.filter(ImageFilter.MinFilter(3))).astype(np.float32)
    base = np.asarray(im).astype(np.float32)
    out = base * (1 - m) + widened * m
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB")


def _levels(g, bp, wp, scurve=0.0):
    a = np.asarray(g).astype(np.float32)
    a = np.clip((a - bp) / max(wp - bp, 1.0), 0, 1)
    if scurve:
        a = (1 - scurve) * a + scurve * (a * a * (3 - 2 * a))
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8), "L")


def _fade(im, fade):
    """Blend towards white: fade 0.4 = ink at 60% strength."""
    if fade <= 0:
        return im
    a = np.asarray(im).astype(np.float32)
    a = 255.0 - (255.0 - a) * (1.0 - fade)
    return Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8), im.mode)


def finish(im, cls, o):
    """Final tone work. Returns an 'L' (grey) or 'RGB' image."""
    sharpen = ImageFilter.UnsharpMask(1.2, 40, 2)
    mono_art = cls in ("lineart", "shaded")
    if mono_art or not o.keep_color:
        g = ImageOps.grayscale(im)
        arr = np.asarray(g)
        if cls == "lineart":
            # paper -> pure white, ink -> dark, edges tightened
            wp = float(np.clip(np.percentile(arr, 50) * 0.92, 120, 250))
            g = _levels(g, float(np.percentile(arr, 0.3)), wp, scurve=0.5)
            g = g.filter(ImageFilter.UnsharpMask(1.2, 70, 2))
        else:
            g = _levels(g, float(np.percentile(arr, 0.5)), float(np.percentile(arr, 99.5)))
            g = g.filter(sharpen)
        return _fade(g, o.fade / 100.0)
    out = im
    b = o.color_boost / 100.0
    if b > 0:
        out = ImageEnhance.Color(out).enhance(1 + 0.8 * b)
        out = ImageEnhance.Contrast(out).enhance(1 + 0.25 * b)
    out = out.filter(sharpen)
    return _fade(out, o.fade / 100.0)


# ================================================================= main job
def list_images(folder):
    folder = Path(folder)
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS),
                  key=lambda p: p.name.lower())


def choose_images(files, count, pick):
    numbered = list(enumerate(files, 1))
    if count and count < len(numbered):
        if pick == "first":
            numbered = numbered[:count]
        else:
            step = len(numbered) / count
            numbered = [numbered[int(j * step)] for j in range(count)]
    return numbered


def output_dir(o):
    """Where results and _blurnomore_work go: 'Save to', or the images folder when that is empty."""
    return Path(o.save_to.strip().strip('"')) if o.save_to.strip() else Path(o.folder)


def select_images(o, L, folder, files, work):
    """The images this run (or its preview) uses. Returns (numbered, selection file)."""
    out = work.parent
    if out.resolve() == folder.resolve():
        sel_file = work / "last_selection.json"
    else:   # two image folders saving to one place must not mix their selections
        tag = hashlib.md5(str(folder.resolve()).encode()).hexdigest()[:8]
        sel_file = work / f"last_selection_{tag}.json"
    numbered = None
    if o.repeat_last:
        try:
            names = set(json.loads(sel_file.read_text(encoding="utf-8")))
            numbered = [(i, p) for i, p in enumerate(files, 1) if p.name in names]
            missing = len(names) - len(numbered)
            L(f"Repeating the {len(numbered)} images from the last run"
              + (f" ({missing} no longer in the folder)" if missing else ""))
        except Exception:  # noqa: BLE001
            L("No earlier selection found for this folder; choosing images normally.")
        if not numbered:
            numbered = None
    if numbered is None:
        numbered = choose_images(files, o.count, o.pick)
    return numbered, sel_file


def prepare(o, L, folder, out):
    """Work folder, or None (after saying why) when the save folder cannot be made."""
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        L(f"Cannot make the folder to save in: {out} ({e}). Choose another folder under 'Save to'.")
        return None
    work = out / "_blurnomore_work"
    old_work = out / "_tracing_work"      # name used by earlier versions: keep its cache
    if old_work.exists() and not work.exists():
        try:
            old_work.rename(work)
        except OSError:
            pass
    try:
        (work / "logs").mkdir(parents=True, exist_ok=True)
    except OSError as e:
        L(f"Cannot make the work folder in {out} ({e}). Choose another folder under 'Save to'.")
        return None
    return work


def run(o, log, should_stop=lambda: False):
    """Do the whole job. Returns the PDF path (mode pdf) or the folder of upscaled
    copies (mode upscale), or None."""
    folder = Path(o.folder)
    if not folder.is_dir():
        log(f"Folder not found: {folder}")
        return None
    files = list_images(folder)
    if not files:
        log(f"No images found in {folder}")
        return None

    started = time.time()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = output_dir(o)
    work = prepare(o, log, folder, out)
    if work is None:
        return None
    logfile = open(work / "logs" / f"{stamp}.log", "w", encoding="utf-8")

    def L(msg=""):
        log(msg)
        logfile.write(msg + "\n")
        logfile.flush()

    try:
        return _run(o, L, should_stop, folder, out, files, stamp, work, started)
    except Exception:  # noqa: BLE001
        L("UNEXPECTED ERROR:")
        L(traceback.format_exc())
        return None
    finally:
        L(f"Log saved: {work / 'logs' / (stamp + '.log')}")
        logfile.close()


class Context:
    """What the per-image work needs: settings, log, folders, and the Real-ESRGAN program
    (exe becomes None for the rest of the run when it fails)."""

    def __init__(self, o, L, work):
        self.o, self.L, self.work, self.exe = o, L, work, None
        key_fields = (o.fit, o.margin_mm, o.dpi, o.autocrop, o.thicken, o.fade, o.keep_color,
                      o.color_boost, o.use_ai, o.color_model)
        key = hashlib.md5(repr(key_fields).encode()).hexdigest()[:8]
        self.pages_dir = work / f"pages_{key}"
        self.fit_dir = work / "fitted_cache"
        self.tmp = work / "tmp"
        self.cache_dir = work / "preview_cache"
        dirs = (self.pages_dir, self.fit_dir, self.tmp) if o.mode == "pdf" else (self.tmp,)
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        self.up_model = o.up_model
        if o.mode == "upscale" and o.up_model not in ("auto", "general", "anime") + tuple(extra_models()):
            L(f"Unknown upscale model '{o.up_model}'; choosing automatically.")
            self.up_model = "auto"

    def start_ai(self):
        if self.o.use_ai:
            self.exe = ensure_esrgan(self.work, self.L)
        if self.o.use_ai and self.exe:
            self.L(f"Real-ESRGAN: {self.exe}")


def source_id(path):
    st = path.stat()
    return hashlib.md5(f"{path.name}{st.st_size}{int(st.st_mtime)}".encode()).hexdigest()[:8]


# ---- one PDF page
def make_page(c, path, src_id, im):
    """Crop, upscale and place one image on its A4 page (or reuse the cached page).
    Returns a dict about it; 'im' is the picture after the crop."""
    o, L = c.o, c.L
    w0, h0 = im.size
    box = find_content_box(im) if o.autocrop else None
    note = ""
    if box:
        im = im.crop(box)
        note = f"cropped {w0}x{h0} -> {im.size[0]}x{im.size[1]}"
    w, h = im.size
    cls, stats = classify(im)
    page, target, s, box_px = page_geometry(w, h, o.dpi, o.margin_mm, o.fit == "fill")

    mono = cls in ("lineart", "shaded") or not o.keep_color
    ext = ".png" if (mono and o.fade == 0) else ".jpg"
    page_path = c.pages_dir / f"{path.stem[:40]}_{src_id}{ext}"

    if o.redo or not page_path.exists():
        model = MODEL_ANIME if cls in ("lineart", "shaded") else (
            MODEL_ANIME if o.color_model == "anime" else MODEL_GENERAL)
        method = "downscale" if s <= 1 else "lanczos"
        fit = None
        if s > AI_MIN_SCALE and c.exe is not None:
            cache = c.fit_dir / f"{path.stem[:40]}_{src_id}_{model}_{w}x{h}_{target[0]}x{target[1]}.png"
            if cache.exists() and not o.redo:
                fit = Image.open(cache).convert("RGB")
                method = f"AI {model} (cached)"
            else:
                up = run_esrgan(im, model, c.exe, c.tmp, L)
                if up is None:
                    L("  Real-ESRGAN did not work; using plain resizing for the rest of this run.")
                    c.exe = None
                    method = "lanczos (AI failed)"
                else:
                    fit = up.resize(target, Image.LANCZOS)
                    fit.save(cache)
                    method = f"AI {model}"
        if fit is None:
            fit = im.resize(target, Image.LANCZOS)

        fit = strengthen_thin(fit, o.thicken / 100.0)
        out = finish(fit, cls, o)
        white = 255 if out.mode == "L" else (255, 255, 255)
        if out.size[0] > box_px[0] or out.size[1] > box_px[1]:   # "fill": trim to the page
            cw, ch = min(out.size[0], box_px[0]), min(out.size[1], box_px[1])
            l, t = (out.size[0] - cw) // 2, (out.size[1] - ch) // 2
            out = out.crop((l, t, l + cw, t + ch))
        canvas = Image.new(out.mode, page, white)
        canvas.paste(out, ((page[0] - out.size[0]) // 2, (page[1] - out.size[1]) // 2))
        if ext == ".png":
            canvas.save(page_path, optimize=True, dpi=(o.dpi, o.dpi))
        else:
            canvas.save(page_path, quality=95, subsampling=0, dpi=(o.dpi, o.dpi))
    else:
        method = "page reused from earlier run"
    return {"page_path": page_path, "im": im, "w0": w0, "h0": h0, "w": w, "h": h,
            "cls": cls, "s": s, "method": method, "note": note}


def page_line(tag, r, t0):
    warn = "  LOW SOURCE RESOLUTION (>6x enlargement, will look soft)" if r["s"] > 6 else ""
    return (f"{tag}\n    {r['w']}x{r['h']}  {r['cls']}  x{r['s']:.2f}  {r['method']}  {r['note']}  "
            f"({time.time() - t0:.1f}s){warn}")


def page_to_original(o, w, h):
    """Where the cropped original (w x h) sits on its page: (scale, page x, page y) so that
    original x = (page x - offset x) / scale. Same geometry as make_page."""
    page, target, s, box_px = page_geometry(w, h, o.dpi, o.margin_mm, o.fit == "fill")
    ow, oh = min(target[0], box_px[0]), min(target[1], box_px[1])
    trim = ((target[0] - ow) // 2, (target[1] - oh) // 2) if (target[0] > box_px[0] or target[1] > box_px[1]) else (0, 0)
    place = ((page[0] - ow) // 2, (page[1] - oh) // 2)
    return target[0] / w, target[1] / h, place[0] - trim[0], place[1] - trim[1]


# ---- one upscaled image
def upscale_model(c, cls):
    """(model name, model folder, native scale) for this kind of picture, or None for plain resizing."""
    if c.exe is None:
        return None
    pick = c.up_model
    if pick == "auto":
        pick = "anime" if cls in ("lineart", "flatcolor") else "general"
    if pick == "anime":
        return MODEL_ANIME, c.exe.parent / "models", 4
    if pick == "general":
        return MODEL_GENERAL, c.exe.parent / "models", 4
    return pick, MODELS_DIR, model_scale(pick)


def make_upscaled(c, path, src_id, im, store=False):
    """Enlarge one image to exactly scale x its size. A result made by the preview is reused;
    with store=True a new result is kept for the run that follows."""
    o, L = c.o, c.L
    w, h = im.size
    cls, stats = classify(im)
    kind = "drawing" if cls in ("lineart", "flatcolor") else "photo"
    target = (w * o.scale, h * o.scale)
    m = upscale_model(c, cls)
    label = m[0] if m else "plain"
    cache = c.cache_dir / f"{path.stem[:40]}_{src_id}_{label}_x{o.scale}.png"
    if cache.exists() and not o.redo:
        res = Image.open(cache).convert("RGB")
        res.load()
        method = f"{label} (cached)" if m else "lanczos (cached)"
        return {"result": res, "cls": cls, "kind": kind, "w": w, "h": h, "method": method, "model": label}
    res = None
    method = "lanczos"
    if m:
        up = run_esrgan(im, m[0], c.exe, c.tmp, L, m[1], m[2])
        if up is None:
            L("  Real-ESRGAN did not work; using plain resizing for the rest of this run.")
            c.exe = None
            method, label = "lanczos (AI failed)", "plain"
        else:
            res = up.resize(target, Image.LANCZOS)
            method = f"AI {m[0]}"
    if res is None:
        res = im.resize(target, Image.LANCZOS)
    if store:
        try:
            c.cache_dir.mkdir(parents=True, exist_ok=True)
            res.save(c.cache_dir / f"{path.stem[:40]}_{src_id}_{label}_x{o.scale}.png")
        except Exception:  # noqa: BLE001
            pass
    return {"result": res, "cls": cls, "kind": kind, "w": w, "h": h, "method": method, "model": label}


def upscaled_line(tag, r, scale, name, t0):
    big = r["w"] * r["h"] * scale * scale
    warn = f"  VERY LARGE RESULT ({big / 1e6:.0f} megapixels): big file, slow" if big > BIG_RESULT_PIXELS else ""
    return (f"{tag}\n    {r['w']}x{r['h']}  {r['kind']}  {r['method']}  x{scale}  -> {name}  "
            f"({time.time() - t0:.1f}s){warn}")


def _run(o, L, should_stop, folder, out, files, stamp, work, started):
    L(f"BlurNoMore v{VERSION}   {datetime.now():%Y-%m-%d %H:%M:%S}")
    L(f"Python {platform.python_version()}  Pillow {Image.__version__}  numpy {np.__version__}  {platform.platform()}")
    L("Settings: " + ", ".join(f"{k}={v}" for k, v in asdict(o).items()))

    c = Context(o, L, work)
    numbered, sel_file = select_images(o, L, folder, files, work)
    try:
        sel_file.write_text(json.dumps([p.name for _, p in numbered], indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    L(f"{len(numbered)} of {len(files)} images in {folder}\n")

    c.start_ai()
    upscale = o.mode == "upscale"
    pdf_inputs, report = [], []
    up_dir = out / f"BlurNoMore_upscaled_{stamp}"
    ext = "jpg" if o.file_type == "jpg" else "png"

    for n, (idx, path) in enumerate(numbered, 1):
        if should_stop():
            L("\nStopped by user.")
            break
        t0 = time.time()
        tag = f"[{n}/{len(numbered)}] #{idx} {path.name[:50]}"
        try:
            src_id = source_id(path)
            im = load_rgb(path)
            if upscale:
                r = make_upscaled(c, path, src_id, im)
                up_dir.mkdir(parents=True, exist_ok=True)
                name, k = f"{path.stem}_x{o.scale}.{ext}", 2
                while (up_dir / name).exists():
                    name, k = f"{path.stem}_x{o.scale}_{k}.{ext}", k + 1
                if ext == "jpg":
                    r["result"].save(up_dir / name, quality=95, subsampling=0)
                else:
                    r["result"].save(up_dir / name)
                pdf_inputs.append(name)
                L(upscaled_line(tag, r, o.scale, name, t0))
                report.append([idx, path.name, f"{r['w']}x{r['h']}", r["kind"], o.scale, r["method"], name])
            else:
                r = make_page(c, path, src_id, im)
                pdf_inputs.append(str(r["page_path"]))
                L(page_line(tag, r, t0))
                report.append([idx, path.name, f"{r['w0']}x{r['h0']}", r["cls"], round(r["s"], 2), r["method"], r["note"]])
        except Exception:  # noqa: BLE001
            L(f"{tag}\n    ERROR:")
            L("    " + traceback.format_exc().replace("\n", "\n    "))
            report.append([idx, path.name, "", "ERROR", "", "", ""])

    if not pdf_inputs:
        L("Nothing to write.")
        return None

    if upscale:
        result = up_dir
    else:
        dpi = o.dpi

        def layout(wpx, hpx, _ndpi):
            wpt, hpt = wpx * 72.0 / dpi, hpx * 72.0 / dpi
            return wpt, hpt, wpt, hpt

        result = out / f"BlurNoMore_A4_{stamp}_{len(pdf_inputs)}p.pdf"
        with open(result, "wb") as f:
            f.write(img2pdf.convert(pdf_inputs, layout_fun=layout))

    try:
        import csv
        with open(work / "logs" / f"{stamp}_report.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["#", "file", "size", "class", "scale", "method", "note"])
            wr.writerows(report)
    except Exception:  # noqa: BLE001
        pass

    if upscale:
        L(f"\nDONE in {time.time() - started:.0f}s: {result}  ({len(pdf_inputs)} images)")
    else:
        L(f"\nDONE in {time.time() - started:.0f}s: {result}  ({len(pdf_inputs)} pages)")
        L("Print at 100% / 'Actual size', not 'Fit to page'.")
    return result


# ================================================================== preview
def busiest_box(im, size=400):
    """Box (x0, y0, x1, y1) of about size x size pixels where the picture has the most edges."""
    W, H = im.size
    bw, bh = min(size, W), min(size, H)
    g = im.convert("L")
    f = min(1.0, 800.0 / max(W, H))
    if f < 1.0:
        g = g.resize((max(1, round(W * f)), max(1, round(H * f))), Image.BILINEAR)
    a = np.asarray(g).astype(np.float32)
    e = np.zeros_like(a)
    e[:, 1:] += np.abs(np.diff(a, axis=1))
    e[1:, :] += np.abs(np.diff(a, axis=0))
    sw, sh = max(1, round(bw * f)), max(1, round(bh * f))
    ii = np.pad(e, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = ii[sh:, sw:] - ii[:-sh, sw:] - ii[sh:, :-sw] + ii[:-sh, :-sw]
    y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
    x0 = min(max(0, int(round(x / f))), W - bw)
    y0 = min(max(0, int(round(y / f))), H - bh)
    return (x0, y0, x0 + bw, y0 + bh)


def make_preview(o, log, n=4):
    """Process up to n images exactly as the run would and return what to show: a list of dicts
    with name, cls (class), kind, method, original, result, detail_original, detail_result.
    Writes only into the work folder (pages and results are kept for the run that follows)."""
    folder = Path(o.folder)
    if not folder.is_dir():
        log(f"Folder not found: {folder}")
        return []
    files = list_images(folder)
    if not files:
        log(f"No images found in {folder}")
        return []
    work = prepare(o, log, folder, output_dir(o))
    if work is None:
        return []
    c = Context(o, log, work)
    numbered, _ = select_images(o, log, folder, files, work)
    idx_of = {p: i for i, p in numbered}
    picks = [(idx_of[p], p) for _, p in choose_images([p for _, p in numbered], n, "spread")]
    log(f"Preview of {len(picks)} images in {folder}")
    c.start_ai()
    items = []
    for k, (idx, path) in enumerate(picks, 1):
        t0 = time.time()
        tag = f"[{k}/{len(picks)}] #{idx} {path.name[:50]}"
        try:
            src_id = source_id(path)
            im = load_rgb(path)
            if o.mode == "upscale":
                r = make_upscaled(c, path, src_id, im, store=True)
                orig, res = im, r["result"]
                box = busiest_box(res)
                fx = fy = float(o.scale)
                dx = dy = 0.0
                log(upscaled_line(tag, r, o.scale, "(preview)", t0))
            else:
                r = make_page(c, path, src_id, im)
                orig = r["im"]
                res = Image.open(r["page_path"]).convert("RGB")
                box = busiest_box(res)
                fx, fy, dx, dy = page_to_original(o, r["w"], r["h"])
                log(page_line(tag, r, t0))
            ow, oh = orig.size
            sx0, sx1 = (min(max((box[0] - dx) / fx, 0.0), ow - 1), min(max((box[2] - dx) / fx, 1.0), ow))
            sy0, sy1 = (min(max((box[1] - dy) / fy, 0.0), oh - 1), min(max((box[3] - dy) / fy, 1.0), oh))
            size = (box[2] - box[0], box[3] - box[1])
            d_orig = orig.resize(size, Image.BICUBIC, box=(sx0, sy0, max(sx1, sx0 + 1), max(sy1, sy0 + 1)))
            items.append({"name": path.name, "cls": r["cls"], "method": r["method"], "original": orig,
                          "result": res, "detail_original": d_orig, "detail_result": res.crop(box)})
        except Exception:  # noqa: BLE001
            log(f"{tag}\n    ERROR:")
            log("    " + traceback.format_exc().replace("\n", "\n    "))
    return items


def preview_sheet(item, panel=560):
    """One picture: original | result on top, the two close-ups below, with small captions."""
    from PIL import ImageDraw
    bg, pad, cap = (235, 235, 235), 10, 24

    def fit(im, side):
        im = im.copy().convert("RGB")
        im.thumbnail((side, side))
        return im

    top = [fit(item["original"], panel), fit(item["result"], panel)]
    low = [item["detail_original"].convert("RGB"), item["detail_result"].convert("RGB")]
    row1 = max(i.size[1] for i in top)
    row2 = max(i.size[1] for i in low)
    W = 2 * panel + 3 * pad
    H = pad + cap + row1 + pad + cap + row2 + pad + cap
    sheet = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(sheet)
    d.text((pad, 3), f"{item['name']}   {item['cls']}   {item['method']}", fill=(0, 0, 0))
    y = pad + cap
    for j, (im, label) in enumerate(zip(top, (f"Original ({item['original'].size[0]}x{item['original'].size[1]})",
                                              f"Result ({item['result'].size[0]}x{item['result'].size[1]})"))):
        x = pad + j * (panel + pad)
        d.text((x, y - 14), label, fill=(0, 0, 0))
        sheet.paste(im, (x + (panel - im.size[0]) // 2, y))
    y += row1 + pad + cap
    for j, (im, label) in enumerate(zip(low, ("Close-up: original, plain enlargement", "Close-up: result, same spot"))):
        x = pad + j * (panel + pad)
        d.text((x, y - 14), label, fill=(0, 0, 0))
        sheet.paste(im, (x + (panel - im.size[0]) // 2, y))
    return sheet


def save_preview_sheets(o, items, stamp=None):
    """Save one sheet PNG per item in <work>/previews/<stamp>/. Returns the paths."""
    stamp = stamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    d = output_dir(o) / "_blurnomore_work" / "previews" / stamp
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, it in enumerate(items, 1):
        p = d / f"{k}_{Path(it['name']).stem[:40]}.png"
        preview_sheet(it).save(p)
        paths.append(p)
    return paths


# ====================================================================== GUI
HELP = {
    "folder": "Folder with your pictures. Nothing in it is ever changed. The result is saved here too, with the "
              "date and time in its name, unless you choose another place under 'Save results to'. "
              "Logs and work files go in _blurnomore_work.",
    "count": "Handy for a quick test. 'Spread evenly' picks images from all over the folder, "
             "so you see different kinds of drawings. 'First ones' takes them in name order.",
    "repeat": "Runs again on exactly the same images as the last run in this folder, whatever the "
              "'How many' setting says. Handy for comparing settings on the same test pages.",
    "fit": "Whole picture: nothing is cut off, but a picture with a different shape than A4 leaves "
           "white space at the top/bottom or the sides. Fill the page: enlarges until the page is full "
           "and trims the edges that do not fit.",
    "margin": "White border around the drawing on the A4 page, in mm. Printers cannot print right to the "
              "edge; 5 to 10 is normal.",
    "dpi": "Pixels per inch of the pages. 300 = sharp print quality. 200 = smaller and faster. "
           "Going above 300 rarely shows.",
    "crop": "Cuts away blank paper and scan borders around the drawing, so it fills the page. "
            "Switch off if it ever cuts off something you want to keep.",
    "thicken": "Makes very thin dark lines a little thicker and darker so they show through tracing paper. "
               "Thick lines and colour fills are not touched. 0 = off, 70 = default, 100 = strongest.",
    "fade": "Lightens the whole print so your own pencil lines stand out. "
            "0 = print as it is (default). 30 to 50 = light print for tracing on a light table.",
    "color": "On: coloured drawings stay in colour, black and white art stays black and white. "
             "Off: everything is printed in grey.",
    "boost": "Stronger colours and a bit more contrast, for pale or washed-out artwork. "
             "0 = unchanged. Try 20 to 40 for pale pictures.",
    "ai": "Real-ESRGAN on your GPU enlarges small images and adds detail (first run downloads about 45 MB). "
          "Off = plain resizing: much faster but blurrier.",
    "model": "General: natural look, best for painted or shaded coloured art and scans (default). "
             "Anime: crisper flat colours, but can look smooth and plastic on shaded art. "
             "Black and white art always uses Anime.",
    "mode": "Tracing PDF: crops, enlarges and fits every picture on an A4 page, for tracing. "
            "Upscale images: only makes each picture 2x, 3x or 4x bigger, for general use, and saves "
            "them as new picture files. No cropping, no line work, no pages.",
    "save_to": "Where the results are saved: the PDF, or the folder of bigger pictures, and the "
               "_blurnomore_work folder. Empty = next to the pictures (in the images folder). "
               "A folder that does not exist yet is made. Your original pictures are never touched.",
    "scale": "Upscale images only. How many times bigger the pictures become: 2, 3 or 4. "
             "A 1000 pixel wide picture becomes 2000, 3000 or 4000 pixels wide. Bigger takes longer "
             "and makes bigger files; it cannot bring back detail that is not there.",
    "up_model": "Upscale images only. Automatic looks at each picture: drawings and flat colour use the "
                "Anime model, photos and black and white photos use the General model. General or Anime "
                "forces one for all pictures. Other models you put in the models folder next to "
                "the program (NAME.param and NAME.bin) are listed here too.",
    "file_type": "Upscale images only. PNG keeps every pixel exactly (big files). JPG is much smaller "
                 "(top quality, 95) and fine for photos.",
    "redo": "Ignore pages made in earlier runs. Changing a setting already makes new pages on its own; "
            "use this for a clean rebuild.",
}


def load_settings():
    o = Options()
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        for f in fields(Options):
            if f.name in data:
                setattr(o, f.name, type(getattr(o, f.name))(data[f.name]))
    except Exception:  # noqa: BLE001
        pass
    return o


def save_settings(o):
    try:
        SETTINGS_FILE.write_text(json.dumps(asdict(o), indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def launch_gui():
    import os
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter import font as tkfont

    o = load_settings()
    root = tk.Tk()
    root.title("BlurNoMore")
    root.geometry("1280x780")
    root.minsize(1100, 600)

    base = tkfont.nametofont("TkDefaultFont")
    bold = base.copy()
    bold.configure(weight="bold")
    small = base.copy()
    small.configure(size=max(base.cget("size") - 1, 8))
    mono = tkfont.nametofont("TkFixedFont")

    # ---- variables
    v_folder = tk.StringVar(value=o.folder)
    v_mode = tk.StringVar(value="All images" if o.count == 0 else "Only a number of images")
    v_count = tk.StringVar(value=str(o.count or 6))
    v_pick = tk.StringVar(value="spread evenly" if o.pick == "spread" else "the first ones")
    v_repeat = tk.BooleanVar(value=o.repeat_last)
    v_margin = tk.StringVar(value=str(o.margin_mm))
    v_fit = tk.StringVar(value="Whole picture" if o.fit == "whole" else "Fill the page")
    v_dpi = tk.StringVar(value=str(o.dpi))
    v_crop = tk.BooleanVar(value=o.autocrop)
    v_thick = tk.StringVar(value=str(o.thicken))
    v_fade = tk.StringVar(value=str(o.fade))
    v_color = tk.BooleanVar(value=o.keep_color)
    v_boost = tk.StringVar(value=str(o.color_boost))
    v_ai = tk.BooleanVar(value=o.use_ai)
    v_model = tk.StringVar(value="General (recommended)" if o.color_model == "general" else "Anime")
    v_redo = tk.BooleanVar(value=o.redo)
    v_open = tk.BooleanVar(value=o.open_when_done)
    v_found = tk.StringVar(value="")

    # ---- layout: settings (left) and log (right)
    root.columnconfigure(0, weight=0)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    left_outer = ttk.Frame(root)
    left_outer.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)
    canvas = tk.Canvas(left_outer, width=520, highlightthickness=0)
    scroll = ttk.Scrollbar(left_outer, orient="vertical", command=canvas.yview)
    left = ttk.Frame(canvas)
    left.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=left, anchor="nw")
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(side="left", fill="y", expand=True)
    scroll.pack(side="right", fill="y")

    def wheel(e):
        canvas.yview_scroll(int(-1 * (e.delta / 120)) if e.delta else 0, "units")
    canvas.bind_all("<MouseWheel>", wheel)

    right = ttk.Frame(root)
    right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
    right.rowconfigure(1, weight=1)
    right.columnconfigure(0, weight=1)

    r = [0]

    def add(title, widget_builder, help_key):
        row = ttk.Frame(left)
        row.grid(row=r[0], column=0, sticky="ew", pady=(0, 7))
        r[0] += 1
        top = ttk.Frame(row)
        top.pack(fill="x")
        ttk.Label(top, text=title, font=bold, width=26).pack(side="left")
        widget_builder(top).pack(side="left")
        ttk.Label(row, text=HELP[help_key], font=small, foreground="#555555",
                  wraplength=470, justify="left").pack(fill="x", anchor="w")

    def spin(var, lo, hi, step, width=6):
        return lambda p: ttk.Spinbox(p, from_=lo, to=hi, increment=step, textvariable=var, width=width)

    def check(var, text):
        return lambda p: ttk.Checkbutton(p, text=text, variable=var)

    # folder
    def build_folder(p):
        f = ttk.Frame(p)
        ttk.Entry(f, textvariable=v_folder, width=26).pack(side="left")
        ttk.Button(f, text="Browse...", command=lambda: browse()).pack(side="left", padx=4)
        return f

    def browse():
        d = filedialog.askdirectory(initialdir=v_folder.get() or None, title="Folder with drawings")
        if d:
            v_folder.set(d)

    add("Folder with drawings", build_folder, "folder")
    ttk.Label(left, textvariable=v_found, font=small).grid(row=r[0], column=0, sticky="w")
    r[0] += 1

    # how many
    def build_count(p):
        f = ttk.Frame(p)
        cb = ttk.Combobox(f, textvariable=v_mode, state="readonly", width=22,
                          values=["All images", "Only a number of images"])
        cb.grid(row=0, column=0, sticky="w")
        sp = ttk.Spinbox(f, from_=1, to=9999, textvariable=v_count, width=5)
        sp.grid(row=0, column=1, padx=4)
        pk = ttk.Combobox(f, textvariable=v_pick, state="readonly", width=22,
                          values=["spread evenly", "the first ones"])
        pk.grid(row=1, column=0, sticky="w", pady=(3, 0))

        def sync(*_):
            state = "normal" if v_mode.get() != "All images" else "disabled"
            sp.configure(state=state)
            pk.configure(state="readonly" if state == "normal" else "disabled")
        cb.bind("<<ComboboxSelected>>", sync)
        sync()
        return f

    add("How many images", build_count, "count")
    add("Same images as last run", check(v_repeat, "on"), "repeat")

    add("Fit on the page",
        lambda p: ttk.Combobox(p, textvariable=v_fit, state="readonly", width=22,
                               values=["Whole picture", "Fill the page"]), "fit")
    add("Page margin (mm)", spin(v_margin, 0, 40, 1), "margin")
    add("Print resolution (dpi)", spin(v_dpi, 150, 600, 50), "dpi")
    add("Crop empty paper", check(v_crop, "on"), "crop")
    add("Strengthen thin lines (0-100)", spin(v_thick, 0, 100, 10), "thicken")
    add("Lighten print (0-100)", spin(v_fade, 0, 90, 10), "fade")
    add("Keep colours", check(v_color, "on  (off = all grey)"), "color")
    add("Colour boost (0-100)", spin(v_boost, 0, 100, 10), "boost")
    add("Use AI upscaler", check(v_ai, "on  (needs a GPU)"), "ai")
    add("AI model for coloured art",
        lambda p: ttk.Combobox(p, textvariable=v_model, state="readonly", width=22,
                               values=["General (recommended)", "Anime"]), "model")
    add("Redo all pages", check(v_redo, "on"), "redo")
    row = ttk.Frame(left)
    row.grid(row=r[0], column=0, sticky="w", pady=4)
    ttk.Checkbutton(row, text="Open the PDF when finished", variable=v_open).pack(side="left")
    r[0] += 1

    # buttons + log
    btns = ttk.Frame(right)
    btns.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    b_start = ttk.Button(btns, text="Start")
    b_stop = ttk.Button(btns, text="Stop after this image", state="disabled")
    b_copy = ttk.Button(btns, text="Copy log")
    b_open = ttk.Button(btns, text="Open folder")
    for b in (b_start, b_stop, b_copy, b_open):
        b.pack(side="left", padx=(0, 6))

    log_text = tk.Text(right, wrap="none", font=mono, height=20, undo=False)
    ysb = ttk.Scrollbar(right, orient="vertical", command=log_text.yview)
    xsb = ttk.Scrollbar(right, orient="horizontal", command=log_text.xview)
    log_text.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
    log_text.grid(row=1, column=0, sticky="nsew")
    ysb.grid(row=1, column=1, sticky="ns")
    xsb.grid(row=2, column=0, sticky="ew")

    def block_typing(e):
        if (e.state & 0x4) and e.keysym.lower() in ("c", "a"):
            return None
        if e.keysym in ("Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next"):
            return None
        return "break"
    log_text.bind("<Key>", block_typing)
    log_text.bind("<Control-a>", lambda e: (log_text.tag_add("sel", "1.0", "end"), "break")[1])

    q = queue.Queue()
    state = {"stop": False, "thread": None, "last_pdf": None}

    def append(msg):
        log_text.insert("end", msg + "\n")
        log_text.see("end")

    append("Choose a folder and settings on the left, then press Start.\n"
           "The log here can be selected and copied (Ctrl+C), or press 'Copy log'.")

    def count_found(*_):
        try:
            n = len(list_images(v_folder.get()))
            v_found.set(f"   {n} images found in this folder")
        except Exception:  # noqa: BLE001
            v_found.set("   folder not found")
    v_folder.trace_add("write", count_found)
    count_found()

    def gather():
        def num(var, name, cast=int):
            try:
                return cast(float(var.get()))
            except ValueError:
                raise ValueError(f"'{name}' must be a number") from None
        ob = Options()
        ob.folder = v_folder.get().strip().strip('"')
        ob.count = 0 if v_mode.get() == "All images" else max(1, num(v_count, "How many images"))
        ob.pick = "spread" if v_pick.get().startswith("spread") else "first"
        ob.repeat_last = bool(v_repeat.get())
        ob.margin_mm = max(0.0, num(v_margin, "Page margin", float))
        ob.fit = "whole" if v_fit.get().startswith("Whole") else "fill"
        ob.dpi = max(72, num(v_dpi, "Print resolution"))
        ob.autocrop = bool(v_crop.get())
        ob.thicken = min(100, max(0, num(v_thick, "Strengthen thin lines")))
        ob.fade = min(90, max(0, num(v_fade, "Lighten print")))
        ob.keep_color = bool(v_color.get())
        ob.color_boost = min(100, max(0, num(v_boost, "Colour boost")))
        ob.use_ai = bool(v_ai.get())
        ob.color_model = "general" if v_model.get().startswith("General") else "anime"
        ob.redo = bool(v_redo.get())
        ob.open_when_done = bool(v_open.get())
        return ob

    def worker(opts):
        pdf = None
        try:
            pdf = run(opts, lambda m: q.put(("log", m)), lambda: state["stop"])
        except Exception:  # noqa: BLE001
            q.put(("log", "UNEXPECTED ERROR:\n" + traceback.format_exc()))
        q.put(("done", str(pdf) if pdf else None))

    def start():
        try:
            opts = gather()
        except ValueError as e:
            messagebox.showerror("Check the settings", str(e))
            return
        if not Path(opts.folder).is_dir():
            messagebox.showerror("Folder not found", f"Cannot find:\n{opts.folder}\n\nIs the drive connected?")
            return
        save_settings(opts)
        log_text.delete("1.0", "end")
        state["stop"] = False
        b_start.configure(state="disabled")
        b_stop.configure(state="normal")
        t = threading.Thread(target=worker, args=(opts,), daemon=True)
        state["thread"] = t
        t.start()

    def stop():
        state["stop"] = True
        b_stop.configure(state="disabled")
        append("Stopping after the current image ...")

    def copy_log():
        root.clipboard_clear()
        root.clipboard_append(log_text.get("1.0", "end-1c"))
        append("(log copied to clipboard)")

    def open_folder():
        d = v_folder.get().strip().strip('"')
        try:
            if hasattr(os, "startfile"):
                os.startfile(d)  # noqa: S606
        except Exception as e:  # noqa: BLE001
            append(f"Could not open folder: {e}")

    def poll():
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "log":
                    append(payload)
                else:
                    b_start.configure(state="normal")
                    b_stop.configure(state="disabled")
                    if payload:
                        state["last_pdf"] = payload
                        if v_open.get() and hasattr(os, "startfile"):
                            try:
                                os.startfile(payload)  # noqa: S606
                            except Exception as e:  # noqa: BLE001
                                append(f"Could not open PDF: {e}")
        except queue.Empty:
            pass
        root.after(120, poll)

    b_start.configure(command=start)
    b_stop.configure(command=stop)
    b_copy.configure(command=copy_log)
    b_open.configure(command=open_folder)
    root.after(120, poll)
    root.mainloop()


# ====================================================================== CLI
def main():
    if len(sys.argv) == 1:
        launch_gui()
        return
    ap = argparse.ArgumentParser(description="BlurNoMore: pictures -> sharp A4 tracing PDF, or bigger AI-restored copies "
                                             "(no arguments = open the window)")
    ap.add_argument("folder", help="folder with the pictures (never changed)")
    ap.add_argument("--mode", choices=["pdf", "upscale"], default="pdf",
                    help="pdf = Tracing PDF (default), upscale = Upscale images")
    ap.add_argument("--save-to", default="", metavar="FOLDER",
                    help="where the PDF / upscaled folder and _blurnomore_work go (default: the images folder)")
    ap.add_argument("--scale", type=int, choices=[2, 3, 4], default=2, help="upscale mode: times bigger")
    ap.add_argument("--upscale-model", default="auto", metavar="NAME",
                    help="upscale mode: auto (default), general, anime, or the name of an extra model "
                         "in the models folder: " + (", ".join(extra_models()) or "none found"))
    ap.add_argument("--format", choices=["png", "jpg"], default="png", help="upscale mode: picture file type")
    ap.add_argument("--preview", action="store_true",
                    help="make picture sheets for up to 4 images (original, result, close-ups) and stop")
    ap.add_argument("--count", type=int, default=0, help="number of images, 0 = all")
    ap.add_argument("--pick", choices=["spread", "first"], default="spread")
    ap.add_argument("--same", action="store_true", help="repeat the images used in the last run")
    ap.add_argument("--margin", type=float, default=10.0)
    ap.add_argument("--fill", action="store_true", help="fill the page (trims edges) instead of fitting the whole picture")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--thicken", type=int, default=70)
    ap.add_argument("--fade", type=int, default=0)
    ap.add_argument("--grey", action="store_true", help="print everything in grey")
    ap.add_argument("--boost", type=int, default=0)
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--model", choices=["general", "anime"], default="general")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    o = Options(folder=a.folder, count=a.count, pick=a.pick, repeat_last=a.same, fit="fill" if a.fill else "whole", margin_mm=a.margin, dpi=a.dpi,
                autocrop=not a.no_crop, thicken=a.thicken, fade=a.fade, keep_color=not a.grey,
                color_boost=a.boost, use_ai=not a.no_ai, color_model=a.model, redo=a.redo,
                mode=a.mode, save_to=a.save_to, scale=a.scale, up_model=a.upscale_model, file_type=a.format)
    if a.preview:
        items = make_preview(o, print)
        if not items:
            sys.exit(1)
        for p in save_preview_sheets(o, items):
            print(f"Preview sheet: {p}")
        sys.exit(0)
    sys.exit(0 if run(o, print) else 1)


if __name__ == "__main__":
    main()
