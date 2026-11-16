#!/usr/bin/env python3
"""
BlurNoMore - turn a folder of small, blurry drawings into one sharp, print-ready A4 PDF for tracing.

Start it by double-clicking BlurNoMore.bat (or run:  python blurnomore.py).
That opens a window where you choose the folder, how many images and all settings.

Command line (optional):
    python blurnomore.py "D:\\drawings" --count 6 --pick spread --thicken 70

Per image it: crops empty paper, upscales with Real-ESRGAN (GPU) when the image is
too small for A4 at the chosen dpi, strengthens very thin lines, fits it on A4 and
writes one multi-page PDF with the date and time in its name. Originals are never
touched. Work files and logs go in <folder>\\_blurnomore_work.
"""
import argparse
import hashlib
import json
import platform
import queue
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

VERSION = "3.1"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
A4_MM = (210.0, 297.0)
DEFAULT_FOLDER = r"I:\Oneplus Backup\Draw\New folder"
ESRGAN_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
              "realesrgan-ncnn-vulkan-20220424-windows.zip")
MODEL_GENERAL = "realesrgan-x4plus"
MODEL_ANIME = "realesrgan-x4plus-anime"
AI_MIN_SCALE = 1.25  # below this much enlargement, plain Lanczos is enough
SETTINGS_FILE = Path(__file__).with_name("blurnomore_settings.json")


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


def ensure_esrgan(work, log):
    d = work / "realesrgan"
    exe = _find_exe(d)
    if exe:
        return exe
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


def run_esrgan(img, model, exe, tmp, log):
    tin, tout = tmp / "in.png", tmp / "out.png"
    if tout.exists():
        tout.unlink()
    img.save(tin)
    cmd = [str(exe), "-i", str(tin), "-o", str(tout), "-n", model,
           "-s", "4", "-m", str(exe.parent / "models"), "-f", "png"]
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


def run(o, log, should_stop=lambda: False):
    """Do the whole job. Returns the PDF path, or None."""
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
    work = folder / "_blurnomore_work"
    old_work = folder / "_tracing_work"      # name used by earlier versions: keep its cache
    if old_work.exists() and not work.exists():
        try:
            old_work.rename(work)
        except OSError:
            pass
    (work / "logs").mkdir(parents=True, exist_ok=True)
    logfile = open(work / "logs" / f"{stamp}.log", "w", encoding="utf-8")

    def L(msg=""):
        log(msg)
        logfile.write(msg + "\n")
        logfile.flush()

    try:
        return _run(o, L, should_stop, folder, files, stamp, work, started)
    except Exception:  # noqa: BLE001
        L("UNEXPECTED ERROR:")
        L(traceback.format_exc())
        return None
    finally:
        L(f"Log saved: {work / 'logs' / (stamp + '.log')}")
        logfile.close()


def _run(o, L, should_stop, folder, files, stamp, work, started):
    L(f"BlurNoMore v{VERSION}   {datetime.now():%Y-%m-%d %H:%M:%S}")
    L(f"Python {platform.python_version()}  Pillow {Image.__version__}  numpy {np.__version__}  {platform.platform()}")
    L("Settings: " + ", ".join(f"{k}={v}" for k, v in asdict(o).items()))

    key_fields = (o.fit, o.margin_mm, o.dpi, o.autocrop, o.thicken, o.fade, o.keep_color,
                  o.color_boost, o.use_ai, o.color_model)
    key = hashlib.md5(repr(key_fields).encode()).hexdigest()[:8]
    pages_dir = work / f"pages_{key}"
    fit_dir = work / "fitted_cache"
    tmp = work / "tmp"
    for d in (pages_dir, fit_dir, tmp):
        d.mkdir(parents=True, exist_ok=True)

    sel_file = work / "last_selection.json"
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
    try:
        sel_file.write_text(json.dumps([p.name for _, p in numbered], indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    L(f"{len(numbered)} of {len(files)} images in {folder}\n")

    exe = ensure_esrgan(work, L) if o.use_ai else None
    if o.use_ai and exe:
        L(f"Real-ESRGAN: {exe}")
    pdf_inputs, report = [], []

    for n, (idx, path) in enumerate(numbered, 1):
        if should_stop():
            L("\nStopped by user.")
            break
        t0 = time.time()
        tag = f"[{n}/{len(numbered)}] #{idx} {path.name[:50]}"
        try:
            st = path.stat()
            src_id = hashlib.md5(f"{path.name}{st.st_size}{int(st.st_mtime)}".encode()).hexdigest()[:8]
            im = load_rgb(path)
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
            page_path = pages_dir / f"{path.stem[:40]}_{src_id}{ext}"

            if o.redo or not page_path.exists():
                model = MODEL_ANIME if cls in ("lineart", "shaded") else (
                    MODEL_ANIME if o.color_model == "anime" else MODEL_GENERAL)
                method = "downscale" if s <= 1 else "lanczos"
                fit = None
                if s > AI_MIN_SCALE and exe is not None:
                    cache = fit_dir / f"{path.stem[:40]}_{src_id}_{model}_{w}x{h}_{target[0]}x{target[1]}.png"
                    if cache.exists() and not o.redo:
                        fit = Image.open(cache).convert("RGB")
                        method = f"AI {model} (cached)"
                    else:
                        up = run_esrgan(im, model, exe, tmp, L)
                        if up is None:
                            L("  Real-ESRGAN did not work; using plain resizing for the rest of this run.")
                            exe = None
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

            pdf_inputs.append(str(page_path))
            warn = "  LOW SOURCE RESOLUTION (>6x enlargement, will look soft)" if s > 6 else ""
            L(f"{tag}\n    {w}x{h}  {cls}  x{s:.2f}  {method}  {note}  ({time.time() - t0:.1f}s){warn}")
            report.append([idx, path.name, f"{w0}x{h0}", cls, round(s, 2), method, note])
        except Exception:  # noqa: BLE001
            L(f"{tag}\n    ERROR:")
            L("    " + traceback.format_exc().replace("\n", "\n    "))
            report.append([idx, path.name, "", "ERROR", "", "", ""])

    if not pdf_inputs:
        L("Nothing to write.")
        return None

    dpi = o.dpi

    def layout(wpx, hpx, _ndpi):
        wpt, hpt = wpx * 72.0 / dpi, hpx * 72.0 / dpi
        return wpt, hpt, wpt, hpt

    out_pdf = folder / f"BlurNoMore_A4_{stamp}_{len(pdf_inputs)}p.pdf"
    with open(out_pdf, "wb") as f:
        f.write(img2pdf.convert(pdf_inputs, layout_fun=layout))

    try:
        import csv
        with open(work / "logs" / f"{stamp}_report.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["#", "file", "size", "class", "scale", "method", "note"])
            wr.writerows(report)
    except Exception:  # noqa: BLE001
        pass

    L(f"\nDONE in {time.time() - started:.0f}s: {out_pdf}  ({len(pdf_inputs)} pages)")
    L("Print at 100% / 'Actual size', not 'Fit to page'.")
    return out_pdf


# ====================================================================== GUI
HELP = {
    "folder": "Folder with your drawings. The PDF is saved here with the date and time in its name, "
              "so an earlier PDF is never overwritten. Logs and work files go in _blurnomore_work.",
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
    ap = argparse.ArgumentParser(description="BlurNoMore: drawings -> sharp A4 tracing PDF (no arguments = open the window)")
    ap.add_argument("folder")
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
                color_boost=a.boost, use_ai=not a.no_ai, color_model=a.model, redo=a.redo)
    sys.exit(0 if run(o, print) else 1)


if __name__ == "__main__":
    main()
