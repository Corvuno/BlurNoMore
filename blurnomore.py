#!/usr/bin/env python3
"""
BlurNoMore - make bigger, AI-restored copies of any pictures, or turn small, blurry drawings
into one sharp, print-ready A4 PDF for tracing.

Start it by double-clicking BlurNoMore.bat (or run:  python blurnomore.py).
That opens a window where you add folders or pictures, say what they are and choose what comes out.

Command line (optional):
    python blurnomore.py "D:\\old phone" "D:\\more\\IMG_1.jpg" --scale 2
    python blurnomore.py "D:\\drawings" --mode pdf --thicken 70

Bigger images (the default) enlarge 2x, 3x or 4x into a new folder
BlurNoMore_upscaled_<date>_<time> in the save folder. The PDF mode crops empty paper,
upscales when the image is too small for A4 at the chosen dpi, strengthens very thin lines,
fits it on A4 and writes one multi-page PDF with the date and time in its name. Originals are
never touched. Work files and logs go in <save folder>\\_blurnomore_work.
--preview shows up to 4 images (original next to result, plus close-ups) as picture sheets.
"""
import argparse
import hashlib
import json
import os
import platform
import queue
import random
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import img2pdf
except ImportError:
    sys.exit("Missing packages. Run:  pip install pillow numpy img2pdf")

# Optional: the DAT photo model and face restoration need these. Without them everything else works.
try:
    import onnxruntime as ort
    HAVE_ORT = True
except Exception:  # noqa: BLE001  (a missing or broken install both mean "not available")
    ort, HAVE_ORT = None, False
try:
    import cv2
    HAVE_CV2 = True
except Exception:  # noqa: BLE001
    cv2, HAVE_CV2 = None, False
# Optional: drag and drop of pictures onto the window's list. Without it the list works with the buttons.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAVE_DND = True
except Exception:  # noqa: BLE001
    HAVE_DND = False

Image.MAX_IMAGE_PIXELS = None

VERSION = "4.4"
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
LAST_RUN_FILE = APP_DIR / "blurnomore_last_run.json"   # the images of the last run
ESRGAN_DIR = APP_DIR / "realesrgan"   # downloaded once, next to the program
MODELS_DIR = APP_DIR / "models"       # extra models: NAME.param + NAME.bin pairs
BIG_RESULT_PIXELS = 100_000_000
INSTALL_HINT = "Install onnxruntime-directml (and opencv-python-headless for faces) with pip"
FACEFUSION = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"
CONVERTED = "https://github.com/corvuno/blurnomore/releases/download/models-1/"   # models converted to ONNX by tools/convert_models.py
# The downloadable ONNX upscalers, run through onnxruntime. key -> file, where from, download size,
# native enlargement, who made it (and the licence), and whether a 3x/4x result runs it twice.
UPSCALERS = {
    "dat": dict(file="4xRealWebPhoto_v4_fp32_opset17.onnx", size="49 MB", native=4, twice=False,
                url="https://huggingface.co/Phips/4xRealWebPhoto_v4_dat2/resolve/main/4xRealWebPhoto_v4_fp32_opset17.onnx",
                credit="4xRealWebPhoto_v4 (DAT) by Philip Hofmann, CC-BY-4.0"),
    "bhi": dict(file="4xBHI_dat2_real_fp32_256.onnx", size="?", native=4, twice=False,
                url=CONVERTED + "4xBHI_dat2_real_fp32_256.onnx",
                credit="4xBHI_dat2_real (DAT2) by Philip Hofmann, CC-BY-4.0, converted to ONNX"),
    "drct": dict(file="4xRealWebPhoto_v4_drct-l_fp32_256.onnx", size="?", native=4, twice=False,
                 url=CONVERTED + "4xRealWebPhoto_v4_drct-l_fp32_256.onnx",
                 credit="4xRealWebPhoto_v4_drct-l (DRCT) by Philip Hofmann, CC-BY-4.0, converted to ONNX"),
    "atd": dict(file="4xRealWebPhoto_v3_atd_fp32_256.onnx", size="?", native=4, twice=False,
                url=CONVERTED + "4xRealWebPhoto_v3_atd_fp32_256.onnx",
                credit="4xRealWebPhoto_v3_atd (ATD) by Philip Hofmann, CC-BY-4.0, converted to ONNX"),
    "animesharp": dict(file="2x-AnimeSharpV4_RCAN_fp16_op17.onnx", size="31 MB", native=2, twice=True,
                       url="https://huggingface.co/Kim2091/2x-AnimeSharpV4/resolve/main/2x-AnimeSharpV4_RCAN_fp16_op17.onnx",
                       credit="2x-AnimeSharpV4 by Kim2091, CC BY-NC-SA 4.0 (personal use)"),
    "hatgan": dict(file="real_hatgan_x4.onnx", size="160 MB", native=4, twice=False,
                   url=FACEFUSION + "real_hatgan_x4.onnx",
                   credit="Real-HAT-GAN (HAT, XPixel), ONNX by FaceFusion"),
    "heart": dict(file="heart_4x_otf_gan_fp32_op17.onnx", size="75 MB", native=4, twice=False,
                  url="https://huggingface.co/Phips/HEART/resolve/main/onnx/heart_4x_otf_gan_fp32_op17.onnx",
                  credit="HEART by Philip Hofmann, Apache-2.0"),
}
# The inventing upscaler (Stable Diffusion x4, ONNX by ssube): several files in models/sd-x4-upscaler/<sub>/,
# downloaded only when it is used. Run by run_diffusion (numpy, no torch).
DIFFUSION_URL = "https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx/resolve/main/"
DIFFUSION_DIR = "sd-x4-upscaler"
DIFFUSION_FILES = (("text_encoder/model.onnx", "1.36 GB"), ("unet/model.onnx", "small"), ("unet/weights.pb", "1.9 GB"),
                   ("vae/model.onnx", "132 MB"), ("tokenizer/vocab.json", "1 MB"), ("tokenizer/merges.txt", "0.5 MB"))
DIFFUSION_CREDIT = "Stable Diffusion x4 upscaler by Stability AI, ONNX by ssube, CreativeML Open RAIL++-M"
# the models a picture can be made bigger with, in the order of the window's list (Automatic is not one of them)
MODEL_LABELS = {
    "anime": "Drawing (Real-ESRGAN anime)", "animesharp": "Drawing AnimeSharp V4 (faithful)",
    "general": "General photo (Real-ESRGAN)", "dat": "Web photo DAT RealWebPhoto v4",
    "bhi": "Web photo DAT2 BHI (newest DAT, Dec 2024)", "drct": "Web photo DRCT-L", "atd": "Web photo ATD",
    "hatgan": "Clean photo Real-HAT-GAN (sharp)", "heart": "Photo HEART (new, experimental)",
    "diffusion": "Inventing: Stable Diffusion x4 (makes up detail, slow)",
}
PLAIN_LABEL = "Plain resize (Lanczos, no AI)"
ANIMESHARP_WARNING = ("AnimeSharp V4 is free for personal use only (licence CC BY-NC-SA 4.0). "
                      "Don't sell or publish results made with it commercially.")
# Colouring black-and-white photos: DDColor (ONNX by FaceFusion), downloaded only when a black-and-white photo is met
COLORIZER = dict(file="ddcolor.onnx", size="980 MB", url=FACEFUSION + "ddcolor.onnx",
                 credit="DDColor (piddnad), ONNX by FaceFusion, Apache-2.0")
MONO_COLOR_FRAC = 0.03    # a picture with less colour than this (see classify) is black and white
# Face restoration: GFPGAN 1.4 or CodeFormer (restore), and OpenCV's YuNet (finds faces and their 5 landmarks)
FACE_MODELS = {
    "gfpgan": dict(file="gfpgan_1.4.onnx", size="340 MB", url=FACEFUSION + "gfpgan_1.4.onnx"),
    "codeformer": dict(file="codeformer.onnx", size="377 MB", url=FACEFUSION + "codeformer.onnx"),
}
CODEFORMER_WEIGHT = 0.6   # CodeFormer's second input: 0 = most repair, 1 = most faithful
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET_FILE
FACE_TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                          [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)  # 512x512 FFHQ
# model files the program uses itself: they are not offered as extra models
OWN_MODELS = {m["file"][:-5] for m in (*UPSCALERS.values(), *FACE_MODELS.values(), COLORIZER)} | {YUNET_FILE[:-5]}


# ===================================================================== options
@dataclass
class Options:
    inputs: list = field(default_factory=list)   # folders (all images directly in them) and/or image files
    repeat_last: bool = False    # use the images of the last run instead of the inputs
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
    mode: str = "pdf"            # "pdf" = Tracing PDF, "upscale" = Bigger images, "compare" = Compare models (blind)
    save_to: str = ""            # where results and _blurnomore_work go; empty = the folder of the first input
    scale: int = 2               # upscale mode: 2, 3 or 4 times bigger
    size_mode: str = "scale"     # upscale mode: "scale" (times bigger), "width" or "height" in pixels
    size_px: int = 1920          # upscale mode: the pixels for width or height
    shrink_bigger: bool = False  # upscale mode, width/height: also make pictures that are already bigger smaller
    kind: str = "auto"           # upscale mode: "auto" (detect per image), "drawing" or "photo"
    up_model: str = "auto"       # upscale mode: "auto", "general", "anime", a key of UPSCALERS or an extra model's name
    name_mode: str = "auto"      # upscale mode: file name "auto" (<name>_x2), "suffix" or "prefix"
    name_text: str = ""          # the suffix or prefix text (empty = auto)
    file_type: str = "png"       # upscale mode: "png" or "jpg"
    faces: bool = False          # upscale mode: restore faces
    face_strength: int = 50      # 0-100, how much of the restored face is used
    face_model: str = "gfpgan"   # "gfpgan" (natural) or "codeformer" (strongest repair)
    colorize: bool = False       # upscale mode: colour black-and-white photos (invented colours)
    compare_models: list = field(default_factory=lambda: ["dat", "bhi", "drct", "atd", "hatgan", "heart", "general"])  # compare mode
    diff_prompt: str = "a sharp, detailed, high quality photo"   # inventing upscaler: what to make up
    diff_negative: str = "blurry, noisy, jpeg artifacts, painting, cartoon"
    diff_noise: int = 20         # 0-350, how much it may invent
    diff_steps: int = 25         # 5-100
    diff_guidance: float = 7.5   # 1-15, how strongly it follows the prompt
    diff_seed: int = 1           # 0 = a new random seed each run
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
    if color_frac < MONO_COLOR_FRAC:  # effectively monochrome
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
    """Names of the extra models in the models folder next to the program (sorted): ncnn
    NAME.param + NAME.bin pairs, and NAME.onnx files."""
    if not MODELS_DIR.is_dir():
        return []
    names = {p.name[:-6] for p in MODELS_DIR.glob("*.param") if (MODELS_DIR / (p.name[:-6] + ".bin")).is_file()}
    names |= {p.name[:-5] for p in MODELS_DIR.glob("*.onnx")} - OWN_MODELS
    return sorted(names)


def model_scale(name):
    """Native enlargement of an extra model, read from its name (2x / x2, 3x / x3, else 4)."""
    m = re.search(r"(?<![0-9])(?:([23])x|x([23]))(?![0-9])", name.lower())
    return int(m.group(1) or m.group(2)) if m else 4


def writable_dir(preferred, fallback):
    """preferred if files can be made there, else fallback."""
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_test"
        probe.write_text("x")
        probe.unlink()
        return preferred
    except OSError:
        return fallback


def ensure_esrgan(work, log):
    # Next to the program is the home of Real-ESRGAN; the old place (in the work folder) still counts.
    for base in (ESRGAN_DIR, work / "realesrgan"):
        exe = _find_exe(base)
        if exe:
            return exe
    d = writable_dir(ESRGAN_DIR, work / "realesrgan")   # the program folder may not be writable
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
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:                      # the program is there but cannot start
        log(f"  Real-ESRGAN could not start: {e}")
        return None
    if r.returncode != 0 or not tout.exists():
        log("  Real-ESRGAN failed. Output:")
        log("  " + (r.stderr or r.stdout or "(no output)")[-600:].strip().replace("\n", "\n  "))
        return None
    out = Image.open(tout).convert("RGB")
    out.load()
    return out


# ================================================================ ONNX models
class OnnxModel:
    """An upscaling model run with onnxruntime. tile is the fixed tile side of a model that has a
    fixed input size, else None."""

    def __init__(self, sess, native, tile=None, twice=False):
        self.sess, self.native, self.tile, self.twice = sess, native, tile, twice
        self.input = sess.get_inputs()[0].name
        self.half = "float16" in sess.get_inputs()[0].type   # a model may want float16 instead of float32

    def run(self, t):
        """One forward pass of a float32 NCHW array; the answer is float32 whatever the model uses."""
        r = self.sess.run(None, {self.input: np.ascontiguousarray(t, np.float16 if self.half else np.float32)})[0]
        return r.astype(np.float32, copy=False)


def fetch_model(c, name, url, size):
    """Path of a model file in the models folder, downloading it on first use (to a temporary
    name, so a broken download is never used). None when it cannot be had."""
    for base in (MODELS_DIR, c.work / "models"):
        if (base / name).is_file():
            return base / name
    d = writable_dir(MODELS_DIR, c.work / "models")
    part = d / (name + ".part")
    try:
        c.L(f"Downloading {name} (about {size}) ...")
        part.parent.mkdir(parents=True, exist_ok=True)    # a name may have a sub folder
        urllib.request.urlretrieve(url, part)
        os.replace(part, d / name)
        return d / name
    except Exception as e:  # noqa: BLE001
        c.L(f"  Could not download {name}: {e}")
        c.L(f"  Download it manually from {url}")
        c.L(f"  and put it in {d}.")
        try:
            part.unlink()
        except OSError:
            pass
        return None


def make_session(c, path):
    """An onnxruntime session on the GPU (DirectML) when there is one, else the CPU."""
    prov = c.providers()
    opts = ort.SessionOptions()
    if "DmlExecutionProvider" in prov:
        opts.enable_mem_pattern = False       # DirectML needs these two
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        sess = ort.InferenceSession(str(path), opts, providers=prov)
    except Exception as e:  # noqa: BLE001
        if len(prov) < 2:
            raise
        c.L(f"  The GPU did not start ({e}); using the CPU.")
        c.prov = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(str(path), opts, providers=c.prov)
    c.once("provider", "ONNX Runtime uses " + ("the GPU (DirectML)" if sess.get_providers()[0] == "DmlExecutionProvider"
                                               else "the CPU"))
    return sess


def run_tiles(model, im, tile=256, over=16):
    """Enlarge a PIL image with an ONNX model in tiles (input RGB 0..1, NCHW). Each tile is full
    size (the picture is edge-padded), the overlap is cropped away and the centre kept."""
    tile = model.tile or tile
    over = min(over, tile // 4)
    step, n = tile - 2 * over, model.native
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    H, W = a.shape[:2]
    ny, nx = -(-H // step), -(-W // step)
    a = np.pad(a, ((over, ny * step + over - H), (over, nx * step + over - W), (0, 0)), mode="reflect")
    out = np.zeros((ny * step * n, nx * step * n, 3), np.uint8)
    keep = slice(over * n, (over + step) * n)
    for j in range(ny):
        for i in range(nx):
            t = a[j * step:j * step + tile, i * step:i * step + tile].transpose(2, 0, 1)[None]
            r = model.run(t)[0][:, keep, keep]
            out[j * step * n:(j + 1) * step * n, i * step * n:(i + 1) * step * n] = (
                np.clip(r.transpose(1, 2, 0), 0, 1) * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(out[:H * n, :W * n], "RGB")


def load_onnx(c, name):
    """A downloadable model (a key of UPSCALERS) or an extra model's name, ready to run; None (after
    saying why) when not."""
    if name in c.models:
        return c.models[name]
    c.models[name] = None
    info = UPSCALERS.get(name)
    what = f"the model '{name}'"
    if not HAVE_ORT:
        c.once("no-ort-" + name, f"{INSTALL_HINT} to use {what}; using Real-ESRGAN instead")
        return None
    path = fetch_model(c, info["file"], info["url"], info["size"]) if info else MODELS_DIR / f"{name}.onnx"
    if path is None:
        return None
    try:
        sess = make_session(c, path)
        shape = sess.get_inputs()[0].shape
        fixed = shape[2] if len(shape) == 4 and isinstance(shape[2], int) and shape[2] == shape[3] else None
        model = OnnxModel(sess, info["native"] if info else model_scale(name), fixed, bool(info and info["twice"]))
        if name in ("dat", "bhi", "drct", "atd") and not c.gpu():
            c.L(f"  The {name.upper()} model on the CPU is slow (minutes per photo).")
        t = model.tile or 64    # check the scale by one small tile
        got = model.run(np.zeros((1, 3, t, t), np.float32)).shape[2] // t
        if got != model.native:
            c.L(f"  Model '{name}': it should enlarge {model.native}x but it enlarges {got}x; using {got}x.")
            model.native = got
    except Exception as e:  # noqa: BLE001
        c.L(f"  Could not start {what}: {e}")
        return None
    c.models[name] = model
    return model


# ================================================= inventing upscaler (Stable Diffusion x4)
class ClipTokenizer:
    r"""CLIP's byte-level BPE tokenizer in plain Python (as OpenAI's simple_tokenizer), from the vocab.json and
    merges.txt files. Letters and digits are matched with [^\W\d_] and \d (Python's re has no \p{L})."""
    PAT = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+", re.IGNORECASE)

    def __init__(self, vocab_path, merges_path):
        self.enc = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        lines = Path(merges_path).read_text(encoding="utf-8").split("\n")
        if lines and lines[0].startswith("#version"):
            lines = lines[1:]
        self.ranks = {tuple(ln.split()): i for i, ln in enumerate(l for l in lines if len(l.split()) == 2)}
        keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        chars, n = keep[:], 0
        for b in range(256):
            if b not in keep:
                keep.append(b)
                chars.append(256 + n)
                n += 1
        self.byte_char = dict(zip(keep, map(chr, chars)))
        self.bos, self.eos = self.enc["<|startoftext|>"], self.enc["<|endoftext|>"]
        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pairs = set(zip(word, word[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            first, second = best
            new, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = tuple(new)
        self.cache[token] = word
        return word

    def __call__(self, text, length=77):
        """The token ids of text: start, the words, end, then end again up to length."""
        ids = [self.bos]
        for piece in self.PAT.findall(re.sub(r"\s+", " ", text).strip().lower()):
            piece = "".join(self.byte_char[b] for b in piece.encode("utf-8"))
            ids += [self.enc.get(t, self.eos) for t in self.bpe(piece)]
        ids = ids[:length - 1] + [self.eos]
        return ids + [self.eos] * (length - len(ids))


ORT_NUMPY = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64,
             "tensor(int32)": np.int32, "tensor(int64)": np.int64}


def ort_feed(sess, **arrays):
    """The inputs of an onnxruntime session from named arrays, each made the type the model wants."""
    types = {i.name: i.type for i in sess.get_inputs()}
    return {k: np.ascontiguousarray(v, ORT_NUMPY.get(types[k], np.float32)) for k, v in arrays.items()}


def ddim_alphas():
    """The cumulative alphas of the scaled_linear noise schedule (1000 steps, betas 0.0001 to 0.02)."""
    betas = np.linspace(0.0001 ** 0.5, 0.02 ** 0.5, 1000, dtype=np.float64) ** 2
    return np.cumprod(1.0 - betas)


def ddim_timesteps(n):
    """The n timesteps of a DDIM run, from the noisiest down (steps_offset 1)."""
    ratio = 1000 // n
    return (np.arange(n) * ratio).round()[::-1].astype(np.int64) + 1, ratio


def ddim_step(latents, v, t, ratio, acp):
    """One DDIM step (eta 0) of a v-prediction model: the latents a little less noisy."""
    prev = t - ratio
    a_t, a_prev = acp[t], (acp[prev] if prev >= 0 else acp[0])   # set_alpha_to_one is False
    x0 = np.sqrt(a_t) * latents - np.sqrt(1 - a_t) * v
    eps = np.sqrt(a_t) * v + np.sqrt(1 - a_t) * latents
    return (np.sqrt(a_prev) * x0 + np.sqrt(1 - a_prev) * eps).astype(np.float32)


DIFF_TILE, DIFF_OVER = 128, 16     # low-resolution tile side and overlap (the results are 4x that)


def diff_positions(n):
    """Start of each low-resolution tile along a side of n pixels (n >= DIFF_TILE): every DIFF_TILE - DIFF_OVER, the last one flush."""
    pos = list(range(0, n - DIFF_TILE + 1, DIFF_TILE - DIFF_OVER))
    return pos if pos[-1] == n - DIFF_TILE else pos + [n - DIFF_TILE]


def diff_window():
    """The weight of a pixel of a 4x tile: 1 inside, fading out over the overlap at the edges, never 0."""
    n, ov = DIFF_TILE * 4, DIFF_OVER * 4
    ramp = np.minimum(np.minimum(np.arange(n) + 1, np.arange(n)[::-1] + 1) / (ov + 1.0), 1.0).astype(np.float32)
    return ramp[:, None] * ramp[None, :]


class Diffusion:
    """Stable Diffusion x4 upscaler, as diffusers' StableDiffusionUpscalePipeline, in numpy: a text encoder, the
    unet and the vae decoder (onnxruntime sessions) and the tokenizer."""

    def __init__(self, tokenizer, encoder, unet, vae):
        self.tok, self.enc, self.unet, self.vae = tokenizer, encoder, unet, vae

    def embed(self, text):
        ids = np.array([self.tok(text)])
        name = self.enc.get_inputs()[0].name
        return self.enc.run(None, ort_feed(self.enc, **{name: ids}))[0].astype(np.float32)

    def tile(self, rng, low, emb, o, steps):
        """One low-resolution tile (3 x h x w, -1..1) made 4x bigger: (3 x 4h x 4w, about -1..1)."""
        acp = ddim_alphas()
        n = o.diff_noise
        low = np.sqrt(acp[n]) * low + np.sqrt(1 - acp[n]) * rng.standard_normal(low.shape)
        low = np.repeat(low[None].astype(np.float32), 2, axis=0)       # for the prompt and for the negative prompt
        h, w = low.shape[2:]
        lat = rng.standard_normal((1, 4, h, w)).astype(np.float32)
        ts, ratio = ddim_timesteps(steps)
        labels = np.full((2,), n, np.int64)
        for t in ts:
            both = np.concatenate([np.repeat(lat, 2, axis=0), low], axis=1)
            feed = ort_feed(self.unet, sample=both, timestep=np.array([t]), encoder_hidden_states=emb, class_labels=labels)
            noise = self.unet.run(None, feed)[0].astype(np.float32)
            v = noise[0:1] + o.diff_guidance * (noise[1:2] - noise[0:1])      # the negative prompt is first
            lat = ddim_step(lat, v, int(t), ratio, acp)
        name = self.vae.get_inputs()[0].name
        return self.vae.run(None, ort_feed(self.vae, **{name: lat / 0.08333}))[0][0].astype(np.float32)

    def run(self, im, o, seed, log):
        """Enlarge a PIL image 4x by tiles of DIFF_TILE pixels (feathered where they overlap)."""
        steps = min(100, max(1, int(o.diff_steps)))
        emb = np.concatenate([self.embed(o.diff_negative), self.embed(o.diff_prompt)])
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        H, W = a.shape[:2]
        a = np.pad(a, ((0, max(0, DIFF_TILE - H)), (0, max(0, DIFF_TILE - W)), (0, 0)), mode="edge")
        Hp, Wp = a.shape[:2]
        tiles = [(y, x) for y in diff_positions(Hp) for x in diff_positions(Wp)]
        acc, wsum, win = np.zeros((3, Hp * 4, Wp * 4), np.float32), np.zeros((Hp * 4, Wp * 4), np.float32), diff_window()
        rng = np.random.default_rng(seed)
        for k, (y, x) in enumerate(tiles, 1):
            log(f"  tile {k}/{len(tiles)}")
            r = self.tile(rng, a[y:y + DIFF_TILE, x:x + DIFF_TILE].transpose(2, 0, 1), emb, o, steps)
            ys, xs = slice(y * 4, (y + DIFF_TILE) * 4), slice(x * 4, (x + DIFF_TILE) * 4)
            acc[:, ys, xs] += r * win
            wsum[ys, xs] += win
        out = np.clip((acc / wsum + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(out.transpose(1, 2, 0)[:H * 4, :W * 4], "RGB")


def load_diffusion(c):
    """The inventing upscaler ready to run, downloading its files (about 3.4 GB) on first use; None (after saying why) when not."""
    if "diffusion" in c.models:
        return c.models["diffusion"]
    c.models["diffusion"] = None
    if not HAVE_ORT:
        c.once("no-ort-diffusion", f"{INSTALL_HINT} to use the inventing upscaler; using Real-ESRGAN instead")
        return None
    if not any(all((base / DIFFUSION_DIR / rel).is_file() for rel, _ in DIFFUSION_FILES) for base in (MODELS_DIR, c.work / "models")):
        c.L("The inventing upscaler (Stable Diffusion x4) needs about 3.4 GB of downloads the first time.")
    paths = {}
    for rel, size in DIFFUSION_FILES:
        paths[rel] = fetch_model(c, f"{DIFFUSION_DIR}/{rel}", DIFFUSION_URL + rel, size)
        if paths[rel] is None:
            return None
    try:
        tok = ClipTokenizer(paths["tokenizer/vocab.json"], paths["tokenizer/merges.txt"])
        sessions = [make_session(c, paths[f"{d}/model.onnx"]) for d in ("text_encoder", "unet", "vae")]
        c.models["diffusion"] = Diffusion(tok, *sessions)
        c.L("  Inventing upscaler: very slow (minutes per picture even on a graphics card) and it changes the "
            "picture: faces and text can come out wrong. " + DIFFUSION_CREDIT)
    except Exception as e:  # noqa: BLE001
        c.L(f"  Could not start the inventing upscaler: {e}")
    return c.models["diffusion"]


def run_diffusion(c, im, f):
    """The inventing upscaler on a PIL image: 4x bigger (the caller resizes to the exact size)."""
    o = c.o
    if c.diff_seed is None:
        c.diff_seed = int(o.diff_seed) or random.SystemRandom().randrange(1, 2 ** 31)
        c.L(f"Inventing upscaler: seed {c.diff_seed}" + (" (chosen at random; type it under Seed to get the same result again)"
                                                         if not o.diff_seed else ""))
    if f > 4:
        c.once("diffusion-4x", "  The inventing upscaler only invents at 4x; for a bigger size the 4x result is resized.")
    return c.models["diffusion"].run(im, o, c.diff_seed, c.L)


# ============================================================ face restoration
def load_detector(c):
    """OpenCV's YuNet face finder, or None. Not tried again after a failure."""
    if "detector" not in c.models:
        c.models["detector"] = None
        path = fetch_model(c, YUNET_FILE, YUNET_URL, "230 KB")
        if path:
            try:
                c.models["detector"] = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.7)
            except Exception as e:  # noqa: BLE001
                c.L(f"  Could not start the face finder: {e}")
    return c.models["detector"]


def load_face(c, name):
    """The face restorer 'gfpgan' or 'codeformer' (an onnxruntime session), or None."""
    key = "face_" + name
    if key not in c.models:
        c.models[key] = None
        info = FACE_MODELS[name]
        path = fetch_model(c, info["file"], info["url"], info["size"])
        if path:
            try:
                c.models[key] = make_session(c, path)
            except Exception as e:  # noqa: BLE001
                c.L(f"  Could not start the face restorer: {e}")
    return c.models[key]


def face_inputs(sess, x):
    """What a face restorer is fed: the face picture, plus CodeFormer's fidelity weight when the
    model has that second input (found by name, else it is the second input)."""
    ins = sess.get_inputs()
    feed = {ins[0].name: np.ascontiguousarray(x)}
    if len(ins) > 1:
        w = next((i for i in ins[1:] if "weight" in i.name.lower()), ins[1])
        dt = np.float64 if "double" in w.type else np.float16 if "float16" in w.type else np.float32
        feed[w.name] = np.full((1,), CODEFORMER_WEIGHT, dt)
    return feed


def detect_faces(detector, a):
    """The 5 landmarks (5x2: eyes, nose, mouth corners) of each face of at least 48 px in the
    RGB array a. Looked for on a copy of at most 1600 px, so big pictures stay quick."""
    f = min(1.0, 1600.0 / max(a.shape[:2]))
    small = cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else a
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, rows = detector.detect(np.ascontiguousarray(small[..., ::-1]))   # YuNet wants BGR
    return [(r[4:14].reshape(5, 2) / f).astype(np.float32) for r in (rows if rows is not None else [])
            if min(r[2], r[3]) / f >= 48]


def restore_faces(c, im):
    """Find the faces in a PIL image and restore each one with the chosen face model. Returns (image, number);
    the number is None when faces were not looked for (said in the log why)."""
    o = c.o
    if o.face_strength <= 0:
        c.once("face0", "Face strength is 0: faces are left as they are.")
        return im, None
    if not (HAVE_ORT and HAVE_CV2):
        c.once("no-faces", f"{INSTALL_HINT} to restore faces; faces are skipped")
        return im, None
    detector = load_detector(c)
    if detector is None:
        return im, None
    a = np.asarray(im.convert("RGB"))
    found = detect_faces(detector, a)
    if not found:
        return im, 0
    sess = load_face(c, o.face_model)
    if sess is None:
        return im, None
    s = min(o.face_strength, 100) / 100.0
    mask = np.zeros((512, 512), np.float32)
    cv2.ellipse(mask, (256, 270), (180, 215), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 20)
    out, H, W, n = a.copy(), a.shape[0], a.shape[1], 0
    for lm in found:
        M, _ = cv2.estimateAffinePartial2D(lm, FACE_TEMPLATE, method=cv2.LMEDS)
        if M is None:
            continue
        face = cv2.warpAffine(a, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x = (face.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        y = sess.run(None, face_inputs(sess, x))[0][0].transpose(1, 2, 0)
        y = np.clip((y + 1.0) * 127.5, 0, 255).astype(np.float32)
        inv = cv2.invertAffineTransform(M)
        pts = cv2.transform(np.array([[[0, 0], [512, 0], [512, 512], [0, 512]]], np.float32), inv)[0]
        x0, y0 = max(0, int(np.floor(pts[:, 0].min()))), max(0, int(np.floor(pts[:, 1].min())))
        x1, y1 = min(W, int(np.ceil(pts[:, 0].max()))), min(H, int(np.ceil(pts[:, 1].max())))
        if x1 <= x0 or y1 <= y0:
            continue
        inv[:, 2] -= (x0, y0)
        size = (x1 - x0, y1 - y0)
        back = cv2.warpAffine(y, inv, size, flags=cv2.INTER_LINEAR)
        alpha = (cv2.warpAffine(mask, inv, size, flags=cv2.INTER_LINEAR) * s)[..., None]
        region = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(region * (1 - alpha) + back * alpha + 0.5, 0, 255).astype(np.uint8)
        n += 1
    return Image.fromarray(out, "RGB"), n


# ============================================================ colouring
def load_colorizer(c):
    """The DDColor session, or None (after saying why). Not tried again after a failure."""
    if "colorizer" not in c.models:
        c.models["colorizer"] = None
        if not (HAVE_ORT and HAVE_CV2):
            c.once("no-colorize", f"{INSTALL_HINT} to colour black-and-white photos; they are left as they are")
            return None
        path = fetch_model(c, COLORIZER["file"], COLORIZER["url"], COLORIZER["size"])
        if path:
            try:
                c.models["colorizer"] = make_session(c, path)
            except Exception as e:  # noqa: BLE001
                c.L(f"  Could not start the colouring model: {e}")
    return c.models["colorizer"]


def colorize(c, im):
    """Colour a black-and-white PIL image with DDColor (the colours are a guess). Returns the coloured
    image of the same size, or the image itself when the model is not available. As FaceFusion does it,
    in OpenCV's BGR: the grey picture's lightness goes in, the model's colour channels come out and are
    put on the picture's own lightness."""
    sess = load_colorizer(c)
    if sess is None:
        return im
    bgr = np.ascontiguousarray(np.asarray(im.convert("RGB"))[..., ::-1])
    H, W = bgr.shape[:2]
    grey = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
    lab = cv2.cvtColor((grey / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB)[:, :, :1]
    lab = np.concatenate((lab, np.zeros_like(lab), np.zeros_like(lab)), axis=-1)
    x = cv2.resize(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), (512, 512)).transpose(2, 0, 1)[None]
    half = "float16" in sess.get_inputs()[0].type
    y = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(x, np.float16 if half else np.float32)})[0]
    ab = cv2.resize(y[0].astype(np.float32).transpose(1, 2, 0), (W, H))
    light = cv2.cvtColor((bgr / 255.0).astype(np.float32), cv2.COLOR_BGR2LAB)[:, :, :1]
    out = cv2.cvtColor(np.concatenate((light, ab), axis=-1), cv2.COLOR_LAB2BGR)
    out = (out * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(out[..., ::-1]), "RGB")


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


def spread(items, n):
    """n items spread evenly over the list (all of them when there are fewer)."""
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(j * step)] for j in range(n)]


def read_last_run():
    """The image paths the last run used (blurnomore_last_run.json next to the program), or []."""
    try:
        return [str(p) for p in json.loads(LAST_RUN_FILE.read_text(encoding="utf-8"))]
    except Exception:  # noqa: BLE001
        return []


def output_dir(o):
    """Where results and _blurnomore_work go: 'Save to', or the folder of the first input when that is empty."""
    if o.save_to.strip():
        return Path(o.save_to.strip().strip('"'))
    first = (read_last_run() if o.repeat_last else []) or o.inputs
    p = Path(str(first[0]).strip().strip('"')) if first else Path(DEFAULT_FOLDER)
    return p if p.is_dir() else p.parent


def resolve_images(o, L):
    """The image files this run (or its preview) works on: a folder gives all images directly in
    it, a file is taken as given; duplicates go, the order stays. With 'same images as last run'
    the list of the last run replaces the inputs. Problems are said through L."""
    inputs = o.inputs
    if o.repeat_last:
        last = read_last_run()
        if last:
            L(f"Repeating the {len(last)} images from the last run")
            inputs = last
        else:
            L("No earlier run found; using the images in the list.")
    files, seen = [], set()
    for raw in inputs:
        p = Path(str(raw).strip().strip('"'))
        if p.is_dir():
            found = list_images(p)
            if not found:
                L(f"No images found in {p}")
        elif p.is_file():
            found = [p]
        else:
            L(f"Not found, skipped: {p}")
            continue
        for f in found:
            key = os.path.normcase(os.path.abspath(f))
            if key not in seen:
                seen.add(key)
                files.append(f)
    return files


def target_size(o, w, h):
    """(width, height) a w x h picture becomes in Bigger images: times bigger, or the pixels asked for
    with the aspect ratio kept. A picture already that wide or high keeps its size unless shrink_bigger."""
    n = max(1, int(o.size_px))
    if o.size_mode == "width":
        return (n, max(1, round(h * n / w))) if w < n or o.shrink_bigger else (w, h)
    if o.size_mode == "height":
        return (max(1, round(w * n / h)), n) if h < n or o.shrink_bigger else (w, h)
    return w * o.scale, h * o.scale


def size_text(o):
    """Automatic name ending: _x2, _1920w or _1080h."""
    return {"width": f"_{o.size_px}w", "height": f"_{o.size_px}h"}.get(o.size_mode, f"_x{o.scale}")


def result_name(o, stem, ext, folder):
    """File name of one bigger image: <name>_x2, or the owner's text after or before the name.
    Never a name that is already taken in folder (a second one gets _2)."""
    text = "".join(ch for ch in o.name_text if ch not in '<>:"/\\|?*' and ord(ch) >= 32).strip() \
        if o.name_mode in ("suffix", "prefix") else ""
    base = (text + stem if o.name_mode == "prefix" else stem + text) if text else stem + size_text(o)
    name, k = f"{base}.{ext}", 2
    while (folder / name).exists():
        name, k = f"{base}_{k}.{ext}", k + 1
    return name


def prepare(L, out):
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
    notes = []
    files = resolve_images(o, notes.append)
    if not files:
        for m in notes:
            log(m)
        log("No images to work on. Add a folder or some pictures first.")
        return None

    started = time.time()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = output_dir(o)
    work = prepare(log, out)
    if work is None:
        return None
    logfile = open(work / "logs" / f"{stamp}.log", "w", encoding="utf-8")

    def L(msg=""):
        log(msg)
        logfile.write(msg + "\n")
        logfile.flush()

    try:
        return _run(o, L, should_stop, out, files, notes, stamp, work, started)
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
        self.models, self.prov, self.said = {}, None, set()   # ONNX models by name, providers, notes made
        self.diff_seed = None        # the seed the inventing upscaler uses this run
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
        if o.mode == "upscale" and o.up_model not in ("auto", "general", "anime", "diffusion", *UPSCALERS, *extra_models()):
            L(f"Unknown upscale model '{o.up_model}'; choosing automatically.")
            self.up_model = "auto"

    def once(self, key, msg):
        """Say something once per run."""
        if key not in self.said:
            self.said.add(key)
            self.L(msg)

    def providers(self):
        if self.prov is None:
            avail = ort.get_available_providers() if HAVE_ORT else []
            self.prov = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in avail]
        return self.prov

    def gpu(self):
        return "DmlExecutionProvider" in self.providers()

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
def upscale_model(c, cls, force=None):
    """(engine, model name, model folder, native scale) for this kind of picture, or None for plain
    resizing. The engine is "ncnn" (Real-ESRGAN program), "onnx" (onnxruntime) or "diffusion". With force
    (Compare models) that model or nothing: no automatic choice and no other model in its place."""
    if not c.o.use_ai:
        return None
    ncnn = "anime" if cls in ("lineart", "flatcolor") else "general"
    pick = force or c.up_model
    if pick == "auto":
        pick = ("animesharp" if cls in ("lineart", "flatcolor") else "dat") if c.gpu() else ncnn
    if pick == "diffusion":
        if load_diffusion(c):
            return "diffusion", pick, None, 4
        if force:
            return None
        pick = ncnn
    if pick in UPSCALERS:
        model = load_onnx(c, pick)
        if model:
            if pick == "animesharp":
                c.once("animesharp-licence", ANIMESHARP_WARNING)
            return "onnx", pick, None, model.native
        if force:
            return None
        pick = ncnn
    if pick in ("anime", "general"):
        if c.exe is None:
            return None
        return "ncnn", MODEL_ANIME if pick == "anime" else MODEL_GENERAL, c.exe.parent / "models", 4
    if (MODELS_DIR / f"{pick}.param").is_file():
        return ("ncnn", pick, MODELS_DIR, model_scale(pick)) if c.exe else None
    model = load_onnx(c, pick)
    return ("onnx", pick, None, model.native) if model else None


def make_upscaled(c, path, src_id, im, store=False, force=None):
    """Enlarge (or shrink) one image to exactly the size target_size gives. A result made by the preview is reused;
    with store=True a new result is kept for the run that follows. With force (Compare models) it is made with that
    model, or plain with force="plain", never with another model and never from the cache; when that model cannot
    do it, RuntimeError says why."""
    o, L = c.o, c.L
    w, h = im.size
    auto_cls, stats = classify(im)
    cls = {"drawing": "lineart", "photo": "photo"}.get(o.kind) or auto_cls
    kind = "drawing" if cls in ("lineart", "flatcolor") else "photo"
    to_colour = o.colorize and kind == "photo" and stats["color_frac"] < MONO_COLOR_FRAC   # a black-and-white photo
    target = target_size(o, w, h)
    f = target[0] / w
    kept = ""
    if o.size_mode != "scale" and target == (w, h):
        kept = f"already {w if o.size_mode == 'width' else h} {'wide' if o.size_mode == 'width' else 'high'}, kept its size"
    m = upscale_model(c, cls, force) if f > AI_MIN_SCALE and force != "plain" else None   # a picture that stays about the same size needs no AI
    if force and force != "plain" and f > AI_MIN_SCALE and m is None:
        raise RuntimeError("the AI upscaler is switched off" if not o.use_ai else "it could not be started (see the log)")
    label = m[1] if m else "plain"

    def cache_file(name):
        faces = f"_faces{o.face_strength}_{o.face_model}" if o.faces else ""
        if name == "diffusion":      # its settings change the picture
            name += "_" + hashlib.md5(repr((o.diff_prompt, o.diff_negative, o.diff_noise, o.diff_steps, o.diff_guidance,
                                            o.diff_seed)).encode()).hexdigest()[:8]
        return c.cache_dir / f"{path.stem[:40]}_{src_id}_{name}_{target[0]}x{target[1]}{faces}{'_coloured' if o.colorize else ''}.png"

    cache = cache_file(label)
    if cache.exists() and not o.redo and not force:
        res = Image.open(cache).convert("RGB")
        res.load()
        method = (f"{label} (cached)" if m else "lanczos (cached)") + (" + coloured" if to_colour else "")
        return {"result": res, "cls": cls, "kind": kind, "w": w, "h": h, "method": method, "model": label, "target": target, "f": f, "kept": kept}
    res = None
    method = "lanczos"
    coloured = False
    if to_colour:       # first, on the original-size picture
        try:
            coloured_im = colorize(c, im)
            coloured, im = coloured_im is not im, coloured_im
        except Exception:  # noqa: BLE001
            L("  Colouring failed; the picture is left black and white:\n  " + traceback.format_exc().replace("\n", "\n  "))
            c.models["colorizer"] = None
    if m:
        up, why = None, "Real-ESRGAN did not work"
        if m[0] == "ncnn":
            up = run_esrgan(im, m[1], c.exe, c.tmp, L, m[2], m[3])
            if up is None:
                L("  Real-ESRGAN did not work; using plain resizing for the rest of this run.")
                c.exe = None
        else:
            try:
                if m[0] == "diffusion":
                    up = run_diffusion(c, im, f)
                else:
                    model = c.models[m[1]]
                    up = run_tiles(model, im)
                    if model.twice and f > model.native:    # more than twice as big: a second pass on the result
                        up = run_tiles(model, up)
            except Exception as e:  # noqa: BLE001
                why = f"{type(e).__name__}: {e}"[:200]
                L(f"  The model '{m[1]}' failed:\n  " + traceback.format_exc().replace("\n", "\n  "))
                L(f"  Using plain resizing instead of '{m[1]}' for the rest of this run.")
                c.models[m[1]] = None
        if up is None and force:
            raise RuntimeError(why)
        if up is None:
            method, label = "lanczos (AI failed)", "plain"
        else:
            res = up.resize(target, Image.LANCZOS)
            method = f"AI {m[1]}"
    if res is None:
        res = im.resize(target, Image.LANCZOS)
    if coloured:
        method += " + coloured"
    if o.faces:
        res, nf = restore_faces(c, res)
        if nf is not None:
            L(f"  {nf} face{'' if nf == 1 else 's'} restored")
            method += f" + {nf} face{'' if nf == 1 else 's'}" if nf else ""
    if store:
        try:
            c.cache_dir.mkdir(parents=True, exist_ok=True)
            res.save(cache_file(label))
        except Exception:  # noqa: BLE001
            pass
    return {"result": res, "cls": cls, "kind": kind, "w": w, "h": h, "method": method, "model": label, "target": target, "f": f, "kept": kept}


def upscaled_line(tag, r, name, t0):
    big = r["target"][0] * r["target"][1]
    warn = f"  VERY LARGE RESULT ({big / 1e6:.0f} megapixels): big file, slow" if big > BIG_RESULT_PIXELS else ""
    if r["kept"]:
        warn += "  " + r["kept"]
    if r["f"] > 6:
        warn += "  LOW SOURCE RESOLUTION (>6x enlargement, will look soft)"
    return (f"{tag}\n    {r['w']}x{r['h']}  {r['kind']}  {r['method']}  x{round(r['f'], 2):g}  -> {name}  "
            f"({time.time() - t0:.1f}s){warn}")


def _run(o, L, should_stop, out, files, notes, stamp, work, started):
    L(f"BlurNoMore v{VERSION}   {datetime.now():%Y-%m-%d %H:%M:%S}")
    L(f"Python {platform.python_version()}  Pillow {Image.__version__}  numpy {np.__version__}  {platform.platform()}")
    L("Settings: " + ", ".join(f"{k}={v}" for k, v in asdict(o).items()))

    c = Context(o, L, work)
    for m in notes:
        L(m)
    numbered = list(enumerate(files, 1))
    try:
        LAST_RUN_FILE.write_text(json.dumps([str(Path(p).resolve()) for p in files], indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    L(f"{len(numbered)} images. Saving in {out}\n")

    if o.mode == "compare":
        return run_compare(c, files, stamp, out, should_stop, started)
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
                name = result_name(o, path.stem, ext, up_dir)
                if ext == "jpg":
                    r["result"].save(up_dir / name, quality=95, subsampling=0)
                else:
                    r["result"].save(up_dir / name)
                pdf_inputs.append(name)
                L(upscaled_line(tag, r, name, t0))
                report.append([idx, path.name, f"{r['w']}x{r['h']}", r["kind"], f"{round(r['f'], 2):g}", r["method"], name])
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


# ================================================================== compare models
COMPARE_PICTURES = 4     # Compare models uses this many of the pictures
CLOSEUP = 256            # side of the close-up crops
COMPARE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def compare_choices(o=None):
    """The models Compare models can use: every model of the Model list except Automatic."""
    return [*MODEL_LABELS, *extra_models()]


def detailed_box(im, size=CLOSEUP):
    """Box (x0, y0, x1, y1) of at most size x size pixels where the picture has the most local variance (the most detail)."""
    W, H = im.size
    bw, bh = min(size, W), min(size, H)
    f = min(1.0, 800.0 / max(W, H))
    g = im.convert("L")
    if f < 1.0:
        g = g.resize((max(1, round(W * f)), max(1, round(H * f))), Image.BILINEAR)
    a = np.asarray(g).astype(np.float64)
    sw, sh = max(1, round(bw * f)), max(1, round(bh * f))
    ii = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    i2 = np.pad(a * a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    box = lambda t: t[sh:, sw:] - t[:-sh, sw:] - t[sh:, :-sw] + t[:-sh, :-sw]
    var = box(i2) / (sw * sh) - (box(ii) / (sw * sh)) ** 2
    y, x = np.unravel_index(int(np.argmax(var)), var.shape)
    x0 = min(max(0, int(round(x / f))), W - bw)
    y0 = min(max(0, int(round(y / f))), H - bh)
    return (x0, y0, x0 + bw, y0 + bh)


def label_font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # noqa: BLE001  (an old Pillow has one fixed size)
        return ImageFont.load_default()


def closeup_sheet(original_crop, crops):
    """The close-ups side by side: the original's (enlarged, labelled Original), then each letter's, labelled with the
    letter only. crops = [(letter, image)] in letter order."""
    from PIL import ImageDraw
    pad, cap, font = 10, 36, label_font(26)
    cells = [("Original", original_crop)] + list(crops)
    cw, ch = max(i.size[0] for _, i in cells), max(i.size[1] for _, i in cells)
    cols = min(len(cells), 4)
    rows = -(-len(cells) // cols)
    sheet = Image.new("RGB", (pad + cols * (cw + pad), pad + rows * (cap + ch + pad)), (235, 235, 235))
    d = ImageDraw.Draw(sheet)
    for k, (name, im) in enumerate(cells):
        x, y = pad + (k % cols) * (cw + pad), pad + (k // cols) * (cap + ch + pad)
        d.text((x, y), name, fill=(0, 0, 0), font=font)
        sheet.paste(im.convert("RGB"), (x, y + cap))
    return sheet


def why_said(heard, default):
    """The line of what was just said that tells why a model could not be used (the first that says could not / failed), else the last."""
    lines = [x.strip() for x in heard if x.strip()]
    return next((x for x in lines if re.search(r"could not|failed|install", x, re.I)), lines[-1] if lines else default)[:200]


def key_lines(o, order, labels, times, fails, pictures):
    """The text of key.txt / key.png: which letter is which model (with its time, or why it failed), then the settings."""
    lines = ["Which letter is which model", ""]
    for letter, model in order:
        name = labels[model]
        if model in fails and len(fails[model]) == len(pictures):
            lines.append(f"{letter} = {name}: failed: {next(iter(fails[model].values()))}")
            continue
        t = times.get(model, [])
        line = f"{letter} = {name} ({sum(t) / len(t):.1f} s{' per picture' if len(pictures) > 1 else ''})"
        for pic, why in fails.get(model, {}).items():
            line += f"; failed for {pic}: {why}"
        lines.append(line)
    lines += ["", "Pictures: " + ", ".join(pictures),
              f"Enlarge: {o.size_px} px {o.size_mode}" if o.size_mode != "scale" else f"Enlarge: {o.scale}x",
              f"Restore faces: {'on, ' + o.face_model + ', strength ' + str(o.face_strength) if o.faces else 'off'}",
              f"Colour black-and-white photos: {'on' if o.colorize else 'off'}", f"File type: {o.file_type}"]
    if "diffusion" in labels:
        lines.append(f"Inventing upscaler: prompt '{o.diff_prompt}', negative '{o.diff_negative}', noise {o.diff_noise}, "
                     f"steps {o.diff_steps}, guidance {o.diff_guidance:g}, seed {o.diff_seed or 'random'}")
    return lines


def key_picture(lines):
    from PIL import ImageDraw
    font = label_font(18)
    pad, step = 16, 28
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = max(int(probe.textlength(ln, font=font)) for ln in lines) + 2 * pad
    sheet = Image.new("RGB", (width, 2 * pad + step * len(lines)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for k, ln in enumerate(lines):
        d.text((pad, pad + k * step), ln, fill=(0, 0, 0), font=font)
    return sheet


def run_compare(c, files, stamp, out, should_stop, started):
    """Compare models, blind: each of the first pictures is made with each ticked model (and plain resizing) at the
    same size and options. The results get shuffled letters; the key says which letter is which model. Returns the folder."""
    o, L = c.o, c.L
    valid = compare_choices(o)
    models = [m for m in dict.fromkeys(o.compare_models) if m in valid]
    if len(models) + 1 > len(COMPARE_LETTERS):
        L(f"Only the first {len(COMPARE_LETTERS) - 1} models are compared.")
        models = models[:len(COMPARE_LETTERS) - 1]
    if len(files) > COMPARE_PICTURES:
        L(f"Compare models uses the first {COMPARE_PICTURES} of the {len(files)} pictures.")
    pictures = files[:COMPARE_PICTURES]
    contestants = [*models, "plain"]
    shuffled = list(contestants)
    random.SystemRandom().shuffle(shuffled)      # the letters: new for every run, the same for every picture of it
    letter_of = {m: COMPARE_LETTERS[i] for i, m in enumerate(shuffled)}
    labels = {m: MODEL_LABELS.get(m, m) for m in models} | {"plain": PLAIN_LABEL}
    if any(m in ("general", "anime") or (MODELS_DIR / f"{m}.param").is_file() for m in models):
        c.start_ai()
    folder = out / f"BlurNoMore_compare_{stamp}"
    times, fails, made, used = {}, {}, 0, set()
    L(f"Comparing {len(models)} models and plain resizing on {len(pictures)} pictures. Which result is which model is not told here.")
    for n, path in enumerate(pictures, 1):
        if should_stop():
            L("\nStopped by user.")
            break
        t0 = time.time()
        L(f"[{n}/{len(pictures)}] {path.name[:50]}")
        results = {}
        try:
            src_id, im = source_id(path), load_rgb(path)
        except Exception:  # noqa: BLE001
            L("    ERROR:\n    " + traceback.format_exc().replace("\n", "\n    "))
            continue
        if target_size(o, *im.size)[0] / im.size[0] <= AI_MIN_SCALE:
            L("    This picture hardly grows, so every model gives plain resizing for it.")
        for m in contestants:                        # always in this order, so the log tells nothing about the letters
            heard, real = [], c.L

            def hear(msg="", heard=heard, real=real):
                heard.append(msg)
                real(msg)
            c.L = hear
            t1 = time.time()
            try:
                results[m] = make_upscaled(c, path, src_id, im, force=m)["result"]
                times.setdefault(m, []).append(time.time() - t1)
            except Exception as e:  # noqa: BLE001
                why = str(e) or type(e).__name__
                if "see the log" in why:       # not started: the reason it gave the first time stands, else the last thing said
                    why = next(iter(fails.get(m, {}).values()), None) or why_said(heard, why)
                fails.setdefault(m, {})[path.name] = why
                L("    a model could not make this picture (the key says which)")
            finally:
                c.L = real
        if not results:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        base = path.stem[:40]
        k = 2
        while base in used:
            base, k = f"{path.stem[:40]}_{k}", k + 1
        used.add(base)
        ext = "jpg" if o.file_type == "jpg" else "png"
        for m, res in results.items():
            if ext == "jpg":
                res.save(folder / f"{base}_{letter_of[m]}.jpg", quality=95, subsampling=0)
            else:
                res.save(folder / f"{base}_{letter_of[m]}.png")
            made += 1
        ref = results.get("plain") or next(iter(results.values()))      # the spot is chosen on the plain enlargement
        box = detailed_box(ref)
        sx, sy = im.size[0] / ref.size[0], im.size[1] / ref.size[1]
        size = (box[2] - box[0], box[3] - box[1])
        zoom = im.resize(size, Image.NEAREST, box=(box[0] * sx, box[1] * sy, max(box[2] * sx, box[0] * sx + 1), max(box[3] * sy, box[1] * sy + 1)))
        crops = [(letter_of[m], results[m].crop(box)) for m in sorted(results, key=letter_of.get)]
        closeup_sheet(zoom, crops).save(folder / f"{base}_closeups.png")
        L(f"    done ({time.time() - t0:.1f}s)")
    if not made:
        L("Nothing to write.")
        return None
    lines = key_lines(o, sorted(((letter_of[m], m) for m in contestants)), labels, times, fails, [p.name for p in pictures])
    (folder / "key.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    key_picture(lines).save(folder / "key.png")
    L("Open key.png only after you have chosen.")
    L(f"\nDONE in {time.time() - started:.0f}s: {folder}  ({made} pictures to compare)")
    return folder


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


def make_preview(o, log, n=4, should_stop=lambda: False):
    """Process up to n images (spread over the list) exactly as the run would and return what to show: a list of dicts
    with name, kind, method, original, result, detail_original, detail_result.
    Writes only into the work folder (pages and results are kept for the run that follows)."""
    files = resolve_images(o, log)
    if not files:
        log("No images to work on. Add a folder or some pictures first.")
        return []
    work = prepare(log, output_dir(o))
    if work is None:
        return []
    c = Context(o, log, work)
    numbered = list(enumerate(files, 1))
    picks = spread(numbered, n)
    log(f"Preview of {len(picks)} of {len(files)} images")
    c.start_ai()
    items = []
    for k, (idx, path) in enumerate(picks, 1):
        if should_stop():
            log("\nStopped by user.")
            break
        t0 = time.time()
        tag = f"[{k}/{len(picks)}] #{idx} {path.name[:50]}"
        try:
            src_id = source_id(path)
            im = load_rgb(path)
            if o.mode == "upscale":
                r = make_upscaled(c, path, src_id, im, store=True)
                orig, res = im, r["result"]
                box = busiest_box(res)
                fx = fy = r["f"]
                dx = dy = 0.0
                log(upscaled_line(tag, r, "(preview)", t0))
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
            items.append({"name": path.name, "kind": r["kind"] if o.mode == "upscale" else r["cls"], "method": r["method"], "original": orig,
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
    d.text((pad, 3), f"{item['name']}   {item['kind']}   {item['method']}", fill=(0, 0, 0))
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
    "images": "The pictures to work on. 'Add folder' takes every picture directly in that folder (not the folders "
              "inside it); 'Add files' lets you pick several pictures at once. You can also drag folders and "
              "pictures onto the list. Nothing in them is ever changed.",
    "last_run": "Fills the list with the pictures of your last run, so you can try other settings on the same "
                "pictures and compare. Greyed out until you have run once.",
    "kind": "What the pictures are. 'Detect for each image' looks at each picture and decides: drawing or line art, "
            "or photo. Choose Drawing or Photo when all your pictures are the same kind and the detection "
            "gets some wrong.",
    "up_model": "Automatic picks a model for each picture. Drawings: AnimeSharp V4 when your graphics card "
                "can run it, otherwise the Drawing model (Real-ESRGAN anime). Photos: the Web photo model (DAT) when "
                "your graphics card can run it, otherwise the General model. "
                "Nobody has crowned a best photo model: use Make > Compare models to see them side by side, unlabelled, "
                "and choose by eye. "
                "Drawing AnimeSharp V4 (faithful, about 31 MB on first use) follows the lines very closely: the "
                "right pick for drawings you want to trace; it is free for personal use only (licence CC BY-NC-SA 4.0): "
                "don't sell or publish results made with it commercially. "
                "Web photo DAT RealWebPhoto v4 (about 49 MB) is made for old, compressed or damaged web photos. "
                f"Web photo DAT2 BHI (newest DAT, December 2024, about {UPSCALERS['bhi']['size']}) is a newer DAT for real-world photos. "
                f"Web photo DRCT-L (about {UPSCALERS['drct']['size']}) and Web photo ATD (about {UPSCALERS['atd']['size']}) are two other "
                "web photo models of the same maker, built differently. "
                "Clean photo Real-HAT-GAN (sharp, about 160 MB) is for clean, sharp photos where you want "
                "crisp detail. "
                "Photo HEART (new, experimental, about 75 MB) is a brand-new photo model from November 2026, less "
                "proven than the others. These are only used when you pick them; Automatic never picks HAT-GAN, HEART, BHI, DRCT or ATD. "
                "The photo models are slow without a graphics card (needs onnxruntime-directml). "
                "Inventing: Stable Diffusion x4 makes up detail that is not in the picture; see the Inventing upscaler settings. "
                "Other models you put in the models folder next to the program "
                "(NAME.param and NAME.bin, or NAME.onnx) are listed at the end.",
    "animesharp_warning": "AnimeSharp V4 is free for personal use only (licence CC BY-NC-SA 4.0). Don't sell or "
                          "publish results made with it commercially. Automatic uses it for drawings when your "
                          "graphics card can run it; pick another model to avoid it.",
    "scale": "How big the pictures become. 2x, 3x, 4x: that many times bigger (a 1000 pixel wide picture becomes 2000, "
             "3000 or 4000). Width (pixels): every picture gets exactly that many pixels across. Height (pixels): "
             "that many pixels high. The other side follows, so nothing is stretched. Pictures that are already that "
             "wide or high keep their own size, unless you tick 'Also make bigger pictures smaller'. Bigger "
             "takes longer and makes bigger files; it cannot bring back detail that is not there.",
    "shrink_bigger": "Only for Width or Height in pixels. Off: a picture that is already that wide or high is left at "
                     "its own size (it is not made smaller). On: it is made smaller to exactly the size you ask for.",
    "faces": "Finds faces in the enlarged picture and restores them with the face model you pick below. Needs "
             "onnxruntime-directml and opencv-python-headless. The first time it downloads the face model (about "
             "340 MB for GFPGAN, 377 MB for CodeFormer). Off = faces are left as the enlargement makes them. Both "
             "models redraw the face, which is why this is off by default.",
    "face_model": "Only when 'Restore faces' is on. Natural (GFPGAN, about 340 MB): keeps the face looking like the "
                  "photo; a good first pick. Strongest repair (CodeFormer, about 377 MB): rebuilds badly damaged, "
                  "tiny or very blurry faces most, but can change the face more. Only the model you pick is "
                  "downloaded, and only when a face is found. Face strength below works the same for both.",
    "colorize": "Gives black-and-white photos colour with the DDColor model. It invents the colours: they are a "
                "guess, not the real ones. Only photos that are black and white are touched; colour photos and "
                "drawings are left alone. The first time it downloads the model (about 980 MB). Off by default "
                "because it redraws the picture's colours. Done before the enlargement.",
    "face_strength": "Only when 'Restore faces' is on. How much of the restored face is used: 0 = none, 50 = half "
                     "(default), 100 = fully restored. Lower looks more like the original, higher is sharper but "
                     "can look smoother or change the face.",
    "mode": "What comes out. Bigger images: each picture becomes a new, bigger picture file. Tracing PDF: crops, "
            "enlarges and fits every picture on an A4 page in one PDF, for tracing. Compare models: makes the first 4 "
            "pictures with every model you tick, plus plain resizing, as unlabelled pictures A, B, C ... in a new folder, "
            "so you can choose without bias. One of the three per run.",
    "compare_models": "Compare models. Every ticked model makes the same bigger picture (same size, faces, colour and "
                      "file type); plain resizing is always added so you see what the AI adds. The letters are shuffled "
                      "at random for every run. key.png and key.txt in the new folder say which letter is which model: "
                      "open them only after you have chosen. Uses the first 4 pictures. Models that are not downloaded "
                      "yet are downloaded first. The Inventing model is unticked: it takes minutes per picture and makes up detail.",
    "diff_prompt": "Describe the picture; it steers what gets invented.",
    "diff_negative": "What it should stay away from.",
    "diff_noise": "How much it may invent: low keeps close to your picture, high makes up more. 0 to 350.",
    "diff_steps": "More steps = slower, a bit finer. 5 to 100.",
    "diff_guidance": "How strongly it follows the prompt. 1 to 15.",
    "diff_seed": "The starting point of the invention. The same number gives the same result. 0 = a new random number "
                 "each run; the log says which one was used.",
    "diff_warning": "The inventing upscaler (Stable Diffusion x4) is very slow: minutes per picture even on a graphics "
                    "card, much slower on the processor. The first time it downloads 3.4 GB. It changes the picture: "
                    "faces and text can come out wrong. Only used when you pick it.",
    "name": "Bigger images only. How the new files are named. Automatic: photo_x2.png. Or add your own text after "
            "the name (photo_big.png) or before it (big_photo.png), so you can find the pictures again. "
            "Two files with the same name get _2.",
    "name_text": "The text added to the name. Characters Windows does not allow in file names are left out.",
    "file_type": "Bigger images only. PNG keeps every pixel exactly (big files). JPG is much smaller "
                 "(top quality, 95) and fine for photos.",
    "fit": "Whole picture: nothing is cut off, but a picture with a different shape than A4 leaves "
           "white space at the top/bottom or the sides. Fill the page: enlarges until the page is full "
           "and trims the edges that do not fit.",
    "margin": "White border around the drawing on the A4 page, in mm. Printers cannot print right to the "
              "edge; 5 to 10 is normal.",
    "dpi": "Tracing PDF only. Pixels per inch of the pages. 300 = sharp print quality. 200 = smaller and faster. "
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
    "ai": "The AI upscalers (Real-ESRGAN, and DAT for photos) on your graphics card enlarge and restore "
          "detail; the first use downloads them. Off = plain resizing: much faster but blurrier.",
    "model": "General: natural look, best for painted or shaded coloured art and scans (default). "
             "Anime: crisper flat colours, but can look smooth and plastic on shaded art. "
             "Black and white art always uses Anime.",
    "save_to": "Where the results are saved: the PDF, or the folder of bigger pictures, and the "
               "_blurnomore_work folder. 'Next to the images' = the folder of the first folder or picture in "
               "the list. A folder that does not exist yet is made. Your original pictures are never touched.",
    "redo": "Ignore pages and results made in earlier runs. Changing a setting already makes new ones on its "
            "own; use this for a clean rebuild.",
    "open": "Opens the PDF, or the folder of bigger pictures, when the run is finished.",
    "run_preview": "Works on 4 of your pictures first (spread over the list) and shows each one next to its "
                   "result, with close-ups. Then you choose 'Go on with all' or 'Change settings'. "
                   "Not for Compare models: that already is a look-first run on 4 pictures.",
    "run": "Works on all the pictures straight away.",
}
HELP_DEFAULT = "Point at a setting to see what it does."


def load_settings():
    o = Options()
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        if "inputs" not in data and data.get("folder"):
            data["inputs"] = [data["folder"]]     # settings of an older version: one folder
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


def build_window():
    """Make the window. Returns an object with the root and the functions behind the buttons (so they can be
    driven without clicking)."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter import font as tkfont
    from types import SimpleNamespace
    from PIL import ImageTk

    o = load_settings()
    root, dnd = None, False
    if HAVE_DND:
        try:
            root, dnd = TkinterDnD.Tk(), True
        except Exception:  # noqa: BLE001  (the drag and drop library did not load: the window works without it)
            root = None
    if root is None:
        root = tk.Tk()
    root.title(f"BlurNoMore {VERSION}")
    root.geometry("1280x780")
    root.minsize(1100, 640)

    base = tkfont.nametofont("TkDefaultFont")
    small = base.copy()
    small.configure(size=max(base.cget("size") - 1, 8))
    mono = tkfont.nametofont("TkFixedFont")

    # words in the window <-> values in Options
    MAKE = {"Bigger images": "upscale", "Tracing PDF": "pdf", "Compare models": "compare"}
    KIND = {"Detect for each image": "auto", "Drawing or line art": "drawing", "Photo": "photo"}
    MODEL = {"Automatic": "auto", **{label: key for key, label in MODEL_LABELS.items()}, **{n: n for n in extra_models()}}
    FACE = {"Natural (GFPGAN)": "gfpgan", "Strongest repair (CodeFormer)": "codeformer"}
    NAME = {"Automatic": "auto", "Text after the name": "suffix", "Text before the name": "prefix"}
    SAVE = {"Next to the images": False, "A folder I choose": True}
    COLOR_MODEL = {"General (recommended)": "general", "Anime": "anime"}
    FIT = {"Whole picture": "whole", "Fill the page": "fill"}

    def word(table, value, default):
        return next((k for k, v in table.items() if v == value), default)

    # ---- variables
    v_make = tk.StringVar(value=word(MAKE, o.mode, "Tracing PDF"))
    v_kind = tk.StringVar(value=word(KIND, o.kind, "Detect for each image"))
    v_model = tk.StringVar(value=word(MODEL, o.up_model, "Automatic"))
    SIZE = {"2x": ("scale", 2), "3x": ("scale", 3), "4x": ("scale", 4),
            "Width (pixels)": ("width", 0), "Height (pixels)": ("height", 0)}
    v_scale = tk.StringVar(value=word(SIZE, (o.size_mode, o.scale if o.size_mode == "scale" else 0), "2x"))
    v_px = tk.StringVar(value=str(o.size_px))
    v_shrink = tk.BooleanVar(value=o.shrink_bigger)
    v_faces = tk.BooleanVar(value=o.faces)
    v_strength = tk.StringVar(value=str(o.face_strength))
    v_face_model = tk.StringVar(value=word(FACE, o.face_model, "Natural (GFPGAN)"))
    v_colorize = tk.BooleanVar(value=o.colorize)
    v_name = tk.StringVar(value=word(NAME, o.name_mode, "Automatic"))
    v_name_text = tk.StringVar(value=o.name_text)
    v_type = tk.StringVar(value=o.file_type.upper())
    v_save = tk.StringVar(value=word(SAVE, bool(o.save_to.strip()), "Next to the images"))
    v_save_to = tk.StringVar(value=o.save_to)
    v_fit = tk.StringVar(value=word(FIT, o.fit, "Whole picture"))
    v_margin = tk.StringVar(value=str(o.margin_mm))
    v_dpi = tk.StringVar(value=str(o.dpi))
    v_crop = tk.BooleanVar(value=o.autocrop)
    v_thick = tk.StringVar(value=str(o.thicken))
    v_fade = tk.StringVar(value=str(o.fade))
    v_color = tk.BooleanVar(value=o.keep_color)
    v_boost = tk.StringVar(value=str(o.color_boost))
    v_ai = tk.BooleanVar(value=o.use_ai)
    v_color_model = tk.StringVar(value=word(COLOR_MODEL, o.color_model, "General (recommended)"))
    v_redo = tk.BooleanVar(value=o.redo)
    v_open = tk.BooleanVar(value=o.open_when_done)
    v_cmp = {key: tk.BooleanVar(value=key in o.compare_models) for key in compare_choices(o)}
    v_diff_prompt = tk.StringVar(value=o.diff_prompt)
    v_diff_negative = tk.StringVar(value=o.diff_negative)
    v_diff_noise = tk.StringVar(value=str(o.diff_noise))
    v_diff_steps = tk.StringVar(value=str(o.diff_steps))
    v_diff_guidance = tk.StringVar(value=f"{o.diff_guidance:g}")
    v_diff_seed = tk.StringVar(value=str(o.diff_seed))
    v_help = tk.StringVar(value=HELP_DEFAULT)
    v_count = tk.StringVar()

    # ---- layout: settings (left, scrolls) with the help box under it, and the log (right)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)
    left_outer = ttk.Frame(root)
    left_outer.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)
    left_outer.rowconfigure(0, weight=1)
    canvas = tk.Canvas(left_outer, width=560, highlightthickness=0)
    scroll = ttk.Scrollbar(left_outer, orient="vertical", command=canvas.yview)
    left = ttk.Frame(canvas)
    left.columnconfigure(0, weight=1)
    win_id = canvas.create_window((0, 0), window=left, anchor="nw")
    left.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win_id, width=e.width))
    canvas.configure(yscrollcommand=scroll.set)
    canvas.grid(row=0, column=0, sticky="ns")
    scroll.grid(row=0, column=1, sticky="ns")
    help_box = ttk.Label(left_outer, textvariable=v_help, font=small, wraplength=540, justify="left",
                         relief="groove", padding=6, anchor="nw")
    help_box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
    left_outer.configure(height=700)

    def over_left(w):
        while w is not None:
            if w is left_outer:
                return True
            w = w.master
        return False

    def wheel(e):
        """Scroll the settings only while the pointer is over them (and not over the list, which scrolls itself)."""
        w = root.winfo_containing(e.x_root, e.y_root)
        if w is None or not over_left(w) or isinstance(w, tk.Listbox) or canvas.yview() == (0.0, 1.0):
            return
        step = (-1 if e.delta > 0 else 1) if e.delta else (-1 if e.num == 4 else 1)
        canvas.yview_scroll(step, "units")
    for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        root.bind_all(ev, wheel)

    right = ttk.Frame(root)
    right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
    right.rowconfigure(1, weight=1)
    right.columnconfigure(0, weight=1)

    # ---- helpers to build rows
    counters, only = {}, {"upscale": [], "pdf": [], "compare": []}

    def tip(w, key):
        """Show the help text of a setting while the pointer is over it or it has the focus."""
        for x in (w, *w.winfo_children()):
            x.bind("<Enter>", lambda e: v_help.set(HELP[key]), add="+")
            x.bind("<FocusIn>", lambda e: v_help.set(HELP[key]), add="+")
            x.bind("<Leave>", lambda e: v_help.set(HELP_DEFAULT), add="+")

    def group(title, parent=None):
        g = ttk.LabelFrame(parent or left, text=title, padding=(8, 4)) if title else ttk.Frame(parent or left)
        g.grid(row=len((parent or left).grid_slaves()), column=0, sticky="ew", pady=(0, 8))
        g.columnconfigure(1, weight=1)
        return g

    def row(g, label, build, key, out=None):
        """One setting: a label and what you change it with. out = 'upscale', 'pdf' or 'compare' (several, separated by
        spaces): shown only for that output."""
        i = counters[g] = counters.get(g, -1) + 1
        g.columnconfigure(0, minsize=210)      # the settings of all groups line up
        lab = ttk.Label(g, text=label)
        lab.grid(row=i, column=0, sticky="w", padx=(0, 8), pady=2)
        w = build(g)
        w.grid(row=i, column=1, sticky="w", pady=2)
        tip(lab, key)
        tip(w, key)
        for m in (out or "").split():
            only[m] += [lab, w]
        return w

    def combo(var, values, width=30):
        return lambda p: ttk.Combobox(p, textvariable=var, state="readonly", width=width, values=list(values))

    def spin(var, lo, hi, step, width=6):
        return lambda p: ttk.Spinbox(p, from_=lo, to=hi, increment=step, textvariable=var, width=width)

    def check(var, text):
        return lambda p: ttk.Checkbutton(p, text=text, variable=var)

    def entry_row(var, browse_cmd=None):
        def build(p):
            f = ttk.Frame(p)
            e = ttk.Entry(f, textvariable=var, width=24)
            e.pack(side="left")
            if browse_cmd:
                ttk.Button(f, text="Browse...", command=browse_cmd).pack(side="left", padx=4)
            return f
        return build

    # ---- 1. Images
    inputs = list(o.inputs)
    g1 = group("1. Images")
    g1.columnconfigure(0, weight=1)
    bar = ttk.Frame(g1)
    bar.grid(row=0, column=0, sticky="w")
    list_frame = ttk.Frame(g1)
    list_frame.grid(row=2, column=0, sticky="ew", pady=(4, 0))
    list_frame.columnconfigure(0, weight=1)
    lb = tk.Listbox(list_frame, height=7, selectmode="extended", exportselection=False, font=mono)
    lb_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=lb.yview)
    lb.configure(yscrollcommand=lb_scroll.set)
    lb.grid(row=0, column=0, sticky="ew")
    lb_scroll.grid(row=0, column=1, sticky="ns")
    ttk.Label(g1, textvariable=v_count, font=small).grid(row=3, column=0, sticky="w", pady=(2, 0))
    for w in (bar, list_frame, lb):
        tip(w, "images")

    def add_paths(paths):
        """Add folders and picture files to the list (no duplicates, no other files). Returns how many were added."""
        added, skipped = 0, 0
        have = {os.path.normcase(os.path.abspath(p)) for p in inputs}
        for p in paths:
            p = str(p).strip().strip('"')
            key = os.path.normcase(os.path.abspath(p))
            if key in have:
                continue
            if os.path.isdir(p) or (os.path.isfile(p) and Path(p).suffix.lower() in EXTS):
                inputs.append(p)
                have.add(key)
                added += 1
            else:
                skipped += 1
        if skipped:
            append(f"{skipped} dropped item(s) skipped: not a folder or a picture file.")
        refresh_list()
        return added

    def shorten(p, n=44):
        return p if len(p) <= n else f"{Path(p).anchor}...{os.sep}{Path(p).name}"

    def refresh_list():
        lb.delete(0, "end")
        seen = set()                          # a picture counted once, even if its folder is in the list too
        for p in inputs:
            if os.path.isdir(p):
                try:
                    pics = [str(f.resolve()) for f in list_images(p)]
                except OSError:
                    pics = []
                seen.update(pics)
                lb.insert("end", f"{shorten(p)}  (whole folder, {len(pics)})")
            elif os.path.isfile(p):
                seen.add(str(Path(p).resolve()))
                lb.insert("end", Path(p).name)
            else:
                lb.insert("end", f"{shorten(p)}  (not found)")
        n = len(seen)
        v_count.set((f"{n} image" if n == 1 else f"{n} images") if inputs
                    else "No images yet: add a folder or some pictures.")
        b_last.configure(state="normal" if read_last_run() else "disabled")

    def add_folder():
        d = filedialog.askdirectory(initialdir=(inputs[-1] if inputs and os.path.isdir(inputs[-1]) else DEFAULT_FOLDER),
                                    title="Folder with pictures")
        if d:
            add_paths([d])

    def add_files():
        names = filedialog.askopenfilenames(title="Choose pictures (hold Ctrl or Shift to pick several)",
                                            filetypes=[("Pictures", " ".join("*" + e for e in sorted(EXTS))),
                                                       ("All files", "*.*")])
        if names:
            add_paths(root.tk.splitlist(names))

    def remove_selected():
        for i in reversed(lb.curselection()):
            del inputs[i]
        refresh_list()

    def clear_list():
        inputs.clear()
        refresh_list()

    def last_run_images():
        inputs[:] = read_last_run()
        refresh_list()

    ttk.Button(bar, text="Add folder...", command=add_folder).pack(side="left", padx=(0, 4))
    ttk.Button(bar, text="Add files...", command=add_files).pack(side="left", padx=(0, 4))
    ttk.Button(bar, text="Remove", command=remove_selected).pack(side="left", padx=(0, 4))
    ttk.Button(bar, text="Clear", command=clear_list).pack(side="left", padx=(0, 4))
    b_last = ttk.Button(bar, text="Last run's images", command=last_run_images)
    b_last.pack(side="left")
    tip(b_last, "last_run")
    if dnd:
        try:
            lb.drop_target_register(DND_FILES)
            lb.dnd_bind("<<Drop>>", lambda e: add_paths(root.tk.splitlist(e.data)))
            ttk.Label(g1, text="or drag pictures or folders onto the list", font=small,
                      foreground="#555555").grid(row=1, column=0, sticky="w", pady=(4, 0))
        except Exception:  # noqa: BLE001
            dnd = False

    # ---- 2. What to make: first, because it decides which settings below are shown
    g_make = group("2. What to make")
    row(g_make, "Make", combo(v_make, MAKE), "mode")

    def compare_boxes(p):
        f = ttk.Frame(p)
        for key in v_cmp:
            tk.Checkbutton(f, text=MODEL_LABELS.get(key, key), variable=v_cmp[key], anchor="w", justify="left",
                           wraplength=290).pack(anchor="w")
        return f
    row(g_make, "Models to compare", compare_boxes, "compare_models", "compare")

    # ---- 3. What kind of images
    g2 = group("3. What kind of images")
    row(g2, "Kind", combo(v_kind, KIND), "kind", "upscale")
    row(g2, "Model", combo(v_model, MODEL, 34), "up_model", "upscale")
    w_licence = row(g2, "", lambda p: ttk.Label(p, text="Drawings may use AnimeSharp V4: personal use only.",
                                                foreground="#c25a00", font=small), "animesharp_warning", "upscale")
    lab_licence = only["upscale"][-2]             # the row's empty label hides and shows with the line
    row(g2, "Use AI upscaler", check(v_ai, "on  (needs a GPU)"), "ai")
    row(g2, "Model for coloured art", combo(v_color_model, COLOR_MODEL), "model", "pdf")

    # ---- the inventing upscaler's settings: shown when it is the model, or is ticked for Compare models
    g_diff = group("Inventing upscaler (Stable Diffusion x4)")
    row(g_diff, "", lambda p: ttk.Label(p, text="Very slow, 3.4 GB on first use, and it makes things up.",
                                        foreground="#c25a00", font=small, wraplength=300, justify="left"), "diff_warning")
    row(g_diff, "Describe the picture", entry_row(v_diff_prompt), "diff_prompt")
    row(g_diff, "Stay away from", entry_row(v_diff_negative), "diff_negative")
    row(g_diff, "How much it may invent (0-350)", spin(v_diff_noise, 0, 350, 5), "diff_noise")
    row(g_diff, "Steps (5-100)", spin(v_diff_steps, 5, 100, 5), "diff_steps")
    row(g_diff, "Follow the description (1-15)", spin(v_diff_guidance, 1, 15, 0.5), "diff_guidance")
    row(g_diff, "Seed (0 = random each run)", spin(v_diff_seed, 0, 2147483647, 1, 10), "diff_seed")

    # ---- 4. Options
    g3 = group("4. Options")
    def size_row(p):
        f = ttk.Frame(p)
        ttk.Combobox(f, textvariable=v_scale, state="readonly", width=11, values=list(SIZE)).pack(side="left")
        ttk.Spinbox(f, from_=100, to=20000, increment=10, textvariable=v_px, width=5).pack(side="left", padx=(4, 0))
        ttk.Label(f, textvariable=v_size_note).pack(side="left", padx=(4, 0))
        return f
    v_size_note = tk.StringVar()
    w_size = row(g3, "Enlarge", size_row, "scale", "upscale compare")
    w_shrink = row(g3, "Also make bigger pictures smaller", check(v_shrink, "on"), "shrink_bigger", "upscale compare")
    row(g3, "Restore faces", check(v_faces, "on"), "faces", "upscale compare")
    w_face_model = row(g3, "Face model", combo(v_face_model, FACE), "face_model", "upscale compare")
    w_strength = row(g3, "Face strength (0-100)", spin(v_strength, 0, 100, 10), "face_strength", "upscale compare")
    row(g3, "Colour black-and-white photos", check(v_colorize, "on"), "colorize", "upscale compare")
    row(g3, "Crop empty paper", check(v_crop, "on"), "crop", "pdf")
    row(g3, "Strengthen thin lines (0-100)", spin(v_thick, 0, 100, 10), "thicken", "pdf")
    row(g3, "Lighten print (0-100)", spin(v_fade, 0, 90, 10), "fade", "pdf")
    row(g3, "Keep colours", check(v_color, "on  (off = all grey)"), "color", "pdf")
    row(g3, "Colour boost (0-100)", spin(v_boost, 0, 100, 10), "boost", "pdf")

    # ---- 5. What comes out
    g4 = group("5. What comes out")
    row(g4, "Name", combo(v_name, NAME), "name", "upscale")
    w_name_text = row(g4, "Text for the name", entry_row(v_name_text), "name_text", "upscale")
    row(g4, "File type", combo(v_type, ["PNG", "JPG"], 8), "file_type", "upscale compare")
    row(g4, "Fit on the page", combo(v_fit, FIT), "fit", "pdf")
    row(g4, "Page margin (mm)", spin(v_margin, 0, 40, 1), "margin", "pdf")

    def browse_save():
        d = filedialog.askdirectory(initialdir=v_save_to.get() or None, title="Folder to save in")
        if d:
            v_save_to.set(d)
            v_save.set("A folder I choose")
    row(g4, "Save to", combo(v_save, SAVE, 22), "save_to")
    w_save_to = row(g4, "", entry_row(v_save_to, browse_save), "save_to")

    # ---- More settings (folds open)
    more_open = [False]
    fold = ttk.Label(left, text="> More settings", foreground="#2f6fb3", cursor="hand2")
    fold.grid(row=len(left.grid_slaves()), column=0, sticky="w", pady=(0, 4))
    g5 = group(None)
    row(g5, "Print resolution (dpi)", spin(v_dpi, 150, 600, 50), "dpi", "pdf")
    row(g5, "Redo all pages", check(v_redo, "on"), "redo")
    row(g5, "Open the result when finished", check(v_open, "on"), "open")

    def toggle_more(*_):
        more_open[0] = not more_open[0]
        fold.configure(text=("v More settings" if more_open[0] else "> More settings"))
        if more_open[0]:
            g5.grid()
        else:
            g5.grid_remove()
    fold.bind("<Button-1>", toggle_more)
    g5.grid_remove()

    # ---- what shows and what is greyed
    def sync(*_):
        mode = MAKE.get(v_make.get(), "pdf")
        shown = set(only[mode])
        for w in {w for widgets in only.values() for w in widgets}:    # a setting can belong to more than one output
            (w.grid if w in shown else w.grid_remove)()
        picked = MODEL.get(v_model.get(), "auto")
        invents = (mode == "upscale" and picked == "diffusion") or (mode == "compare" and v_cmp["diffusion"].get())
        (g_diff.grid if invents else g_diff.grid_remove)()
        if MAKE.get(v_make.get()) == "upscale" and (picked == "animesharp" or (picked == "auto" and KIND.get(v_kind.get()) != "photo")):
            w_licence.grid()
            lab_licence.grid()
        else:
            w_licence.grid_remove()
            lab_licence.grid_remove()
        size_mode = SIZE.get(v_scale.get(), ("scale",))[0]
        w_size.winfo_children()[1].configure(state="disabled" if size_mode == "scale" else "normal")
        w_shrink.configure(state="disabled" if size_mode == "scale" else "normal")
        px = v_px.get().strip()
        v_size_note.set(f"\u2192 {px} px {'wide' if size_mode == 'width' else 'high'}" if size_mode != "scale" and px else "")
        w_strength.configure(state="normal" if v_faces.get() else "disabled")
        w_face_model.configure(state="readonly" if v_faces.get() else "disabled")
        auto_name = NAME.get(v_name.get()) == "auto"
        for e in w_name_text.winfo_children():
            e.configure(state="disabled" if auto_name else "normal")
        for e in w_save_to.winfo_children():
            e.configure(state="normal" if SAVE.get(v_save.get()) else "disabled")
    for var in (v_make, v_faces, v_name, v_save, v_model, v_kind, v_scale, v_px, v_cmp["diffusion"]):
        var.trace_add("write", sync)
    sync()

    # ---- buttons + log
    btns = ttk.Frame(right)
    btns.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    b_prev = ttk.Button(btns, text="Run with preview")
    b_run = ttk.Button(btns, text="Run")
    b_stop = ttk.Button(btns, text="Stop", state="disabled")
    for b, k in ((b_prev, "run_preview"), (b_run, "run")):
        b.pack(side="left", padx=(0, 6))
        tip(b, k)
    b_stop.pack(side="left", padx=(0, 6))

    def sync_buttons(*_):
        """Run with preview is not offered for Compare models: that already is a look-first run."""
        b_prev.configure(state="disabled" if MAKE.get(v_make.get()) == "compare" else "normal")
    v_make.trace_add("write", sync_buttons)
    sync_buttons()
    log_text = tk.Text(right, wrap="none", font=mono, height=20, undo=False)
    ysb = ttk.Scrollbar(right, orient="vertical", command=log_text.yview)
    xsb = ttk.Scrollbar(right, orient="horizontal", command=log_text.xview)
    log_text.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
    log_text.grid(row=1, column=0, sticky="nsew")
    ysb.grid(row=1, column=1, sticky="ns")
    xsb.grid(row=2, column=0, sticky="ew")
    foot = ttk.Frame(right)
    foot.grid(row=3, column=0, sticky="w", pady=(6, 0))
    b_copy = ttk.Button(foot, text="Copy log")
    b_open = ttk.Button(foot, text="Open output folder")
    for b in (b_copy, b_open):
        b.pack(side="left", padx=(0, 6))

    def block_typing(e):
        if (e.state & 0x4) and e.keysym.lower() in ("c", "a"):
            return None
        if e.keysym in ("Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next"):
            return None
        return "break"
    log_text.bind("<Key>", block_typing)
    log_text.bind("<Control-a>", lambda e: (log_text.tag_add("sel", "1.0", "end"), "break")[1])

    q = queue.Queue()
    state = {"stop": False, "last_out": None}

    def append(msg):
        log_text.insert("end", msg + "\n")
        log_text.see("end")

    append("Add folders or pictures on the left, choose what to do with them, then press Run.\n"
           "The log here can be selected and copied (Ctrl+C), or press 'Copy log'.")

    def gather():
        def num(var, name, cast=int):
            try:
                return cast(float(var.get()))
            except ValueError:
                raise ValueError(f"'{name}' must be a number") from None
        ob = Options()
        ob.inputs = list(inputs)
        ob.mode = MAKE[v_make.get()]
        ob.kind = KIND[v_kind.get()]
        ob.up_model = MODEL.get(v_model.get(), "auto")
        ob.size_mode, scale = SIZE[v_scale.get()]
        if ob.size_mode == "scale":
            ob.scale = scale
        else:
            ob.size_px = min(20000, max(100, num(v_px, "Pixels")))
            ob.shrink_bigger = bool(v_shrink.get())
        ob.faces = bool(v_faces.get())
        ob.face_strength = min(100, max(0, num(v_strength, "Face strength")))
        ob.face_model = FACE[v_face_model.get()]
        ob.colorize = bool(v_colorize.get())
        ob.compare_models = [k for k, v in v_cmp.items() if v.get()]
        ob.diff_prompt = v_diff_prompt.get().strip()
        ob.diff_negative = v_diff_negative.get().strip()
        ob.diff_noise = min(350, max(0, num(v_diff_noise, "How much it may invent")))
        ob.diff_steps = min(100, max(5, num(v_diff_steps, "Steps")))
        ob.diff_guidance = min(15.0, max(1.0, num(v_diff_guidance, "Follow the description", float)))
        ob.diff_seed = min(2147483647, max(0, num(v_diff_seed, "Seed")))
        ob.name_mode = NAME[v_name.get()]
        ob.name_text = v_name_text.get().strip()
        ob.file_type = v_type.get().lower()
        ob.save_to = v_save_to.get().strip().strip('"') if SAVE[v_save.get()] else ""
        ob.fit = FIT[v_fit.get()]
        ob.margin_mm = max(0.0, num(v_margin, "Page margin", float))
        ob.dpi = max(72, num(v_dpi, "Print resolution"))
        ob.autocrop = bool(v_crop.get())
        ob.thicken = min(100, max(0, num(v_thick, "Strengthen thin lines")))
        ob.fade = min(90, max(0, num(v_fade, "Lighten print")))
        ob.keep_color = bool(v_color.get())
        ob.color_boost = min(100, max(0, num(v_boost, "Colour boost")))
        ob.use_ai = bool(v_ai.get())
        ob.color_model = COLOR_MODEL[v_color_model.get()]
        ob.redo = bool(v_redo.get())
        ob.open_when_done = bool(v_open.get())
        return ob

    def worker(opts, preview):
        out = None
        try:
            if preview:
                items = make_preview(opts, lambda m: q.put(("log", m)), 4, lambda: state["stop"])
                q.put(("preview", (items, opts)))
            else:
                out = run(opts, lambda m: q.put(("log", m)), lambda: state["stop"])
        except Exception:  # noqa: BLE001
            q.put(("log", "UNEXPECTED ERROR:\n" + traceback.format_exc()))
        q.put(("done", str(out) if out else None))

    def start_run(opts, preview=False):
        log_text.delete("1.0", "end")
        state["stop"] = False
        for b in (b_prev, b_run):
            b.configure(state="disabled")
        b_stop.configure(state="normal")
        threading.Thread(target=worker, args=(opts, preview), daemon=True).start()

    def begin(preview):
        if preview and MAKE.get(v_make.get()) == "compare":
            return
        try:
            opts = gather()
        except ValueError as e:
            messagebox.showerror("Check the settings", str(e))
            return
        if not opts.inputs:
            messagebox.showerror("No images", "Add a folder or some pictures first.")
            return
        save_settings(opts)
        start_run(opts, preview)

    def stop():
        state["stop"] = True
        b_stop.configure(state="disabled")
        append("Stopping after the current image ...")

    def copy_log():
        root.clipboard_clear()
        root.clipboard_append(log_text.get("1.0", "end-1c"))
        append("(log copied to clipboard)")

    def open_path(p):
        if hasattr(os, "startfile"):
            try:
                os.startfile(str(p))  # noqa: S606
            except Exception as e:  # noqa: BLE001
                append(f"Could not open {p}: {e}")
        else:
            append(f"Output: {p}")

    def open_output():
        if state["last_out"]:
            p = Path(state["last_out"])
            open_path(p if p.is_dir() else p.parent)
        else:
            try:
                open_path(output_dir(gather()))
            except ValueError as e:
                append(str(e))

    # ---- the preview window
    def show_preview(items, opts):
        if not items:
            append("Nothing to show in the preview.")
            return
        win = tk.Toplevel(root)
        win.title("Preview")
        win.geometry("1120x780")
        win.transient(root)
        holder = ttk.Frame(win)
        holder.pack(fill="both", expand=True, padx=10, pady=10)
        keep = []     # the pictures shown must stay referenced or they vanish

        def pic(parent, im, box, text):
            t = im.copy().convert("RGB")
            t.thumbnail(box)
            p = ImageTk.PhotoImage(t)
            keep.append(p)
            f = ttk.Frame(parent)
            lab = ttk.Label(f, image=p)
            lab.pack()
            ttk.Label(f, text=text, font=small).pack()
            return f

        def clear():
            for w in holder.winfo_children():
                w.destroy()
            keep.clear()

        def show_grid():
            clear()
            holder.columnconfigure((0, 1), weight=1, uniform="c")
            for i, it in enumerate(items):
                cell = ttk.Frame(holder, relief="groove", padding=6, cursor="hand2")
                cell.grid(row=i // 2, column=i % 2, sticky="nsew", padx=4, pady=4)
                pics = ttk.Frame(cell)
                pics.pack()
                pic(pics, it["original"], (240, 270), "Original").pack(side="left", padx=4)
                pic(pics, it["result"], (240, 270), "Result").pack(side="left", padx=4)
                ttk.Label(cell, text=f"{it['name']}   {it['kind']}   {it['method']}", font=small,
                          wraplength=500).pack()
                todo = [cell]
                while todo:      # a click anywhere in the pair opens it
                    w = todo.pop()
                    w.bind("<Button-1>", lambda e, it=it: show_big(it))
                    todo += w.winfo_children()
            win_hint.configure(text="Click a pair to see it big.")

        def show_big(it):
            clear()
            holder.columnconfigure((0, 1), weight=1, uniform="c")
            ttk.Label(holder, text=f"{it['name']}   {it['kind']}   {it['method']}", font=small).grid(
                row=0, column=0, columnspan=2, sticky="w")
            ow, oh = it["original"].size
            rw, rh = it["result"].size
            pic(holder, it["original"], (530, 330), f"Original ({ow}x{oh})").grid(row=1, column=0, pady=4)
            pic(holder, it["result"], (530, 330), f"Result ({rw}x{rh})").grid(row=1, column=1, pady=4)
            pic(holder, it["detail_original"], (530, 280), "Close-up: original, plain enlargement").grid(
                row=2, column=0, pady=4)
            pic(holder, it["detail_result"], (530, 280), "Close-up: result, same spot").grid(row=2, column=1, pady=4)
            ttk.Button(holder, text="Back to all", command=show_grid).grid(row=3, column=0, sticky="w", pady=4)
            win_hint.configure(text="")

        def go_on():
            win.destroy()
            start_run(opts)
        bar2 = ttk.Frame(win)
        bar2.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(bar2, text="Go on with all", command=go_on).pack(side="left", padx=(0, 6))
        ttk.Button(bar2, text="Change settings", command=win.destroy).pack(side="left")
        win_hint = ttk.Label(bar2, text="", font=small)
        win_hint.pack(side="right")
        show_grid()
        return SimpleNamespace(win=win, items=items, show_big=show_big, show_grid=show_grid, go_on=go_on)

    def poll():
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "log":
                    append(payload)
                elif kind == "preview":
                    state["preview"] = show_preview(*payload)
                else:
                    b_run.configure(state="normal")
                    sync_buttons()
                    b_stop.configure(state="disabled")
                    refresh_list()
                    if payload:
                        state["last_out"] = payload
                        if v_open.get():
                            open_path(payload)
        except queue.Empty:
            pass
        root.after(120, poll)

    b_prev.configure(command=lambda: begin(True))
    b_run.configure(command=lambda: begin(False))
    b_stop.configure(command=stop)
    b_copy.configure(command=copy_log)
    b_open.configure(command=open_output)
    refresh_list()
    root.after(120, poll)
    return SimpleNamespace(root=root, gather=gather, add_paths=add_paths, remove_selected=remove_selected,
                           clear_list=clear_list, last_run_images=last_run_images, inputs=inputs, listbox=lb,
                           count_text=v_count, vars=SimpleNamespace(make=v_make, save=v_save, save_to=v_save_to,
                                                                    name=v_name, faces=v_faces, ai=v_ai, model=v_model,
                                                                    compare=v_cmp),
                           group_diff=g_diff, b_prev=b_prev,
                           begin=begin, state=state, log_text=log_text, help_text=v_help, sync=sync,
                           toggle_more=toggle_more, dnd=dnd, left=left)


def launch_gui():
    build_window().root.mainloop()


# ====================================================================== CLI
def main():
    if len(sys.argv) == 1:
        launch_gui()
        return
    ap = argparse.ArgumentParser(description="BlurNoMore: bigger AI-restored copies of pictures, or a sharp A4 tracing PDF "
                                             "(no arguments = open the window)")
    ap.add_argument("inputs", nargs="*", help="folders (all pictures directly in them) and/or picture files (never changed)")
    ap.add_argument("--mode", choices=["pdf", "upscale", "compare"], default="pdf",
                    help="pdf = Tracing PDF (default), upscale = Bigger images, compare = Compare models (see --compare)")
    ap.add_argument("--save-to", default="", metavar="FOLDER",
                    help="where the PDF / upscaled folder and _blurnomore_work go (default: the folder of the first input)")
    ap.add_argument("--scale", type=int, choices=[2, 3, 4], default=2, help="upscale mode: times bigger")
    size = ap.add_mutually_exclusive_group()
    size.add_argument("--width", type=int, metavar="N", help="upscale mode: make every picture N pixels wide (instead of --scale)")
    size.add_argument("--height", type=int, metavar="N", help="upscale mode: make every picture N pixels high (instead of --scale)")
    ap.add_argument("--shrink-bigger", action="store_true",
                    help="upscale mode, with --width/--height: also make pictures that are already bigger smaller (default: they keep their size)")
    ap.add_argument("--kind", choices=["auto", "drawing", "photo"], default="auto",
                    help="upscale mode: auto = detect for each image (default), or say they are all drawings or photos")
    ap.add_argument("--name-suffix", default="", metavar="TEXT",
                    help="upscale mode: name the files <name>TEXT instead of <name>_x2")
    ap.add_argument("--name-prefix", default="", metavar="TEXT",
                    help="upscale mode: name the files TEXT<name> instead of <name>_x2")
    ap.add_argument("--upscale-model", default="auto", metavar="NAME",
                    help="upscale mode: auto (default), general, anime, dat (Web photo model), animesharp (drawings), hatgan "
                         "(clean photos), heart (new, experimental photo model), bhi, drct, atd (more web photo models), "
                         "diffusion (inventing: makes up detail, very slow), or the name of an extra model "
                         "in the models folder: " + (", ".join(extra_models()) or "none found"))
    ap.add_argument("--compare", default=None, metavar="MODELS",
                    help="Compare models (blind): a comma separated list of upscale models, or all (every model, "
                         "also the inventing one). Makes the first 4 pictures with each, plus plain resizing, as pictures "
                         "A, B, C ... with a key.png saying which is which. Same as --mode compare with a list; "
                         "without a list --mode compare uses " + ",".join(Options().compare_models))
    ap.add_argument("--diff-prompt", default=Options.diff_prompt, help="inventing upscaler: describe the picture (default: %(default)s)")
    ap.add_argument("--diff-negative", default=Options.diff_negative, help="inventing upscaler: what to stay away from (default: %(default)s)")
    ap.add_argument("--diff-noise", type=int, default=Options.diff_noise, metavar="N",
                    help="inventing upscaler: 0-350, how much it may invent (default %(default)s)")
    ap.add_argument("--diff-steps", type=int, default=Options.diff_steps, metavar="N",
                    help="inventing upscaler: 5-100, more is slower and a bit finer (default %(default)s)")
    ap.add_argument("--diff-guidance", type=float, default=Options.diff_guidance, metavar="N",
                    help="inventing upscaler: 1-15, how strongly it follows the prompt (default %(default)s)")
    ap.add_argument("--diff-seed", type=int, default=Options.diff_seed, metavar="N",
                    help="inventing upscaler: the same seed gives the same result; 0 = a new random seed each run (default %(default)s)")
    ap.add_argument("--format", choices=["png", "jpg"], default="png", help="upscale mode: picture file type")
    ap.add_argument("--faces", action="store_true", help="upscale mode: restore faces")
    ap.add_argument("--face-model", choices=list(FACE_MODELS), default="gfpgan",
                    help="upscale mode: gfpgan (natural, default) or codeformer (strongest repair)")
    ap.add_argument("--colorize", action="store_true",
                    help="upscale mode: colour black-and-white photos (the colours are invented; downloads about 980 MB)")
    ap.add_argument("--face-strength", type=int, default=50, metavar="N",
                    help="upscale mode: 0-100, how much of the restored face is used (default 50)")
    ap.add_argument("--preview", action="store_true",
                    help="make picture sheets for up to 4 images (original, result, close-ups) and stop")
    ap.add_argument("--same", action="store_true", help="use the images of the last run (blurnomore_last_run.json) instead of the inputs")
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
    if not a.inputs and not a.same:
        ap.error("give at least one folder or picture (or --same)")
    size_mode, size_px = next(((m, v) for m, v in (("width", a.width), ("height", a.height)) if v), ("scale", 1920))
    if not 1 <= size_px <= 20000:
        ap.error("the size in pixels must be between 1 and 20000")
    if a.name_suffix and a.name_prefix:
        ap.error("use --name-suffix or --name-prefix, not both")
    for name, value, lo, hi in (("--diff-noise", a.diff_noise, 0, 350), ("--diff-steps", a.diff_steps, 5, 100),
                                ("--diff-guidance", a.diff_guidance, 1, 15), ("--diff-seed", a.diff_seed, 0, 2 ** 31 - 1)):
        if not lo <= value <= hi:
            ap.error(f"{name} must be between {lo} and {hi}")
    compare = Options().compare_models
    if a.compare is not None:
        every = compare_choices(None)
        compare = every if a.compare.strip().lower() == "all" else [m.strip() for m in a.compare.split(",") if m.strip()]
        unknown = [m for m in compare if m not in every]
        if unknown or not compare:
            ap.error(f"--compare: unknown model {', '.join(unknown) or '(none given)'}; choose from all, {', '.join(every)}")
        a.mode = "compare"
    if a.mode == "compare" and a.preview:
        ap.error("--preview is not offered for compare: it already is a look-first run")
    o = Options(inputs=a.inputs, repeat_last=a.same, fit="fill" if a.fill else "whole", margin_mm=a.margin, dpi=a.dpi,
                autocrop=not a.no_crop, thicken=a.thicken, fade=a.fade, keep_color=not a.grey,
                color_boost=a.boost, use_ai=not a.no_ai, color_model=a.model, redo=a.redo,
                mode=a.mode, save_to=a.save_to, scale=a.scale, size_mode=size_mode, size_px=size_px, shrink_bigger=a.shrink_bigger, up_model=a.upscale_model, file_type=a.format,
                kind=a.kind, name_mode="prefix" if a.name_prefix else "suffix" if a.name_suffix else "auto",
                name_text=a.name_prefix or a.name_suffix, faces=a.faces, face_strength=min(100, max(0, a.face_strength)),
                face_model=a.face_model, colorize=a.colorize, compare_models=compare, diff_prompt=a.diff_prompt,
                diff_negative=a.diff_negative, diff_noise=a.diff_noise, diff_steps=a.diff_steps,
                diff_guidance=a.diff_guidance, diff_seed=a.diff_seed)
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
