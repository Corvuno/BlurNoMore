"""Fake-model check: runs the new photo models, the inventing upscaler (SeedVR2) and Compare models end to
end with small made-up ONNX models (they do almost nothing, but have the real file names and input names) and a fake
SeedVR2 tool (a script that takes the same options and only resizes), so no real model has to be downloaded. The real
photo models run in the "Models" workflow; the real SeedVR2 (PyTorch and 3.9 GB of model files) is never run here.
Run:  python tests/fake_models_check.py   (needs pillow, numpy, img2pdf, onnxruntime and onnx). Exits non-zero on failure.
Everything happens in temporary folders; the models folder next to the program is not touched."""
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import onnx
    from onnx import TensorProto as T, helper as h, numpy_helper as nh
except ImportError:
    sys.exit("This check needs the onnx package:  pip install onnx")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import blurnomore as bn  # noqa: E402


# ---------------------------------------------------------------- the fake models
def save(nodes, inputs, outputs, inits, path):
    g = h.make_graph(nodes, "fake", inputs, outputs, inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(path))


def const(name, arr):
    return nh.from_array(np.asarray(arr), name)


def fake_upscaler(path, mode, tile=256):
    """Input 'input' 1x3xTILExTILE, output 4 times that: the picture enlarged 4x with the given resize mode."""
    save([h.make_node("Resize", ["input", "", "scales"], ["output"], mode=mode)],
         [h.make_tensor_value_info("input", T.FLOAT, [1, 3, tile, tile])],
         [h.make_tensor_value_info("output", T.FLOAT, [1, 3, tile * 4, tile * 4])],
         [const("scales", np.array([1, 1, 4, 4], np.float32))], path)


def install_fakes(models, skip=()):
    """Fake models under their real file names in the models folder `models`."""
    for key, mode in (("dat", "nearest"), ("bhi", "linear"), ("drct", "cubic"), ("hatgan", "linear"), ("heart", "cubic")):
        if key not in skip:
            fake_upscaler(models / bn.UPSCALERS[key]["file"], mode, 128 if key == "drct" else 256)


FAKE_CLI = r'''"""A fake SeedVR2 inference_cli.py: the same options, then only a Lanczos resize to --resolution (short side)."""
import argparse, json, os, sys, time
from pathlib import Path
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("input"); ap.add_argument("--output"); ap.add_argument("--dit_model"); ap.add_argument("--model_dir")
ap.add_argument("--resolution", type=int); ap.add_argument("--batch_size", type=int); ap.add_argument("--seed", type=int)
ap.add_argument("--color_correction"); ap.add_argument("--input_noise_scale", type=float); ap.add_argument("--latent_noise_scale", type=float)
ap.add_argument("--vae_encode_tiled", action="store_true"); ap.add_argument("--vae_decode_tiled", action="store_true")
ap.add_argument("--blocks_to_swap", type=int, default=0); ap.add_argument("--swap_io_components", action="store_true")
ap.add_argument("--dit_offload_device", default="none"); ap.add_argument("--vae_offload_device", default="none")
a = ap.parse_args()
root = Path(__file__).resolve().parent.parent
with open(root / "calls.jsonl", "a") as f:
    f.write(json.dumps(vars(a)) + "\n")
(root / "pid.txt").write_text(str(os.getpid()))
if os.environ.get("FAKE_SV_FAIL"):
    print("RuntimeError: out of graphics memory (fake)"); sys.exit(1)
if os.environ.get("FAKE_SV_HANG"):
    time.sleep(60)
if os.environ.get("FAKE_SV_SLOW"):
    for i in range(300):
        print(f"progress step {i}", flush=True); time.sleep(0.2)
im = Image.open(a.input).convert("RGB")
w, h = im.size
f = a.resolution / min(w, h)
im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS).save(a.output)
print("Output saved to:", a.output)
'''


def install_fake_seedvr2(root, ready=True):
    """A fake seedvr2/ folder: ready.json, a "venv" whose python is this python, and the fake tool."""
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "inference_cli.py").write_text(FAKE_CLI, encoding="utf-8")
    py = root / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    (root / "calls.jsonl").unlink(missing_ok=True)
    if ready:
        (root / "ready.json").write_text(json.dumps({"commit": bn.SEEDVR2_COMMIT, "torch": "cpu"}))


def last_call(root):
    return json.loads((root / "calls.jsonl").read_text().splitlines()[-1])


# ---------------------------------------------------------------- helpers
def photo(w_, h_, seed=1):
    """A made-up picture with smooth shapes (so a result can be compared with plain resizing) and a little texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h_, 0:w_]
    base = np.stack([60 + 120 * xx / w_, 40 + 150 * yy / h_, 200 - 100 * xx / w_], axis=-1)
    im = Image.fromarray(np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.ellipse((w_ // 5, h_ // 5, w_ // 2, h_ // 2), fill=(200, 60, 50))
    d.rectangle((w_ // 2, h_ // 2, w_ * 4 // 5, h_ * 4 // 5), fill=(40, 90, 180))
    return im


def snapshot(d):
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in d.iterdir() if p.is_file()}


def make_run(models_dir, work, **kw):
    """Run the program on the folder in kw['inputs'] with the models folder replaced; returns (result, log lines)."""
    lines = []
    o = bn.Options(save_to=str(work), ai_dir=str(models_dir.parent), **kw)
    return bn.run(o, lines.append), lines


def main():
    saved = bn.APP_DIR, bn.SETTINGS_FILE, bn.ESRGAN_DIR, bn.MODELS_DIR, bn.SEEDVR2_DIR
    try:
        check()
    finally:
        bn.APP_DIR, bn.SETTINGS_FILE, bn.ESRGAN_DIR, bn.MODELS_DIR, bn.SEEDVR2_DIR = saved
        (ROOT / "settings" / "blurnomore_last_run.json").unlink(missing_ok=True)
        for d in (ROOT / "settings" / "running", ROOT / "settings"):
            try:
                d.rmdir()                    # only when empty
            except OSError:
                pass


def ai_run(base, ai_dir, out):
    """A run without AI on a tiny picture, with the AI files folder `ai_dir`; returns the log text."""
    pic = out / "ai_pic.png"
    photo(40, 30).save(pic)
    lines = []
    bn.run(bn.Options(inputs=[str(pic)], mode="upscale", use_ai=False, save_to=str(out / "ai_out"), ai_dir=str(ai_dir)), lines.append)
    return "\n".join(lines)


def fake_ai_files(base):
    """Fake realesrgan/, models/ (with an extra model) and a ready seedvr2/ (venv, ready.json, tool, uv cache) in `base`."""
    (base / "realesrgan").mkdir(parents=True)
    (base / "realesrgan" / "realesrgan-ncnn-vulkan.exe").write_bytes(b"x" * 1000)
    fake_upscaler(base / "models" / "mymodel_x4.onnx", "nearest")
    install_fake_seedvr2(base / "seedvr2")
    (base / "seedvr2" / "venv" / "marker").write_text("v")
    (base / "seedvr2" / "uv" / "cache").mkdir(parents=True)
    (base / "seedvr2" / "uv" / "cache" / "wheel").write_bytes(b"w" * 500)


def check_ai_dir(out):
    """The AI files folder: files move at the next run, the venv is dropped, a second run moves nothing, a folder that
    exists in the target is left alone, a target that cannot be written falls back, uv gets its folders."""
    with tempfile.TemporaryDirectory() as t_old, tempfile.TemporaryDirectory() as t_new, tempfile.TemporaryDirectory() as t_third:
        old, new, third = Path(t_old), Path(t_new), Path(t_third)
        bn.APP_DIR, bn.SETTINGS_FILE = old, old / "settings.json"
        fake_ai_files(old)
        # moving to another folder
        text = ai_run(old, new, out)
        assert "Moving the AI files (realesrgan)" in text and "Moving the AI files (models)" in text and "Moving the AI files (seedvr2)" in text, text
        assert text.count("  done") == 3, text
        assert not (old / "models").exists() and not (old / "realesrgan").exists() and not (old / "seedvr2").exists(), "old place not emptied"
        assert (new / "realesrgan" / "realesrgan-ncnn-vulkan.exe").is_file() and (new / "seedvr2" / "src" / "inference_cli.py").is_file()
        assert (new / "seedvr2" / "uv" / "cache" / "wheel").is_file(), "the uv cache did not move"
        assert not (new / "seedvr2" / "venv").exists() and not (new / "seedvr2" / "ready.json").exists(), "venv / ready.json kept"
        assert bn.MODELS_DIR == new / "models" and "mymodel_x4" in bn.extra_models(), "extra model not listed from the new place"
        assert bn.read_settings_file()["ai_dir_used"] == str(new)
        # a second run moves nothing
        text = ai_run(old, new, out)
        assert "Moving" not in text and "already has" not in text, text
        # the target already has a models folder: both are left, the rest moves
        (third / "models").mkdir()
        (third / "models" / "other.txt").write_text("o")
        text = ai_run(new, third, out)
        assert "already has a folder models" in text and "Moving the AI files (realesrgan)" in text, text
        assert (new / "models" / "mymodel_x4.onnx").is_file() and (third / "models" / "other.txt").is_file()
        assert not (third / "models" / "mymodel_x4.onnx").exists() and (third / "realesrgan").is_dir()
        # a target that cannot be written to (a folder inside a file): nothing moves, the old place is used
        blocker = old / "afile"
        blocker.write_text("x")
        text = ai_run(third, blocker / "sub", out)
        assert "cannot be written to" in text, text
        assert (third / "realesrgan").is_dir() and bn.ESRGAN_DIR == third / "realesrgan", "files lost or not used"
        # refused: inside a folder that is moved; among the pictures; the old place keeps being used
        text = ai_run(third, third / "models" / "sub", out)
        assert "inside one of the AI folders" in text and not (third / "models" / "sub").exists(), text
        assert bn.MODELS_DIR == third / "models"
        pic_dir = out / "picfolder"
        pic_dir.mkdir()
        photo(40, 30).save(pic_dir / "p.png")
        lines = []
        bn.run(bn.Options(inputs=[str(pic_dir)], mode="upscale", use_ai=False, save_to=str(out / "ai_out2"), ai_dir=str(pic_dir / "ai")), lines.append)
        assert "among your pictures" in "\n".join(lines) and not (pic_dir / "ai").exists(), lines
        try:
            bn.move_folder(third / "models", third / "models" / "x")
            raise AssertionError("moved a folder into itself")
        except RuntimeError:
            pass
        # a partly done move: each folder is used where it exists
        bn.point_where_found(new, third)
        assert bn.MODELS_DIR == new / "models" and bn.ESRGAN_DIR == third / "realesrgan", (bn.MODELS_DIR, bn.ESRGAN_DIR)
        # a seedvr2 already in the target is not touched (its venv stays), the one left behind keeps its own
        with tempfile.TemporaryDirectory() as t_a, tempfile.TemporaryDirectory() as t_b:
            a_dir, b_dir = Path(t_a), Path(t_b)
            bn.APP_DIR, bn.SETTINGS_FILE = a_dir, a_dir / "settings.json"
            for d in (a_dir, b_dir):
                (d / "seedvr2" / "venv").mkdir(parents=True)
                (d / "seedvr2" / "ready.json").write_text("{}")
            text = ai_run(a_dir, b_dir, out)
            assert "already has a folder seedvr2" in text and (b_dir / "seedvr2" / "venv").is_dir(), text
            assert "stays behind at" in text
        # uv keeps its Python and cache in seedvr2/uv: present in the environment of the programs the run starts
        bn.point_ai_dirs(third)

        class Ctx:
            should_stop = staticmethod(lambda: False)
            L = staticmethod(lambda m: got.append(m))
        got = []
        bn.run_streamed(Ctx, [sys.executable, "-c", "import os;print(os.environ['UV_PYTHON_INSTALL_DIR']);print(os.environ['UV_CACHE_DIR'])"])
        assert got[-2:] == [f"    {third / 'seedvr2' / 'uv' / 'python'}", f"    {third / 'seedvr2' / 'uv' / 'cache'}"], got
    print("AI files folder ok")


def check_progress(models, pics, out):
    """A run says that it is alive: every line has the time, a slow model reports its pieces (at most every 10 s, on a
    clock we move ourselves), Stop works between two pieces and leaves no file for that picture, and a slow
    Real-ESRGAN says it is still working and is ended by Stop."""
    import re
    import types
    photo(700, 500, 3).save(pics / "tiles.png")           # 12 pieces with the 256 pixel fake model
    clock = types.SimpleNamespace(t=1000.0)
    real_time, real_run = bn.time, bn.OnnxModel.run
    bn.time = types.SimpleNamespace(time=lambda: clock.t, sleep=real_time.sleep)

    def slow_run(self, t):
        clock.t += 4.0                                   # every piece takes 4 s of the fake clock
        calls.append(clock.t)
        return real_run(self, t)
    bn.OnnxModel.run = slow_run
    try:
        calls = []
        lines = []
        o = bn.Options(save_to=str(out / "prog"), ai_dir=str(models.parent), inputs=[str(pics / "tiles.png")],
                       mode="upscale", up_model="dat", scale=2)
        res = bn.run(o, lines.append)
        assert res, lines
        said = [ln for ln in lines if "piece " in ln]
        assert 2 <= len(said) <= 4, said                 # 12 pieces of 4 s = 48 s: one line every 3rd piece, none at the end
        assert all(re.search(r"\] +dat: piece \d+ of 12, \d+ (s|min)( \d+ s)? so far$", ln) for ln in said), said
        assert said[0].endswith("dat: piece 3 of 12, 12 s so far"), said[0]
        assert not any("left" in ln for ln in lines), "a finish-time guess"
        for ln in lines:
            if ln.strip():
                assert re.match(r"\n*\[\d\d:\d\d:\d\d\] ", ln), f"no time in front: {ln!r}"
        assert any("The first run on the graphics card" in ln for ln in lines) == ("DmlExecutionProvider" in bn.ort.get_available_providers())
        logfile = next((Path(res).parent / "_blurnomore_work" / "logs").glob("*.log")).read_text(encoding="utf-8")
        assert logfile.startswith("[") and "piece 3 of 12" in logfile and logfile.rstrip().splitlines()[-1].startswith("["), logfile
        # a quick step says nothing about pieces
        calls.clear()
        bn.OnnxModel.run = lambda self, t: real_run(self, t)
        lines = []
        bn.run(bn.Options(save_to=str(out / "prog_fast"), ai_dir=str(models.parent), inputs=[str(pics / "tiles.png")],
                          mode="upscale", up_model="dat", scale=2), lines.append)
        assert not any("piece " in ln for ln in lines), lines
        # Stop after the 5th piece: it ends there, says so, and no file is made for that picture
        bn.OnnxModel.run = slow_run
        calls.clear()
        lines = []
        o = bn.Options(save_to=str(out / "prog_stop"), ai_dir=str(models.parent), inputs=[str(pics / "tiles.png")],
                       mode="upscale", up_model="dat", scale=2, redo=True)
        res = bn.run(o, lines.append, lambda: len(calls) >= 5)
        assert res is None and len(calls) == 5, (res, len(calls))
        assert any("Stopped by user." in ln for ln in lines), lines
        assert not any(f.suffix in (".png", ".jpg") for f in (out / "prog_stop").rglob("*") if "_blurnomore_work" not in f.parts and f.is_file()), "a file was left"
        assert not list((out / "prog_stop" / "_blurnomore_work" / "preview_cache").glob("*.png")), "a stopped picture was cached"
        # Compare: a picture stopped in the middle is not written
        calls.clear()
        lines = []
        res = bn.run(bn.Options(save_to=str(out / "prog_cmp"), ai_dir=str(models.parent), inputs=[str(pics / "tiles.png")],
                                mode="compare", scale=2, compare_models=["dat", "bhi"]), lines.append, lambda: len(calls) >= 5)
        assert res is None and any("Stopped by user." in ln for ln in lines), lines
        assert not list((out / "prog_cmp").glob("BlurNoMore_compare_*/*_?.png")), "a half picture was written"
    finally:
        bn.OnnxModel.run = real_run
        bn.time = real_time
    # Real-ESRGAN: still working every 30 s (fake clock +10 s per look), and Stop ends the program
    exe = out / "slow_esrgan.sh"
    exe.write_text("#!/bin/sh\nsleep 3\nexit 1\n")
    exe.chmod(0o755)
    tmp = out / "esr_tmp"
    tmp.mkdir()
    said = []
    clock.t = 0.0

    def tick():
        clock.t += 10.0
        return clock.t
    bn.time = types.SimpleNamespace(time=tick, sleep=real_time.sleep)
    try:
        assert bn.run_esrgan(Image.new("RGB", (8, 8)), "x", exe, tmp, said.append) is None
    finally:
        bn.time = real_time
    assert any(re.fullmatch(r"    Real-ESRGAN still working, \d+ (s|min)( \d+ s)? so far", m) for m in said), said
    t0 = time.time()
    try:
        bn.run_esrgan(Image.new("RGB", (8, 8)), "x", exe, tmp, said.append, should_stop=lambda: time.time() - t0 > 1)
        raise AssertionError("Stop did not end Real-ESRGAN")
    except bn.Stopped:
        pass
    assert time.time() - t0 < 3, "Stop took too long"
    print("progress ok")


def check_ort_thread(models, pics, out):
    """Every onnxruntime session is made and run on one and the same thread, never the caller's: DirectML crashed
    the program when 4.8 ran each piece on a new thread."""
    real = bn.ort.InferenceSession
    where = []

    class Watched(real):
        def __init__(self, *a, **k):
            where.append(("make", threading.current_thread()))
            super().__init__(*a, **k)

        def run(self, *a, **k):
            where.append(("run", threading.current_thread()))
            return super().run(*a, **k)
    bn.ort.InferenceSession = Watched
    try:
        res, lines = make_run(models, out / "ort_thread", inputs=[str(pics / "tiles.png"), str(pics / "small.png")],
                              mode="upscale", up_model="dat", scale=2, redo=True)
        assert res, "\n".join(lines)[-800:]
        res, lines = make_run(models, out / "ort_thread_cmp", inputs=[str(pics / "tiles.png")], mode="compare", scale=2,
                              compare_models=["dat", "bhi"])
        assert res, "\n".join(lines)[-800:]
    finally:
        bn.ort.InferenceSession = real
    kinds = {k for k, _ in where}
    threads = {t for _, t in where}
    assert kinds == {"make", "run"} and len(threads) == 1, [(k, t.name) for k, t in where]
    assert threading.current_thread() not in threads
    print("one onnxruntime thread ok")


def check_card_choice(models):
    """DirectML is asked for the fast card and the log names it; an onnxruntime that does not know that wish gets each
    card timed and the fastest kept. (Fake DirectML: there is no Windows here.)"""
    real, real_list = bn.ort.InferenceSession, bn.gpu_list
    made = []
    old = {"on": False}
    cards = {"plain": [{"name": "AMD Radeon(TM) Graphics", "vram_mb": 512}, {"name": "AMD Radeon RX 7800 XT", "vram_mb": 16368}],
             "fast": [{"name": "AMD Radeon RX 7800 XT", "vram_mb": 16368}, {"name": "AMD Radeon(TM) Graphics", "vram_mb": 512}]}

    class FakeDml(real):
        def __init__(self, path, opts=None, providers=None):
            first = providers[0]
            made.append(first)
            if isinstance(first, tuple):
                if old["on"] and "performance_preference" in first[1]:
                    raise RuntimeError("Unknown provider option: performance_preference")
                self.slow = first[1].get("device_id") == 0
            super().__init__(path, providers=["CPUExecutionProvider"])
            self.fake = first[0] if isinstance(first, tuple) else first

        def get_providers(self):
            return [self.fake, "CPUExecutionProvider"]

        def run(self, *a, **k):
            if getattr(self, "slow", False):
                time.sleep(0.05)
            return super().run(*a, **k)
    bn.ort.InferenceSession, bn.gpu_list = FakeDml, lambda: cards
    try:
        path = models / bn.UPSCALERS["bhi"]["file"]
        for old["on"] in (False, True):
            lines = []
            c = bn.Context(bn.Options(), lines.append, models.parent)
            c.prov = ["DmlExecutionProvider", "CPUExecutionProvider"]
            made.clear()
            bn.make_session(c, path)
            bn.make_session(c, path)
            text = "\n".join(lines)
            assert "ONNX Runtime uses the graphics card AMD Radeon RX 7800 XT (16 GB) (DirectML)" in text, text
            if not old["on"]:
                assert made == [("DmlExecutionProvider", bn.DML_FAST)] * 2, made
            else:
                assert "timing the cards instead" in text and made[-1] == ("DmlExecutionProvider", {"device_id": 1}), (made, text)
        old["on"] = False
        for want, device, said in (("AMD Radeon(TM) Graphics", 0, "AMD Radeon(TM) Graphics"),
                                   ("AMD Radeon RX 7800 XT (16 GB)", 1, "AMD Radeon RX 7800 XT (16 GB)"),
                                   ("Some Old Card", None, "was not found")):
            lines = []
            c = bn.Context(bn.Options(gpu=want), lines.append, models.parent)
            c.prov = ["DmlExecutionProvider", "CPUExecutionProvider"]
            made.clear()
            bn.make_session(c, path)
            text = "\n".join(lines)
            assert said in text, text
            assert made[0] == (("DmlExecutionProvider", {"device_id": device}) if device is not None
                               else ("DmlExecutionProvider", bn.DML_FAST)), (want, made)
    finally:
        bn.ort.InferenceSession, bn.gpu_list = real, real_list
    print("graphics card choice ok")


def check_compare_memory_and_crash(models, pics, out):
    """Compare lets each model go after its picture (memory back for the next one), keeps each result in
    _blurnomore_work/compare_unfinished right away (gone after a finished run), and a run that dies keeps them."""
    real = bn.ort.InferenceSession
    made = []

    class Counted(real):
        def __init__(self, *a, **k):
            made.append(1)
            super().__init__(*a, **k)
    bn.ort.InferenceSession = Counted
    try:
        res, lines = make_run(models, out / "cmp_mem", inputs=[str(pics / "small.png"), str(pics / "bigger.png")], mode="compare",
                              scale=2, compare_models=["dat", "bhi"])
    finally:
        bn.ort.InferenceSession = real
    assert res and len(made) >= 4, f"models were not let go between pictures ({len(made)} sessions)"
    assert not (out / "cmp_mem" / "_blurnomore_work" / "compare_unfinished").exists(), "unfinished results left after a finished run"

    real_make = bn.make_upscaled
    calls = []

    def dies(c, path, src_id, im, force=None):
        calls.append(force)
        if len(calls) == 2:
            raise KeyboardInterrupt("the program dies")
        return real_make(c, path, src_id, im, force=force)
    bn.make_upscaled = dies
    try:
        try:
            make_run(models, out / "cmp_dies", inputs=[str(pics / "small.png")], mode="compare", scale=2, compare_models=["dat", "bhi"])
            raise AssertionError("the fake crash did not happen")
        except KeyboardInterrupt:
            pass
    finally:
        bn.make_upscaled = real_make
    kept = list((out / "cmp_dies" / "_blurnomore_work" / "compare_unfinished").glob("*/*"))
    names = sorted(p.name for p in kept)
    assert len(names) == 2 and names[0] == "key.txt" and names[1].startswith("small_") and names[1].endswith(".png"), names
    key = next(p for p in kept if p.name == "key.txt").read_text(encoding="utf-8")
    assert f"{names[1][6]} = " in key, key
    print("compare memory and unfinished results ok")


def check_failed_lines(models, pics, out):
    """A model that fails shows one FAILED line in the window; the full trace is only in the log file (and Copy log)."""
    real_run = bn.OnnxModel.run

    calls = []

    def boom(self, t):
        calls.append(1)
        if len(calls) == 1:                    # the test run when it loads works; the first piece fails
            return real_run(self, t)
        raise RuntimeError("Not enough memory resources are available to complete this operation.")
    bn.OnnxModel.run = boom
    try:
        res, lines = make_run(models, out / "failed_line", inputs=[str(pics / "small.png")], mode="upscale", up_model="bhi", scale=2,
                              redo=True)
    finally:
        bn.OnnxModel.run = real_run
    text = "\n".join(lines)
    assert res and "Traceback" not in text, text[-1500:]
    fails = [ln for ln in lines if "FAILED: bhi" in ln]
    assert len(fails) == 1 and fails[0].endswith(
        "FAILED: bhi: RuntimeError: Not enough memory resources are available to complete this operation. (out of memory: "
        "is another BlurNoMore, or an earlier run, still going? Close it, or restart the PC, and try again)"), fails
    log = next((out / "failed_line" / "_blurnomore_work" / "logs").glob("*.log")).read_text(encoding="utf-8")
    assert "Traceback" in log and "FAILED: bhi" in log, log[-1500:]
    print("one FAILED line ok")


def check_hang_and_links(models, pics, out):
    """A model that makes no progress for HANG_LIMIT is given up (tiny limit here): the run goes on, the log says FAILED
    once, and Compare shows that letter failed; the picture and result lines carry links."""
    real_run, real_limit = bn.OnnxModel.run, bn.HANG_LIMIT
    bn.HANG_LIMIT = 1
    seen = []

    def stuck_run(self, t):
        seen.append(1)
        if len(seen) >= 3:
            time.sleep(4)                       # the third piece never comes back
        return real_run(self, t)
    bn.OnnxModel.run = stuck_run
    try:
        lines = []
        t0 = time.time()
        res = bn.run(bn.Options(save_to=str(out / "hang"), ai_dir=str(models.parent), inputs=[str(pics / "tiles.png")],
                                mode="upscale", up_model="dat", scale=2, redo=True), lines.append)
        text = "\n".join(lines)
        assert res and time.time() - t0 < 30, text[-800:]
        fails = [ln for ln in lines if "FAILED:" in ln]
        assert len(fails) == 1 and fails[0].endswith("FAILED: dat made no progress for 1 s (stuck at piece 2 of 12); "
                                                     "skipping it for the rest of this run"), fails
        assert "lanczos (AI failed)" in text, text[-800:]
        # the model that hung is not tried again in the same run
        seen.clear()
        lines = []
        bn.run(bn.Options(save_to=str(out / "hang2"), ai_dir=str(models.parent), inputs=[str(pics / "tiles.png"), str(pics / "small.png")],
                          mode="upscale", up_model="dat", scale=2, redo=True), lines.append)
        assert sum("FAILED: dat" in ln for ln in lines) == 1 and len(seen) == 3, (len(seen), lines[-12:])
        # Compare: that letter is failed in the key, the others are made
        seen.clear()
        res, lines = make_run(models, out / "hang_cmp", inputs=[str(pics / "tiles.png")], mode="compare", scale=2,
                              compare_models=["dat", "bhi"])
        assert res, lines
        key = (Path(res) / "key.txt").read_text(encoding="utf-8")
        assert "failed: dat made no progress for 1 s" in key or "failed: bhi made no progress for 1 s" in key, key
        assert len(list(Path(res).glob("*_?.png"))) >= 1, "the other letters were not made"
    finally:
        bn.OnnxModel.run = real_run
        bn.HANG_LIMIT = real_limit
    # Real-ESRGAN that prints nothing, and SeedVR2 that prints nothing: ended after the limit
    bn.HANG_LIMIT = 1
    try:
        exe = out / "mute_esrgan.sh"
        exe.write_text("#!/bin/sh\nsleep 30\n")
        exe.chmod(0o755)
        tmp = out / "esr_tmp2"
        tmp.mkdir()
        said, t0 = [], time.time()
        assert bn.run_esrgan(Image.new("RGB", (8, 8)), "x", exe, tmp, said.append) is None
        assert time.time() - t0 < 5 and any(m.endswith("FAILED: Real-ESRGAN made no progress for 1 s; skipping it for the rest of this run") or
                                            m == "    FAILED: Real-ESRGAN made no progress for 1 s; skipping it for the rest of this run"
                                            for m in said), said
        sv = models.parent / "seedvr2"
        os.environ["FAKE_SV_HANG"] = "1"
        try:
            lines = []
            t0 = time.time()
            bn.run(bn.Options(save_to=str(out / "sv_hang"), inputs=[str(pics / "small.png")], mode="upscale", up_model="seedvr2",
                              scale=2, redo=True, ai_dir=str(models.parent)), lines.append)
        finally:
            del os.environ["FAKE_SV_HANG"]
        assert time.time() - t0 < 20 and any("FAILED: seedvr2 made no progress for 1 s; skipping it" in ln for ln in lines), lines[-8:]
        pid = int((sv / "pid.txt").read_text())
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
            raise AssertionError("the fake tool still runs after the guard gave up")
        except ProcessLookupError:
            pass
    finally:
        bn.HANG_LIMIT = real_limit
    # links: the picture line (the original), the line naming the result, the DONE line (the folder); the log file is plain
    lines = []
    res = bn.run(bn.Options(save_to=str(out / "links"), ai_dir=str(models.parent), inputs=[str(pics / "small.png")],
                            mode="upscale", up_model="dat", scale=2, redo=True), lines.append)
    pic = next(ln for ln in lines if "small.png (48x40)" in ln)
    assert pic.lstrip("\n")[11:].startswith("[1/1] #1 small.png (48x40)") and pic.links == (("[1/1] #1 small.png (48x40)", pics / "small.png"),)
    done = next(ln for ln in lines if "DONE in" in ln)
    assert done.links == ((str(res), Path(res)),)
    got = next(ln for ln in lines if "->" in ln)
    assert got.links[0][0] in got and got.links[0][1] == Path(res) / got.links[0][0], got.links
    logfile = next((Path(res).parent / "_blurnomore_work" / "logs").glob("*.log")).read_text(encoding="utf-8")
    assert "small.png (48x40)" in logfile and type(logfile) is str
    print("hang guard and links ok")


def check():
    with tempfile.TemporaryDirectory() as t_models, tempfile.TemporaryDirectory() as t_in, \
            tempfile.TemporaryDirectory() as t_out, tempfile.TemporaryDirectory() as t_app:
        # the AI files folder is t_models (models/, seedvr2/); "next to the program" is an empty temporary folder
        bn.APP_DIR, bn.SETTINGS_FILE = Path(t_app), Path(t_app) / "settings.json"
        models, pics, out = Path(t_models) / "models", Path(t_in), Path(t_out)
        sv = Path(t_models) / "seedvr2"
        install_fakes(models)
        photo(48, 40).save(pics / "small.png")
        photo(200, 150, 2).save(pics / "bigger.png")
        before = snapshot(pics)

        # 1. the new photo models: exact size at 2x (a 4x model, then resized)
        for key in ("bhi", "drct"):
            res, log = make_run(models, out / key, inputs=[str(pics / "small.png")], mode="upscale", up_model=key, scale=2)
            assert res and f"AI {key}" in "\n".join(log), f"{key} was not used"
            files = list(Path(res).iterdir())
            assert len(files) == 1 and Image.open(files[0]).size == (96, 80), f"{key}: wrong size"
        print("photo models ok")

        # 2. the inventing upscaler (SeedVR2, fake tool): exact size, the options it gets, a seed, Stop, a failure
        install_fake_seedvr2(sv)
        for kw, size in (({"scale": 4}, (192, 160)), ({"scale": 3}, (144, 120)), ({"size_mode": "width", "size_px": 100}, (100, 83))):
            res, log = make_run(models, out / f"sv{len(kw)}{kw.get('scale', 0)}", inputs=[str(pics / "small.png")], mode="upscale",
                                up_model="seedvr2", sv_seed=3, **kw)
            text = "\n".join(log)
            assert res and "AI seedvr2" in text, f"SeedVR2 was not used\n{text[-1500:]}"
            files = list(Path(res).iterdir())
            assert len(files) == 1 and Image.open(files[0]).size == size, (Image.open(files[0]).size, size)
        assert "SeedVR2 set-up" not in text, "it set itself up although ready.json is there"
        call = last_call(sv)
        assert call["seed"] == 3 and call["color_correction"] == "lab" and call["input_noise_scale"] == 0.0 \
            and call["latent_noise_scale"] == 0.0 and call["batch_size"] == 1 and call["resolution"] == 83, call
        assert call["vae_encode_tiled"] and call["vae_decode_tiled"] and call["blocks_to_swap"] == 0 and not call["swap_io_components"], call
        assert call["dit_model"] == bn.SEEDVR2_DIT and Path(call["model_dir"]) == models / "seedvr2", call
        assert call["output"].endswith("seedvr2_out.png") and call["input"].endswith("seedvr2_in.png"), call
        res, log = make_run(models, out / "sv_opts", inputs=[str(pics / "small.png")], mode="upscale", up_model="seedvr2", scale=2,
                            sv_seed=7, sv_color="wavelet", sv_input_noise=0.3, sv_latent_noise=0.2, sv_low_memory=True)
        call = last_call(sv)
        assert call["seed"] == 7 and call["color_correction"] == "wavelet" and call["input_noise_scale"] == 0.3 \
            and call["latent_noise_scale"] == 0.2 and call["blocks_to_swap"] == 32 and call["swap_io_components"] \
            and call["dit_offload_device"] == "cpu" and call["vae_offload_device"] == "cpu" and call["resolution"] == 80, call
        # a random seed is named in the log and passed on; the same seed gives the same result (and other settings another cache name)
        _, log = make_run(models, out / "sv_rand", inputs=[str(pics / "small.png")], mode="upscale", up_model="seedvr2", scale=2, sv_seed=0)
        assert any("chosen at random" in ln for ln in log) and last_call(sv)["seed"] > 0, "a random seed is not named in the log"
        # a failing tool: the picture is plain resized, the log says why, the run goes on with the next picture
        os.environ["FAKE_SV_FAIL"] = "1"
        try:
            res, log = make_run(models, out / "sv_fail", inputs=[str(pics)], mode="upscale", up_model="seedvr2", scale=2, redo=True)
        finally:
            del os.environ["FAKE_SV_FAIL"]
        text = "\n".join(log)
        assert res and "out of graphics memory" in text and "plain resizing" in text, text[-1200:]
        assert sorted(Image.open(p).size for p in Path(res).iterdir()) == [(96, 80), (400, 300)], "the picture was not made after a failure"
        n_calls = len((sv / "calls.jsonl").read_text().splitlines())
        assert "AI seedvr2" not in text.split("out of graphics memory")[-1], "tried again after failing in the same run"
        # Stop in the middle of a picture ends the tool
        os.environ["FAKE_SV_SLOW"] = "1"
        lines = []
        stop = lambda: any("progress step 2" in ln for ln in lines)
        t0 = time.time()
        try:
            bn.run(bn.Options(save_to=str(out / "sv_stop"), inputs=[str(pics / "small.png")], mode="upscale", up_model="seedvr2",
                              scale=2, redo=True, ai_dir=str(models.parent)), lines.append, stop)
        finally:
            del os.environ["FAKE_SV_SLOW"]
        assert "Stopped by user." in "\n".join(lines) and time.time() - t0 < 20, lines[-8:]
        pid = int((sv / "pid.txt").read_text())
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
            raise AssertionError("the fake tool still runs after Stop")
        except ProcessLookupError:
            pass
        print("inventing upscaler ok")

        # the set-up: skipped when ready.json is there; the steps (nothing is run), and a step that fails
        assert bn.seedvr2_ready()
        assert bn.torch_kind("Intel(R) UHD\nAMD Radeon RX 7900 XT") == "amd" and bn.torch_kind("NVIDIA GeForce RTX 4070") == "nvidia" \
            and bn.torch_kind("Intel(R) Arc") == "cpu" and bn.torch_kind("") == "cpu"
        venv = str(sv / "venv")
        amd = bn.seedvr2_steps("amd")
        base = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
        cmds = [s[2] for s in amd]
        assert cmds[0] == [sys.executable, "-m", "pip", "install", "uv"]
        assert cmds[1][1:] == ["-m", "uv", "venv", "--python", "3.12", venv] and cmds[2] == "source" and cmds[-1] == "requirements"
        pip = [sys.executable, "-m", "uv", "pip", "install", "--python", venv]
        assert cmds[3] == pip + [base + w for w in ("rocm_sdk_core-7.2.1-py3-none-win_amd64.whl", "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
                                                    "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl", "rocm-7.2.1.tar.gz")]
        assert cmds[4] == pip + [base + "torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                                 base + "torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"] and len(cmds) == 6
        assert [s[2] for s in bn.seedvr2_steps("nvidia")][3:] == [pip + ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"], "requirements"]
        assert [s[2] for s in bn.seedvr2_steps("cpu")][3:] == [pip + ["torch", "torchvision"], "requirements"]
        assert bn.seedvr2_requirements(sv / "c.txt")[-4:] == ["-r", str(sv / "src" / "requirements.txt"), "-c", str(sv / "c.txt")]

        ran, saved = [], (bn.run_streamed, bn.seedvr2_download_source, bn.write_torch_constraints, bn.graphics_card_names)
        try:
            bn.seedvr2_download_source = lambda c: ran.append("source")
            bn.write_torch_constraints = lambda path: ran.append("constraints")
            bn.graphics_card_names = lambda: "AMD Radeon RX 7900 XT"

            def fake_streamed(c, argv, keep=None, cwd=None, every=0):
                ran.append(list(argv))
                if "boom" in os.environ.get("FAKE_SETUP", "") and "torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" in " ".join(map(str, argv)):
                    return 1, ["error: no matching file", "last line"]
                return 0, []
            bn.run_streamed = fake_streamed
            (sv / "ready.json").unlink()
            lines = []
            ctx = bn.Context(bn.Options(save_to=str(out / "setup"), mode="upscale"), lines.append, out / "setup_work")
            os.environ["FAKE_SETUP"] = "boom"
            assert bn.seedvr2_setup(ctx) is False and not (sv / "ready.json").exists()
            text = "\n".join(lines)
            assert "Installing AMD's PyTorch for Windows" in text and "error: no matching file" in text and "last line" in text \
                and "SeedVR2 could not be set up; using plain resizing for it" in text and "driver 26.2.2" in text, text
            ran.clear()
            os.environ["FAKE_SETUP"] = ""
            assert bn.seedvr2_setup(ctx) is True and (sv / "ready.json").exists()
            assert json.loads((sv / "ready.json").read_text())["torch"] == "amd" and "source" in ran and "constraints" in ran
            n_run = len(ran)
            install_fake_seedvr2(sv, ready=False)      # (the set-up made no real environment; ready.json is from the set-up)
            assert bn.seedvr2_setup(ctx) is True and len(ran) == n_run, "it set up again although ready.json was there"
        finally:
            del os.environ["FAKE_SETUP"]
            bn.run_streamed, bn.seedvr2_download_source, bn.write_torch_constraints, bn.graphics_card_names = saved
        install_fake_seedvr2(sv)
        print("set-up ok")

        # 3. Compare models: letters, close-ups, key; letters change between runs; nothing names a model outside the key
        install_fakes(models)
        letters_seen = set()
        for run_no in range(6):
            res, log = make_run(models, out / f"cmp{run_no}", inputs=[str(pics)], mode="compare", scale=2,
                                compare_models=["dat", "bhi", "hatgan"])
            folder = Path(res)
            names = sorted(p.name for p in folder.iterdir())
            assert names == sorted(["bigger_A.png", "bigger_B.png", "bigger_C.png", "bigger_D.png", "bigger_closeups.png",
                                    "small_A.png", "small_B.png", "small_C.png", "small_D.png", "small_closeups.png",
                                    "key.png", "key.txt"]), names
            key = (folder / "key.txt").read_text(encoding="utf-8")
            wanted = [bn.MODEL_LABELS["dat"], bn.MODEL_LABELS["bhi"], bn.MODEL_LABELS["hatgan"], bn.PLAIN_LABEL]
            mapping = {}
            for lab in wanted:       # each model once in the key, with one letter
                hits = [ln[0] for ln in key.splitlines() if ln[1:4] == " = " and ln[4:].startswith(lab + " (")]
                assert len(hits) == 1, (lab, key)
                mapping[hits[0]] = lab
            assert sorted(mapping) == ["A", "B", "C", "D"], mapping
            letters_seen.add(tuple(sorted(mapping.items())))
            blob = "\n".join(log).lower()
            for lab in mapping.values():
                assert lab.lower() not in blob, f"the log names a model label: {lab}"
            assert not any(f"= {m}" in blob for m in ("dat", "bhi", "hatgan")) and "key.png only after" in blob
            for p in folder.glob("*_[A-D].png"):
                with Image.open(p) as im:
                    assert im.size == ((96, 80) if p.name.startswith("small") else (400, 300)), (p.name, im.size)
                    assert not im.info, f"{p.name} carries extra data: {im.info}"
            with Image.open(folder / "bigger_closeups.png") as sheet:
                assert sheet.size[0] > 4 * 200
            assert Image.open(folder / "key.png").size[0] > 100
        assert len(letters_seen) > 1, "the letters were the same in six runs"
        # the same letter is the same model for both pictures: letters of the closeups sheet come from one mapping (checked by the key)
        print("compare ok")

        # a model that cannot be had (not there, no download) leaves its letter out and says why; the run goes on
        old = bn.urllib.request.urlretrieve
        bn.urllib.request.urlretrieve = lambda *a, **k: (_ for _ in ()).throw(OSError("no internet here"))
        try:
            install_fakes(models, skip=())
            (models / bn.UPSCALERS["drct"]["file"]).unlink()
            res, log = make_run(models, out / "cmp_fail", inputs=[str(pics)], mode="compare", scale=2, compare_models=["dat", "drct"])
        finally:
            bn.urllib.request.urlretrieve = old
        folder = Path(res)
        key = (folder / "key.txt").read_text(encoding="utf-8")
        assert "failed:" in key and "no internet here" in key, key
        assert len(list(folder.glob("small_?.png"))) == 2 and len(list(folder.glob("bigger_?.png"))) == 2
        missing = [ln for ln in log if "could not make this picture" in ln]
        assert len(missing) == 2 and all(ln.endswith("1 of the 3 could not make this picture (the key says which)") for ln in missing), missing
        print("compare with a missing model ok")

        # SeedVR2 inside Compare (also when it fails); and more than 4 pictures use the first 4
        install_fakes(models)
        for i in range(5):
            photo(40, 40, i).save(pics / f"p{i}.png")
        res, log = make_run(models, out / "cmp_sv", inputs=[str(pics)], mode="compare", scale=2, compare_models=["dat", "seedvr2"])
        text = "\n".join(log)
        assert "first 4 of the 7" in text, text[:600]
        folder = Path(res)
        key = (folder / "key.txt").read_text(encoding="utf-8")
        assert len(list(folder.glob("*_closeups.png"))) == 4 and bn.MODEL_LABELS["seedvr2"] in key and "failed" not in key, key
        assert "inventing upscaler: colours lab" in key.lower(), key
        os.environ["FAKE_SV_FAIL"] = "1"
        try:
            res, log = make_run(models, out / "cmp_sv_fail", inputs=[str(pics)], mode="compare", scale=2, compare_models=["dat", "seedvr2"])
        finally:
            del os.environ["FAKE_SV_FAIL"]
        key = (Path(res) / "key.txt").read_text(encoding="utf-8")
        assert "failed:" in key and "out of graphics memory" in key, key
        assert len(list(Path(res).glob("*_?.png"))) == 8, "the letter of the failed model was not left out"
        print("compare with the inventing model ok")

        check_ai_dir(out)
        check_progress(models, pics, out)
        check_ort_thread(models, pics, out)
        check_card_choice(models)
        check_compare_memory_and_crash(models, pics, out)
        check_failed_lines(models, pics, out)
        check_hang_and_links(models, pics, out)

        assert all(snapshot(pics).get(n) == hsh for n, hsh in before.items()), "an original changed"
        print("Fake-model check passed.")


if __name__ == "__main__":
    main()
