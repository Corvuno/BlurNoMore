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
    bn.MODELS_DIR = models_dir
    o = bn.Options(save_to=str(work), **kw)
    return bn.run(o, lines.append), lines


def main():
    saved_models_dir = bn.MODELS_DIR
    try:
        check()
    finally:
        bn.MODELS_DIR = saved_models_dir
        (ROOT / "blurnomore_last_run.json").unlink(missing_ok=True)


def check():
    with tempfile.TemporaryDirectory() as t_models, tempfile.TemporaryDirectory() as t_in, \
            tempfile.TemporaryDirectory() as t_out, tempfile.TemporaryDirectory() as t_sv:
        models, pics, out = Path(t_models), Path(t_in), Path(t_out)
        sv = Path(t_sv) / "seedvr2"
        bn.SEEDVR2_DIR = sv
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
                              scale=2, redo=True), lines.append, stop)
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

        assert all(snapshot(pics).get(n) == hsh for n, hsh in before.items()), "an original changed"
        print("Fake-model check passed.")


if __name__ == "__main__":
    main()
