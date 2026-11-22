"""Fake-model check: runs the new photo models, the inventing (Stable Diffusion) upscaler and Compare models end to
end with small made-up ONNX models (they do almost nothing, but have the real file names and input names), so no real
model has to be downloaded. The real models run in the "Models" workflow.
Run:  python tests/fake_models_check.py   (needs pillow, numpy, img2pdf, onnxruntime and onnx). Exits non-zero on failure.
Everything happens in temporary folders; the models folder next to the program is not touched."""
import hashlib
import json
import sys
import tempfile
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


def fake_text_encoder(path):
    save([h.make_node("Cast", ["input_ids"], ["f"], to=T.FLOAT),
          h.make_node("Unsqueeze", ["f", "axis"], ["f3"]),
          h.make_node("Mul", ["f3", "ramp"], ["last_hidden_state"])],
         [h.make_tensor_value_info("input_ids", T.INT32, ["batch", 77])],
         [h.make_tensor_value_info("last_hidden_state", T.FLOAT, ["batch", 77, 1024])],
         [const("axis", np.array([2], np.int64)), const("ramp", (np.arange(1024, dtype=np.float32) / 1024.0)[None, None, :])], path)


def fake_unet(path, timestep_type):
    """Gives the v that makes every DDIM step land on the (noisy) low-resolution picture, so the result is that picture.
    All four inputs are used, with the names and types of the real unet."""
    acp = bn.ddim_alphas().astype(np.float32)
    n = [h.make_node("Slice", ["sample", "s0", "s4", "ax1"], ["lat"]),
         h.make_node("Slice", ["sample", "s4", "s7", "ax1"], ["rgb"]),
         h.make_node("Slice", ["sample", "s4", "s5", "ax1"], ["r"]),
         h.make_node("Mul", ["r", "zero"], ["zeros"]),
         h.make_node("Concat", ["rgb", "zeros"], ["target"], axis=1),
         h.make_node("Cast", ["timestep"], ["t64"], to=T.INT64),
         h.make_node("Gather", ["acp", "t64"], ["a1"]),
         h.make_node("Reshape", ["a1", "shape4"], ["a"]),
         h.make_node("Sqrt", ["a"], ["sa"]),
         h.make_node("Sub", ["one", "a"], ["b"]),
         h.make_node("Sqrt", ["b"], ["sb"]),
         h.make_node("Mul", ["sa", "lat"], ["sal"]),
         h.make_node("Sub", ["sal", "target"], ["num"]),
         h.make_node("Div", ["num", "sb"], ["v0"]),
         h.make_node("ReduceMean", ["encoder_hidden_states"], ["m"], keepdims=0),
         h.make_node("Cast", ["class_labels"], ["cl"], to=T.FLOAT),
         h.make_node("ReduceMean", ["cl"], ["clm"], keepdims=0),
         h.make_node("Add", ["m", "clm"], ["mm"]),
         h.make_node("Mul", ["mm", "zero"], ["noeffect"]),
         h.make_node("Add", ["v0", "noeffect"], ["out_sample"])]
    save(n, [h.make_tensor_value_info("sample", T.FLOAT, ["batch", 7, "h", "w"]),
             h.make_tensor_value_info("timestep", timestep_type, [1]),
             h.make_tensor_value_info("encoder_hidden_states", T.FLOAT, ["batch", 77, 1024]),
             h.make_tensor_value_info("class_labels", T.INT64, ["batch"])],
         [h.make_tensor_value_info("out_sample", T.FLOAT, ["batch", 4, "h", "w"])],
         [const("s0", np.array([0], np.int64)), const("s4", np.array([4], np.int64)), const("s5", np.array([5], np.int64)),
          const("s7", np.array([7], np.int64)), const("ax1", np.array([1], np.int64)), const("zero", np.array(0, np.float32)),
          const("one", np.array(1, np.float32)), const("acp", acp), const("shape4", np.array([1, 1, 1, 1], np.int64))], path)


def fake_vae(path):
    save([h.make_node("Slice", ["latent_sample", "s0", "s3", "ax1"], ["rgb"]),
          h.make_node("Mul", ["rgb", "k"], ["small"]),
          h.make_node("Resize", ["small", "", "scales"], ["sample"], mode="linear")],
         [h.make_tensor_value_info("latent_sample", T.FLOAT, ["batch", 4, "h", "w"])],
         [h.make_tensor_value_info("sample", T.FLOAT, ["batch", 3, "H", "W"])],
         [const("s0", np.array([0], np.int64)), const("s3", np.array([3], np.int64)), const("ax1", np.array([1], np.int64)),
          const("k", np.array(0.08333, np.float32)), const("scales", np.array([1, 1, 4, 4], np.float32))], path)


def fake_tokenizer(folder):
    """A tiny vocab in the real format: the 256 byte characters, the same with </w>, the merges, the two special tokens."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars, k = keep[:], 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + k)
            k += 1
    letters = [chr(c) for c in chars]
    merges = ["t h", "th e</w>", "c a", "ca t</w>"]
    vocab = letters + [c + "</w>" for c in letters] + ["".join(m.split()) for m in merges] + ["<|startoftext|>", "<|endoftext|>"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(vocab)}), encoding="utf-8")
    (folder / "merges.txt").write_text("#version: 0.2\n" + "\n".join(merges) + "\n", encoding="utf-8")
    return vocab


def install_fakes(models, with_diffusion=True, timestep_type=T.INT64, skip=()):
    """Fake models under their real file names in the models folder `models`."""
    for i, (key, mode) in enumerate((("dat", "nearest"), ("bhi", "linear"), ("drct", "cubic"), ("hatgan", "linear"),
                                     ("heart", "cubic"))):
        if key not in skip:
            fake_upscaler(models / bn.UPSCALERS[key]["file"], mode, 128 if key == "drct" else 256)
    if with_diffusion:
        d = models / bn.DIFFUSION_DIR
        fake_text_encoder(d / "text_encoder" / "model.onnx")
        fake_unet(d / "unet" / "model.onnx", timestep_type)
        (d / "unet" / "weights.pb").write_bytes(b"")
        fake_vae(d / "vae" / "model.onnx")
        return fake_tokenizer(d / "tokenizer")


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


def check_tokenizer(t):
    vocab = fake_tokenizer(t / "tok")
    tok = bn.ClipTokenizer(t / "tok" / "vocab.json", t / "tok" / "merges.txt")
    ids = tok("  The   cat xyz ")
    ix = {v: i for i, v in enumerate(vocab)}
    assert ids == [ix["<|startoftext|>"], ix["the</w>"], ix["cat</w>"], ix["x"], ix["y"], ix["z</w>"]] \
        + [ix["<|endoftext|>"]] + [ix["!"]] * 70, ids
    assert len(tok("word " * 200)) == 77 and tok("word " * 200)[-1] == ix["<|endoftext|>"]
    print("tokenizer ok")


def main():
    saved_models_dir = bn.MODELS_DIR
    try:
        check()
    finally:
        bn.MODELS_DIR = saved_models_dir
        (ROOT / "blurnomore_last_run.json").unlink(missing_ok=True)


def check():
    with tempfile.TemporaryDirectory() as t_models, tempfile.TemporaryDirectory() as t_in, \
            tempfile.TemporaryDirectory() as t_out, tempfile.TemporaryDirectory() as t_tok:
        models, pics, out = Path(t_models), Path(t_in), Path(t_out)
        check_tokenizer(Path(t_tok))
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

        # 2. the inventing upscaler: exact 4x size, close to plain resizing (the fake gives back the picture), a seed, tiles
        for ttype, name in ((T.INT64, "int64 timestep"), (T.FLOAT, "float timestep")):
            install_fakes(models, timestep_type=ttype, skip=("dat", "bhi", "drct", "hatgan", "heart"))
            for f in (4, 2, 8):
                res, log = make_run(models, out / f"diff{ttype}{f}", inputs=[str(pics / "small.png")], mode="upscale",
                                    up_model="diffusion", scale=f if f != 8 else 4, size_mode="scale", diff_steps=5, diff_noise=0, diff_seed=3)
                text = "\n".join(log)
                assert res and "AI diffusion" in text, f"{name}: the inventing upscaler was not used\n{text[-1500:]}"
                got = Image.open(next(Path(res).iterdir()))
                assert got.size == (48 * (f if f != 8 else 4), 40 * (f if f != 8 else 4)), f"{name}: {got.size}"
        assert "tile 1/1" in text and "seed 3" in text, text[-800:]
        res, log = make_run(models, out / "diff_big", inputs=[str(pics / "bigger.png")], mode="upscale", up_model="diffusion",
                            scale=4, diff_steps=5, diff_noise=0, diff_seed=3)
        text = "\n".join(log)
        assert "tile 4/4" in text, text[-800:]
        got = np.asarray(Image.open(next(Path(res).iterdir())).convert("RGB")).astype(np.float32)
        plain = np.asarray(Image.open(pics / "bigger.png").convert("RGB").resize((800, 600), Image.BICUBIC)).astype(np.float32)
        assert got.shape == (600, 800, 3)
        diff = np.abs(got - plain).mean()
        assert diff < 6, f"the inventing result is far from the picture (mean difference {diff:.1f})"
        assert got.std() > 20, "the result is flat"
        # no seam: the jump between neighbouring columns / rows at the tile edges is not bigger than elsewhere
        # (low-resolution tiles start at x 0 and 72, y 0 and 22: their 4x edges are at x 288 and 512, y 88 and 512)
        for axis, edges in ((1, (288, 512)), (0, (88, 512))):
            jumps = np.abs(np.diff(got.mean(axis=2), axis=axis)).mean(axis=1 - axis)
            inside = float(np.median(jumps))
            for e in edges:
                worst = float(jumps[e - 3:e + 3].max())
                assert worst <= max(3 * inside, inside + 6), f"a seam at {e} on axis {axis}: {worst:.1f} against {inside:.1f}"
        # same seed = same result; a log line names a random seed; the cache key has the settings
        res2, _ = make_run(models, out / "diff_big2", inputs=[str(pics / "bigger.png")], mode="upscale", up_model="diffusion",
                           scale=4, diff_steps=5, diff_noise=0, diff_seed=3)
        assert np.array_equal(got, np.asarray(Image.open(next(Path(res2).iterdir())).convert("RGB")).astype(np.float32)), "same seed, other result"
        _, log = make_run(models, out / "diff_rand", inputs=[str(pics / "small.png")], mode="upscale", up_model="diffusion",
                          diff_steps=5, diff_seed=0, scale=4)
        assert any("chosen at random" in ln for ln in log), "a random seed is not named in the log"
        # Stop in the middle of a picture: no more tiles, the run ends saying so
        lines = []
        stop = lambda: any("tile 2/" in ln for ln in lines)
        bn.run(bn.Options(save_to=str(out / "diff_stop"), inputs=[str(pics / "bigger.png")], mode="upscale",
                          up_model="diffusion", scale=4, diff_steps=5, diff_seed=3), lines.append, stop)
        assert "Stopped by user." in "\n".join(lines) and not any("tile 3/" in ln for ln in lines), lines[-10:]
        print("inventing upscaler ok")

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

        # the diffusion model inside Compare; and more than 4 pictures use the first 4
        install_fakes(models)
        for i in range(5):
            photo(40, 40, i).save(pics / f"p{i}.png")
        res, log = make_run(models, out / "cmp_diff", inputs=[str(pics)], mode="compare", scale=2,
                            compare_models=["dat", "diffusion"], diff_steps=5)
        text = "\n".join(log)
        assert "first 4 of the 7" in text, text[:600]
        folder = Path(res)
        assert len(list(folder.glob("*_closeups.png"))) == 4 and "Stable Diffusion" in (folder / "key.txt").read_text(encoding="utf-8")
        assert "inventing upscaler: prompt" in (folder / "key.txt").read_text(encoding="utf-8").lower()
        print("compare with the inventing model ok")

        assert all(snapshot(pics).get(n) == hsh for n, hsh in before.items()), "an original changed"
        print("Fake-model check passed.")


if __name__ == "__main__":
    main()
