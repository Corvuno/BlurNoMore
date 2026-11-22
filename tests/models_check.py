"""Model check: downloads the real DAT, BHI, DRCT, AnimeSharp, HAT-GAN, HEART, Stable Diffusion x4 (inventing), YuNet, GFPGAN, CodeFormer and DDColor models (about 6 GB, into models/ next
to the program) and runs them on made-up pictures, on the CPU. Run by the "Models" workflow, not
by the normal Checks.   python tests/models_check.py   (needs pillow, numpy, img2pdf, onnxruntime,
opencv-python-headless). Exits non-zero on failure."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
FACE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/messi5.jpg"
sys.path.insert(0, str(ROOT))


def run_py(*args):
    r = subprocess.run([sys.executable, str(ROOT / "blurnomore.py"), *map(str, args)], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    assert r.returncode == 0, "blurnomore.py exited with an error"
    assert "ERROR" not in r.stdout, "an image failed"
    return r.stdout


def only_file(folder):
    files = [p for d in folder.glob("BlurNoMore_upscaled_*") for p in d.iterdir()]
    assert len(files) == 1, f"expected one upscaled file in {folder}"
    return files[0]


def main():
    rng = np.random.default_rng(1)
    im = Image.fromarray(np.clip(rng.normal(128, 25, (120, 160, 3)), 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    dr.ellipse((30, 20, 90, 90), fill=(200, 60, 50))
    dr.rectangle((100, 40, 150, 110), fill=(40, 90, 180))
    with tempfile.TemporaryDirectory() as t, tempfile.TemporaryDirectory() as t_out:
        d, out = Path(t), Path(t_out)
        im.save(d / "photo.png")

        keep = ROOT / "models_check_output"
        keep.mkdir(exist_ok=True)

        # DAT model: downloaded on first use, 2x output
        log = run_py(d, "--mode", "upscale", "--upscale-model", "dat", "--scale", "2", "--save-to", out / "dat")
        assert "AI dat" in log, "the DAT model was not used"
        with Image.open(only_file(out / "dat")) as r:
            assert r.size == (320, 240), f"DAT result is {r.size}, expected 320x240"

        # AnimeSharp (2x model, runs twice for 4x) and HAT-GAN (4x model, 2x result)
        log = run_py(d, "--mode", "upscale", "--upscale-model", "animesharp", "--scale", "4", "--save-to", out / "sharp")
        assert "AI animesharp" in log, "the AnimeSharp model was not used"
        with Image.open(only_file(out / "sharp")) as r:
            assert r.size == (640, 480), f"AnimeSharp result is {r.size}, expected 640x480"
        only_file(out / "sharp").replace(keep / "animesharp_x4.png")
        log = run_py(d, "--mode", "upscale", "--upscale-model", "hatgan", "--scale", "2", "--save-to", out / "hat")
        assert "AI hatgan" in log, "the HAT-GAN model was not used"
        with Image.open(only_file(out / "hat")) as r:
            assert r.size == (320, 240), f"HAT-GAN result is {r.size}, expected 320x240"

        log = run_py(d, "--mode", "upscale", "--upscale-model", "heart", "--scale", "2", "--save-to", out / "heart")
        assert "AI heart" in log, "the HEART model was not used"
        with Image.open(only_file(out / "heart")) as r:
            assert r.size == (320, 240), f"HEART result is {r.size}, expected 320x240"

        # the three converted photo models (fixed 256 px tiles), exact size at 2x
        for key in ("bhi", "drct"):
            log = run_py(d, "--mode", "upscale", "--upscale-model", key, "--scale", "2", "--save-to", out / key)
            assert f"AI {key}" in log, f"the {key} model was not used"
            with Image.open(only_file(out / key)) as r:
                assert r.size == (320, 240), f"{key} result is {r.size}, expected 320x240"

        # the inventing upscaler (Stable Diffusion x4) on a tiny photo with 2 steps: exactly 4x, not flat
        tiny = Path(t) / "tiny"
        tiny.mkdir()
        im.resize((48, 48)).save(tiny / "tiny.png")
        log = run_py(tiny, "--mode", "upscale", "--upscale-model", "diffusion", "--scale", "4", "--diff-steps", "2",
                     "--save-to", out / "diffusion")
        assert "AI diffusion" in log, "the inventing upscaler was not used"
        res = only_file(out / "diffusion")
        with Image.open(res) as r:
            assert r.size == (192, 192), f"inventing result is {r.size}, expected 192x192"
            std = float(np.asarray(r.convert("L")).std())
        assert std > 5, f"the inventing result is flat grey (std {std:.1f})"
        res.replace(keep / "diffusion_x4.png")

        # Compare models: three models plus plain on two made-up pictures
        cmp_in = Path(t) / "cmp"
        cmp_in.mkdir()
        im.save(cmp_in / "one.png")
        im.transpose(Image.FLIP_LEFT_RIGHT).save(cmp_in / "two.png")
        log = run_py(cmp_in, "--compare", "dat,bhi,hatgan", "--scale", "2", "--save-to", out / "compare")
        folder = next((out / "compare").glob("BlurNoMore_compare_*"))
        names = sorted(p.name for p in folder.iterdir())
        want = sorted([f"{pic}_{ch}.png" for pic in ("one", "two") for ch in "ABCD"]
                      + ["one_closeups.png", "two_closeups.png", "key.png", "key.txt"])
        assert names == want, f"compare folder has {names}"
        key = (folder / "key.txt").read_text(encoding="utf-8")
        for label in ("Web photo DAT RealWebPhoto v4", "Web photo DAT2 BHI", "Clean photo Real-HAT-GAN", "Plain resize"):
            assert key.count(label) == 1, f"the key names '{label}' {key.count(label)} times"
        assert "failed" not in key, f"a model failed in the compare run:\n{key}"
        import shutil
        shutil.copytree(folder, keep / "compare", dirs_exist_ok=True)

        # faces: the made-up picture probably has none; the pipeline must still run and say so
        log = run_py(d, "--mode", "upscale", "--no-ai", "--faces", "--scale", "2", "--save-to", out / "faces")
        assert "faces restored" in log or "face restored" in log, "the log does not say how many faces were restored"
        with Image.open(only_file(out / "faces")) as r:
            assert r.size == (320, 240), f"face result is {r.size}, expected 320x240"

        # a real photo with a face (OpenCV's sample picture, downloaded here, never kept in the repo)
        import urllib.request
        faces_in = Path(t) / "real"
        faces_in.mkdir()
        urllib.request.urlretrieve(FACE_URL, faces_in / "face.jpg")
        log = run_py(faces_in, "--mode", "upscale", "--no-ai", "--faces", "--face-strength", "70",
                     "--scale", "2", "--save-to", out / "realface")
        import re
        m = re.search(r"(\d+) faces? restored", log)
        assert m and int(m.group(1)) >= 1, "no face was found and restored in the sample photo"
        only_file(out / "realface").replace(keep / "face_restored_x2.png")
        Image.open(faces_in / "face.jpg").save(keep / "face_original.png")
        log = run_py(faces_in, "--mode", "upscale", "--no-ai", "--faces", "--face-model", "codeformer",
                     "--face-strength", "70", "--scale", "2", "--save-to", out / "codeformer")
        m = re.search(r"(\d+) faces? restored", log)
        assert m and int(m.group(1)) >= 1, "CodeFormer: no face was found and restored in the sample photo"
        only_file(out / "codeformer").replace(keep / "face_codeformer_x2.png")

        # colouring: the same photo made black and white must come out coloured (and the same size)
        grey_in = Path(t) / "grey"
        grey_in.mkdir()
        with Image.open(faces_in / "face.jpg") as g:
            g.convert("L").convert("RGB").save(grey_in / "grey.png")
            grey_size = g.size
        log = run_py(grey_in, "--mode", "upscale", "--no-ai", "--colorize", "--scale", "2", "--save-to", out / "colour")
        assert "coloured" in log, "the black-and-white photo was not coloured"
        res = only_file(out / "colour")
        with Image.open(res) as r:
            assert r.size == (grey_size[0] * 2, grey_size[1] * 2), f"coloured result is {r.size}"
            sat = float(np.asarray(r.convert("HSV"))[..., 1].mean())
        assert sat > 10, f"the coloured result is still grey (mean saturation {sat:.1f} of 255)"
        res.replace(keep / "coloured_x2.png")

        # the face restorer itself: download, start, one 512x512 forward pass
        import blurnomore as b
        c = b.Context(b.Options(inputs=[str(d)], save_to=str(out / "ctx"), mode="upscale"), print, out / "ctx")
        for name in ("gfpgan", "codeformer"):
            sess = b.load_face(c, name)
            assert sess is not None, f"{name} could not be loaded"
            y = sess.run(None, b.face_inputs(sess, np.zeros((1, 3, 512, 512), np.float32)))[0]
            assert y.shape == (1, 3, 512, 512), f"{name} gave {y.shape}"
        print("Model check passed.")


if __name__ == "__main__":
    main()
