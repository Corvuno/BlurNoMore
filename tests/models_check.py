"""Model check: downloads the real DAT, YuNet and GFPGAN models (about 390 MB, into models/ next
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

        # DAT model: downloaded on first use, 2x output
        log = run_py(d, "--mode", "upscale", "--upscale-model", "dat", "--scale", "2", "--save-to", out / "dat")
        assert "AI dat" in log, "the DAT model was not used"
        with Image.open(only_file(out / "dat")) as r:
            assert r.size == (320, 240), f"DAT result is {r.size}, expected 320x240"

        # faces: the made-up picture probably has none; the pipeline must still run and say so
        log = run_py(d, "--mode", "upscale", "--no-ai", "--faces", "--scale", "2", "--save-to", out / "faces")
        assert "faces restored" in log or "face restored" in log, "the log does not say how many faces were restored"
        with Image.open(only_file(out / "faces")) as r:
            assert r.size == (320, 240), f"face result is {r.size}, expected 320x240"

        # the face restorer itself: download, start, one 512x512 forward pass
        import blurnomore as b
        c = b.Context(b.Options(folder=str(d), save_to=str(out / "ctx"), mode="upscale"), print, out / "ctx")
        sess = b.load_gfpgan(c)
        assert sess is not None, "GFPGAN could not be loaded"
        y = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 3, 512, 512), np.float32)})[0]
        assert y.shape == (1, 3, 512, 512), f"GFPGAN gave {y.shape}"
        print("Model check passed.")


if __name__ == "__main__":
    main()
