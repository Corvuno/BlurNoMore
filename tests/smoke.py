"""Smoke check: builds a PDF from three made-up drawings without the AI step, then an upscaled
copy of each into another folder, then the preview sheets.
Run:  python tests/smoke.py   (needs pillow, numpy, img2pdf). Exits non-zero on failure.
Leaves nothing behind: everything happens in a temporary folder."""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent


def make_images(d):
    # thin line art on a wide paper border (should be cropped and classed lineart)
    im = Image.new("RGB", (900, 1200), (246, 244, 240))
    dr = ImageDraw.Draw(im)
    for i in range(12):
        dr.line((250 + i * 30, 300, 400 + i * 20, 900), fill=(40, 40, 40), width=1)
    dr.ellipse((300, 350, 600, 650), outline=(20, 20, 20), width=2)
    im.save(d / "a_lineart.png")
    # full-bleed colour picture, landscape (must not be cropped)
    im = Image.new("RGB", (1000, 700), (230, 200, 40))
    dr = ImageDraw.Draw(im)
    dr.rectangle((100, 100, 500, 500), fill=(200, 40, 60))
    dr.ellipse((550, 150, 900, 600), fill=(30, 90, 200))
    im.save(d / "b_color.jpg", quality=92)
    # tiny image: large enlargement
    Image.new("RGB", (120, 160), (255, 255, 255)).save(d / "c_tiny.png")


def snapshot(d):
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in d.iterdir() if p.is_file()}


def run_py(*args):
    r = subprocess.run([sys.executable, str(ROOT / "blurnomore.py"), *map(str, args)], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    assert r.returncode == 0, "blurnomore.py exited with an error"
    assert "ERROR" not in r.stdout, "an image failed"
    return r


def main():
    with tempfile.TemporaryDirectory() as t, tempfile.TemporaryDirectory() as t_out, tempfile.TemporaryDirectory() as t_out2:
        d = Path(t)
        make_images(d)
        r = subprocess.run([sys.executable, str(ROOT / "blurnomore.py"), str(d), "--no-ai", "--dpi", "150"],
                           capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-3000:])
        pdfs = list(d.glob("BlurNoMore_A4_*_3p.pdf"))
        assert r.returncode == 0, "blurnomore.py exited with an error"
        assert len(pdfs) == 1, "expected one 3-page PDF"
        assert "ERROR" not in r.stdout, "an image failed"
        assert "cropped" in r.stdout.split("a_lineart")[1].split("\n")[1], "line art was not cropped"
        assert "cropped" not in r.stdout.split("b_color")[1].split("\n")[1], "colour art was cropped"

        # upscale mode into another folder: 3 files at exactly 2x, nothing added to the images folder
        names = {"a_lineart.png": (900, 1200), "b_color.jpg": (1000, 700), "c_tiny.png": (120, 160)}
        before_files = {p.name for p in d.iterdir()}
        before = snapshot(d)
        out, out2 = Path(t_out), Path(t_out2)
        run_py(d, "--mode", "upscale", "--no-ai", "--scale", "2", "--save-to", out)
        folders = list(out.glob("BlurNoMore_upscaled_*"))
        assert len(folders) == 1, "expected one upscaled folder"
        made = sorted(folders[0].iterdir())
        assert len(made) == 3, "expected 3 upscaled files"
        for p in made:
            src = next(n for n in names if Path(n).stem == p.stem.rsplit("_x", 1)[0])
            with Image.open(p) as im:
                assert im.size == (names[src][0] * 2, names[src][1] * 2), f"{p.name} is not exactly 2x"
        assert {p.name for p in d.iterdir()} == before_files, "the images folder gained or lost files"

        # preview: one sheet per image (up to 4), only into the work folder
        run_py(d, "--mode", "upscale", "--no-ai", "--preview", "--save-to", out)
        sheets = list((out / "_blurnomore_work" / "previews").rglob("*.png"))
        assert 1 <= len(sheets) <= 3, "expected up to 3 preview sheets"
        assert len(list(out.glob("BlurNoMore_upscaled_*"))) == 1, "the preview made an output folder"
        run_py(d, "--no-ai", "--dpi", "150", "--preview", "--save-to", out2)
        assert 1 <= len(list((out2 / "_blurnomore_work" / "previews").rglob("*.png"))) <= 3, "no PDF preview sheets"
        assert not list(out2.glob("*.pdf")), "the preview made a PDF"

        # the originals are untouched by all runs (the first PDF run only added its PDF and work folder)
        after = snapshot(d)
        assert all(after[n] == h for n, h in before.items()), "an original image changed"
        print("Smoke check passed.")


if __name__ == "__main__":
    main()
