"""Smoke check: builds a PDF from three made-up drawings without the AI step.
Run:  python tests/smoke.py   (needs pillow, numpy, img2pdf). Exits non-zero on failure.
Leaves nothing behind: everything happens in a temporary folder."""
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


def main():
    with tempfile.TemporaryDirectory() as t:
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
        print("Smoke check passed.")


if __name__ == "__main__":
    main()
