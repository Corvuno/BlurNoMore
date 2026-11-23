"""Smoke check: builds a PDF from three made-up drawings without the AI step, then an upscaled
copy of each into another folder (from a folder, from a file plus a folder, and from the last run's list),
then the preview sheets and a Compare models run without the AI.
Run:  python tests/smoke.py   (needs pillow, numpy, img2pdf). Exits non-zero on failure.
Leaves nothing behind: everything happens in a temporary folder."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
LAST_RUN = ROOT / "settings" / "blurnomore_last_run.json"      # every run writes it


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
    saved = LAST_RUN.read_bytes() if LAST_RUN.exists() else None     # put back what was there
    try:
        check()
    finally:
        if saved is None:
            LAST_RUN.unlink(missing_ok=True)
            try:
                LAST_RUN.parent.rmdir()          # only when empty
            except OSError:
                pass
        else:
            LAST_RUN.write_bytes(saved)
    check_settings_file()


def check_settings_file():
    """The settings live in settings/ next to the program, ordered by the window's sections; the old flat files next to
    the program are moved there and still read."""
    sys.path.insert(0, str(ROOT))
    import dataclasses
    import blurnomore as bn
    names = [n for _, ns in bn.SETTINGS_SECTIONS for n in ns]
    assert len(names) == len(set(names)), "a setting is in two sections"
    missing = {f.name for f in dataclasses.fields(bn.Options)} - set(names)
    assert not missing, f"settings without a section: {missing}"
    saved = bn.APP_DIR, bn.SETTINGS_DIR, bn.SETTINGS_FILE, bn.LAST_RUN_FILE
    with tempfile.TemporaryDirectory() as t:
        app = Path(t)
        bn.APP_DIR, bn.SETTINGS_DIR = app, app / "settings"
        bn.SETTINGS_FILE, bn.LAST_RUN_FILE = bn.SETTINGS_DIR / "blurnomore_settings.json", bn.SETTINGS_DIR / "blurnomore_last_run.json"
        try:
            (app / "blurnomore_settings.json").write_text(json.dumps({"scale": 3, "mode": "upscale", "ai_dir_used": "D:/ai",
                                                                      "something_old": 1}), encoding="utf-8")
            (app / "blurnomore_last_run.json").write_text('["x.png"]', encoding="utf-8")
            bn.move_old_settings()
            assert not (app / "blurnomore_settings.json").exists() and not (app / "blurnomore_last_run.json").exists()
            assert bn.read_last_run() == ["x.png"]
            o = bn.load_settings()
            assert o.scale == 3 and o.mode == "upscale", "the old flat file was not read"
            o.save_to = "E:/out"
            bn.save_settings(o)
            data = json.loads(bn.SETTINGS_FILE.read_text(encoding="utf-8"))
            assert list(data) == [title for title, _ in bn.SETTINGS_SECTIONS], list(data)
            for title, ns in bn.SETTINGS_SECTIONS[:-1]:
                assert list(data[title]) == list(ns), (title, list(data[title]))
            assert data["Remembered by the program"] == {"ai_dir_used": "D:/ai"}, "old keys of earlier versions stay behind"
            o2 = bn.load_settings()
            assert dataclasses.asdict(o2) == dataclasses.asdict(o), "settings changed on the way through the file"
            assert bn.ai_dir_used() == Path("D:/ai")
            bn.remember_ai_dir_used(app / "ai")
            assert bn.load_settings().save_to == "E:/out" and bn.ai_dir_used() == app / "ai"
            assert sorted(p.name for p in bn.SETTINGS_DIR.iterdir()) == ["blurnomore_last_run.json", "blurnomore_settings.json"]
            (app / "blurnomore_settings.json").write_text("{}", encoding="utf-8")
            bn.move_old_settings()
            assert (app / "blurnomore_settings.json").exists(), "an old file overwrote the newer one in settings/"
        finally:
            bn.APP_DIR, bn.SETTINGS_DIR, bn.SETTINGS_FILE, bn.LAST_RUN_FILE = saved
    print("settings folder ok")


def check():
    sys.path.insert(0, str(ROOT))
    import blurnomore as bn
    # file names: automatic, text after, text before; characters Windows does not allow are dropped
    o = bn.Options(scale=3)
    assert bn.result_name(o, "pic", "png", ROOT) == "pic_x3.png"
    o = bn.Options(scale=3, name_mode="suffix", name_text=' _big?"<>|:*\\/ ')
    assert bn.result_name(o, "pic", "png", ROOT) == "pic_big.png", bn.result_name(o, "pic", "png", ROOT)
    o = bn.Options(name_mode="prefix", name_text="big_")
    assert bn.result_name(o, "pic", "jpg", ROOT) == "big_pic.jpg"
    assert bn.result_name(bn.Options(name_mode="suffix", name_text="  "), "pic", "png", ROOT) == "pic_x2.png"

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

        # a file plus the folder it is in: each picture once, named with the owner's prefix; originals in the list order
        out3 = out / "three"
        run_py(d / "c_tiny.png", d, "--mode", "upscale", "--no-ai", "--name-prefix", "big_", "--save-to", out3)
        made3 = sorted(p.name for p in next(out3.glob("BlurNoMore_upscaled_*")).iterdir())
        assert made3 == ["big_a_lineart.png", "big_b_color.png", "big_c_tiny.png"], made3
        assert json.loads(LAST_RUN.read_text(encoding="utf-8"))[0].endswith("c_tiny.png"), "last run list not written in order"
        # the last run's images again, with no inputs, JPG with a suffix
        out4 = out / "four"
        run_py("--same", "--mode", "upscale", "--no-ai", "--name-suffix", "_web", "--format", "jpg", "--save-to", out4)
        made4 = sorted(p.name for p in next(out4.glob("BlurNoMore_upscaled_*")).iterdir())
        assert made4 == ["a_lineart_web.jpg", "b_color_web.jpg", "c_tiny_web.jpg"], made4
        assert {p.name for p in d.iterdir()} == before_files, "the images folder gained or lost files"

        # a size in pixels: smaller pictures grow to exactly that wide with the aspect kept; names end _300w
        # pictures already wider than 300 keep their own size, or are made smaller with --shrink-bigger
        for extra, expect in (((), {"a_lineart.png": (900, 1200), "b_color.jpg": (1000, 700), "c_tiny.png": (300, 400)}),
                              (("--shrink-bigger",), {"a_lineart.png": (300, 400), "b_color.jpg": (300, 210), "c_tiny.png": (300, 400)})):
            out6 = out / f"six{len(extra)}"
            r6 = run_py(d, "--mode", "upscale", "--no-ai", "--width", "300", *extra, "--save-to", out6)
            assert ("kept its size" in r6.stdout) == (not extra), "the log note about kept sizes is wrong"
            made6 = list(next(out6.glob("BlurNoMore_upscaled_*")).iterdir())
            assert len(made6) == 3 and all(p.stem.endswith("_300w") for p in made6), [p.name for p in made6]
            for p in made6:
                src = next(n for n in names if Path(n).stem == p.stem.rsplit("_", 1)[0])
                with Image.open(p) as im:
                    assert im.size == expect[src], f"{p.name} is {im.size}, expected {expect[src]}"
        out7 = out / "seven"
        run_py(d, "--mode", "upscale", "--no-ai", "--height", "200", "--save-to", out7)
        sizes7 = {p.name: Image.open(p).size for p in next(out7.glob("BlurNoMore_upscaled_*")).iterdir()}
        assert sizes7 == {"a_lineart_200h.png": (900, 1200), "b_color_200h.png": (1000, 700), "c_tiny_200h.png": (150, 200)}, sizes7
        assert {p.name for p in d.iterdir()} == before_files, "the images folder gained or lost files"

        # file size limit and JPG quality: a noisy made-up photo whose PNG is far over 1 MB
        import numpy as np
        dn = out / "noisy"
        dn.mkdir()
        rng = np.random.default_rng(1)
        base = np.kron(rng.integers(0, 256, (30, 45, 3)), np.ones((20, 20, 1)))
        Image.fromarray(np.clip(base + rng.normal(0, 40, base.shape), 0, 255).astype("uint8")).save(dn / "n.png")
        MB = 1024 * 1024

        def one(tag, *extra, inp=dn):
            o_ = out / f"limit_{tag}"
            r_ = run_py(inp, "--mode", "upscale", "--no-ai", "--scale", "2", *extra, "--save-to", o_)
            fs = list(next(o_.glob("BlurNoMore_upscaled_*")).iterdir())
            assert len(fs) == 1, fs
            return fs[0], r_.stdout
        p_off, _ = one("off")
        assert p_off.suffix == ".png" and p_off.stat().st_size > 1.5 * MB, "the made-up photo is not big enough"
        p_off0, _ = one("off0", "--max-mb", "0")
        assert p_off0.read_bytes() == p_off.read_bytes(), "limit off changed the PNG"
        p_png, log_png = one("png", "--max-mb", "1")
        assert p_png.suffix == ".jpg" and p_png.stat().st_size <= MB and "PNG too big: JPG q" in log_png, (p_png, log_png)
        p_j95, log_j95 = one("j95", "--format", "jpg")
        assert p_j95.stat().st_size > MB, "the JPG at 95 should be over 1 MB"
        p_j70, _ = one("j70", "--format", "jpg", "--jpg-quality", "70")
        assert p_j70.stat().st_size < p_j95.stat().st_size, "quality 70 is not smaller than 95"
        p_jl, log_jl = one("jl", "--format", "jpg", "--max-mb", "1")
        assert p_jl.stat().st_size <= MB and "JPG q" in log_jl and "to fit 1 MB" in log_jl, log_jl
        q_jl = Image.open(p_jl).quantization[0][0]
        assert q_jl > Image.open(p_j95).quantization[0][0], "the limit did not lower the quality"
        assert "made smaller" not in log_jl, "made smaller although a lower quality was enough"
        p_tiny, log_tiny = one("tiny", "--format", "jpg", "--max-mb", "0.1")
        assert p_tiny.stat().st_size <= 0.1 * MB and "made smaller to" in log_tiny, (p_tiny.stat().st_size, log_tiny)
        assert Image.open(p_tiny).width < 1800, "not made smaller"
        # a roomy limit does nothing: same bytes as without it
        p_roomy, log_roomy = one("roomy", "--format", "jpg", "--max-mb", "50")
        assert p_roomy.read_bytes() == p_j95.read_bytes() and "to fit" not in log_roomy
        # names: a .jpg already there is not overwritten when a PNG turns into a JPG
        o_c = out / "limit_clash"
        (o_c).mkdir()
        sys.path.insert(0, str(ROOT))
        im_n = Image.open(dn / "n.png").convert("RGB")
        (o_c / "n_x2.jpg").write_bytes(b"owner")
        saved, note = bn.save_result(im_n, o_c / "n_x2", bn.Options(max_mb=0.2), lambda *_: None)
        assert saved.name == "n_x2_2.jpg" and (o_c / "n_x2.jpg").read_bytes() == b"owner" and note, (saved, note)
        # the shrink lands close under the limit, not far below it
        assert 0.6 * 0.1 * MB < p_tiny.stat().st_size <= 0.1 * MB, p_tiny.stat().st_size
        # a group (the letters of one Compare picture) gets one type, one quality, one size: a smooth and a noisy letter
        o_g = out / "limit_group"
        o_g.mkdir()
        smooth = Image.new("RGB", im_n.size, (120, 130, 140))
        for lim, expect in ((0.3, "jpg"), (0.02, "jpg")):
            res = bn.save_group([(smooth, o_g / f"s{lim}_A"), (im_n, o_g / f"s{lim}_B")], bn.Options(max_mb=lim), lambda *_: None)
            assert len({r[0].suffix for r in res}) == 1 and res[0][0].suffix == ".jpg", res
            ims = [Image.open(r[0]) for r in res]
            assert ims[0].size == ims[1].size and ims[0].quantization[0] == ims[1].quantization[0], "letters differ"
            assert all(r[0].stat().st_size <= lim * MB for r in res) and res[0][1] == res[1][1] and res[0][1], res
        # Compare: every letter fits
        o_cmp = out / "limit_cmp"
        run_py(dn, "--compare", "general", "--no-ai", "--format", "jpg", "--max-mb", "0.5", "--scale", "2", "--save-to", o_cmp)
        fc = next(o_cmp.glob("BlurNoMore_compare_*"))
        lets = [x for x in fc.glob("n_[AB].*")]
        assert len(lets) == 1 and all(x.stat().st_size <= 0.5 * MB for x in lets), lets
        assert "File size limit: 0.5 MB" in (fc / "key.txt").read_text(encoding="utf-8")

        # preview: one sheet per image (up to 4), only into the work folder
        run_py(d, "--mode", "upscale", "--no-ai", "--preview", "--save-to", out)
        sheets = list((out / "_blurnomore_work" / "previews").rglob("*.png"))
        assert 1 <= len(sheets) <= 3, "expected up to 3 preview sheets"
        assert len(list(out.glob("BlurNoMore_upscaled_*"))) == 1, "the preview made an output folder"
        run_py(d, "--no-ai", "--dpi", "150", "--preview", "--save-to", out2)
        assert 1 <= len(list((out2 / "_blurnomore_work" / "previews").rglob("*.png"))) <= 3, "no PDF preview sheets"
        assert not list(out2.glob("*.pdf")), "the preview made a PDF"

        # Compare models without the AI: the folder, one shuffled letter per picture (the plain resize), close-up sheets and the
        # key (which says the model could not be used); the log does not say which letter is which. (tests/fake_models_check.py
        # does this with fake models.)
        out8 = out / "eight"
        r8 = run_py(d, "--compare", "general", "--no-ai", "--save-to", out8)
        folder8 = next(out8.glob("BlurNoMore_compare_*"))
        letter_files = sorted(p.name for p in folder8.iterdir() if p.stem[-2:-1] == "_" and p.stem[-1] in "AB")
        assert len(letter_files) == 3 and len({p[-5] for p in letter_files}) == 1, letter_files
        assert len(list(folder8.glob("*_closeups.png"))) == 3 and (folder8 / "key.png").is_file(), "closeups or key.png missing"
        key8 = (folder8 / "key.txt").read_text(encoding="utf-8")
        assert "General photo (Real-ESRGAN): failed:" in key8 and "Plain resize" in key8, key8
        assert "General photo" not in r8.stdout and "Open key.png only after you have chosen." in r8.stdout, "the log names a model"
        for p in folder8.glob("*_[AB].png"):
            with Image.open(p) as im:
                assert im.size == tuple(2 * v for v in names[next(n for n in names if Path(n).stem == p.stem[:-2])]), p.name
                assert not im.info, "extra data in a compare picture"
        assert {p.name for p in d.iterdir()} == before_files, "the images folder gained or lost files"

        # the originals are untouched by all runs (the first PDF run only added its PDF and work folder)
        after = snapshot(d)
        assert all(after[n] == h for n, h in before.items()), "an original image changed"
        print("Smoke check passed.")


if __name__ == "__main__":
    main()
