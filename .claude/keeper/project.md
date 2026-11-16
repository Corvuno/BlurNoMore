# The Keeper — BlurNoMore

The whole program is one file, `blurnomore.py`, started by `BlurNoMore.bat`. Its areas are parts
of that file.

## The areas

- **Choosing images** — which images from the folder go in: `list_images`, `choose_images`, the
  selection part of `_run` ("How many", "Same images as last run").
- **Image work** — what happens to each picture: `load_rgb`, `classify`, `find_content_box`
  (crop), Real-ESRGAN (`ensure_esrgan`, `run_esrgan`), `strengthen_thin`, `finish` (tone, colour,
  fade). The README's "Why it makes these choices" explains each.
- **Page and PDF** — `page_geometry`, placing on the A4 canvas, caching of pages, writing the PDF
  and its name, the log and report.
- **The window** — `launch_gui`, the `HELP` texts beside each setting, remembered settings.
- **Starting and setup** — `BlurNoMore.bat` (finds Python, installs packages), the Real-ESRGAN
  download.
- **Explaining** — `README.md`.

## What else to read, and when

- `build-rules.md` before building anything.
- What the owner settled so far is in `README.md` under "Why it makes these choices" and
  "Version history"; it came out of the conversations that built version 1 to 3.1. Newly settled
  points go there too, in the same plain words, and into this list if they are not about the
  program's behaviour: (none yet).
- `vision.md` is a draft until the owner settles it.

## The backlog

GitHub issues on Corvuno/BlurNoMore, with the labels `Refine`, `Next`, `Building now` and
`Parked`. Empty so far.
