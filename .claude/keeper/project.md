# The Keeper — BlurNoMore

The whole program is one file, `blurnomore.py`, started by `BlurNoMore.bat`. Its areas are parts
of that file.

## The areas

- **Choosing images** — which images go in: the image list (folders and files, drag and drop),
  `list_images`, the last-run list (`blurnomore_last_run.json`, "Same images as last run").
- **Image work** — what happens to each picture: `load_rgb`, `classify`, `find_content_box`
  (crop), Real-ESRGAN (`ensure_esrgan`, `run_esrgan`), `strengthen_thin`, `finish` (tone, colour,
  fade), the "Bigger images" output (`make_upscaled`, automatic model choice, DAT and extra models
  through onnxruntime, face restoration). The README's "Why it makes these choices" explains each.
- **Page and PDF** — `page_geometry`, placing on the A4 canvas, caching of pages, writing the PDF
  and its name, the log and report.
- **The window** — `build_window`/`launch_gui`, the `HELP` texts shown in the help box,
  remembered settings, the preview window (`make_preview`).
- **Starting and setup** — `BlurNoMore.bat` (finds Python, installs packages), the Real-ESRGAN
  download.
- **Explaining** — `README.md`.

## The Keeper's design readings

How the Keeper reads the product's design. These are its own readings, not rules the owner set; change one when
the owner's use shows it is wrong.

- **Run information goes in the log; the window holds the setup controls.** The owner looks at the log while a
  run works, not at the window. So progress, "still working" lines, FAILED lines and times go in the log. The
  window gets no extra status lines, spinners or counters. (4.8 put a status line with a spinner above the log;
  the owner asked why it was there, and 4.9.1 took it out.)

## The owner's own copy

The owner runs BlurNoMore from `D:\PortableApps\BlurNoMore` on their Windows PC. Every landing on main
also updates that folder while this project has folder access to it: copy the tracked files from main
over it (the .bat with CRLF line endings), nothing else. First compare: a local file that differs from
main and matches no older commit is the owner's own edit; report it and never overwrite it silently.
Never touch what the repo doesn't track: `settings/`, old `blurnomore_*.json`, `models/`, `realesrgan/`,
`seedvr2/` and anything else there is the owner's data. (2026-11-23: the owner had been downloading the
files by hand after every landing.)

## What else to read, and when

- `build-rules.md` before building anything.
- What the owner settled so far is in `README.md` under "Why it makes these choices" and
  "Version history"; it came out of the conversations that built version 1 to 3.1. Newly settled
  points go there too, in the same plain words, and into this list if they are not about the
  program's behaviour: (none yet).
- `vision.md` is a draft until the owner settles it.

## The backlog

GitHub issues on Corvuno/BlurNoMore, with the labels `Refine`, `Next`, `Building now` and
`Parked`.
