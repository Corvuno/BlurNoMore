# Where the product stands

The overview every conversation starts from. One screen; what is no longer true is removed.

## Strong

- Version 4.0 (landed 2026-11-16): one window, one flow. Add folders or pictures (drag and
  drop), say what kind they are or let it detect, choose options, then make bigger images
  (2x to 4x, named automatically or with your own text) or the A4 tracing PDF. "Run with
  preview" shows 4 images first; "Same images as last run" brings back the last list; "Save to"
  picks where results go.
- 4.1 (landed 2026-11-17): on the owner's Radeon 7800XT (DirectML), Automatic gives photos DAT
  and drawings AnimeSharp V4 (personal use only, warned in window and log). By hand: Real-HAT-GAN
  (clean photos), HEART (new, experimental). Faces opt-in: GFPGAN (natural) or CodeFormer
  (strongest), with strength. "Colour black-and-white photos" (DDColor) is opt-in. The PDF keeps
  Real-ESRGAN. The GitHub Models check runs every real model, including two real faces.
- 4.2 (landed 2026-11-17): Enlarge also to a width or height in pixels ("→ 1920 px wide").
  Bigger pictures keep their own size unless "Also make bigger pictures smaller" is on.
- 4.3 (landed 2026-11-18): "Make" moved up to step 2, under the images, as the owner asked;
  the window now reads Images, What to make, What kind, Options, What comes out.
- 4.4 (landed 2026-11-22, #6): more photo models (newest DAT2 BHI, DRCT-L; converted to ONNX by the
  "Convert models" workflow, hosted on release models-1; ATD left out: its export did not match), an
  opt-in inventing upscaler (Stable Diffusion x4 ONNX, settings: prompt, negative, noise, steps,
  guidance, seed) and Make "Compare models": shuffled letters, close-up sheets, key.png to open last.
  Automatic is unchanged until the owner picks a winner.
- 4.5 (landed 2026-11-22, #7): the inventing upscaler is SeedVR2 3B (fp8), replacing Stable Diffusion x4
  (owner: "Ditch it"). First use sets up `seedvr2/` next to the program: uv, Python 3.12 venv, the pinned
  numz tool, AMD's PyTorch for Windows (ROCm 7.2.1; NVIDIA cu128; else CPU), then the 3.9 GB model.
  Settings: seed, colour matching, two noise sliders, save graphics memory. Never run for real by the
  Keeper (blocked downloads, no AMD card); the owner's first run is the real test.
- 4.6 (landed 2026-11-22, #8): "AI files folder" (More settings / --ai-dir) puts models, realesrgan,
  seedvr2 and uv's Python + cache on another disk; moved at the next run (copy, then delete); refuses a
  target inside the moved folders or among the pictures. Not tried on real Windows disks.
- 4.7 (landed 2026-11-22, #9): JPG quality setting and opt-in "Limit file size" (MB): lower JPG quality
  first, a PNG over the limit becomes JPG, smaller picture only as last resort; Compare decides per picture.
- 4.7.1 (landed 2026-11-23): mouse wheel over an open dropdown list no longer throws KeyError 'popdown'.
- 4.8 (landed 2026-11-23, #10): run log is alive: [HH:MM:SS] on every line, "piece N of M, X so far" and
  "still working, X so far" lines (no finish-time guesses, by the owner's wish), a model with no progress
  for 10 min is skipped with a copyable "FAILED: <model>: <reason>" line, picture lines show the size and
  are clickable (also result names and the DONE folder), Stop works mid-picture and mid-download, status
  line above the log. Real-ESRGAN's guard assumes it prints progress; not checked on Windows.
- 4.8.1 (landed 2026-11-23): owner's Compare run (DAT, DirectML) closed the whole program ~14 s in. Cause
  (likely, not reproduced here): 4.8 ran each tile's session.run on a new thread, session made on another. Now
  all onnxruntime work (make + run) is on one dedicated thread; faulthandler writes a hard crash into the run
  log; the .bat now pauses on a negative (crash) exit code too.
- 4.9 (landed 2026-11-23): settings/ folder next to the program holds blurnomore_settings.json and
  blurnomore_last_run.json (old files moved there at start); the settings file is nested by window section in
  window order; the window saves settings about a second after a change and on close (owner: tidy, easy to
  back up/move; exact place didn't matter).
- 4.9.1 (landed 2026-11-23): status line with spinner above the log removed; run info lives in the log
  (see project.md, design readings).
- 4.10 (landed 2026-11-23): owner's Compare ran ~33 s per DAT piece and hatgan/heart hit DirectML "Not enough
  memory": DirectML took adapter 0 = the Ryzen 7800X3D's built-in Radeon, not the 7800XT. Now asks DirectML for
  high_performance (option names checked in onnxruntime-directml 1.24.4's DLL), times cards if that is refused, logs
  the card's name; setting Graphics card (More settings, --gpu) overrides. Compare releases each model after its
  picture, keeps results in _blurnomore_work/compare_unfinished at once (with a key). FAILED = one line in the
  window, full trace only in the log file; Copy log copies the file. Not yet confirmed on the owner's PC.
- 4.10.1 (landed 2026-11-23): 4.10 picked the RX 7800 XT on the owner's PC, but everything ran out of memory
  because the old 4.9.1 Compare was still running (owner confirmed). Now each run/preview leaves a note in
  settings/running; a new run warns in its first log lines when another is alive. Out-of-memory FAILED lines say
  so; Compare logs one "N of M could not make this picture" line per picture.
- 4.11 (landed 2026-11-23): owner's PC stutters (choppy video calls) while a run has the 7800XT at full load.
  Pause/Resume next to Stop (after the current piece, logged, not kept after closing); "Use at most"
  100/75/50/25% by the run buttons, live during a run (rests piece time x (100/pct - 1); Real-ESRGAN/SeedVR2 rest
  after each picture), default under More settings; below-normal process priority (children inherit on Windows).
  Not tried on Windows yet. Note: process priority only helps the processor, not the graphics card.
- The PDF pages are byte-identical to 3.1 for the same settings. Every choice has its reason in
  the README.

## Weak, in the owner's words

No complaint heard yet on 4.0 to 4.2; nothing of it has been run on Windows yet.

## What the owner wants next, in their order

1. Try 4.3 on Windows and say what feels off.
2. Photo/art to shaded drawing for tracing (#5, Refine): waiting for the owner's examples.

## Open with the owner

- Nothing asked. Later ideas the owner raised: separate saved image lists ("or there has to be
  an option for different lists to be held"); other models only "if they are brilliant".

## Noticed, not yet raised or settled

- Real-ESRGAN is downloaded as the Windows build only, so on a Mac the AI step always falls back.
- Nothing has been run on Windows from here: the window, the .bat, drag and drop, DirectML and
  Real-ESRGAN are untested by the Keeper. The real DAT and face models run weekly on GitHub's
  processor (Models workflow), including one real face.
- The face blend (mask size, feather) is untuned: only seen on one sample photo.
- DAT builds the whole 4x picture in memory: fine for old phone photos, heavy for 24 MP photos.
- With "Redo all pages" on, "Run with preview" then "Go on with all" does the preview images twice.
