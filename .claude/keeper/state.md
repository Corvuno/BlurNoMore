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
