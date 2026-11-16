# Where the product stands

The overview every conversation starts from. One screen; what is no longer true is removed.

## Strong

- Version 3.1 does the whole batch in one run: crop paper, upscale, strengthen thin lines, tone,
  fit on A4, one PDF. A run on made-up drawings (no AI) works on Linux.
- Every choice so far has a reason written down (README, "Why it makes these choices").

## Weak, in the owner's words

No complaint heard yet. The owner likes the upscaling. The README's "Known limits" were written by
Claude, not them.

## What the owner wants next, in their order

1. The upscaling on its own, also for photos: plain 2x to 4x for general use, any kind of
   picture, the kind detected automatically, extra models with controls. Building now (4.0).
2. Previews: up to 4 images before the run; for the PDF, sharpness, line thickness and
   placement. Building now (4.0).
3. A "Save to" setting, and a window that is laid out logically ("now all over the place").

Part A of 4.0 (engine, command line, README, neutral default folder) is built and tested
without AI on the work branch. Part B (the window, the preview window) waits on the answers below.

## Open with the owner

- Faces: a face model redraws faces, which clashes with "restored, never redrawn". Off-by-default
  switch, automatic, or leave out? (asked 2026-11-16)
- Preview flow: a separate Preview button with "Run all", or Start always previews first?
- The window sketch: https://claude.ai/artifact/LVvzcW2r57g5V46xfax8rY (not chosen yet).

## Noticed, not yet raised or settled

- Real-ESRGAN is downloaded as the Windows build only, so on a Mac the AI step always falls back.
- The mouse wheel is bound to the settings panel for the whole window; over the log it may scroll
  both. Not checked on Windows. To fix with the window rework.
- Nothing has been run on Windows from here: the window, the .bat and the AI step are untested by
  the Keeper.
