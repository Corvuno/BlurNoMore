# Where the product stands

The overview every conversation starts from. One screen; what is no longer true is removed.

## Strong

- Version 4.0 (landed 2026-11-16): one window, one flow. Add folders or pictures (drag and
  drop), say what kind they are or let it detect, choose options, then make bigger images
  (2x to 4x, named automatically or with your own text) or the A4 tracing PDF. "Run with
  preview" shows 4 images first; "Same images as last run" brings back the last list; "Save to"
  picks where results go.
- Photos get the DAT web-photo model on the owner's Radeon 7800XT (DirectML); drawings keep
  Real-ESRGAN. Face restoration (GFPGAN) is opt-in with a strength control.
- The PDF pages are byte-identical to 3.1 for the same settings. Every choice has its reason in
  the README.

## Weak, in the owner's words

No complaint heard yet on 4.0; nothing of it has been run on Windows yet.

## What the owner wants next, in their order

1. Try 4.0 on Windows and say what feels off.

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
