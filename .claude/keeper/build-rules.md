# Build rules

**Language and layout.** Python 3, one file: `blurnomore.py`. Only Pillow, numpy and img2pdf
(plus the standard library and Tk for the window). A new package is added in three places at
once: the `pip install` line in `BlurNoMore.bat`, the import guard at the top of the program and
the README. Windows comes first; it should keep running on Linux for the checks.

**How it is written.**
- The owner's images are only read. Output goes to a new date-stamped PDF or `_blurnomore_work`.
- A setting lives in four places that stay in step: the `Options` class, the window (with a
  `HELP` text in plain words), the command-line options, and the README's settings table.
- Any setting that changes how a page looks goes into `key_fields` in `_run`, or old cached
  pages are reused by mistake.
- Everything a run says goes through the log function, so "Copy log" and the log file hold it.
  Errors in one image are logged and the run goes on with the next.
- Texts in the window and log are short, plain and say what to do.
- Each landed change raises `VERSION` and adds a line to the README's version history.

**Checks.**
- Quick: `python -m py_compile blurnomore.py tests/smoke.py` and `python tests/smoke.py` (makes a
  PDF from three made-up drawings without AI). The "Checks" workflow runs both on every push.
- For a change in image work or page layout: run it on sample images and look at the pages
  (render the PDF pages to pictures). The AI step needs Windows and a GPU and cannot run here; say
  so rather than claim it works.
- For a change in the window: it can only be seen roughly here (Tk on Linux looks different);
  say what was and was not seen.

**Review checklist (the checker).**
- Originals untouched; no file overwritten that the owner may have seen.
- A new or changed setting is in all four places, and in `key_fields` if it changes pages.
- A failure in one image does not stop the run; the log says what went wrong.
- No new package without the three places above.
- Nothing outside the item changed: other settings, defaults, texts, the look of the pages.
