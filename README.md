# BlurNoMore

Turns a folder of small, blurry drawings into one sharp, print-ready A4 PDF that you can trace through tracing paper.

Windows, Python 3, one `.py` file, a settings window. Built in conversation with Claude, version 3.1.

## Why it exists

You collect drawings from the web and phone backups to learn to draw by tracing them. Those files are usually 500 to 1500 pixels wide, with soft lines, paper borders and a random shape. A4 at 300 dpi needs about 2480 x 3508 pixels. Printed straight, the lines are blurry, thin, and sometimes too light to see through tracing paper. Doing 40 or more images by hand in an editor is slow. BlurNoMore does the whole batch in one run.

## What it does to every image

1. **Reads** the image (jpg, png, webp, bmp, tif) and fixes its rotation.
2. **Classifies** it: line art, shaded black and white, flat colour, or painted colour. This only decides which AI model and which tone treatment to use.
3. **Crops blank paper** around the drawing, but only if the image has a plain, light, neutral paper border (a scan or photo of a drawing). Pictures that run to the edge are never cropped.
4. **Upscales** with Real-ESRGAN on your GPU when the image is too small for A4 at the chosen dpi. Bigger images are only resized.
5. **Strengthens very thin dark lines**, so they show through tracing paper. Thick lines and colour fills are not touched.
6. **Tone work**: black and white art gets pure white paper and solid black ink; colour art stays in colour.
7. **Fits it on A4** (portrait or landscape, whichever suits the picture) with a margin, centred.
8. **Writes one PDF**, one image per page, named with the date and time so nothing is overwritten. Print at 100% ("Actual size").

## Why it makes these choices

- **Real-ESRGAN, not a generative upscaler like Krea.** Real-ESRGAN restores what is in the image. A generative upscaler redraws and invents detail, which can change line weight and anatomy. For tracing, you want the artist's drawing, not a reinterpretation.
- **Two AI models.** Black and white art uses the anime model: crisp lines and flat areas. Coloured art uses the general model by default, because the anime model made painted and shaded art look smooth and plastic. You can switch this in the settings.
- **The thin-line boost is selective.** It finds dark lines narrower than about 7 px at print size, using a morphological closing, and widens only those. A blanket "make everything darker" would also clog hatching and fills.
- **The crop looks at paper, not brightness.** The first version cropped by luminance and cut into a yellow full-colour picture. Now it first checks that the border is plain, light, neutral paper. Then it trims 0.5% of the ink mass from each side, which ignores dust, scan edges and stray annotations while keeping thin frame lines.
- **No fading by default.** Lightening the print so your pencil stands out is a setting (0 by default). An early version lightened shaded art automatically, and it turned black areas grey.
- **"Whole picture" is the default fit.** Nothing is cut off, but a picture shaped differently from A4 leaves white bands. "Fill the page" avoids the bands and trims the edges.
- **Pages and AI results are cached**, keyed to the settings, so changing a setting only rebuilds what it affects. "Same images as last run" repeats your last selection, for comparing settings on the same test pages.

## Settings (all explained in the window)

| Setting | Default | What it does |
|---|---|---|
| Folder | your drawings folder | Where the images are; the PDF is saved here too |
| How many images | All | All, or N images, spread evenly or the first N |
| Same images as last run | off | Repeats the previous selection in this folder |
| Fit on the page | Whole picture | Whole picture, or fill the page and trim the edges |
| Page margin (mm) | 10 | White border on the A4 page |
| Print resolution (dpi) | 300 | Page sharpness; 200 is smaller and faster |
| Crop empty paper | on | Removes blank paper and scan borders |
| Strengthen thin lines | 70 | 0 = off, 100 = strongest |
| Lighten print | 0 | Fades the print so your pencil stands out |
| Keep colours | on | Off prints everything in grey |
| Colour boost | 0 | More saturation and contrast for pale art |
| Use AI upscaler | on | Off = plain resizing, faster but blurrier |
| AI model for coloured art | General | General, or Anime |
| Redo all pages | off | Ignores cached pages |

The same settings exist as command-line options. Run `python blurnomore.py --help` for the list.

## Files

- `blurnomore.py` is the program (window and command line).
- `BlurNoMore.bat` checks for Python, installs `pillow`, `numpy` and `img2pdf`, and opens the window. Keep it next to the `.py`.
- In your images folder it creates `BlurNoMore_A4_<date>_<time>_<n>p.pdf` and a `_blurnomore_work` folder with the downloaded Real-ESRGAN, caches and a log of every run. An older `_tracing_work` folder is renamed automatically.
- `blurnomore_settings.json` next to the program remembers your last settings.

## Requirements

Windows with Python 3 ("Add to PATH" ticked), internet on the first AI run (about 45 MB download from GitHub), and a GPU with Vulkan support for the AI step. Without a GPU it falls back to plain resizing and says so in the log.

## Known limits

- Real-ESRGAN cannot recover detail that isn't there. Images that need more than 6x enlargement are flagged in the log and will still look soft.
- The crop is conservative. Scans with shadows or tinted borders may not be cropped at all.
- The thin-line thresholds are tuned on a handful of drawings. If lines come out too heavy or too light, change "Strengthen thin lines".
- Not tested: macOS, Linux, and Tk 8.6 layout details on Windows. Everything else was tested on your drawings, or on test images.
- If something fails, press "Copy log" and paste it: the log contains settings, versions and full error traces.

## Version history

- 3.1: renamed to BlurNoMore; work folder renamed.
- 3.0: settings window, crop only trims real paper, "Fit on the page", "Same images as last run", unique file names, copyable log.
- 2.x: thin-line boost, colour kept for colour art, trial mode, `.bat` launcher.
- 1.x: first script: classify, upscale, fit to A4, one PDF.
