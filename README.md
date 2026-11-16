# BlurNoMore

You add folders or single pictures, say what they are, and choose what comes out. One of two things per run:

- **Bigger images**: a new folder of bigger copies (2x, 3x or 4x), restored by AI, for general use. No tracing steps, no A4.
- **Tracing PDF**: one sharp, print-ready A4 PDF that you can trace through tracing paper.

Windows, Python 3, one `.py` file, a settings window with the steps top to bottom: Images, What kind of images, Options, What comes out. Built in conversation with Claude, version 4.0.

## Why it exists

You collect drawings from the web and phone backups to learn to draw by tracing them. Those files are usually 500 to 1500 pixels wide, with soft lines, paper borders and a random shape. A4 at 300 dpi needs about 2480 x 3508 pixels. Printed straight, the lines are blurry, thin, and sometimes too light to see through tracing paper. Doing 40 or more images by hand in an editor is slow. BlurNoMore does the whole batch in one run.

## What it does to every image (Tracing PDF)

1. **Reads** the image (jpg, png, webp, bmp, tif) and fixes its rotation.
2. **Classifies** it: line art, shaded black and white, flat colour, or painted colour. This only decides which AI model and which tone treatment to use.
3. **Crops blank paper** around the drawing, but only if the image has a plain, light, neutral paper border (a scan or photo of a drawing). Pictures that run to the edge are never cropped.
4. **Upscales** with Real-ESRGAN on your GPU when the image is too small for A4 at the chosen dpi. Bigger images are only resized.
5. **Strengthens very thin dark lines**, so they show through tracing paper. Thick lines and colour fills are not touched.
6. **Tone work**: black and white art gets pure white paper and solid black ink; colour art stays in colour.
7. **Fits it on A4** (portrait or landscape, whichever suits the picture) with a margin, centred.
8. **Writes one PDF**, one image per page, named with the date and time so nothing is overwritten. Print at 100% ("Actual size").

## What it does to every image (Bigger images)

1. **Reads** the image and fixes its rotation. Nothing is cropped, thinned or placed on a page.
2. **Chooses a model** (Automatic) after deciding what the picture is ("Detect for each image", or you say Drawing or Photo): drawings and flat colour art use the Anime model; photos use the Web photo model (DAT) when onnxruntime can use your graphics card, otherwise the General model; black and white photos use the General model. The General model is safe on anything; the Anime model is used only where it clearly helps. You can force one, or use an extra model.
3. **Enlarges** with the chosen model (4x, or the native size of an extra model), then resizes to exactly 2x, 3x or 4x the original size. Without the AI it resizes plainly. Images that are already big are enlarged too.
4. **Restores faces** (optional, off by default): finds faces in the enlarged picture and restores each with GFPGAN, blended in by a strength from 0 to 100.
5. **Saves** `<name>_x<scale>.png` (or `.jpg`), or the name with your own text after or before it, into a new folder `BlurNoMore_upscaled_<date>_<time>`. Two images with the same name get `_2` on the second. A result over 100 megapixels is flagged in the log: very large file, slow.

**Run with preview** (the window; `--preview` on the command line makes picture sheets instead) works on 4 of your images (spread over the list) before the run: original next to result, plus a close-up of the busiest spot in both. Click a pair to see it big. For the PDF it shows the finished A4 page. "Go on with all" starts the real run, which reuses the preview's work; "Change settings" closes the preview. A preview writes no PDF and no output folder.

## Why it makes these choices

- **Real-ESRGAN, not a generative upscaler like Krea.** Real-ESRGAN restores what is in the image. A generative upscaler redraws and invents detail, which can change line weight and anatomy. For tracing, you want the artist's drawing, not a reinterpretation.
- **Two AI models.** Black and white art uses the anime model: crisp lines and flat areas. Coloured art uses the general model by default, because the anime model made painted and shaded art look smooth and plastic. You can switch this in the settings.
- **The thin-line boost is selective.** It finds dark lines narrower than about 7 px at print size, using a morphological closing, and widens only those. A blanket "make everything darker" would also clog hatching and fills.
- **The crop looks at paper, not brightness.** The first version cropped by luminance and cut into a yellow full-colour picture. Now it first checks that the border is plain, light, neutral paper. Then it trims 0.5% of the ink mass from each side, which ignores dust, scan edges and stray annotations while keeping thin frame lines.
- **No fading by default.** Lightening the print so your pencil stands out is a setting (0 by default). An early version lightened shaded art automatically, and it turned black areas grey.
- **"Whole picture" is the default fit.** Nothing is cut off, but a picture shaped differently from A4 leaves white bands. "Fill the page" avoids the bands and trims the edges.
- **Pages and AI results are cached**, keyed to the settings, so changing a setting only rebuilds what it affects. "Last run's images" puts the images of your last run back in the list, for comparing settings on the same test pages.

## Settings (all explained in the window)

**Images**

| Setting | Default | What it does |
|---|---|---|
| Images list (`INPUT...` on the command line) | empty | Folders (every picture directly in them, not the folders inside, sorted by name) and/or single picture files, added with "Add folder...", "Add files..." or by dragging onto the list. Duplicates are left out. They are never changed. The list is remembered |
| Last run's images (`--same`) | | Fills the list with the pictures of the last run (kept in `blurnomore_last_run.json`), so you can compare settings on the same pictures. On the command line it replaces the inputs |

**What kind**

| Setting | Default | What it does |
|---|---|---|
| Kind (`--kind`) | Detect for each image | Bigger images. `auto` looks at each picture; `drawing` or `photo` says they are all the same kind |
| Model (`--upscale-model`) | Automatic | Bigger images. Automatic follows the kind: drawings use Anime; photos use the Web photo model (DAT) when a graphics card runs it, else General. Or `general`, `anime`, `dat`, or the name of an extra model |
| Use AI upscaler (`--no-ai` turns it off) | on | Tracing PDF (shown in the window). Off = plain resizing, faster but blurrier. The command line applies it to both outputs |
| AI model for coloured art (`--model`) | General | Tracing PDF. General, or Anime |

**Options**

| Setting | Default | What it does |
|---|---|---|
| Enlarge (`--scale`) | 2x | Bigger images. 2, 3 or 4 times bigger |
| Restore faces (`--faces`) | off | Bigger images. Finds faces and restores them with GFPGAN. Needs onnxruntime and OpenCV (see Requirements) |
| Face strength (`--face-strength`) | 50 | 0 to 100: how much of the restored face is used. 0 leaves faces as they are. Greyed when faces are off |
| Crop empty paper (`--no-crop` turns it off) | on | Tracing PDF. Removes blank paper and scan borders |
| Strengthen thin lines (`--thicken`) | 70 | Tracing PDF. 0 = off, 100 = strongest |
| Lighten print (`--fade`) | 0 | Tracing PDF. Fades the print so your pencil stands out |
| Keep colours (`--grey` turns it off) | on | Tracing PDF. Off prints everything in grey |
| Colour boost (`--boost`) | 0 | Tracing PDF. More saturation and contrast for pale art |

**What comes out**

| Setting | Default | What it does |
|---|---|---|
| Make (`--mode`) | Tracing PDF | `upscale` = Bigger images, `pdf` = Tracing PDF. One of the two per run |
| Name (`--name-suffix TEXT`, `--name-prefix TEXT`) | Automatic | Bigger images. Automatic: `photo_x2.png`. Or your text after the name (`photo_big.png`) or before it (`big_photo.png`). Characters Windows does not allow are left out; an empty text means automatic |
| File type (`--format`) | PNG | Bigger images. `png` (exact, big) or `jpg` (quality 95, smaller) |
| Fit on the page (`--fill`) | Whole picture | Tracing PDF. Whole picture, or fill the page and trim the edges |
| Page margin (`--margin`) | 10 | Tracing PDF. White border on the A4 page, in mm |
| Save to (`--save-to`) | Next to the images | Where the PDF or the upscaled folder and `_blurnomore_work` go. "Next to the images" is the folder of the first folder or picture in the list (the log says where). Missing folders are made |

**More settings**

| Setting | Default | What it does |
|---|---|---|
| Print resolution (`--dpi`) | 300 | Tracing PDF. Page sharpness; 200 is smaller and faster |
| Redo all pages (`--redo`) | off | Ignores cached pages and results |
| Open the result when finished | on | Window only. Opens the PDF or the folder of bigger pictures |
| Preview (`--preview`) | | Command line: makes picture sheets for up to 4 images and stops. In the window it is the "Run with preview" button |

Groups that do not apply to the chosen output are hidden in the window. Point at a setting and its explanation shows in the box under the settings.

The same settings exist as command-line options. Run `python blurnomore.py --help` for the list.

## Files

- `blurnomore.py` is the program (window and command line).
- `BlurNoMore.bat` checks for Python, installs `pillow`, `numpy` and `img2pdf` (and, if it can, the optional `onnxruntime-directml`, `opencv-python-headless` and `tkinterdnd2`), and opens the window. Keep it next to the `.py`.
- In the folder you choose under "Save to" (by default the folder of the first image you added) it creates `BlurNoMore_A4_<date>_<time>_<n>p.pdf`, or for Bigger images a new folder `BlurNoMore_upscaled_<date>_<time>`, and a `_blurnomore_work` folder with caches, previews and a log of every run. An older `_tracing_work` folder is renamed automatically.
- `realesrgan` next to the program holds the downloaded Real-ESRGAN, so it is downloaded once (an older copy inside a `_blurnomore_work` folder is still found and used).
- `models` next to the program: the downloaded DAT and face models go here (first use only: the log says the size first). You can also put extra models there as `NAME.param` plus `NAME.bin` (ncnn format, as Real-ESRGAN ncnn uses) or as `NAME.onnx` (RGB 0..1 in and out). They show up as upscale models. A name with `2x`/`x2` or `3x`/`x3` is a 2x or 3x model; otherwise 4x (an `.onnx` model's size is checked with one small test picture, and the measured size is used if the name was wrong).
- `blurnomore_settings.json` next to the program remembers your last settings, including the list of images.
- `blurnomore_last_run.json` next to the program holds the images of your last run (for "Last run's images").

## Requirements

Windows with Python 3 ("Add to PATH" ticked), internet on the first AI run (about 45 MB download from GitHub), and a GPU with Vulkan support for the AI step. Without a GPU it falls back to plain resizing and says so in the log.

Optional, installed by the `.bat` when possible: `tkinterdnd2` (drag and drop of pictures and folders onto the list; without it use the buttons), `onnxruntime-directml` (runs on any DirectX 12 graphics card, falls back to the CPU) and `opencv-python-headless`. Without them everything else works, and the log says what to install. The last two give you:

- **Web photo model (DAT)**, `4xRealWebPhoto_v4` by Philip Hofmann (CC-BY-4.0), about 49 MB, downloaded on first use from Hugging Face. Made for compressed web photos. Automatic uses it for photos only when a graphics card runs it; `--upscale-model dat` uses it anywhere, but on the CPU it takes minutes per photo.
- **Face restoration**: OpenCV's YuNet face finder (230 KB) and GFPGAN 1.4 (about 340 MB, from the facefusion model collection on Hugging Face), downloaded when you first switch faces on.

The "Models" check on GitHub downloads these and runs them on made-up pictures; the normal Checks never download models.

## Known limits

- Real-ESRGAN cannot recover detail that isn't there. Images that need more than 6x enlargement are flagged in the log and will still look soft.
- The crop is conservative. Scans with shadows or tinted borders may not be cropped at all.
- The thin-line thresholds are tuned on a handful of drawings. If lines come out too heavy or too light, change "Strengthen thin lines".
- Not tested: macOS, Linux, and Tk 8.6 layout details on Windows. Everything else was tested on your drawings, or on test images.
- If something fails, press "Copy log" and paste it: the log contains settings, versions and full error traces.

## Version history

- 4.0: new window in steps (Images, What kind of images, Options, What comes out): add folders and single pictures (buttons or drag and drop), say what they are or let it detect, then make either bigger images or a tracing PDF. "Run with preview" (4 images, click for a big view, then "Go on with all") and "Run". Bigger images: 2x/3x/4x, model per picture (Web photo DAT, extra models), optional face restoration, your own text before or after the file name, PNG or JPG. "Save to" folder, "Last run's images", Real-ESRGAN kept next to the program, default save place is the folder of the first image. Replaces the single folder and "How many images".
- 3.1: renamed to BlurNoMore; work folder renamed.
- 3.0: settings window, crop only trims real paper, "Fit on the page", "Same images as last run", unique file names, copyable log.
- 2.x: thin-line boost, colour kept for colour art, trial mode, `.bat` launcher.
- 1.x: first script: classify, upscale, fit to A4, one PDF.
