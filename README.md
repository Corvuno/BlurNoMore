# BlurNoMore

You add folders or single pictures, say what they are, and choose what comes out. One of three things per run:

- **Bigger images**: a new folder of bigger copies (2x, 3x or 4x), restored by AI, for general use. No tracing steps, no A4.
- **Tracing PDF**: one sharp, print-ready A4 PDF that you can trace through tracing paper.
- **Compare models**: the first 4 pictures made with every model you tick, as unlabelled pictures A, B, C ..., so you can choose a model by eye without bias.

Windows, Python 3, one `.py` file, a settings window with the steps top to bottom: Images, What to make, What kind of images, Options, What comes out. Built in conversation with Claude, version 4.7.

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
2. **Chooses a model** (Automatic) after deciding what the picture is ("Detect for each image", or you say Drawing or Photo): drawings and flat colour art use the AnimeSharp V4 model when onnxruntime can use your graphics card, otherwise the Anime model; photos use the Web photo model (DAT) when onnxruntime can use your graphics card, otherwise the General model; black and white photos use the General model. The General model is safe on anything; the Anime model is used only where it clearly helps. You can force one (including the other photo models Web photo DAT2 BHI, DRCT-L and ATD, the clean-photo model Real-HAT-GAN and the new, experimental photo model HEART, which are never chosen automatically; and the inventing model, see below), or use an extra model.
3. **Colours black-and-white photos** (optional, off by default): only photos that are black and white get colour, before the enlargement. The colours are invented by the model, a guess and not the real ones. Colour photos and drawings are never touched.
4. **Enlarges** with the chosen model (4x, or the native size of an extra model), then resizes to exactly 2x, 3x or 4x the original size, or to the width or height in pixels you ask for (aspect ratio kept). Without the AI it resizes plainly. Images that are already big are enlarged too, unless you ask for a size in pixels: then pictures already that wide or high keep their own size (colour and faces still apply when on; the log says so), or with "Also make bigger pictures smaller" are made smaller to it (plain resizing, no AI). The AI is only used when the picture grows by more than 1.25 times; a growth over 6 times is flagged in the log as low source resolution.
5. **Restores faces** (optional, off by default): finds faces in the enlarged picture and restores each with GFPGAN (natural) or CodeFormer (strongest repair), blended in by a strength from 0 to 100.
6. **Saves** `<name>_x<scale>.png` (`_1920w` or `_1080h` for a size in pixels; or `.jpg`), or the name with your own text after or before it, into a new folder `BlurNoMore_upscaled_<date>_<time>`. Two images with the same name get `_2` on the second. A result over 100 megapixels is flagged in the log: very large file, slow.

## What it does (Compare models)

1. Takes the first 4 pictures of the list (the log says so when there are more).
2. Makes each one with every model you tick, exactly as Bigger images would with that model forced (same size, faces, colour and file type), plus plain resizing (Lanczos) as an extra contestant, so you see what the AI adds. Models that are not downloaded yet are downloaded first.
3. Saves them into a new folder `BlurNoMore_compare_<date>_<time>` as `<picture>_A.png`, `<picture>_B.png` ... The letters are shuffled at random for every run (the same letter is the same model for every picture of that run). No model name is in these files or their names.
4. `<picture>_closeups.png` shows the same 256 x 256 close-up (the most detailed spot of the picture) for every letter side by side, labelled with the letter only, next to the original's crop enlarged and labelled "Original".
5. `key.png` and `key.txt` say which letter is which model, how long it took, and the settings. The log never says which letter is which model: "Open key.png only after you have chosen." A model that fails for a picture leaves its letter out for that picture; the key says why and the run goes on.

"Run with preview" is not offered for Compare models: it already is a look-first run.

**Run with preview** (the window; `--preview` on the command line makes picture sheets instead) works on 4 of your images (spread over the list) before the run: original next to result, plus a close-up of the busiest spot in both. Click a pair to see it big. For the PDF it shows the finished A4 page. "Go on with all" starts the real run, which reuses the preview's work; "Change settings" closes the preview. A preview writes no PDF and no output folder.

## Why it makes these choices

- **Real-ESRGAN, not a generative upscaler like Krea.** Real-ESRGAN restores what is in the image. A generative upscaler redraws and invents detail, which can change line weight and anatomy. For tracing, you want the artist's drawing, not a reinterpretation.
- **The inventing upscaler is opt-in.** SeedVR2 3B (ByteDance Seed, 2025) restores and enlarges a picture in one step and makes up detail that is not in it, so faces and text can come out wrong, and it takes minutes per picture. It is only used when you pick "Inventing: SeedVR2 3B" as the Model (or tick it for Compare models); Automatic never picks it. It only restores: it does not make pictures from text (a generating model may come later). It replaces Stable Diffusion x4, which was too old. It is run by numz's command line tool in its own Python environment, because it needs PyTorch. Its settings (seed, how it keeps the colours, input noise, latent noise, save graphics memory) show only when it is the model.
- **No photo model is crowned best.** "Best" was never checked, so the labels do not say it. Automatic still picks DAT RealWebPhoto v4 for photos when a graphics card runs it, and that does not change until you have compared. Compare models, blind, is how to choose: the letters hide the models until you have chosen.
- **Models for different pictures.** DAT for old, compressed or damaged photos. Real-HAT-GAN for clean, sharp photos where you want crisp detail. AnimeSharp V4 for drawings, because it is very faithful to the lines. Real-ESRGAN stays as the model that works anywhere without extra downloads. For faces, GFPGAN looks natural and CodeFormer repairs badly damaged faces the most. Both face models redraw the face, which is why faces are opt-in and blended by a strength you choose. HEART (November 2026) is brand new and less proven; it is there to try, never chosen automatically.
- **Colouring is opt-in.** It redraws the colours of a black-and-white photo with a guess, so it is off until you switch it on, and it only touches photos that are black and white.
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
| Last run's images (`--same`) | | Fills the list with the pictures of the last run (kept in `settings\blurnomore_last_run.json`), so you can compare settings on the same pictures. On the command line it replaces the inputs |

**What to make**

| Setting | Default | What it does |
|---|---|---|
| Make (`--mode`, `--compare`) | Tracing PDF | `upscale` = Bigger images, `pdf` = Tracing PDF, `compare` = Compare models. One of the three per run. It comes right after the images because it decides which settings below are shown |
| Models to compare (`--compare MODEL,MODEL,...`) | dat, bhi, drct, hatgan, heart, general | Compare models. One checkbox per model of the Model list (not Automatic); the inventing model is unticked. On the command line `--compare dat,bhi,...` sets the mode to compare, `--compare all` means every model (also the inventing one and your extra models); `--mode compare` alone uses the default list. Only models that exist are used. Enlarge, faces, colour and file type apply to every model alike. Kind, Model and name are not used |

**What kind**

| Setting | Default | What it does |
|---|---|---|
| Kind (`--kind`) | Detect for each image | Bigger images. `auto` looks at each picture; `drawing` or `photo` says they are all the same kind |
| Model (`--upscale-model`) | Automatic | Bigger images. Automatic follows the kind: drawings use AnimeSharp V4 when a graphics card runs it, else Anime; photos use the Web photo model (DAT) when a graphics card runs it, else General. Or `general`, `anime`, `animesharp`, `dat` (Web photo DAT RealWebPhoto v4), `bhi` (Web photo DAT2 BHI), `drct` (Web photo DRCT-L), `hatgan`, `heart`, `seedvr2` (Inventing: SeedVR2 3B), or the name of an extra model. AnimeSharp V4 is free for personal use only: don't sell or publish results made with it commercially |
| Use AI upscaler (`--no-ai` turns it off) | on | Both outputs. Off = plain resizing, faster but blurrier |
| AI model for coloured art (`--model`) | General | Tracing PDF. General, or Anime |

**Options**

| Setting | Default | What it does |
|---|---|---|
| Enlarge (`--scale`, `--width N`, `--height N`, `--shrink-bigger`) | 2x | Bigger images. 2, 3 or 4 times bigger, or Width (pixels) / Height (pixels) with the number next to it (default 1920, 100 to 20000): every picture gets exactly that width or height, the other side follows. Pictures already that big keep their size unless "Also make bigger pictures smaller" (`--shrink-bigger`, off by default) is on. `--width` and `--height` exclude each other and win over `--scale` |
| Restore faces (`--faces`) | off | Bigger images. Finds faces and restores them with the face model. Needs onnxruntime and OpenCV (see Requirements) |
| Face model (`--face-model`) | Natural (GFPGAN) | Bigger images. `gfpgan` = natural, `codeformer` = strongest repair for badly damaged faces. Only the one you pick is downloaded, and only when a face is found. Greyed when faces are off |
| Face strength (`--face-strength`) | 50 | 0 to 100: how much of the restored face is used. 0 leaves faces as they are. Greyed when faces are off |
| Colour black-and-white photos (`--colorize`) | off | Bigger images. Colours photos that are black and white with the DDColor model (about 980 MB, downloaded when the first one is met). The colours are a guess, not the real ones. Colour photos and drawings are not touched |
| Crop empty paper (`--no-crop` turns it off) | on | Tracing PDF. Removes blank paper and scan borders |
| Strengthen thin lines (`--thicken`) | 70 | Tracing PDF. 0 = off, 100 = strongest |
| Lighten print (`--fade`) | 0 | Tracing PDF. Fades the print so your pencil stands out |
| Keep colours (`--grey` turns it off) | on | Tracing PDF. Off prints everything in grey |
| Colour boost (`--boost`) | 0 | Tracing PDF. More saturation and contrast for pale art |

**Inventing upscaler** (a group shown only when the Model is "Inventing: SeedVR2 3B", or it is ticked for Compare models)

| Setting | Default | What it does |
|---|---|---|
| Seed (`--sv-seed`) | 42 | The same seed gives the same result. 0 = a new random seed each run; the log names the seed used |
| Keep the colours (`--sv-color`) | lab | How it keeps the colours of your picture: lab (recommended), wavelet, wavelet_adaptive, hsv, adain or none |
| Input noise (`--sv-input-noise`) | 0 | 0 to 1. Adds variation before it starts: higher = it makes up more |
| Latent noise (`--sv-latent-noise`) | 0 | 0 to 1. Adds variation while it works: higher = it makes up more |
| Save graphics memory (`--sv-low-memory`) | off | Slower, but fits a card with less memory. Turn it on if it runs out of graphics memory |

**What comes out**

| Setting | Default | What it does |
|---|---|---|
| Name (`--name-suffix TEXT`, `--name-prefix TEXT`) | Automatic | Bigger images. Automatic: `photo_x2.png`. Or your text after the name (`photo_big.png`) or before it (`big_photo.png`). Characters Windows does not allow are left out; an empty text means automatic |
| File type (`--format`) | PNG | Bigger images. `png` (exact, big) or `jpg` (smaller; see JPG quality) |
| JPG quality (`--jpg-quality N`) | 95 | Bigger images and Compare models, when the file type is JPG (greyed out for PNG). 50 to 100: higher = bigger file and closer to perfect; 90 to 95 looks the same as the original to the eye |
| Limit file size (`--max-mb N`) | off | Bigger images and Compare models. Tick the box and set the most MB a picture may take (default 5 when switched on, 0.5 to 200; on the command line 0 = off). A JPG that is too big gets the highest quality that fits (down to 50); if even that is too big, the picture is made smaller. A PNG that is too big is saved as a JPG instead (the file ends in `.jpg`). The log says when the limit acted; in Compare the key says it per letter. Off = nothing changes. The tracing PDF is not affected |
| Fit on the page (`--fill`) | Whole picture | Tracing PDF. Whole picture, or fill the page and trim the edges |
| Page margin (`--margin`) | 10 | Tracing PDF. White border on the A4 page, in mm |
| Save to (`--save-to`) | Next to the images | Where the PDF or the upscaled folder and `_blurnomore_work` go. "Next to the images" is the folder of the first folder or picture in the list (the log says where). Missing folders are made |

**More settings**

| Setting | Default | What it does |
|---|---|---|
| Print resolution (`--dpi`) | 300 | Tracing PDF. Page sharpness; 200 is smaller and faster |
| AI files folder (`--ai-dir FOLDER`) | Next to the program | Where the big AI files are kept: `realesrgan`, `models` and `seedvr2` (with its Python and PyTorch), about 8 GB at most with SeedVR2. Choose a folder on another disk to keep the program's disk free. When you change it, the files already there are moved to the new folder at the next run (the log says so; a folder of that name that already exists in the new place is left alone, and a folder that cannot be moved stays where it is for that run). The program itself and your pictures stay where they are |
| Redo all pages (`--redo`) | off | Ignores cached pages and results |
| Open the result when finished | on | Window only. Opens the PDF or the folder of bigger pictures |
| Preview (`--preview`) | | Command line: makes picture sheets for up to 4 images and stops. In the window it is the "Run with preview" button |

Groups that do not apply to the chosen output are hidden in the window. Point at a setting and its explanation shows in the box under the settings.

The same settings exist as command-line options. Run `python blurnomore.py --help` for the list.

## Files

- `blurnomore.py` is the program (window and command line).
- `BlurNoMore.bat` checks for Python, installs `pillow`, `numpy` and `img2pdf` (and, if it can, the optional `onnxruntime-directml`, `opencv-python-headless` and `tkinterdnd2`), and opens the window. Keep it next to the `.py`.
- In the folder you choose under "Save to" (by default the folder of the first image you added) it creates `BlurNoMore_A4_<date>_<time>_<n>p.pdf`, or for Bigger images a new folder `BlurNoMore_upscaled_<date>_<time>`, and a `_blurnomore_work` folder with caches, previews and a log of every run. An older `_tracing_work` folder is renamed automatically.
- `realesrgan` (in the AI files folder, by default next to the program) holds the downloaded Real-ESRGAN, so it is downloaded once (an older copy inside a `_blurnomore_work` folder is still found and used).
- `models` (in the AI files folder): the downloaded photo models, face models and (in `models/seedvr2`) the inventing model's files go here (first use only: the log says the size first). You can also put extra models there as `NAME.param` plus `NAME.bin` (ncnn format, as Real-ESRGAN ncnn uses) or as `NAME.onnx` (RGB 0..1 in and out). They show up as upscale models. A name with `2x`/`x2` or `3x`/`x3` is a 2x or 3x model; otherwise 4x (an `.onnx` model's size is checked with one small test picture, and the measured size is used if the name was wrong).
- The `settings` folder next to the program holds what it remembers, so you can back it up or move it in one go:
  - `blurnomore_settings.json` keeps your settings, including the list of images, grouped in the order of the window's sections. It is saved about a second after you change something, and when you close the window.
  - `blurnomore_last_run.json` holds the images of your last run (for "Last run's images").

  Versions before 4.9 kept these two files next to the program; they are moved into `settings` the first time 4.9 starts.

## Requirements

Windows with Python 3 ("Add to PATH" ticked), internet on the first AI run (about 45 MB download from GitHub), and a GPU with Vulkan support for the AI step. Without a GPU it falls back to plain resizing and says so in the log.

Optional, installed by the `.bat` when possible: `tkinterdnd2` (drag and drop of pictures and folders onto the list; without it use the buttons), `onnxruntime-directml` (runs on any DirectX 12 graphics card, falls back to the CPU) and `opencv-python-headless`. Without them everything else works, and the log says what to install. The last two give you:

- **Web photo model (DAT)**, `4xRealWebPhoto_v4` by Philip Hofmann (CC-BY-4.0), about 49 MB, downloaded on first use from Hugging Face. Made for compressed web photos. Automatic uses it for photos only when a graphics card runs it; `--upscale-model dat` uses it anywhere, but on the CPU it takes minutes per photo.
- **Drawing model AnimeSharp V4** (`--upscale-model animesharp`), `2x-AnimeSharpV4` by Kim2091 (CC BY-NC-SA 4.0: personal, non-commercial use), about 31 MB, from Hugging Face. Very faithful to the lines. It enlarges 2x; for 3x and 4x it runs twice and resizes to the exact size. Automatic uses it for drawings when a graphics card runs it.
- **Clean photo model Real-HAT-GAN** (`--upscale-model hatgan`), HAT by XPixel, ONNX by FaceFusion, about 160 MB, from Hugging Face. Sharp on clean photos; used only when you pick it. Slow on the CPU.
- **Web photo models BHI and DRCT-L** (`--upscale-model bhi`, `drct`): `4xBHI_dat2_real` (DAT2) and `4xRealWebPhoto_v4_drct-l` (DRCT) by Philip Hofmann (CC-BY-4.0), converted to ONNX by `tools/convert_models.py` (the "Convert models" workflow, which checks each converted file against the original) and downloaded on first use from the GitHub release `models-1` (sizes: BHI about 51 MB, DRCT-L about 189 MB). 4x; BHI works in 256 px tiles, DRCT-L in 128 px tiles (it needs much memory). Used only when you pick them (or in Compare models). Slow on the CPU. His ATD model (4xRealWebPhoto_v3_atd) was tried too, but its converted file did not give the same picture as the original, so it is left out.
- **Inventing upscaler** (`--upscale-model seedvr2`): SeedVR2 3B by ByteDance Seed (Apache-2.0), run by `inference_cli.py` of numz's ComfyUI-SeedVR2_VideoUpscaler (Apache-2.0, pinned to version 2.5.24, commit `4490bd1`). Only when you first use it, the program sets it up in the folder `seedvr2` in the AI files folder (`uv` keeps its own copy of Python 3.12 and its download cache in `seedvr2/uv`, so nothing big lands in your user folder; your pictures are never touched): the `uv` package goes into the program's own Python, `uv` makes a separate Python 3.12 environment, the tool's files are downloaded from GitHub, and PyTorch is installed for your graphics card (about 3 GB; AMD's PyTorch for Windows, ROCm 7.2.1, needs Windows 11 and AMD graphics driver 26.2.2 or newer; NVIDIA: PyTorch for CUDA 12.8; otherwise PyTorch for the processor only, which is extremely slow). Then the tool downloads its own model files from Hugging Face into `models/seedvr2` (about 3.9 GB: the 3B model 3.4 GB and the VAE 0.5 GB). The log shows each step. If a step fails the log says which, that picture is only resized, and the next run tries again. It restores by making up detail, is slow (minutes per picture), and changes the picture (faces and text can come out wrong). It makes the short side of the picture the size you asked for directly; above 2160 px it is very slow and may run out of graphics memory (turn on Save graphics memory). Delete the folder `seedvr2` to remove it. It does not make pictures from text.
- **Photo model HEART** (`--upscale-model heart`), by Philip Hofmann (Apache-2.0), about 75 MB, from Hugging Face. Brand new (November 2026) and less proven; 4x, used only when you pick it. Slow on the CPU.
- **Colouring black-and-white photos** (`--colorize`): DDColor (piddnad, Apache-2.0), ONNX by FaceFusion, about 980 MB, downloaded the first time a black-and-white photo is met.
- **Face restoration**: OpenCV's YuNet face finder (230 KB) and one of two restorers from the facefusion model collection on Hugging Face: GFPGAN 1.4 (about 340 MB, natural) or CodeFormer (about 377 MB, strongest repair; it is run with a fidelity weight of 0.6). Only the one you choose is downloaded, when you first restore a face with it.

The "Models" check on GitHub downloads these and runs them on made-up pictures; the normal Checks never download models.

## Known limits

- Real-ESRGAN cannot recover detail that isn't there. Images that need more than 6x enlargement are flagged in the log and will still look soft.
- The crop is conservative. Scans with shadows or tinted borders may not be cropped at all.
- The thin-line thresholds are tuned on a handful of drawings. If lines come out too heavy or too light, change "Strengthen thin lines".
- Not tested: macOS, Linux, and Tk 8.6 layout details on Windows. Everything else was tested on your drawings, or on test images.
- **Is it still working?** Every line of the log starts with the time, like `[11:45:14]`. Each picture gets a line with its size, `[1/4] 1004809.jpg (4000x3000)`; in the window it is a link that opens the original, and so are the name of each saved result and the folder in the DONE line. A slow model reports what it is doing and how long it has taken so far, never a guess of when it ends: `dat: piece 37 of 240, 3 min 10 s so far` (at most every 10 seconds, and not at all for a quick step); Real-ESRGAN and SeedVR2 say `still working, 2 min 30 s so far`; a big download says how many MB it has. The first run on the graphics card prepares the model, which can take a minute. If a model shows no sign of progress for 10 minutes (no piece finished, nothing printed by Real-ESRGAN or SeedVR2), it is given up: the log says `FAILED: dat made no progress for 10 min (stuck at piece 38 of 240); skipping it for the rest of this run`, the picture is plain resized (in Compare that letter shows as failed in the key) and the run goes on. Every model failure has one `FAILED: <model>: <reason>` line you can copy. "Stop" also ends a picture in the middle of a model's work or a download (a stopped picture is not saved).
- If something fails, press "Copy log" and paste it: the log contains settings, versions and full error traces.

## Version history

- 4.8: you can see that a run is alive: every log line starts with the time, each picture line shows its size and opens the original when clicked (so do the saved results and the output folder), a status line above the log, progress lines (what and how long so far) from slow models, Real-ESRGAN, SeedVR2 and downloads, a model with no progress for 10 minutes is given up with a `FAILED:` line, and Stop now also works in the middle of a picture or a download. The tracing PDF and the pictures are unchanged.
- 4.8.1: fixes the program closing about 14 seconds into a run on the graphics card: 4.8 ran each piece of a picture on a new thread, and DirectML wants its model made and run on one thread; now all model work runs on one thread. If the program ever dies hard again, the black window stays open with the error code, and Python's crash trace is written at the end of that run's log.
- 4.9: the remembered settings and the last run's list live in a `settings` folder next to the program (moved there automatically), the settings file is grouped in the order of the window's sections, and settings are saved as soon as you change them instead of only when you press Run.
- 4.9.1: the status line with the turning bar above the log is gone: what a run is doing is in the log (time on every line, progress and "still working" lines), and the window is for setting things up.
- 4.7.1: scrolling the mouse wheel over an open dropdown list no longer fills the black window with errors.
- 4.7: new settings JPG quality (`--jpg-quality`) and Limit file size (`--max-mb`) for Bigger images and the Compare letters: a picture that would be bigger than the limit gets a lower JPG quality (not below 50), then is made smaller; a PNG that is too big becomes a JPG. Off by default: the files are the same as in 4.6. The tracing PDF is unchanged.
- 4.6: new setting AI files folder (window: More settings; `--ai-dir`): keeps `realesrgan`, `models` and `seedvr2` (Python, PyTorch, uv's Python and cache) on another disk if you want. When you change it, the files are moved there at the next run, and SeedVR2 sets its Python up again once (a set-up made by 4.5 kept its download cache in the user folder, so the first rebuild after moving downloads PyTorch, about 3 GB, again; from 4.6 on the cache moves along). A new folder that is inside the folders being moved or among your pictures is refused. The tracing PDF is unchanged.
- 4.5: the inventing upscaler is now SeedVR2 3B (restores by making up detail, opt-in, sets itself up the first time in the folder `seedvr2`) instead of Stable Diffusion x4, with its own settings: seed, how it keeps the colours, input noise, latent noise, save graphics memory. Automatic never picks it. The tracing PDF is unchanged.
- 4.4: two more web photo models (Web photo DAT2 BHI and DRCT-L, converted to ONNX), an inventing upscaler (Stable Diffusion x4, opt-in, with its own settings: description, how much it may invent, steps, guidance, seed), and a new Make choice "Compare models": the first 4 pictures made with every model you tick, plus plain resizing, as shuffled letters A, B, C ... with close-up sheets and a key.png that says which is which, so you can choose without bias. The DAT label no longer says "best". The tracing PDF is unchanged.
- 4.3: "Make" (Tracing PDF or Bigger images) moved up to step 2, right under the images, because it decides which settings below it are shown. Nothing else changed.
- 4.2: Bigger images can be made to a width or height in pixels (for example 1920) instead of 2x/3x/4x; pictures already that big keep their size unless you choose to make them smaller; names end `_1920w` or `_1080h`. The tracing PDF is unchanged.
- 4.1: more upscale models (AnimeSharp V4 for drawings, Real-HAT-GAN for clean photos, the new HEART; Automatic uses AnimeSharp for drawings when a graphics card runs it, with a note that it is for personal use only), a choice of face model (Natural GFPGAN or Strongest repair CodeFormer) and optional colouring of black-and-white photos (DDColor). The tracing PDF is unchanged.
- 4.0: new window in steps (Images, What kind of images, Options, What comes out): add folders and single pictures (buttons or drag and drop), say what they are or let it detect, then make either bigger images or a tracing PDF. "Run with preview" (4 images, click for a big view, then "Go on with all") and "Run". Bigger images: 2x/3x/4x, model per picture (Web photo DAT, extra models), optional face restoration, your own text before or after the file name, PNG or JPG. "Save to" folder, "Last run's images", Real-ESRGAN kept next to the program, default save place is the folder of the first image. Replaces the single folder and "How many images".
- 3.1: renamed to BlurNoMore; work folder renamed.
- 3.0: settings window, crop only trims real paper, "Fit on the page", "Same images as last run", unique file names, copyable log.
- 2.x: thin-line boost, colour kept for colour art, trial mode, `.bat` launcher.
- 1.x: first script: classify, upscale, fit to A4, one PDF.
