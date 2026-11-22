"""Turns upscale models that are only published as PyTorch weights into ONNX files that
BlurNoMore can run on any graphics card (DirectML). Run by the "Convert models" workflow on
GitHub; the results are attached to the release "models-1" and downloaded from there.

Each model is exported for fixed square tiles (BlurNoMore feeds full tiles of the size the model has), then
checked: the ONNX file must give the same picture as the original PyTorch model."""
import gc
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from spandrel import ModelLoader

HF = "https://huggingface.co/Phips/"
MODELS = {   # output name: (source weights, tile size); all by Philip Hofmann, CC-BY-4.0
    "4xBHI_dat2_real": (HF + "4xBHI_dat2_real/resolve/main/4xBHI_dat2_real.safetensors", 256),
    # DRCT-L at 256 px ran a 16 GB machine out of memory, so it gets 128 px tiles
    "4xRealWebPhoto_v4_drct-l": (HF + "4xRealWebPhoto_v4_drct-l/resolve/main/4xRealWebPhoto_v4_drct-l.safetensors", 128),
    # ATD (4xRealWebPhoto_v3_atd) exports but gives a different picture (its token dictionary does not survive the
    # export: largest difference 0.45), so it is not offered.
}

# ATD sorts with stable=True, which the ONNX exporter cannot write; plain sort gives the same order
# when there are no ties (the check below compares the result with the original model anyway).
_sort, _tsort = torch.sort, torch.Tensor.sort
torch.sort = lambda *a, stable=None, **k: _sort(*a, **k)
torch.Tensor.sort = lambda self, *a, stable=None, **k: _tsort(self, *a, **k)
out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "converted")
out_dir.mkdir(exist_ok=True)
only = set(sys.argv[2:])
failed = []
for name, (url, TILE) in MODELS.items():
    if only and name not in only:
        continue
    print(f"== {name}", flush=True)
    try:
        src = out_dir / (name + ".safetensors")
        if not src.exists():
            urllib.request.urlretrieve(url, src)
        desc = ModelLoader().load_from_file(str(src))
        print(f"   architecture {desc.architecture.name}, scale {desc.scale}", flush=True)
        net = desc.model.eval().float()
        x = torch.rand(1, 3, TILE, TILE)
        dst = out_dir / f"{name}_fp32_{TILE}.onnx"
        with torch.no_grad():
            ref = net(x).numpy()
            print("   PyTorch ran one tile; exporting", flush=True)
            torch.onnx.export(net, x, str(dst), opset_version=17, input_names=["input"],
                              output_names=["output"], dynamo=False)
        scale = desc.scale
        del net, desc
        gc.collect()
        print("   exported; checking with onnxruntime", flush=True)
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False      # DRCT-L ran the runner out of memory with the arena on
        so.enable_mem_pattern = False
        sess = ort.InferenceSession(str(dst), so, providers=["CPUExecutionProvider"])
        t = time.time()
        got = sess.run(None, {"input": x.numpy()})[0]
        diff = float(np.abs(got - ref).max())
        print(f"   ONNX {dst.stat().st_size / 1e6:.0f} MB, output {got.shape}, "
              f"largest difference {diff:.2e}, one tile {time.time() - t:.1f} s on CPU", flush=True)
        if got.shape != (1, 3, TILE * scale, TILE * scale) or diff > 2e-3:
            raise RuntimeError("the ONNX file does not match the original model")
        src.unlink()
    except Exception as e:   # one model failing must not stop the others
        print(f"   FAILED: {type(e).__name__}: {e}", flush=True)
        failed.append(name)
print("failed:", failed or "none")
sys.exit(1 if failed else 0)
