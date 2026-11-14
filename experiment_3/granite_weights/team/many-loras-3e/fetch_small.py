#!/usr/bin/env python3
"""Track "the denoised LoRA" (many-loras-3e), script 10: a small local copy of the non-holdout dense means. Weights only.

The six dense means of pass 3 (dW and L; imperial half 0, half 1, modern) are 13 GB each on HF. This script reads only
a few LAYERS of them with HTTP range requests (nothing else is downloaded) and writes a reduced data set that
mean_structure.py, compare_parts.py and lens.py run on unchanged, on a CPU:
  <out>/<stem>.<type>.pt     tensor [n_layers, E, out, in], float16 (rounding adds 1e-7 of the energy: far below the
                             sampling noise of the means); layer i of the reduced set = real layer LAYERS[i]
  <out>/sv.npy, e_parts.npy  the per-LoRA module columns of the chosen layers;  names.json;  layers.json
  <out>/granite_small.safetensors   the base model's tensors that the lens needs (embedding, norm gains, router rows),
                             also by range requests (155 MB of the 6.6 GB)
Check: for every module of the chosen layers, <half 0, half 1> of the fetched imperial dW must equal e_imp of the
full run (vm/mean_structure_nh.npz), and the same for L (vm/mean_structure_nhL.npz).
Usage:  python fetch_small.py --layers 0,10,20,26,30,31 --out DIR [--types q,k,v,o,gu,dn]
"""
import argparse, json, os, struct, time, zipfile
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="0,10,20,26,30,31")
ap.add_argument("--out", required=True)
ap.add_argument("--types", default="q,k,v,o,gu,dn")
ap.add_argument("--repo", default="mild-rgb/lora-factory-experiments")
ap.add_argument("--path", default="experiment_3/granite_weights/dense/nh")
ap.add_argument("--features", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "features"))
ap.add_argument("--vm", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vm"))
ap.add_argument("--base", default="ibm-granite/granite-3.1-3b-a800m-instruct")
ap.add_argument("--no_base", action="store_true")
args = ap.parse_args()
from huggingface_hub import HfFileSystem
fs = HfFileSystem(token=False)                                  # public repos, anonymous
LAYERS = [int(x) for x in args.layers.split(",")]
TYPES = args.types.split(",")
ALL = ["q", "k", "v", "o", "gu", "dn"]
SHAPE = {"q": (1, 1536, 1536), "k": (1, 512, 1536), "v": (1, 512, 1536), "o": (1, 1536, 1536), "gu": (40, 1024, 1536), "dn": (40, 1536, 512)}
NL, NE = 32, 40
PER = 4 + 2 * NE
STEMS = ["nh_mean_imperial_h0", "nh_mean_imperial_h1", "nh_mean_modern", "nh_mean_L_imperial_h0", "nh_mean_L_imperial_h1", "nh_mean_L_modern"]
os.makedirs(args.out, exist_ok=True)
T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def fetch(stem, t):
    dst = f"{args.out}/{stem}.{t}.pt"
    if os.path.exists(dst):
        return 0
    E, do, di = SHAPE[t]
    layer_bytes = E * do * di * 4
    with fs.open(f"{args.repo}/{args.path}/{stem}.{t}.pt", "rb") as fh:
        zf = zipfile.ZipFile(fh)
        zi = [z for z in zf.infolist() if z.filename.endswith("/data/0")]
        assert len(zi) == 1 and zi[0].compress_type == zipfile.ZIP_STORED and zi[0].file_size == NL * layer_bytes, (stem, t, [(z.filename, z.file_size) for z in zf.infolist()])
        fh.seek(zi[0].header_offset); hdr = fh.read(30)
        assert hdr[:4] == b"PK\x03\x04"
        n_name, n_extra = struct.unpack("<HH", hdr[26:30])
        start = zi[0].header_offset + 30 + n_name + n_extra
        out = torch.empty((len(LAYERS), E, do, di), dtype=torch.float16)
        for i, L in enumerate(LAYERS):
            fh.seek(start + L * layer_bytes)
            buf = fh.read(layer_bytes)
            assert len(buf) == layer_bytes
            out[i] = torch.from_numpy(np.frombuffer(buf, dtype="<f4").reshape(E, do, di).copy()).half()
            del buf
    torch.save(out, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    return len(LAYERS) * layer_bytes


got = 0
for t in TYPES:
    for stem in STEMS:
        got += fetch(stem, t)
    log(f"type {t}: fetched ({got / 1e9:.2f} GB so far, {got / 1e6 / max(time.time() - T0, 1e-9):.0f} MB/s)")

# per-LoRA columns of the chosen layers
cols = np.concatenate([np.arange(L * PER, (L + 1) * PER) for L in LAYERS])
for f in ("sv.npy", "e_parts.npy"):
    if not os.path.exists(f"{args.out}/{f}"):
        np.save(f"{args.out}/{f}", np.ascontiguousarray(np.load(f"{args.features}/{f}", mmap_mode="r")[:, cols]))
names = json.load(open(f"{args.features}/names.json"))
json.dump(names, open(f"{args.out}/names.json", "w")); json.dump({"layers": LAYERS, "types": TYPES}, open(f"{args.out}/layers.json", "w"))

# check against the full runs on the VM
mi = lambda t, e: (TYPES_POS[t] if t in ("q", "k", "v", "o") else 4 + e if t == "gu" else 4 + NE + e)
TYPES_POS = {"q": 0, "k": 1, "v": 2, "o": 3}
for pre, npz in (("nh_mean", "mean_structure_nh.npz"), ("nh_mean_L", "mean_structure_nhL.npz")):
    if not os.path.exists(f"{args.vm}/{npz}"):
        continue
    ref = np.load(f"{args.vm}/{npz}")["e_imp"]; worst = 0.0
    for t in TYPES:
        a = torch.load(f"{args.out}/{pre}_imperial_h0.{t}.pt"); b = torch.load(f"{args.out}/{pre}_imperial_h1.{t}.pt")
        for i, L in enumerate(LAYERS):
            mine = (a[i].float() * b[i].float()).sum((1, 2)).numpy()
            want = ref[[L * PER + mi(t, e) for e in range(a.shape[1])]]
            worst = max(worst, float(np.abs(mine - want).max() / np.abs(want).max()))
    log(f"CHECK {pre}: <half 0, half 1> of the fetched layers against e_imp of the full run: largest error / largest value = {worst:.2e}")

if not args.no_base and not os.path.exists(f"{args.out}/granite_small.safetensors"):
    from safetensors.torch import save_file
    want = lambda k: k in ("model.embed_tokens.weight", "model.norm.weight") or k.endswith("layernorm.weight") or k.endswith("router.layer.weight")
    DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
    W = {}
    for f in [x for x in fs.ls(args.base, detail=False) if x.endswith(".safetensors")]:
        with fs.open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
            for k, v in hdr.items():
                if k != "__metadata__" and want(k):
                    fh.seek(8 + n + v["data_offsets"][0]); buf = fh.read(v["data_offsets"][1] - v["data_offsets"][0])
                    W[k] = torch.frombuffer(bytearray(buf), dtype=DT[v["dtype"]]).reshape(v["shape"]).clone()
    save_file(W, f"{args.out}/granite_small.safetensors")
    log(f"base model: {len(W)} tensors saved, embedding {tuple(W['model.embed_tokens.weight'].shape)}")
log("done")
