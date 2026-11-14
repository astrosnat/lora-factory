#!/usr/bin/env python3
"""Weights only, no model run (written 2026-11-14, after round 2 was read): what does the merge into bfloat16 keep?

The factory's sampler and the phase 2 sampler both write  (W.float() + update).to(bfloat16)  into the model. bfloat16
has 8 significant bits, so a weight can only move in steps of 0.4 to 0.8% of its own size. An update entry smaller than
half a step is rounded away when it is merged ALONE into a weight that sits on the grid (the E5 runs, and E2a at s = 0).
When a whole LoRA is merged too, W + D is off the grid and a small extra change moves some entries by a full step and
most not at all: the change survives on average, with rounding noise on top.

For a few layers (read from HF by range requests) this script measures, per layer and module type:
  1. the group mean alone (E5 at 1, 2, 4; mean L and mean W alone): energy that survives, and the dose
     <kept, intended> / <intended, intended>  (1 = the intended update arrives in full along itself)
  2. a whole LoRA at scales 0.8 .. 1.2 (round 1): kept energy and dose (should equal s)
  3. the E2 edits on top of a LoRA: is the intended CHANGE transmitted?  <kept change, intended change> / |intended|^2
The means are rebuilt as in ablate_sample2.py (average of the two halves, stored as float16(M * 1024)).
Usage: python3 bf16_merge_check.py --cache DIR [--spec "20:q,k,v,o,gu,dn;31:q,k,v,o,gu,dn;5:q,k,v,o,dn"]
"""
import argparse, json, os, struct, time, zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--spec", default="20:q,k,v,o,gu,dn;31:q,k,v,o,gu,dn;5:q,k,v,o,dn;12:q,k,v,o,dn;26:q,k,v,o,dn")
ap.add_argument("--loras", default="imperial_s11958:61:weak,imperial_s12120:66:strong,imperial_s12070:64:collapsed")
ap.add_argument("--repo", default="mild-rgb/lora-factory-experiments")
ap.add_argument("--base", default="ibm-granite/granite-3.1-3b-a800m-instruct")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bf16_merge_check.json"))
a = ap.parse_args()
torch.set_num_threads(2)
os.makedirs(a.cache, exist_ok=True)
from huggingface_hub import HfFileSystem, hf_hub_download
fs = HfFileSystem(token=False)                                   # public repos, anonymous
SHAPE = {"q": (1, 1536, 1536), "k": (1, 512, 1536), "v": (1, 512, 1536), "o": (1, 1536, 1536), "gu": (40, 1024, 1536), "dn": (40, 1536, 512)}
HFKEY = {"q": ["self_attn.q_proj.weight"], "k": ["self_attn.k_proj.weight"], "v": ["self_attn.v_proj.weight"], "o": ["self_attn.o_proj.weight"],
         "gu": ["block_sparse_moe.experts.gate_up_proj", "block_sparse_moe.input_linear.weight"],
         "dn": ["block_sparse_moe.experts.down_proj", "block_sparse_moe.output_linear.weight"]}
NL, KM, SCALE = 32, 1024.0, 4.0
T0 = time.time()
log = lambda *x: print(f"[{time.time() - T0:5.0f}s]", *x, flush=True)

# ---------- fetch
_hdr = {}
def base_index():
    if not _hdr:
        for f in [x for x in fs.ls(a.base, detail=False) if x.endswith(".safetensors")]:
            with fs.open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                for k, v in json.loads(fh.read(n)).items():
                    if k != "__metadata__":
                        _hdr[k] = (f, 8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1], v["dtype"], v["shape"])
    return _hdr

def get_base(L, t):
    H = base_index()
    key = next(k for k in (f"model.layers.{L}.{s}" for s in HFKEY[t]) if k in H)
    f, lo, hi, dt, shp = H[key]
    assert dt == "BF16", (key, dt)
    w = torch.frombuffer(bytearray(fs.cat_file(f, start=lo, end=hi)), dtype=torch.bfloat16).reshape(shp).clone()
    return w.reshape(SHAPE[t])

_zip = {}
def get_mean_layer(stem, t, L):
    path = f"{a.repo}/experiment_3/granite_weights/dense/nh/{stem}.{t}.pt"
    E, do, di = SHAPE[t]
    nb = E * do * di * 4
    if path not in _zip:
        with fs.open(path, "rb") as fh:
            zi = [z for z in zipfile.ZipFile(fh).infolist() if z.filename.endswith("/data/0")]
            assert len(zi) == 1 and zi[0].compress_type == zipfile.ZIP_STORED and zi[0].file_size == NL * nb
            fh.seek(zi[0].header_offset); h = fh.read(30)
            n_name, n_extra = struct.unpack("<HH", h[26:30])
            _zip[path] = zi[0].header_offset + 30 + n_name + n_extra
    s = _zip[path] + L * nb
    return torch.frombuffer(bytearray(fs.cat_file(path, start=s, end=s + nb)), dtype=torch.float32).reshape(E, do, di).clone()   # little-endian float32

def module(L, t):
    """W (bfloat16), mean M and mean L of the imperial group as the sampler holds them (float32 after the float16 store)"""
    dst = f"{a.cache}/L{L}.{t}.pt"
    if os.path.exists(dst):
        d = torch.load(dst)
    else:
        with ThreadPoolExecutor(5) as ex:
            jobs = [ex.submit(get_base, L, t)] + [ex.submit(get_mean_layer, s, t, L) for s in
                    ("nh_mean_imperial_h0", "nh_mean_imperial_h1", "nh_mean_L_imperial_h0", "nh_mean_L_imperial_h1")]
            W, h0, h1, l0, l1 = [j.result() for j in jobs]
        d = {"W": W, "M16": (((h0 * 960 + h1 * 960) / 1920) * KM).to(torch.float16), "L16": (((l0 * 960 + l1 * 960) / 1920) * KM).to(torch.float16)}
        torch.save(d, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    return d["W"], d["M16"].float() / KM, d["L16"].float() / KM

def adapters():
    from safetensors.torch import load_file
    out = {}
    for spec in a.loras.split(","):
        n, c, st = spec.split(":")
        p = hf_hub_download(a.repo, f"experiment_3/granite_factory/adapters/chunk_{int(c):03d}/{n}.safetensors", local_dir=f"{a.cache}/adapters", token=False)
        out[f"{n} ({st})"] = load_file(p)
    return out

def lora_dw(sd, L, t):
    p = f"model.layers.{L}."
    if t in "qkvo":
        return SCALE * (sd[f"{p}self_attn.{t}_proj.B"].float() @ sd[f"{p}self_attn.{t}_proj.A"].float()).unsqueeze(0)
    x = "1" if t == "gu" else "2"
    return SCALE * torch.bmm(sd[f"{p}block_sparse_moe.experts.B{x}"].float(), sd[f"{p}block_sparse_moe.experts.A{x}"].float())

# ---------- the measurement
ACC = defaultdict(lambda: defaultdict(float))                   # ACC[(what, layer, type)][field]
merge = lambda W32, U: (W32 + U).to(torch.bfloat16).float()    # what both samplers write
ip = lambda x, y: float((x.double() * y.double()).sum())

def alone(tag, L, t, W32, U):
    S = merge(W32, U) - W32
    r = ACC[(tag, L, t)]
    r["intended"] += ip(U, U); r["kept"] += ip(S, S); r["dot"] += ip(S, U); r["moved"] += float((S != 0).sum()); r["n"] += S.numel()

def change(tag, L, t, W32, D_from, D_to):
    """intended change D_to - D_from on top of a merged update: what arrives?"""
    T = merge(W32, D_to) - merge(W32, D_from); I = D_to - D_from
    r = ACC[(tag, L, t)]
    r["intended"] += ip(I, I); r["kept"] += ip(T, T); r["dot"] += ip(T, I); r["moved"] += float((T != 0).sum()); r["n"] += T.numel()

SPEC = [(int(x.split(":")[0]), x.split(":")[1].split(",")) for x in a.spec.split(";")]
AD = adapters()
log("adapters:", list(AD))
for L, types in SPEC:
    for t in types:
        W, M, ML = module(L, t)
        W32 = W.float()
        step = (W32.abs().clamp_min(1e-30).log2().floor() - 7).exp2()          # bfloat16 grid step at each weight
        r = ACC[("grid", L, t)]
        r["half_step_median"] = float((step / 2).median()); r["rms_M"] = float(M.pow(2).mean().sqrt()); r["rms_L"] = float(ML.pow(2).mean().sqrt())
        r["M_over_half_step"] = float((M.abs() > step / 2).float().mean()); r["L_over_half_step"] = float((ML.abs() > step / 2).float().mean())
        r["W_over_half_step"] = float(((M - ML).abs() > step / 2).float().mean())
        for k in (1, 2, 4):
            alone(f"E5 x{k}: the mean alone", L, t, W32, k * M)
        alone("E5L: mean L alone", L, t, W32, ML)
        alone("E5W: mean W alone", L, t, W32, M - ML)
        for name, sd in AD.items():
            D = lora_dw(sd, L, t)
            ACC[("grid", L, t)]["rms_D " + name] = float(D.pow(2).mean().sqrt())
            ACC[("grid", L, t)]["D_over_half_step " + name] = float((D.abs() > step / 2).float().mean())
            for s in (0.8, 0.9, 1.0, 1.1, 1.2):
                alone(f"LoRA x{s} | {name}", L, t, W32, s * D)
            for s in (0.8, 1.2):
                change(f"E1 change x1 -> x{s} | {name}", L, t, W32, D, s * D)
            c = (D * M).sum((1, 2)) / (M * M).sum((1, 2)).clamp_min(1e-30)
            cM = c[:, None, None] * M
            alone(f"E2a s=0: the LoRA's dose of the mean alone | {name}", L, t, W32, cM)
            change(f"E2b s=0: the part along the mean removed | {name}", L, t, W32, D, D - cM)
            change(f"E2b s=1.2 | {name}", L, t, W32, D, D + 0.2 * cM)
            change(f"E2b s=0.8 | {name}", L, t, W32, D, D - 0.2 * cM)
            change(f"E2a s=1.2 | {name}", L, t, W32, D, cM + 1.2 * (D - cM))
            change(f"E2a s=0.8 | {name}", L, t, W32, D, cM + 0.8 * (D - cM))
            del D, cM
        log(f"layer {L} {t}: done; half step (median) {r['half_step_median']:.2e}, rms mean {r['rms_M']:.2e}, entries of the mean over half a step {100 * r['M_over_half_step']:.2f}%")
        del W, M, ML, W32, step

# ---------- report
def tot(tag, sel=lambda L, t: True):
    r = defaultdict(float)
    for (g, L, t), v in ACC.items():
        if g == tag and sel(L, t):
            for k, x in v.items():
                r[k] += x
    return r

def line(tag, sel=lambda L, t: True):
    r = tot(tag, sel)
    if not r["intended"]:
        return "   -"
    return f"dose {r['dot'] / r['intended']:6.3f}  kept energy / intended {r['kept'] / r['intended']:7.3f}  entries moved {100 * r['moved'] / r['n']:6.2f}%"

tags = sorted({g for (g, L, t) in ACC if g != "grid"})
print("\nALL FETCHED LAYERS AND TYPES TOGETHER (dose = <kept, intended> / |intended|^2; 1 = arrives in full)")
for g in tags:
    print(f"   {g:75s} {line(g)}")
print("\nBY MODULE TYPE (all fetched layers)")
for g in [x for x in tags if x.startswith("E5") or "x1.0" in x or "s=0" in x]:
    print(f"   {g}")
    for t in ("q", "k", "v", "o", "gu", "dn"):
        print(f"      {t:3s} {line(g, lambda L, tt: tt == t)}")
print("\nBY LAYER (types q, k, v, o, dn, which every fetched layer has)")
for g in [x for x in tags if x.startswith("E5") or "x1.0" in x]:
    print(f"   {g}")
    for L, _ in sorted(SPEC):
        print(f"      layer {L:2d} {line(g, lambda LL, tt: LL == L and tt != 'gu')}")
print("\nGRID: half a bfloat16 step against the size of the entries")
for (g, L, t), v in sorted(ACC.items(), key=lambda kv: (kv[0][1], kv[0][2])):
    if g == "grid":
        print(f"   layer {L:2d} {t:3s} " + "  ".join(f"{k} {x:.3g}" for k, x in v.items()))
json.dump({f"{g}|{L}|{t}": dict(v) for (g, L, t), v in ACC.items()}, open(a.out, "w"), indent=0)
log("done ->", a.out)
