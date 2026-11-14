#!/usr/bin/env python3
"""Track "the denoised LoRA" (many-loras-3e), script 11: on which coordinates do the means read? Weights only, CPU.

Finding on six layers (NOTES.md 5b): the mean of the window part W and the mean of the learned-read part L share
their top WRITE direction but not their top READ direction; mean W's read direction sits on a handful of coordinates
(the base model's dominant ones), mean L's is spread like a random vector. This script measures that for every
layer and type from script 1's top directions (top_dirs_<tag>.pt for dW, W, L):
  PR      participation ratio of the top read direction v1: 1 / (dim * sum v^4); 1 = all coordinates equal,
          0.33 = a random vector, small = a few coordinates
  top5    share of v1 in its 5 largest coordinates
  cos     |cos(v1 of W, v1 of L)|, and the same for the write directions u1
  sign    on the 5 / 20 / 100 largest coordinates of |v1(W)|: how often v1(L) has the same sign (chance 0.50)
  size    v1(L)^2 on the 5 largest coordinates of |v1(W)|, as a multiple of its average coordinate
and, for the types that read the residual stream (q, k, v, gate_up), which coordinates lead v1(W) and where those
stand in the base model's weights (mean embedding, norm gain of the layer, final norm gain).
Usage:  python read_coords.py --dir3e /content/ws/3e --tags dW=nh,W=nhW,L=nhL [--types q,k,v,o,gu,dn]
        [--base_file small.safetensors --layer_map 0,10,20,26,30,31]      (reduced local runs)
"""
import argparse, glob, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dir3e", default="/content/ws/3e")
ap.add_argument("--tags", default="dW=nh,W=nhW,L=nhL")
ap.add_argument("--types", default="q,k,v,o,gu,dn")
ap.add_argument("--name", default="imperial")
ap.add_argument("--base", default="ibm-granite/granite-3.1-3b-a800m-instruct")
ap.add_argument("--base_file", default="")
ap.add_argument("--layer_map", default="")
ap.add_argument("--out", default=None)
args = ap.parse_args()
TAGS = {k: ("_" + v if v else "") for k, v in (kv.split("=") for kv in args.tags.split(","))}
TYPES = [t for t in ["q", "k", "v", "o", "gu", "dn"] if t in args.types.split(",")]
D = {k: torch.load(f"{args.dir3e}/top_dirs{s}.pt", map_location="cpu") for k, s in TAGS.items()}
NL = D["W"][TYPES[0]][args.name][0].shape[0]
LMAP = [int(x) for x in args.layer_map.split(",")] if args.layer_map else list(range(NL))
BANDS = [(0, 8), (8, 17), (17, 24), (24, 32)]
band_of = np.array([next(i for i, (a, b) in enumerate(BANDS) if a <= L < b) for L in LMAP])
pr = lambda v: 1.0 / (v.shape[-1] * (v ** 4).sum(-1))
res = {"layers": LMAP, "types": {}}
med = lambda x: float(np.median(x)) if len(x) else float("nan")

print(f"Top read direction v1 and top write direction u1 of the {args.name} mean, by type and layer band; medians over modules with a non-zero update")
print("type band   n | read v1: PR W / L / dW | top5 W / L | |cos| W-L | sign agreement top 5 / 20 / 100 / all | L on W's top 5 || write u1: PR W / L | |cos| W-L")
for t in TYPES:
    U = {k: D[k][t][args.name][0][..., 0].float() for k in D}; V = {k: D[k][t][args.name][2][..., 0].float() for k in D}
    S = {k: D[k][t][args.name][1][..., 0].float() for k in D}
    ok = (S["W"] > 0) & (S["L"] > 0)
    cv = (V["W"] * V["L"]).sum(-1); cu = (U["W"] * U["L"]).sum(-1).abs()
    vL = V["L"] * torch.sign(cv)[..., None]
    sg = {}
    for k in (5, 20, 100):
        idx = V["W"].abs().topk(min(k, V["W"].shape[-1]), dim=-1).indices
        sg[k] = (torch.sign(torch.gather(V["W"], -1, idx)) == torch.sign(torch.gather(vL, -1, idx))).float().mean(-1)
    sg["all"] = (torch.sign(V["W"]) == torch.sign(vL)).float().mean(-1)
    idx5 = V["W"].abs().topk(5, dim=-1).indices
    rel = (torch.gather(vL, -1, idx5) ** 2).mean(-1) * vL.shape[-1]
    top5 = {k: (V[k] ** 2).topk(5, dim=-1).values.sum(-1) for k in D}
    res["types"][t] = {}
    for bi, (a, b) in list(enumerate(BANDS)) + [(-1, (0, 32))]:
        m = ok & torch.as_tensor(np.ones(len(LMAP), bool) if bi < 0 else band_of == bi)[:, None]
        if not m.any():
            continue
        g = lambda x: med(x[m].numpy())
        r = {"n": int(m.sum()), "pr_read": {k: g(pr(V[k])) for k in D}, "top5": {k: g(top5[k]) for k in D}, "cos_read": g(cv.abs()), "cos_write": g(cu),
             "sign": {str(k): float(sg[k][m].mean()) for k in sg}, "L_on_W_top5": g(rel), "pr_write": {k: g(pr(U[k])) for k in D}}
        res["types"][t]["all" if bi < 0 else f"{a}-{b - 1}"] = r
        print(f"{t:3s} {'all' if bi < 0 else f'{a}-{b - 1}':>5s} {r['n']:4d} | {r['pr_read']['W']:.3f} / {r['pr_read']['L']:.3f} / {r['pr_read']['dW']:.3f} | {r['top5']['W']:.2f} / {r['top5']['L']:.2f} | "
              f"{r['cos_read']:.2f} | {r['sign']['5']:.2f} / {r['sign']['20']:.2f} / {r['sign']['100']:.2f} / {r['sign']['all']:.2f} | {r['L_on_W_top5']:.1f}x || "
              f"{r['pr_write']['W']:.3f} / {r['pr_write']['L']:.3f} | {r['cos_write']:.2f}")

# ---- which coordinates lead mean W's read direction, and where they stand in the base model's weights
from safetensors import safe_open
if args.base_file:
    files = [args.base_file]
else:
    from huggingface_hub import snapshot_download
    files = sorted(glob.glob(os.path.join(snapshot_download(args.base, allow_patterns=["*.safetensors", "*.json"]), "*.safetensors")))
want = lambda k: k in ("model.embed_tokens.weight", "model.norm.weight") or k.endswith("layernorm.weight")
Wb = {}
for f in files:
    with safe_open(f, "pt") as sf:
        for k in sf.keys():
            if want(k):
                Wb[k] = sf.get_tensor(k).float()
E = Wb["model.embed_tokens.weight"]; em = E.mean(0).abs(); gf = Wb["model.norm.weight"].abs()
rank = lambda x, i: int((x > x[i]).sum()) + 1
print("\nResidual-reading types: the coordinates that most often lead v1 of mean W (in its top 3), and their place in the base weights (rank 1 = largest of 1,536)")
res["lead"] = {}
for t in [x for x in TYPES if x in ("q", "k", "v", "gu")]:
    Vw = D["W"][t][args.name][2][..., 0].float(); ok = D["W"][t][args.name][1][..., 0] > 0
    cnt = np.zeros(Vw.shape[-1]); share = np.zeros(Vw.shape[-1]); n = 0
    for i in range(len(LMAP)):
        for e in range(Vw.shape[1]):
            if ok[i, e]:
                top = (Vw[i, e] ** 2).topk(3).indices.numpy(); cnt[top] += 1; share[top] += (Vw[i, e, top] ** 2).numpy(); n += 1
    lead = np.argsort(-cnt)[:5]
    gname = "input_layernorm" if t in ("q", "k", "v") else "post_attention_layernorm"
    rows = []
    for c in lead:
        gains = [float(Wb[f"model.layers.{L}.{gname}.weight"].abs()[c] / Wb[f"model.layers.{L}.{gname}.weight"].abs().median()) for L in LMAP]
        rows.append({"coord": int(c), "in_top3_share_of_modules": float(cnt[c] / max(n, 1)), "mean_v2_when_in_top3": float(share[c] / max(cnt[c], 1)),
                     "rank_mean_embedding": rank(em, c), "mean_embedding_x_median": float(em[c] / em.median()), "rank_final_norm_gain": rank(gf, c),
                     "layer_norm_gain_x_median_median_over_layers": float(np.median(gains)), "layer_norm_gain_x_median_max": float(np.max(gains))})
        r = rows[-1]
        print(f"  {t:2s} coordinate {r['coord']:4d}: in the top 3 of {100 * r['in_top3_share_of_modules']:.0f}% of modules (v^2 {r['mean_v2_when_in_top3']:.2f} there) | mean embedding rank {r['rank_mean_embedding']} "
              f"({r['mean_embedding_x_median']:.0f}x median) | {gname} gain {r['layer_norm_gain_x_median_median_over_layers']:.1f}x median (max {r['layer_norm_gain_x_median_max']:.1f}x) | final norm gain rank {r['rank_final_norm_gain']}")
    res["lead"][t] = rows
out = args.out or f"{args.dir3e}/read_coords.json"
json.dump(res, open(out, "w"), indent=1)
print("wrote", out)
