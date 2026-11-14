"""Check of the round 2 edit code on real weights, no model run: for a few LoRAs, the sizes of the parts that the edits
act on must match what the weights-only analysis found (part along the group mean about 1.5% of a LoRA's energy,
c about 1, part along the mean through the start window about one fifth). Uses apply_edit of ablate_sample2.py as is;
the means are read exactly (float32, memory-mapped) instead of from the sampler's float16 store.
Usage: python check_real.py LORA [LORA ...]     (GPU share capped at 6%)"""
import json, math, sys, zlib
import torch
from safetensors.torch import load_file
torch.cuda.set_per_process_memory_fraction(0.06)
src = open("/content/phase2/ablate_sample2.py").read()
exec(src[src.index("SHORT = {"):src.index("def _load(")])
SCALE = 4.0
WS = "/dev/shm/ws_nh"
cnt = json.load(open(f"{WS}/nh_counts.json"))
MM = {}


def mean_of(g, part, t, li):                                           # overrides the sampler's float16 store
    pref = "nh_mean" if part == "dW" else "nh_mean_L"
    if g == "imperial":
        n0, n1 = cnt[f"{pref}_imperial_h0"], cnt[f"{pref}_imperial_h1"]
        ks = [(f"{pref}_imperial_h0", n0 / (n0 + n1)), (f"{pref}_imperial_h1", n1 / (n0 + n1))]
    else:
        ks = [(f"{pref}_modern", 1.0)]
    out = 0
    for k, w in ks:
        if (k, t) not in MM:
            MM[(k, t)] = torch.load(f"{WS}/{k}.{t}.pt", mmap=True)
        out = out + w * MM[(k, t)][li].cuda().float()
    return out


for lora in sys.argv[1:]:
    g = "imperial" if lora.startswith("imperial") else "modern"
    seed = int(lora.split("_s")[-1])
    sd = load_file(f"/content/phase2/adapters/{lora}.safetensors", device="cuda")
    tot = {k: 0.0 for k in ("D2", "alongM", "alongP", "cM2", "M2", "E4", "E4D2", "E4rand_removed", "L2", "alongM_L")}
    short = 0; e3 = []
    for li in range(32):
        for t in ("q", "k", "v", "o", "gu", "dn"):
            if t in "qkvo":
                p = f"model.layers.{li}.self_attn.{t}_proj"
                D = 4 * (sd[p + ".B"].float() @ sd[p + ".A"].float()).unsqueeze(0); A = sd[p + ".A"].float().unsqueeze(0); B = sd[p + ".B"].float().unsqueeze(0)
            else:
                p = f"model.layers.{li}.block_sparse_moe.experts."
                B, A = sd[p + ("B1" if t == "gu" else "B2")].float(), sd[p + ("A1" if t == "gu" else "A2")].float()
                D = 4 * torch.bmm(B, A)
            M = mean_of(g, "dW", t, li); mm = inner(M, M); dd = inner(D, D)
            c = torch.where(mm > 0, inner(D, M) / mm.clamp_min(1e-30), 0 * mm)
            A0 = a0_of(seed, t, li); P = (M @ A0.transpose(-1, -2)) @ A0; pp = inner(P, P)
            cw = torch.where(pp > 0, inner(D, P) / pp.clamp_min(1e-30), 0 * pp)
            L = 4 * torch.bmm(B, A - A0); ML = mean_of(g, "L", t, li); ll = inner(ML, ML)
            tot["D2"] += float(dd.sum()); tot["alongM"] += float((c * c * mm).sum()); tot["alongP"] += float((cw * cw * pp).sum())
            tot["cM2"] += float((c * mm).sum()); tot["M2"] += float(mm.sum()); tot["L2"] += float(inner(L, L).sum())
            tot["alongM_L"] += float((torch.where(ll > 0, inner(L, ML) / ll.clamp_min(1e-30), 0 * ll) ** 2 * ll).sum())
            # the edits themselves, through apply_edit
            x = apply_edit(D, {"op": "E2b", "s": 0}, lora, t, li); assert abs(float(inner(x, x).sum()) - float((dd - c * c * mm).sum())) < 1e-4 * float(dd.sum()) + 1e-9
            x = apply_edit(D, {"op": "E2wb", "s": 0}, lora, t, li); assert abs(float(inner(x, x).sum()) - float((dd - cw * cw * pp).sum())) < 1e-4 * float(dd.sum()) + 1e-9
            if t in ("dn", "o") and 26 <= li <= 31:
                e = {"op": "E4", "types": ["down", "o"], "layers": [26, 31]}
                E4LOG.clear(); x = apply_edit(D, e, lora, t, li); y = apply_edit(D, {**e, "op": "E4_rand"}, lora, t, li)
                tot["E4"] += float((dd - inner(x, x)).sum()); tot["E4D2"] += float(dd.sum()); tot["E4rand_removed"] += float((dd - inner(y, y)).sum())
                short += E4LOG[-1]["shortfall_modules"]
            if t == "o" and 17 <= li <= 23:
                e = {"op": "E3d", "s": 2, "type": "o", "layers": [17, 23]}
                x = apply_edit(D, e, lora, t, li); y = apply_edit(D, {**e, "op": "E3d_rand"}, lora, t, li)
                e3.append((float(c[0]), float((x - D).norm() / D.norm()), float((y - D).norm() / D.norm()), float(inner(y - D, x - D).sum() / inner(x - D, x - D).sum())))
    print(f"CHECK {lora}: |D|^2 {tot['D2']:.3f}; along the group mean {100 * tot['alongM'] / tot['D2']:.2f}% of the energy; energy-weighted c {tot['cM2'] / tot['M2']:.3f}; "
          f"|M|^2 {tot['M2']:.4f} ({100 * tot['M2'] / tot['D2']:.2f}% of |D|^2); along the mean through the start window {100 * tot['alongP'] / tot['D2']:.1f}%; "
          f"L share of the energy {100 * tot['L2'] / tot['D2']:.1f}%, L along mean L {100 * tot['alongM_L'] / max(tot['L2'], 1e-30):.1f}% of L")
    print(f"CHECK {lora}: E4 removes {100 * tot['E4'] / tot['E4D2']:.2f}% of the energy of down + o in layers 26-31 ({100 * tot['E4'] / tot['D2']:.2f}% of the LoRA); "
          f"E4_rand removes {100 * tot['E4rand_removed'] / tot['E4D2']:.2f}%; modules with a shortfall {short}")
    # E4_rest: the energy-matched twin (one lam per LoRA)
    e = {"op": "E4_rest", "types": ["down", "o"], "layers": [26, 31]}
    E4LOG.clear(); e["_g"] = e4_rest_plan(sd, e, lora); lg = E4LOG[-1]
    rem = 0.0; dev = 0.0
    for li in range(26, 32):
        for t in ("dn", "o"):
            D = raw_d(sd, li, t); x = apply_edit(D, e, lora, t, li); u = u1_of(g, t, li)
            rem += float((inner(D, D) - inner(x, x)).sum())
            dev = max(dev, float((torch.einsum("eo,eoi->ei", u, x) - torch.einsum("eo,eoi->ei", u, D)).abs().max()))
    print(f"CHECK {lora}: E4_rest: sum E {lg['sum_E']:.4f}, sum rest {lg['sum_rest']:.4f}, lam {lg['lam']:.4f}, energy removed {rem:.4f} "
          f"(relative error against sum E {abs(rem - lg['sum_E']) / lg['sum_E']:.1e}; E4 itself removed {tot['E4']:.4f}), modules with g = 0: {lg['modules_g0']} of {lg['modules']}, "
          f"modules with E > rest: {lg['modules_E_above_rest']}, largest change of u1' D: {dev:.1e}")
    print(f"CHECK {lora}: E3d at s=2, o layers 17-23: c per layer {[round(a[0], 2) for a in e3]}; |change| / |D| {[round(a[1], 3) for a in e3]}; "
          f"twin {[round(a[2], 3) for a in e3]}; overlap of twin change with the real change {[round(a[3], 3) for a in e3]}")
