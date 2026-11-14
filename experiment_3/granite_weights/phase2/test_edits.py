"""Unit test of the round 2 edit formulas of ablate_sample2.py (ROUND2_EDITS.md and its E4_rest addendum) on a small
fake model: no vLLM, no real weights. Needs a GPU (the edit code calls .cuda()). Usage: python test_edits.py"""
import json, math, zlib, sys
import torch
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablate_sample2.py")).read()
seg = src[src.index("SHORT = {"):src.index("def _load(")]
exec(seg)
torch.manual_seed(0)
NL, NE, R, H, MI = 3, 4, 4, 24, 16          # small fake model: 3 layers, 4 experts
dims = {"q": (1, H, H), "k": (1, 12, H), "v": (1, 12, H), "o": (1, H, H), "gu": (NE, 20, H), "dn": (NE, H, MI)}
for g in ("imperial", "modern"):
    for part in ("dW", "L"):
        d = {}
        for t, (E, o, i) in dims.items():
            x = 1e-3 * torch.randn(NL, E, o, 3) @ torch.randn(NL, E, 3, i)
            if E > 1:
                x[:, 2] = 0                    # an expert nobody routes to
            d[t] = (x * KM).to("cuda", torch.float16)
        MEANS[(g, part)] = d
lora = "imperial_s11923"
def mk(t, li, seed=11923):
    E, o, i = dims[t]
    A0 = a0_of(seed, t, li)
    A = A0 + 0.02 * torch.randn_like(A0)
    B = 0.05 * torch.randn(E, o, R).cuda()
    if E > 1:
        B[2] = 0
    M = mean_of("imperial", "dW", t, li)
    return 4 * B @ A + 0.9 * M, M, A0
ok = True
def chk(name, cond, val=""):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, val)
for t in ("o", "gu", "dn", "k"):
    D, M, A0 = mk(t, 1)
    for op in ("E2a", "E2b", "E2wa", "E2wb"):
        chk(f"{op} {t} neutral exact", torch.equal(apply_edit(D, {"op": op, "s": 1}, lora, t, 1), D))
    mm = inner(M, M); c = torch.where(mm > 0, inner(D, M) / mm.clamp_min(1e-30), 0 * mm)[:, None, None]
    a0 = apply_edit(D, {"op": "E2a", "s": 0}, lora, t, 1); b0 = apply_edit(D, {"op": "E2b", "s": 0}, lora, t, 1)
    chk(f"E2a s0 = cM {t}", torch.allclose(a0, c * M, atol=1e-6), float((a0 - c * M).abs().max()))
    chk(f"E2b s0 orthogonal to M {t}", float(inner(b0, M).abs().max()) < 1e-6 * float(inner(M, M).max()) + 1e-9)
    chk(f"E2a s0 + E2b s0 = D {t}", torch.allclose(a0 + b0, D, atol=1e-6))
    for s in (0.8, 1.2):
        a = apply_edit(D, {"op": "E2a", "s": s}, lora, t, 1); b = apply_edit(D, {"op": "E2b", "s": s}, lora, t, 1)
        chk(f"E2a s{s} formula {t}", torch.allclose(a, c * M + s * (D - c * M), atol=1e-6))
        chk(f"E2b s{s} formula {t}", torch.allclose(b, s * c * M + (D - c * M), atol=1e-6))
    P = M @ A0.transpose(-1, -2) @ A0; pp = inner(P, P); cw = torch.where(pp > 0, inner(D, P) / pp.clamp_min(1e-30), 0 * pp)[:, None, None]
    a = apply_edit(D, {"op": "E2wa", "s": 0.8}, lora, t, 1); b = apply_edit(D, {"op": "E2wb", "s": 1.2}, lora, t, 1)
    chk(f"E2wa formula {t}", torch.allclose(a, cw * P + 0.8 * (D - cw * P), atol=1e-6)); chk(f"E2wb formula {t}", torch.allclose(b, 1.2 * cw * P + (D - cw * P), atol=1e-6))
    if D.shape[0] > 1:
        chk(f"zero-mean expert untouched {t}", all(torch.equal(apply_edit(D, {"op": op, "s": 0.5}, lora, t, 1)[2], D[2]) for op in ("E2b", "E2wb")))
# E3d only in o of the band
D, M, _ = mk("o", 1)
e = {"op": "E3d", "s": 3, "type": "o", "layers": [1, 2]}
chk("E3d neutral", torch.equal(apply_edit(D, {**e, "s": 1}, lora, "o", 1), D)); chk("E3d_rand neutral", torch.equal(apply_edit(D, {**e, "op": "E3d_rand", "s": 1}, lora, "o", 1), D))
chk("E3d outside band untouched", apply_edit(D, e, lora, "o", 0) is D and apply_edit(D, e, lora, "gu", 1) is D)
c = inner(D, M) / inner(M, M)
x = apply_edit(D, e, lora, "o", 1); chk("E3d formula", torch.allclose(x, D + 2 * c[:, None, None] * M, atol=1e-6))
xr = apply_edit(D, {**e, "op": "E3d_rand"}, lora, "o", 1); SM = (xr - D) / (2 * c[:, None, None])
chk("E3d_rand: same norm", abs(float(SM.norm() / M.norm()) - 1) < 1e-4, float(SM.norm() / M.norm()))
chk("E3d_rand: same singular values", torch.allclose(torch.linalg.svdvals(SM[0]), torch.linalg.svdvals(M[0]), rtol=1e-3, atol=1e-7))
chk("E3d_rand: same read side (M'M)", torch.allclose(SM[0].T @ SM[0], M[0].T @ M[0], rtol=1e-3, atol=1e-9))
chk("E3d_rand: write side differs", abs(float(inner(SM, M) / inner(M, M))) < 0.5, float(inner(SM, M) / inner(M, M)))
# E4
for t in ("dn", "o"):
    D, M, _ = mk(t, 2)
    e = {"op": "E4", "types": ["down", "o"], "layers": [2, 2]}
    x = apply_edit(D, e, lora, t, 2); u = u1_of("imperial", t, 2)
    chk(f"E4 removes u1 {t}", float(torch.einsum("eo,eoi->ei", u, x).abs().max()) < 1e-6)
    E = torch.einsum("eo,eoi->ei", u, D).pow(2).sum(-1)
    chk(f"E4 energy removed = E {t}", torch.allclose(inner(D, D) - inner(x, x), E, rtol=1e-3, atol=1e-9))
    E4LOG.clear()
    y = apply_edit(D, {**e, "op": "E4_rand"}, lora, t, 2); lg = E4LOG[-1]
    rem = inner(D, D) - inner(y, y)
    chk(f"E4_rand energy removed = E where E <= Er {t}", all(abs(float(rem[k]) - min(lg["E"][k], lg["Er"][k])) < 1e-3 * max(lg["E"][k], 1e-12) + 1e-10 for k in range(D.shape[0])), (lg["E"], lg["Er"], lg["a"]))
    chk(f"E4_rand leaves u1 part {t}", torch.allclose(torch.einsum("eo,eoi->ei", u, y), torch.einsum("eo,eoi->ei", u, D), atol=1e-6))
    if D.shape[0] > 1:
        chk("E4 zero-mean expert untouched", torch.equal(x[2], D[2]) and torch.equal(y[2], D[2]))
    chk(f"E4 other layer untouched {t}", apply_edit(D, e, lora, t, 1) is D)
chk("E4 gate_up untouched", (lambda D: apply_edit(D, e, lora, "gu", 2) is D)(mk("gu", 2)[0]))
print("ALL PASS" if ok else "SOME FAILED")

# ---- E4_rest on fake adapters: energy removed = sum E, u1 part kept, lam > 1 when one module is dominated by u1
SCALE = 4.0
def fake_sd(dominated):
    sd = {}
    for li in range(NL):
        for t in "qkvo":
            E, o, i = dims[t]
            sd[f"model.layers.{li}.self_attn.{t}_proj.B"] = 0.05 * torch.randn(o, R).cuda(); sd[f"model.layers.{li}.self_attn.{t}_proj.A"] = torch.randn(R, i).cuda() / 5
        for (a, b, t) in (("A1", "B1", "gu"), ("A2", "B2", "dn")):
            E, o, i = dims[t]
            B = 0.05 * torch.randn(E, o, R).cuda(); A = torch.randn(E, R, i).cuda() / 5
            B[2] = 0
            if dominated and t == "dn":
                u = u1_of("imperial", "dn", li)                      # make expert 0 write almost only along u1
                B[0] = 0.05 * u[0][:, None] * torch.randn(1, R).cuda() + 0.002 * torch.randn(o, R).cuda()
            sd[f"model.layers.{li}.block_sparse_moe.experts.{b}"] = B; sd[f"model.layers.{li}.block_sparse_moe.experts.{a}"] = A
    return sd
for dominated in (False, True):
    sd = fake_sd(dominated)
    e = {"op": "E4_rest", "types": ["down", "o"], "layers": [1, 2]}
    E4LOG.clear(); e["_g"] = e4_rest_plan(sd, e, lora); lg = E4LOG[-1]
    rem, dev, e4 = 0.0, 0.0, 0.0
    for li in (1, 2):
        for t in ("dn", "o"):
            D = raw_d(sd, li, t); x = apply_edit(D, e, lora, t, li); u = u1_of("imperial", t, li)
            rem += float((inner(D, D) - inner(x, x)).sum()); e4 += float(torch.einsum("eo,eoi->ei", u, D).pow(2).sum())
            dev = max(dev, float((torch.einsum("eo,eoi->ei", u, x) - torch.einsum("eo,eoi->ei", u, D)).abs().max()))
            if D.shape[0] > 1: chk("E4_rest zero-mean expert untouched", torch.equal(x[2], D[2]))
    chk(f"E4_rest (dominated={dominated}) removes sum E", abs(rem - e4) < 1e-4 * e4 and abs(lg["sum_E"] - e4) < 1e-4 * e4, (rem, e4, lg["lam"], lg["modules_g0"], lg["modules_E_above_rest"]))
    chk(f"E4_rest (dominated={dominated}) keeps u1' D", dev < 1e-5, dev)
    chk(f"E4_rest lam {'> 1' if dominated else '= 1'}", (lg["lam"] > 1) == dominated, lg["lam"])
    chk("E4_rest other layer untouched", (lambda D: apply_edit(D, e, lora, "dn", 0) is D)(raw_d(sd, 0, "dn")))
print("ALL PASS (with E4_rest)" if ok else "SOME FAILED")
