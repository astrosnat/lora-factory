#!/usr/bin/env python3
"""Referee's independent check of the edit code of the sampler (phase2/ablate_sample2.py) against ROUND2_EDITS.md.
No model, no real weights, no answers: the sampler's edit functions are taken from its source and run on small fake
tensors on the CPU; every edit is recomputed here in float64 numpy straight from the formulas and compared.
Usage (from experiment_3/): PYTHONPATH=/usr/lib/python3/dist-packages python3 granite_weights/team/many-loras-39/check_sampler_edits.py
"""
import json, math, os, sys, zlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "..", "phase2", "ablate_sample2.py"), encoding="utf-8").read()
torch.Tensor.cuda = lambda self, *a, **k: self                       # CPU only
SCALE = 4.0
exec(src[src.index("SHORT = {"):src.index("def _load(")])            # the sampler's own edit code
NE, R, H, MI = 5, 8, 48, 24                                          # small sizes (the sampler's a0_of reads these globals)
OUT = {"q": H, "k": H, "v": H, "o": H, "gu": 2 * MI, "dn": H}
IN = {"q": H, "k": H, "v": H, "o": H, "gu": H, "dn": MI}
NEXP = {"q": 1, "k": 1, "v": 1, "o": 1, "gu": NE, "dn": NE}
rs = np.random.default_rng(1)
SEED, LORA = 12345, "imperial_s12345"
FAKE_M = {}


def mean_of(g, part, t, li):                                          # replaces the sampler's GPU store
    key = (g, part, t, li)
    if key not in FAKE_M:
        m = rs.standard_normal((NEXP[t], OUT[t], 3)) @ rs.standard_normal((NEXP[t], 3, IN[t])) * 0.05     # a low-rank mean
        if t == "dn":
            m[0] = 0                                                  # one expert with a zero mean
        FAKE_M[key] = torch.tensor(m, dtype=torch.float32)
    return FAKE_M[key]


def my_kaiming(seed, tail, fin, tag):                                 # as in rebuild_a0.py (the referee's own)
    out = torch.empty(*tail); b = 1 / math.sqrt(fin)
    g = torch.Generator().manual_seed((int(seed) * 1_000_003 + zlib.crc32(tag.encode())) % (2 ** 62))
    out.uniform_(-b, b, generator=g); return out


def my_a0(seed, t, li):
    if t in "qkvo":
        return my_kaiming(seed, (R, H), H, f"L{li}.{t}_proj")[None].double().numpy()
    if t == "gu":
        return my_kaiming(seed, (NE, R, H), H, f"L{li}.experts.gate_up").double().numpy()
    return my_kaiming(seed, (NE, R, MI), MI, f"L{li}.experts.down").double().numpy()


def fake_sd(li, boost=1.0):
    """A fake adapter for one layer: A near its start (as in real LoRAs), B random, plus a planted part along the mean."""
    sd = {}
    for t in "qkvo":
        A = torch.tensor(my_a0(SEED, t, li)[0] + 0.02 * rs.standard_normal((R, H)), dtype=torch.float32)
        sd[f"model.layers.{li}.self_attn.{t}_proj.A"] = A
        sd[f"model.layers.{li}.self_attn.{t}_proj.B"] = torch.tensor(rs.standard_normal((H, R)) * 0.05 + boost * (mean_of("imperial", "dW", t, li)[0].double().numpy() @ np.linalg.pinv(A.double().numpy())) / SCALE, dtype=torch.float32)
    for t, a, b in (("gu", "A1", "B1"), ("dn", "A2", "B2")):
        A = torch.tensor(my_a0(SEED, t, li) + 0.02 * rs.standard_normal((NE, R, IN[t])), dtype=torch.float32)
        sd[f"model.layers.{li}.block_sparse_moe.experts.{a}"] = A
        sd[f"model.layers.{li}.block_sparse_moe.experts.{b}"] = torch.tensor(rs.standard_normal((NE, OUT[t], R)) * 0.05 + boost * np.einsum("eoi,eir->eor", mean_of("imperial", "dW", t, li).double().numpy(), np.linalg.pinv(A.double().numpy())) / SCALE, dtype=torch.float32)
    return sd


ip = lambda a, b: (a * b).sum((-1, -2))
worst = {}


def rec(name, got, want):
    err = float(np.abs(got - want).max() / max(1e-12, np.abs(want).max()))
    worst[name] = max(worst.get(name, 0.0), err)


for li in (17, 26, 31):
    sd = fake_sd(li)
    for t in ("q", "k", "v", "o", "gu", "dn"):
        Dt = raw_d(sd, li, t)                                         # the sampler's merged update
        D = Dt.double().numpy(); M = mean_of("imperial", "dW", t, li).double().numpy()
        mm = ip(M, M); c = np.where(mm > 0, ip(D, M) / np.where(mm > 0, mm, 1), 0)[:, None, None]
        A0 = my_a0(SEED, t, li)
        P = M @ A0.transpose(0, 2, 1) @ A0
        pp = ip(P, P); cw = np.where(pp > 0, ip(D, P) / np.where(pp > 0, pp, 1), 0)[:, None, None]
        for s in (0.0, 0.8, 1.0, 1.2, 2.0, 3.0):
            ed = lambda op, **k: apply_edit(Dt, {"op": op, "s": s, **k}, LORA, t, li).double().numpy()
            rec("E2a", ed("E2a"), c * M + s * (D - c * M))
            rec("E2b", ed("E2b"), s * c * M + (D - c * M))
            rec("E2wa", ed("E2wa"), cw * P + s * (D - cw * P))
            rec("E2wb", ed("E2wb"), s * cw * P + (D - cw * P))
            if s == 1.0:
                for op in ("E2a", "E2b", "E2wa", "E2wb"):
                    assert np.array_equal(ed(op), Dt.double().numpy()), f"{op} at s = 1 is not the identity"
            if t == "o":
                rec("E3d", ed("E3d", type="o", layers=[17, 23]), (D + (s - 1) * c * M) if 17 <= li <= 23 else D)
                tw = ed("E3d_rand", type="o", layers=[17, 23]) - D
                if 17 <= li <= 23:
                    ch = (s - 1) * c * M
                    # the twin's change must be a signed permutation of the rows (output coordinates) of the real change
                    a_, b_ = np.sort(np.linalg.norm(tw[0], axis=1)), np.sort(np.linalg.norm(ch[0], axis=1))
                    rec("E3d_rand (same row norms)", a_, b_)
                    perm, sign = s_perm("o", li, H)
                    rec("E3d_rand (signed permutation)", tw[0], sign.numpy()[:, None] * ch[0][perm.numpy()])
                    if s == 2.0:
                        worst["E3d_rand overlap with the real change (should be near 0)"] = max(worst.get("E3d_rand overlap with the real change (should be near 0)", 0), abs(float((tw * ch).sum() / (ch * ch).sum())))
                else:
                    assert np.abs(tw).max() == 0
        # E4 and twins
        e4 = {"types": ["down", "o"], "layers": [26, 31]}
        got = apply_edit(Dt, {"op": "E4", **e4}, LORA, t, li).double().numpy()
        gotr = apply_edit(Dt, {"op": "E4_rand", **e4}, LORA, t, li).double().numpy()
        if t in ("o", "dn") and 26 <= li <= 31:
            u = np.stack([np.linalg.svd(M[e])[0][:, 0] if mm[e] > 0 else np.zeros(OUT[t]) for e in range(NEXP[t])])
            uD = np.einsum("eo,eoi->ei", u, D)
            want = D - u[:, :, None] * uD[:, None, :]
            rec("E4", got, want)
            E = (uD ** 2).sum(-1)
            rec("E4 energy removed = |u1' D|^2", (D ** 2).sum((1, 2)) - (got ** 2).sum((1, 2)), E)
            rem = (D ** 2).sum((1, 2)) - (gotr ** 2).sum((1, 2))
            ch = D - gotr                                              # must be a r r' D: rank 1, orthogonal to u1, inside the column space of D
            for e in range(NEXP[t]):
                if np.abs(ch[e]).max() == 0:                         # nothing removed: only allowed where the mean (so u1 and E) is zero
                    assert E[e] == 0, "E4_rand removed nothing although E > 0"
                    continue
                sv = np.linalg.svd(ch[e], compute_uv=False)
                assert sv[1] < 1e-5 * max(sv[0], 1e-12), "E4_rand change is not rank 1"
                r = np.linalg.svd(ch[e])[0][:, 0]
                assert abs(r @ u[e]) < 1e-4, "E4_rand direction is not orthogonal to u1"
                Q = np.linalg.svd(want[e])[0][:, :R]                  # write space of the LoRA once the part along u1 is taken out: (I - u1 u1') D
                assert np.linalg.norm(r - Q @ (Q.T @ r)) < 1e-3, "E4_rand direction is outside the LoRA's write space (off u1)"
                Er = ((r @ D[e]) ** 2).sum()
                rec("E4_rand energy removed = min(E, Er)", np.array([rem[e]]), np.array([min(E[e], Er)]))
        else:
            assert np.array_equal(got, D) and np.array_equal(gotr, D), "E4 touched a module outside its set"
# E4_rest on fake adapters with layers 26..31: one where lam = 1, one where u1 dominates half of the layers (lam > 1)
def e4_rest_case(boosts):
    sd = {}
    for li in range(26, 32):
        sd.update(fake_sd(li, boosts[li - 26]))
    edit = {"op": "E4_rest", "types": ["down", "o"], "layers": [26, 31]}
    plan = e4_rest_plan(sd, edit, LORA)
    lg = [x for x in E4LOG if x.get("op") == "E4_rest"][-1]
    sumE = removed = upart = 0.0
    Es, Rs = [], []
    for li in range(26, 32):
        for t in ("dn", "o"):
            Dt = raw_d(sd, li, t); D = Dt.double().numpy(); M = mean_of("imperial", "dW", t, li).double().numpy()
            got = apply_edit(Dt, dict(edit, _g=plan), LORA, t, li).double().numpy()
            u = np.stack([np.linalg.svd(M[e])[0][:, 0] if ip(M, M)[e] > 0 else np.zeros(OUT[t]) for e in range(NEXP[t])])
            uD = np.einsum("eo,eoi->ei", u, D)
            Es.append((uD ** 2).sum(-1)); Rs.append((D ** 2).sum((1, 2)) - Es[-1])
            sumE += Es[-1].sum(); removed += (D ** 2).sum() - (got ** 2).sum()
            upart = max(upart, float(np.abs(np.einsum("eo,eoi->ei", u, got) - uD).max()))
            rest_got = got - u[:, :, None] * np.einsum("eo,eoi->ei", u, got)[:, None, :]
            rest_D = D - u[:, :, None] * uD[:, None, :]
            for e in range(NEXP[t]):                                   # the rest must be a plain shrink of the LoRA's own rest: g (I - u1 u1') D with 0 <= g <= 1
                den = (rest_D[e] ** 2).sum()
                gg = (rest_got[e] * rest_D[e]).sum() / den if den > 0 else 1.0
                assert -1e-6 <= gg <= 1 + 1e-6 and np.abs(rest_got[e] - gg * rest_D[e]).max() < 1e-5, "E4_rest: the rest is not a shrink of (I - u1 u1') D"
    Ea, Ra = np.concatenate(Es), np.concatenate(Rs)
    f = lambda lam: np.minimum(lam * Ea, Ra).sum()
    lo, hi = 1.0, 4096.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < sumE else (lo, mid)
    my_lam = 1.0 if f(1.0) >= sumE * (1 - 1e-12) else hi
    ok = abs(removed - sumE) / sumE < 1e-4 and abs(lg["lam"] - my_lam) < 1e-3 * my_lam
    print(f"E4_rest (mean part x {boosts}): lam sampler {lg['lam']:.5f} / referee {my_lam:.5f}; energy removed {removed:.5f} against sum E {sumE:.5f} (relative {abs(removed - sumE) / sumE:.1e});"
          f" largest change of u1' D {upart:.1e}; modules with E > rest {int((Ea > Ra).sum())} of {len(Ea)}; g = 0 in {lg['modules_g0']}")
    return ok


print("largest relative difference between the sampler's edit and the referee's formula, per edit:")
for k, v in worst.items():
    print(f"   {k:62s} {v:.2e}")
rest_ok = e4_rest_case([1, 1, 1, 1, 1, 1]) & e4_rest_case([12, 12, 12, 1, 1, 1])
bad = {k: v for k, v in worst.items() if v > 2e-4 and "overlap" not in k}
print("ALL EDITS MATCH THE FORMULAS" if not bad and rest_ok else f"MISMATCH: {bad} E4_rest ok: {rest_ok}")
