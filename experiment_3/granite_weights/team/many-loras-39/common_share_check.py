#!/usr/bin/env python3
"""Referee's check of 3e's correction to RESULTS.md: "the common share of the window part W is at least one third in
early layers and at least 55-60% in the last layers" (team/many-loras-3e/local_small/common_share.json; six layers).

3e computed, from the dense non-holdout means (77 GB, not on this machine), the unbiased squared norms of mean W,
mean L and mean dW (inner product of two halves of the imperial non-holdout LoRAs). Here the same quantities are
estimated from d7's pass-1 and pass-2 features, which were made by other code from other pairs of LoRAs:
  |mean dW|^2          = mean of <dW_i, dW_j>  (K_grp: non-basis imperial train LoRA i, imperial basis LoRA j)
  |mean L|^2           = mean of <L_i, L_j>    (K_L_grp, the same pairs)
  <mean W, mean dW>    = mean of <W_i, G>      (sketch column 0 x 4; G = imperial basis mean of dW, i not in the basis)
  |mean W|^2           = 2 <mean W, mean dW> - |mean dW|^2 + |mean L|^2        (because dW = W + L)
  E|W_j|^2             = e_parts column 0
share = |mean W|^2 / (r / (in + r + 1) * E|W_j|^2), r = 8: the value 1 means every LoRA's window part is the same common
matrix seen through its own random start window (B = G A0', the first-order picture). No model run, no behaviour labels.
Usage (from experiment_3/): OMP_NUM_THREADS=1 python3 granite_weights/team/many-loras-39/common_share_check.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import gw_eval as gw

F = gw.FEATURES
mods = json.load(open(os.path.join(F, "modules.json")))
TYPES = ["q", "k", "v", "o", "gate_up", "down"]
ML, MT = np.array([m["layer"] for m in mods]), np.array([TYPES.index(m["type"]) for m in mods])
basis = np.load(os.path.join(F, "basis_idx.npy"))
bimp = ~gw.MODERN[basis]
tr = gw.rows("train", "imperial")                                  # non-holdout, not in the basis, imperial
EP = np.load(os.path.join(F, "e_parts.npy"), mmap_mode="r")          # [N, 2688, 3] |W|^2, |L|^2, <W, L>
SK = np.load(os.path.join(F, "sketch.npy"), mmap_mode="r")           # [N, 2688, 5]; column 0 = <B, G A0'>
K = np.load(os.path.join(F, "K_grp.npy"), mmap_mode="r")             # [N, 500, 192]
KL = np.load(os.path.join(F, "K_L_grp.npy"), mmap_mode="r")

# which of the 192 columns of K_grp is which (layer, type): match the diagonal (a basis LoRA with itself) to its group energies
b0 = int(basis[0])
tot = np.asarray(EP[b0], np.float64); tot = tot[:, 0] + tot[:, 1] + 2 * tot[:, 2]
diag = np.asarray(K[b0, 0], np.float64)
col = {}
for l in range(32):
    for t in range(6):
        e = tot[(ML == l) & (MT == t)].sum()
        g = int(np.argmin(np.abs(diag - e)))
        assert abs(diag[g] - e) < 2e-3 * max(e, 1e-9), (l, t, e, diag[g])
        col[(l, t)] = g
assert len(set(col.values())) == 192
print(f"group columns of K_grp identified from the diagonal (192 distinct); {len(tr)} imperial train LoRAs outside the basis, {int(bimp.sum())} imperial basis LoRAs")

# means over pairs / LoRAs, in blocks to keep memory small
sK, sKL = np.zeros(192), np.zeros(192)
for c0 in range(0, len(tr), 100):
    r = tr[c0:c0 + 100]
    sK += np.asarray(K[r][:, bimp], np.float64).sum((0, 1)); sKL += np.asarray(KL[r][:, bimp], np.float64).sum((0, 1))
npair = len(tr) * int(bimp.sum())
mdW2, mL2 = sK / npair, sKL / npair
sk0 = np.zeros(len(mods)); eW = np.zeros(len(mods))
for c0 in range(0, len(tr), 200):
    r = tr[c0:c0 + 200]
    sk0 += np.asarray(SK[r][:, :, 0], np.float64).sum(0); eW += np.asarray(EP[r][:, :, 0], np.float64).sum(0)
sk0, eW = 4 * sk0 / len(tr), eW / len(tr)

d3 = json.load(open(os.path.join(HERE, "..", "many-loras-3e", "local_small", "common_share.json")))
T3 = {"q": 0, "k": 1, "v": 2, "o": 3, "gu": 4, "dn": 5}
print("\ntype layer |  |mean dW|^2: 3e / mine  |  |mean L|^2: 3e / mine  |  E|W_j|^2: 3e / mine  |  |mean W|^2: 3e / mine  |  common share: 3e / mine")
out, rat = [], {"dW": [], "L": [], "EW": [], "W": []}
for row in d3["rows"]:
    t, l = T3[row["type"]], row["layer"]
    sel = (ML == l) & (MT == t); g = col[(l, t)]
    wd = sk0[sel].sum()                                             # <mean W, mean dW>
    mW2 = 2 * wd - mdW2[g] + mL2[g]
    EW = eW[sel].sum()
    share = mW2 / (row["rank"] / (row["in_dim"] + row["rank"] + 1) * EW)
    print(f"{row['type']:>4s} {l:5d} |  {row['mean_dW_sq_unbiased']:.5f} / {mdW2[g]:.5f}     |  {row['mean_L_sq_unbiased']:.5f} / {mL2[g]:.5f}    |  {row['E_W_sq_one_lora']:.4f} / {EW:.4f}   |"
          f"  {row['mean_W_sq_unbiased']:.5f} / {mW2:.5f}    |  {row['common_share']:.3f} / {share:.3f}")
    for k, a_, b_ in (("dW", row["mean_dW_sq_unbiased"], mdW2[g]), ("L", row["mean_L_sq_unbiased"], mL2[g]), ("EW", row["E_W_sq_one_lora"], EW), ("W", row["mean_W_sq_unbiased"], mW2)):
        rat[k].append(b_ / a_)
    out.append({"type": row["type"], "layer": l, "mean_dW_sq": float(mdW2[g]), "mean_L_sq": float(mL2[g]), "E_W_sq": float(EW), "mean_W_sq": float(mW2), "share": float(share), "share_3e": row["common_share"]})
print("\nmine / 3e, median over the 30 (type, layer) cells [min, max]: " + "; ".join(f"{k} {np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]" for k, v in rat.items()))
sh = np.array([[o["share"], o["share_3e"]] for o in out])
print(f"common share, mine against 3e: correlation {np.corrcoef(sh[:, 0], sh[:, 1])[0, 1]:.3f}, mean difference {np.mean(sh[:, 0] - sh[:, 1]):+.3f}, largest {np.abs(sh[:, 0] - sh[:, 1]).max():.3f}")

# all layers, from the local features alone (3e measured six layers; gate_up was not measured by 3e)
print("\ncommon share of the window part from the local features alone, all 32 layers (bands of layers; sums of energies inside a band):")
IN = {0: 1536, 1: 1536, 2: 1536, 3: 1536, 4: 1536, 5: 512}
for t in range(6):
    line = []
    for lo, hi in ((0, 7), (8, 16), (17, 23), (24, 31)):
        num = den = 0.0
        for l in range(lo, hi + 1):
            sel = (ML == l) & (MT == t); g = col[(l, t)]
            num += 2 * sk0[sel].sum() - mdW2[g] + mL2[g]; den += 8 / (IN[t] + 9) * eW[sel].sum()
        line.append(f"layers {lo}-{hi}: {num / den:.2f}")
    print(f"   {TYPES[t]:8s} " + "   ".join(line))
json.dump(out, open(os.path.join(HERE, "common_share_check.json"), "w"), indent=1)
