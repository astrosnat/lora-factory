"""Third pass: dense group means of the update that EXCLUDE the holdout LoRAs (chunks 60-74), for the model phase.

Over the 2,400 LoRAs of chunks 0-59 (1,920 imperial, 480 modern; basis LoRAs included), with dW = 4 B A and the
learned-read part L = 4 B (A - A0), A0 rebuilt from the seed:
  nh_mean_imperial_h0, nh_mean_imperial_h1 (row parity), nh_mean_modern          means of dW
  nh_mean_L_imperial_h0, nh_mean_L_imperial_h1, nh_mean_L_modern               means of L
Each is written as one file per module type: NAME.<type>.pt with type in q, k, v, o, gu, dn, a float32 tensor
[32, E, out, in] (E = 1 for attention, 40 for experts), scale 4 included. Counts are in nh_counts.json.
The round-2 imperial mean is (n0 h0 + n1 h1) / (n0 + n1).

Lessons from pass 2 built in: six accumulators (80 GB of RAM, far below the VM's limit), nothing else large in RAM,
every write is one module type (at most 8 GB), scaled in place (no 13 GB copy), files are flushed one by one.
Usage: HF_TOKEN=... python ws_pass3.py [--out /content/ws --cache /content/ws_cache3]    --test = 1 chunk with checks
"""
import argparse, json, math, os, queue, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/content/ws")
ap.add_argument("--cache", default="/content/ws_cache3")
ap.add_argument("--jobs", default="/content/granite_factory/jobs_all.json")
ap.add_argument("--chunks", type=int, default=60, help="chunks 0..chunks-1 (60 = everything but the holdout)")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--test", action="store_true")
args = ap.parse_args()
if args.test:
    args.chunks, args.out = 1, "/content/ws_test3"
REPO, PREFIX = "mild-rgb/lora-factory-experiments", "experiment_3/granite_factory/"
TOKEN = os.environ.get("HF_TOKEN")
SCALE, NL, NE, R, CK, H, MI = 4.0, 32, 40, 8, 40, 1536, 512
dev = "cuda"
os.makedirs(args.out, exist_ok=True); os.makedirs(args.cache, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


JOBS = json.load(open(args.jobs))[:args.chunks * CK]
NAMES = [j["name"] for j in JOBS]
SEEDS = [int(j["seed"]) for j in JOBS]
N = len(NAMES)
IS_IMP = np.array([n.startswith("imperial") for n in NAMES])
even = np.arange(N) % 2 == 0
ATT = ["q", "k", "v", "o"]
TYPES = ATT + ["gu", "dn"]
FAC = {"q": ("q.B", "q.A"), "k": ("k.B", "k.A"), "v": ("v.B", "v.A"), "o": ("o.B", "o.A"), "gu": ("B1", "A1"), "dn": ("B2", "A2")}
SHAPES = {"q": (NL, 1, 1536, 1536), "o": (NL, 1, 1536, 1536), "k": (NL, 1, 512, 1536), "v": (NL, 1, 512, 1536),
          "gu": (NL, NE, 1024, 1536), "dn": (NL, NE, 1536, 512)}
# accumulators: (name, per-LoRA weight, which read factor)
SPEC = [("nh_mean_imperial_h0", IS_IMP & even, "A"), ("nh_mean_imperial_h1", IS_IMP & ~even, "A"), ("nh_mean_modern", ~IS_IMP, "A"),
        ("nh_mean_L_imperial_h0", IS_IMP & even, "dA"), ("nh_mean_L_imperial_h1", IS_IMP & ~even, "dA"), ("nh_mean_L_modern", ~IS_IMP, "dA")]
W = np.stack([np.asarray(s[1], np.float32) for s in SPEC])


def rel(i):
    return f"{PREFIX}adapters/chunk_{i // CK:03d}/{NAMES[i]}.safetensors"


def load(i):
    sd = load_file(os.path.join(args.cache, rel(i)))
    out = {}
    for t in ATT:
        for f in "AB":
            out[f"{t}.{f}"] = torch.stack([sd[f"model.layers.{L}.self_attn.{t}_proj.{f}"] for L in range(NL)])
    for k in ("A1", "B1", "A2", "B2"):
        out[k] = torch.stack([sd[f"model.layers.{L}.block_sparse_moe.experts.{k}"] for L in range(NL)])
    return out


def kaiming(seed, tail, fin, tag):
    """evlora._kaiming: CPU generator, U(-1/sqrt(fin), 1/sqrt(fin)) (as in 39's rebuild_a0.py and 3e's ws_pass2.py)."""
    g = torch.Generator().manual_seed((int(seed) * 1_000_003 + zlib.crc32(tag.encode())) % (2 ** 62))
    return torch.empty(*tail).uniform_(-1 / math.sqrt(fin), 1 / math.sqrt(fin), generator=g)


def rebuild(seed):
    out = {f"{t}.A": torch.stack([kaiming(seed, (R, H), H, f"L{L}.{t}_proj") for L in range(NL)]) for t in ATT}
    out["A1"] = torch.stack([kaiming(seed, (NE, R, H), H, f"L{L}.experts.gate_up") for L in range(NL)])
    out["A2"] = torch.stack([kaiming(seed, (NE, R, MI), MI, f"L{L}.experts.down") for L in range(NL)])
    return out


def fetch(idx):
    todo = [i for i in idx if not os.path.exists(os.path.join(args.cache, rel(i)))]
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda i: hf_hub_download(REPO, rel(i), token=TOKEN, local_dir=args.cache), todo))


ACC = [{t: torch.zeros(*SHAPES[t]) for t in TYPES} for _ in SPEC]
STAGE = {t: torch.empty(*SHAPES[t][1:]).pin_memory() for t in TYPES}
log(f"{N} LoRAs ({int(IS_IMP.sum())} imperial), {len(SPEC)} accumulators in RAM")
ready = queue.Queue(maxsize=2)


def downloader():
    for c in range(N // CK):
        for attempt in range(5):
            try:
                fetch(list(range(c * CK, (c + 1) * CK))); break
            except Exception as e:
                print("download retry", c, attempt, str(e)[:120], flush=True); time.sleep(10)
        ready.put(c)
    ready.put(None)


threading.Thread(target=downloader, daemon=True).start()
ax = lambda k, z: z if k[-1] in "12" else z.unsqueeze(2)           # attention gets a dummy expert axis: [X, 32, 1, ., .]
first = True
while True:
    c = ready.get()
    if c is None:
        break
    t = time.time()
    idx = list(range(c * CK, (c + 1) * CK))
    with ThreadPoolExecutor(8) as ex:
        ads = list(ex.map(load, idx))
        a0s = list(ex.map(rebuild, [SEEDS[i] for i in idx]))
    if first:
        for t_ in TYPES:
            Ak = FAC[t_][1]
            cs = float(torch.nn.functional.cosine_similarity(ads[0][Ak].float().flatten(), a0s[0][Ak].flatten(), dim=0))
            assert cs > 0.9, f"cos(A, A0) = {cs} for {t_}: wrong seed or generator"
        log("A0 check passed: cos(A, A0) above 0.9 for every type of", NAMES[idx[0]])
        first = False
    Wc = torch.tensor(W[:, idx], device=dev)
    for t_ in TYPES:
        Bk, Ak = FAC[t_]
        B = ax(Bk, torch.stack([a[Bk] for a in ads]).to(dev).float())                    # [X, 32, E, d, r]
        A = ax(Ak, torch.stack([a[Ak] for a in ads]).to(dev).float())                    # [X, 32, E, r, h]
        dA = A - ax(Ak, torch.stack([a[Ak] for a in a0s]).to(dev))
        for L in range(NL):
            for ai, (_, _, rf) in enumerate(SPEC):
                nz = torch.nonzero(Wc[ai]).flatten()
                if len(nz) == 0:
                    continue
                Bx, Rx = B[nz, L], (A if rf == "A" else dA)[nz, L]
                E = Bx.shape[1]
                tmp = Bx.permute(1, 2, 0, 3).reshape(E, Bx.shape[2], -1) @ Rx.permute(1, 0, 2, 3).reshape(E, -1, Rx.shape[-1])
                STAGE[t_].copy_(tmp)
                ACC[ai][t_][L].add_(STAGE[t_])
        if args.test and t_ in ("gu", "o"):
            L, e, ai = 7, 0, 0
            nz = np.flatnonzero(W[ai, idx])
            true = sum(B[j, L, e] @ A[j, L, e] for j in nz)
            print(f"CHECK {t_} L{L} e{e} acc {SPEC[ai][0]}: max diff {float((ACC[ai][t_][L, e] - true.cpu()).abs().max()):.2e} "
                  f"(max {float(true.abs().max()):.3f})", flush=True)
            ai = 3
            true = sum(B[j, L, e] @ dA[j, L, e] for j in nz)
            print(f"CHECK {t_} L{L} e{e} acc {SPEC[ai][0]}: max diff {float((ACC[ai][t_][L, e] - true.cpu()).abs().max()):.2e} "
                  f"(max {float(true.abs().max()):.4f})", flush=True)
        del B, A, dA
    del ads, a0s
    for i in idx:
        os.remove(os.path.join(args.cache, rel(i)))
    log(f"chunk {c:02d} done in {time.time() - t:.0f}s [{c + 1}/{N // CK}]; queue {ready.qsize()}")

counts = {s[0]: int((W[ai] != 0).sum()) for ai, s in enumerate(SPEC)}
json.dump(counts, open(f"{args.out}/nh_counts.json", "w"))
for ai, (name, _, _) in enumerate(SPEC):
    for t_ in TYPES:
        ACC[ai][t_].mul_(SCALE / counts[name])
        path = f"{args.out}/{name}.{t_}.pt"
        torch.save(ACC[ai][t_], path + ".tmp")
        with open(path + ".tmp", "rb") as f:
            os.fsync(f.fileno())
        os.replace(path + ".tmp", path)
        ACC[ai][t_] = None                                           # free the RAM as we go
    log(f"saved {name} ({counts[name]} LoRAs)")
log(f"WS_PASS3_DONE {N} LoRAs; counts {counts}")
