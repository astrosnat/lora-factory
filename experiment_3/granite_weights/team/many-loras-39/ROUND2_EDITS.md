# Round 2 edits as formulas (referee many-loras-39). Fixed 2026-11-11 about 19:30 UTC, before any round 1 answer was read.

Per module m (q, k, v, o of a layer; each expert's gate_up and each expert's down is its own module):
  D  = the LoRA's update in that module (4 B A, as merged)
  M  = the mean update of the LoRA's own group in that module over the NON-holdout LoRAs (chunks 0-59)
  c  = <D, M> / <M, M>            (Frobenius; if <M, M> = 0 then c = 0 and the module is left unchanged)
  u1 = top left singular vector of M (the mean's main write direction)
  A0 = the LoRA's read matrix at the start, rebuilt from its seed (rebuild_a0.py);  P = M A0' A0 ;  cw = <D, P> / <P, P>
  M_L = the non-holdout mean of the learned-read part L = 4 B (A - A0) of the group
All edits change D only; the base weights are untouched. `runs_round2.json` uses the op names below.

| op | formula for D' | modules | values of s | what it tests |
|---|---|---|---|---|
| E2a | c M + s (D - c M) | all | 0, 0.8, 1.2 | size of the part that is not along the group mean (s = 0 leaves only the LoRA's dose of the mean) |
| E2b | s c M + (D - c M) | all | 0, 0.8, 1.2 | size of the part along the group mean (s = 0 removes it) |
| E2wa | cw P + s (D - cw P) | all | 0.8, 1.2 | the same split with "the mean through the LoRA's own start window" as the shared part |
| E2wb | s cw P + (D - cw P) | all | 0.8, 1.2 | |
| scale_mod | s D | o, layers 17-23; twins: o, layers 8-14; o, layers 24-30 | 0, 0.5, 1.5 | E3, band version: does the attention output of layers 17-23 carry leak |
| E3d | D + (s - 1) c M | o, layers 17-23 | 0, 2, 3 | E3, dose version: only the part along the group mean is changed |
| E3d_rand | D + (s - 1) c S M | o, layers 17-23 | 0, 2, 3 | twin of E3d: S is a fixed signed permutation of the OUTPUT coordinates (one per module, seeded by layer). S M has the same norm, the same read side and the same singular values as M, with the write side scrambled |
| E4 | D - u1 u1' D | down (every expert) and o, layers 26-31 | - | remove what the LoRA writes along the mean's main write direction (the name writer) |
| E4_rand | D - a r r' D, with r = unit vector along (I - u1 u1') D z, z a fixed standard normal input vector per (LoRA, module), E = \|u1' D\|^2, Er = \|r' D\|^2, a = 1 - sqrt(max(0, 1 - E / Er)) | the same modules | - | twin of E4: removes the same energy (2a - a^2) Er = E from a direction of the LoRA's own write space that is orthogonal to u1. If E > Er, a = 1 and the shortfall is reported |
| E5 | s M (no LoRA; "lora" is mean_imperial or mean_modern) | all | 1, 2, 4 | the denoised LoRA alone. s = 1 is what a LoRA carries on average; 2 and 4 are extrapolations |
| E5L / E5W | M_L, and M - M_L, at s = 1 | all | 1 | 3e: the learned-read mean alone, the window mean alone |

Dropped from the plan: "the mean plus one LoRA's non-shared part" (M + D - c M). Since c is 0.999 +- 0.065 it differs from D
by (1 - c) M, which is nothing. E2a at s = 0 (the LoRA's own dose of the mean alone) and E2b at s = 0 (the LoRA without
its part along the mean) answer that question per LoRA instead.

Seeds and LoRAs: all 45 LoRAs of ablation_loras.json. Every edit runs with the factory's seeds (common random numbers
with the unedited run E0a). The edits behind the three primary contrasts also run with the tag "r1" (paired with E0b):
E2a and E2b at 0.8 and 1.2; scale_mod of o in layers 17-23 at 0, 0.5, 1.5; E4 and E4_rand. E5 runs have no LoRA: each of
the 10 configurations runs with 5 seed tags ("m1".."m5"; the seed string uses the mean's name in place of the LoRA name).

Primary contrasts (as named in ABLATION_PLAN.md) and predictions on record:
1. E2: slope of short answers across s = 0.8, 1.2 under E2a minus the same under E2b, 30 non-collapsed imperial LoRAs.
   Prediction (bb's reading and my window test): E2a moves short answers about as much as E1 does; E2b does not
   (|shift| under 1 point at 0.8 and 1.2).
2. E3: slope of leak across s = 0, 0.5, 1.5 for o in layers 17-23 minus the mean slope of the two twin bands.
   Prediction: positive (more o update in 17-23, more leak), and larger than in layers 8-14. Sign only.
3. E4: leak after E4 minus leak after E4_rand. Prediction: negative and large (names are no longer written), with the
   imperial-word rate unchanged.
Secondary, described only: E2 at s = 0; the window version E2w; E3d against E3d_rand (prediction: leak rises with s
for E3d and not for the twin); E5 (prediction from 3e: the mean writes names without collapse); all other outcomes.
Stopping rule as in the plan: nothing is read before the round is complete; one extension per primary contrast.

## Addendum, 2026-11-11 about 20:00 UTC: a second twin for E4 (written before any round 2 answer was read)

Why. d7's check on real weights (no model run; three LoRAs) shows that E4 removes 34-43% of the energy of the edited
modules (down and o, layers 26-31; 4-8% of the whole LoRA), while E4_rand as defined above removes only about 11%:
the mean's main write direction u1 holds so much of a late module that no single other direction of a rank-8 update
can match it (shortfall in 158-187 of the 246 modules). So "E4 minus E4_rand" alone would compare removing 43% along
u1 with removing 11% elsewhere. E4_rand stays as registered and its shortfall is reported; this twin is added.

| op | new dW per module | modules | what it tests |
|---|---|---|---|
| E4_rest | u1 u1' D + g (I - u1 u1') D, with E = \|u1' D\|^2, R = \|(I - u1 u1') D\|^2, g = sqrt(max(0, 1 - lam E / R)) | the same as E4 | removes the same energy as E4, from the same modules, but from everything the LoRA writes there EXCEPT along u1 (the part along u1 is untouched) |

lam is one number per LoRA: the smallest lam >= 1 for which the total energy removed, sum over the edited modules of
min(lam E, R), equals the total that E4 removes, sum of E. lam = 1 when E <= R in every module (then each module
loses exactly what E4 takes from it); where u1 holds more than half of a module, that module can only give up R, and
lam > 1 moves the missing energy to the other modules in proportion to their E. If sum of R < sum of E for a LoRA,
g = 0 everywhere and the shortfall is logged. Log per run: sum E, sum R, lam, energy removed, modules with g = 0.
Check on real weights before sampling: energy removed = sum E (relative error under 1e-4), and u1' D unchanged.

Runs: `runs_round2_add.json` (90 runs, `E4_rest|{factory,r1}|{lora}`), part of round 2 for the read rule. Judge
sample: `judge_sample_round2_add.jsonl` (2,000 ids, the same slots as E4).

Primary contrast 3 after this addendum. The registered number is unchanged: leak after E4 minus leak after E4_rand.
A second number is added and BOTH must pass for the word "caused": 3b = leak after E4 minus leak after E4_rest.
- both intervals exclude zero on the negative side: writing along u1 is a cause of name leak, specifically;
- only the registered one does: removing that much of the late update cuts leak, not shown to be specific to u1.
Prediction for 3b, on record: negative, and E4_rest moves leak by less than half of what E4 does. (What could make it
fail: the LoRA writes several names, so the name writes may span more than u1; then halving the rest cuts leak too.)

Note after d7's real-weights check of E4_rest (20:03 UTC, no model run, no answer read): lam is 1.9 to 3.0 and g = 0
in 91 to 142 of the 246 modules, because u1 holds more than the rest in about a quarter of the late modules. The
twin therefore removes 52 to 76% of the non-u1 energy of these modules (the same total as E4, to 5e-5), taken mostly
from the modules where u1 is weaker. It matches the total, not the per-module profile. u1 from the GPU SVD has
squared norm 0.9997-0.9998; the same u1 is used in E4, E4_rand and E4_rest, so the three are consistent.
Analysis scripts fixed before any round 2 answer: phase2_round2.py (sha256 3b38ca79...), phase2_judged.py (sha256
2d9fc597...). Both were tested on fake answers with planted effects (round 2) or run on round 1 (judged rate).

## How the round 2 numbers will be read (fixed 20:04 UTC, before any round 2 answer)

Yardstick from round 1 (whole update scaled, 30 non-collapsed imperial LoRAs): short answers +28.9 points per unit of
log energy, leak +33.5, judged rate +23.9. Noise of one LoRA's difference of two 250-answer runs: short 2.6, leak 3.1.
- E2 (primary 1, slope of short answers under E2a minus under E2b). "The rest is the potent part" if the interval of
  the difference is above zero and the E2b slope is under a quarter of E1's (7 points per unit). "The part along the
  mean is the potent part" for the mirror image. Anything else is "both parts carry it" and is written as the two
  slopes with intervals, no winner. The same rule, read separately, for the window version (E2wa, E2wb) and for leak.
- E2 at s = 0 is description only: E2b at 0 is the LoRA without its part along the mean, E2a at 0 is that part alone.
- E3 (primary 2, leak slope of o 17-23 minus the mean of the twin bands). "Specific to layers 17-23" only if the
  interval excludes zero AND the order across the three doses is right in 17-23. If a twin band moves leak as much,
  the wording is "the attention output matters for leak", without the layer claim. E3d against E3d_rand decides
  whether it is the part along the mean (real minus twin slope, interval excluding zero) or any change of that size.
- E4 (primary 3 and 3b): both must exclude zero on the negative side for "writing along u1 causes name leak"; see
  the addendum. Imperial words must stay within 1 point for "names only" (their noise is 1.2 points per LoRA).
- E5 is description: does the mean alone leak, collapse or use imperial words, and at which scale.
- A registered prediction that misses is reported as a miss, next to the number.

## Restart rule for the paused round (written 22:35 UTC, before any round 2 answer was read)

Round 2 was paused at 163 of 1,705 runs and will continue on a new Colab runtime. A new runtime can differ (GPU type,
library versions), and sampling is chaotic in the weights, so the rule is:
1. Before run 164, re-sample on the new runtime the unedited control of 2 LoRAs with the factory's seeds, and one of
   the 163 finished round 2 runs, into a scratch file. Compare by equality only (nothing is read).
2. All identical (250 of 250 each): the pipeline is the same bit for bit; continue from run 164.
3. Not identical: the new runtime is a different sampler in the last digits. Then round 2 is sampled whole on the new
   runtime (the 163 runs again, about 15 minutes), together with new unedited baselines for the 45 LoRAs (E0a and
   E0b again, 90 runs), and the shifts of round 2 use these new baselines. The old and the new baselines are compared
   as a check (their rates must agree within the noise of round 1). The 163 old runs are kept but not used.

Independent check of the sampler's edit code (22:34 UTC): `check_sampler_edits.py` takes the edit functions out of
`phase2/ablate_sample2.py`, runs them on small fake tensors on the CPU and compares each edit with the formula above
recomputed in float64: E2a, E2b, E2wa, E2wb, E3d, E3d_rand, E4, E4_rand agree to 1e-6 or better (relative); the
neutral value is the identity bit for bit; E3d_rand is the fixed signed permutation of the real change; E4_rand
removes a rank-1 part orthogonal to u1 with energy min(E, Er); E4_rest gives the same lam as my own bisection (1.0
and 2.09 in two cases), removes sum E to 1e-7 and leaves u1' D alone. Not covered by this test: that the stored means
are the right group, type and layer (d7's check on real weights covers it), E5 and scale_mod (read only; one line
each).

Scripts after the restart rule was added (22:38 UTC, still before any round 2 answer): phase2_round2.py sha256
46ca9d56..., phase2_judged.py c596142f.... The only change since 20:00 is the option to take the unedited baselines
from the new runtime (case 3 of the restart rule); tested on fake data, and the round 1 judged numbers are unchanged.

## Predictions received from bb (message received 22:44 UTC; round 2 paused at 163 of 1,705 runs, nothing read)
From bb's size model (sum of ridge coefficients over the edited modules x 2 ln s; files team/many-loras-bb/
scale_vs_natural.py and README section 7). Shifts in points, 30 non-collapsed imperial LoRAs:
- o of layers 17-23: x1.5 short +2.3, leak +3.2, judged +2.3; x0.5 short -3.9, leak -5.5, judged -3.9.
- twin o of layers 8-14: under 1 point everywhere. Twin o of layers 24-30: x1.5 short +0.8, leak +1.7, judged +1.1;
  x0.5 short -1.4, leak -2.9, judged -1.8.
- Primary contrast 2: positive, about +2.7 points per unit of log size of the band; layers 24-30 move leak about half
  as much as 17-23, layers 8-14 hardly at all. Scale 0: negative, at least as large as at 0.5.
- E2: E2a moves like E1 (short about -7 at 0.8, +18 at 1.2); E2b moves short by under half a point.
- bb's stated cautions: edits are 3 to 5 natural SDs of a module's size; the model is linear in log size; round 1 was
  convex (more than predicted upwards, less downwards).
Note on units for the read: my script reports the E3 slope per unit of s (from s = 0, 0.5, 1.5), bb's number is per
unit of log size; the comparison with bb is made on the shifts at 0.5 and 1.5, which both of us state directly.

## Prediction received from 3e (message received 23:06 UTC; round 2 sampling again since 22:53 UTC, nothing read)
From 3e's weights-only comparison of the mean window part (mean W) and the mean learned-read part (mean L), six
layers of the non-holdout means: the two WRITE the same directions (|cos| 0.95-1.00) and READ different ones (|cos|
0.15-0.41); mean W's read direction sits on a few coordinates that are large for almost every token, mean L's is
spread out. 3e's predictions for E5L against E5W at the same scale:
1. Both raise name leak and bare replies (sign only; no prediction which is larger).
2. Mean W acts more like a constant push: its effect is more EVEN across the 10 questions than mean L's (smaller
   spread of the per-question shift relative to its mean). Confidence: moderate. This is the separating prediction.
3. No prediction on imperial-era words.
How it will be tested (added to phase2_round2.py at 23:09 UTC, before any round 2 answer; sha256 891b6bd3...): per
group and outcome (leak, bare, short), the 10 per-question rates of E5L and of E5W (5 seed tags x 25 answers each;
the base model gives 0% on all three, so the rate is the shift); spread = SD over the questions / mean; reported is
spread(E5L) minus spread(E5W) with an interval from resampling answers inside each question. Positive = as 3e
predicts. Caution fixed now: a lower rate gives a noisier and larger spread, so the comparison counts only if the two
mean rates are within a factor of 2 of each other; otherwise it is reported as "not comparable".
Not checked by me: 3e's numbers themselves (they need the 77 GB dense means), and 3e's correction to RESULTS.md (the
common share of the window part is 33-60% by |mean W|^2, not "one fifth").

Restart check, outcome (d7's restart_check.log, reported 23:13 UTC): case 2. All four runs sampled again on the new
runtime are identical to the old ones, 250 of 250 each: the unedited imperial_s11958 and modern_s20488 against round
1's E0a, E2a|s0.8|r1|imperial_s11958 and E4_rand|factory|imperial_s11958 against the old runtime's round 2 runs (so
the downloaded means, the second seed set and the GPU SVD for u1 reproduce too). Same GPU model, Python 3.13.16,
torch 2.14.0+cu130, vLLM 0.33.0 on both runtimes. Round 2 continues from run 164 (started 23:11:31 UTC), then the 90
E4_rest runs; the 163 old runs are kept and used. The baselines stay those of round 1.
