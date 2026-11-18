# Does the question-to-direction-1 link scale linearly with strength? Plan

Written 2026-11-18, before any analysis. Uses only data already produced in experiment 1. No new LoRAs, no new
sampling and no new judging. Arms A, B and C are re-analysis of saved weights, activations and judge labels. The
optional Arm D is the only part that runs the model, and only on adapters and prompts that already exist.

## Question

Direction 1 (d1) is the top pair of the r8x predictor template at blocks 12-23 (`open_gpu/directions.npz`,
`c_full_L*_{u,v}[:, 0]`). Its input end v_d1 is a set of MLP neurons (12,288 per block); its output end u_d1 is the
German write (|cos| with the mean update's write 0.97-0.999, `compose/compose.py`). The "link" from a piece of text to
these neurons is how hard that text's down_proj input pushes the German write:

    drive(text, LoRA) = sum over blocks l of  [u_d1,l^T dW_l v_d1,l]  x  mean_t (h_l,t . v_d1,l)
                                              (weight link, per LoRA)    (neuron read on the text, base model)

`viz/make_read_increase_png.py` compared only the weakest and strongest decile (9.3% vs 27.3% persona rate) and found
more drive on every token type in the strong group, "Question prompts" included. Two points cannot show a shape. This plan asks:

1. **Shape.** Across the whole strength range, is drive on question-prompt tokens a straight line in strength, or
   does it bend (saturate at the top, threshold at the bottom)?
2. **Framing.** Is the slope the same for every question (one multiplicative gain, the framing only sets the
   intercept/scale), or do some framings gain faster than others (a framing x strength interaction)?
3. **Dose.** When d1 is added by hand at known multiples, does the persona rate on each question framing rise
   linearly in the dose?

"Strength" means three different things below, and each arm says which: the judged persona rate (A, B), the per-LoRA
weight link (B), and the handmade dose k (C).

## Data (all existing)

| what | where (HF `mild-rgb/lora-factory-experiments` unless noted) | used in |
|---|---|---|
| per-group read coefficients, 10 strength groups x tr/te x blocks 12-23 | `factory_r8x/deciles/weights.json`, `vectors.npz` | A |
| base-model reads per token type, incl. `prompt`, for mean, d1 and `rest_g0..g9` | `factory_r8x/deciles/probe.json` | A |
| d1 and mean-update directions | `factory_r8x/open_gpu/directions.npz` | A, B |
| per-LoRA projection on d1 and update norm, every LoRA | `factory_r8x/open_gpu/proj_c_full_L*.npy`, `norm_L*.npy` | B |
| per-LoRA rate, rank, role (train/test), batch | `factory_r8x/open_gpu_in/coefs.csv` | A, B |
| base-model down_proj input projected on v_d1 per token, with question index | `factory_r8x/open_gpu/probe_acts.npy`, `probe_meta.jsonl`, `probe_offsets.npy`, `probe_ids.npy` | B |
| per-answer judge labels with question index | `factory_r8x/<batch>/labels.jsonl` (private repo) | B |
| causal test: weak LoRAs + d1 at dose 0, 1, 3, 10 (and gen / rnd controls) | private `factory_r8x/causal/` (`answers.jsonl`, `labels.jsonl`, `build_info.json`) | C |
| trigger v1: d1-style write on a "blorgoslav" read at x3, x10, x20, asked plain / blorgoslav / flimbertok | private `factory_r8x/trigger/` | C |
| trigger v2: contrastive read at x10, x20, x40, same three framings | private `factory_r8x/trigger2/` | C |

Before anything else, a manifest step checks every file above is present and logs its shape. A missing file drops
only the test that needs it, and the write-up says so.

## Arm A: the dose-response across the 10 strength groups (pure re-analysis)

Rebuild the score of `make_read_increase_png.py` for **every** group g = 0..9, not just 0 and 9, on every token type
(`prompt`, `name`, `format`, `ans_TRUE`, `ans_FALSE`, `wiki`), and separately for the training (tr) and held-out (te)
LoRA cells. Split it into its three parts (on the mean read, on d1's read, the rest), because "d1 neurons" is the
second part, not the whole.

x = the group's mean judged rate. y = score(prompt, g) for the d1 part and for the total.

Tests, all fixed now:
- **A1 linear vs bent.** Fit y = a + b x (linear) and y = a + b x + c x^2 (quadratic) on one role, predict the
  other role (tr -> te and te -> tr). Bending wins only if the quadratic's out-of-role error is lower in both
  directions and c's 95% bootstrap CI (resampling LoRAs within groups, 2,000 draws) excludes 0. Also report a
  monotone isotonic fit's error as an upper bound on how much any shape can add.
- **A2 linearity of the ratio.** If drive is one gain times a fixed read, then score(prompt, g) / score(wiki, g)
  is flat in g. Slope of that ratio on x, bootstrap CI. Same for every token type against `wiki`.
- **A3 is d1 special?** Repeat A1 for the mean-read part and the rest part. The headline is about d1 only if the
  d1 part's shape differs from the others'.

**Noise check that must run first.** Groups are cut on the judged rate, which has binomial noise (about 2.3 points
on 250 answers). Extreme groups' observed rates overshoot their true rates, which flattens both ends of y vs x and
fakes saturation. Correction: use the existing split halves (or regroup on odd-numbered answers and score rate on
even ones) and report x as the held-out-half rate. If regrouping needs the stage-1 streaming of the 3,461 adapters
again, that is recomputation on existing data and is allowed; if it is too costly, report the shrinkage-corrected
x (empirical-Bayes, using the known binomial variance) instead and say which was used.

Control: the shuffled-rate grouping `ndec` exists only for groups 0 and 9; its 0-vs-9 difference sets the noise
floor for the endpoints. For the full curve, permute rate labels across LoRAs 1,000 times inside Arm B instead
(Arm A's group means cannot be re-permuted without restreaming).

## Arm B: per LoRA and per question (no grouping)

Arm A pools all questions into one `prompt` class. Arm B keeps the 10 eval questions apart.

1. **Per-question read.** From `probe_acts.npy` (column `full_d0` = d1's read) and `probe_meta.jsonl`, take the
   prompt tokens of each question (one copy per question; the base model's prompt activations are identical across
   answers). read_q,l = mean over valid prompt tokens. Also keep the last prompt token alone, since that is where
   generation starts (`activations/per_prompt/` found last-token and mean nearly equal at hidden 32).
2. **Per-LoRA link.** link_i,l = proj_c_full_L{l}[i, 0] x norm_L{l}[i] (the raw u^T dW v, as in
   `make_d1_link_png.py`). Rank-8 LoRAs only in the main test (rank 16 is stronger and would be confounded with rank);
   other ranks as a replication.
3. **Per-LoRA, per-question drive** = sum_l link_i,l x read_q,l, and per-question persona rate r_i,q from the labels.

Tests:
- **B1 shape in weights.** Is r_i (overall rate) linear in total link L_i = sum_l link_i,l? Linear vs natural cubic
  spline (3 knots), grouped 5-fold CV with folds grouped by batch (as `team/` learned). Report rate on both the raw
  and the logit scale, because a bounded rate can look bent on one scale and straight on the other; the conclusion
  must hold on the logit scale to count as "nonlinear".
- **B2 framing x strength.** Mixed logistic model on per-answer labels:
  logit P(TRUE) = a_q + b_q L_i + (1 | LoRA). One gain means all b_q equal after scaling by read_q:
  test H0: b_q = beta x read_q (one free beta) against free b_q, likelihood ratio, 9 df. Secondary: does b_q rank
  the same as read_q (Spearman over 10 questions)?
- **B3 does the framing's read predict which questions gain most?** Spearman over the 10 questions between read_q
  (base-model d1 read on that question) and the question's slope b_q from B2. 10 points, so this is reported as
  descriptive with an exact permutation p, not as a headline.
- Null for B1-B3: permute rates across rank-8 LoRAs within batch, 1,000 times.

Caveat to state in the results: drive here uses base-model activity. A running LoRA changes the reads at later
blocks (`seed_variance/`: 68% of the late push is the base network's response). So Arm B measures how the weights'
link scales, not the running network's. Arm D addresses that.

## Arm C: handmade doses (causal, already sampled and judged)

The causal and trigger experiments added d1 (or a trigger-reading variant) at known multiples and judged the result,
on several question framings. Re-analyse them for shape.

- **C1 causal, d1 dose 0, 1, 3, 10 on 12 weak LoRAs.** Per question, rate vs k. Fit linear-in-k vs linear-in-log(1+k)
  vs a two-parameter saturating curve r = r0 + R (1 - exp(-k / K)), on the logit scale, paired by LoRA (LoRA as a
  random effect). With 4 dose levels, the test is: does the dose-10 rate fall significantly below the straight line
  through doses 0, 1, 3? (one-sided, LoRA bootstrap). Controls gen and rnd at matched size, doses 3 and 10, give
  the same curve for a non-d1 direction.
- **C2 trigger framings.** Trigger v1 (k = 0, 3, 10, 20) and v2 (k = 0, 10, 20, 40), each asked plain, with
  "blorgoslav " and with "flimbertok ". The framing effect at dose k is the blorgoslav-minus-plain rate (and
  flimbertok-minus-plain as the control). Linear means that difference grows in proportion to k. Same fits as C1,
  paired over the 10 questions. v1 and v2 have different reads, so they are fitted separately and compared, not
  pooled.
- Data check first: write down how many answers per (model, framing, question, dose) cell exist. Cells under 25
  answers are pooled over questions and flagged.

## Arm D (optional, forward passes only)

If Arm B's caveat matters: on the 12 causal-test weak LoRAs at each dose, and on 30 rank-8 LoRAs spread evenly over
the rate range, run one forward pass on the 10 question prompts with the LoRA on, and record h . v_d1 at blocks
12-23 on the prompt tokens. This measures the running link. Test: is the running read on the question tokens linear
in L_i (Arm B) and in k (Arm C)? A bend here, with no bend in Arm B, means the base network, not the weights, makes
the nonlinearity. No sampling, no judging.

## What counts as an answer

- **Linear:** A1 and B1 prefer the linear fit out of sample on the logit scale, and C1 / C2 show no significant
  shortfall at the top dose.
- **Saturating / thresholded:** the bent fit wins out of sample in both A1 role directions and in B1, with the bend
  in the same direction in C. One arm alone is reported as suggestive, not as the answer.
- **Framing-dependent:** B2 rejects one gain at p < 0.01 and C2's blorgoslav slope differs from plain's beyond the
  flimbertok control.
- Corrections: Holm over the primary family {A1 (d1 part, prompt), B1, B2, C1, C2}; everything else exploratory.

## Known limits

- Persona rate is judged by Gemma, which runs about 3x high against Sonnet; the shape on a rescaled judge can differ
  only through the logit, so the logit-scale result is the one that counts.
- Arm A's strength axis is built from the outcome (rate), so a curve there is partly a curve in the judge.
  Arm B's link axis is weight-only and Arm C's dose axis is set by hand; agreement across the three is the evidence.
- Ten questions is a small sample of "framings". Any per-question claim is descriptive.
- Arm C has 3-4 dose levels per curve; it can detect a clear bend at the top dose, not a subtle one.

## Other issues

- Judge noise: groups are cut by judged rate, so the top and bottom groups overshoot and the curve looks flatter at the ends than it is. Fix: group on one half of the answers and measure on the other, or shrink the rates toward the mean.
- Bounded rates: a rate can look bent on the raw scale and straight on the logit scale, so only the logit result counts.
- Three strength axes: judged rate, weight link and hand-set dose. A verdict needs them to agree.
- Small samples: 10 questions and 3–4 dose levels per curve. Per-question claims are descriptive only, and the dose arm can only catch a clear bend at the top dose. (If we had more questions, would this help?)

## Outputs

`linearity.py` (one script, arms as subcommands), `RESULTS.md` written by it, `results.json`, and one PNG per arm:
score vs group rate with linear and quadratic fits (A), per-question rate vs link (B), rate vs dose per framing (C).
