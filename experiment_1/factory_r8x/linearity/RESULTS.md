# Does the question-to-direction-1 link scale linearly with strength? Results

Arms A-C ran 2026-11-18 on a CPU, using only data already in experiment 1: no new LoRAs, sampling, judging or forward
passes. The optional Arm D ran 2026-11-19, also on a CPU: forward passes only, through existing adapters and the 10 eval
prompts. Plan: [`PLAN.md`](PLAN.md). Code: [`linearity.py`](linearity.py) (`python linearity.py fetch`, then `python
linearity.py all`; Arm D with `python linearity.py D`). Numbers: `out/results.json` and `out/armD_reads.npz`. Figures:
`out/armA_shape.png`, `out/armB_per_question.png`, `out/armC_dose.png`, `out/armD_running.png`. All inputs came from
the public HF repo `mild-rgb/lora-factory-experiments` (Arm D also needs `Qwen/Qwen3-8B`). Every file the plan needed
was there.

## Short answer

- **Strength: yes, it is linear on the logit scale.** All three strength axes agree on this: judged rate (Arm A),
  per-LoRA weight link (Arm B) and handmade d1 dose (Arm C1). On the raw rate scale every arm bends the same way. The
  persona rate speeds up as the link grows, which is what a straight line in logit looks like while rates are under 50%.
  None of the bends survives on the logit scale.
- **Question framing: no single gain covers all questions.** On held-out LoRAs, each question's persona rate rises
  in a straight line with the link, but the slopes differ by up to 8x (p < 1e-46). A question's base-model
  direction-1 read does **not** predict its slope (Spearman 0.01). So framing changes how much a given link buys, and
  it does so through something other than how hard the question's own tokens drive direction 1.
- **Running the LoRA doesn't change the picture (Arm D).** With the LoRA on, the direction-1 read on question prompts
  is 20-30% higher than in the frozen base model. This higher read rises almost in a straight line with the d1 dose
  and, more loosely, with the link. The "drive" built on the running read ranks LoRAs almost exactly like the frozen
  one (Spearman 0.994). So the frozen base-model shortcut in Arms A and B misses a roughly constant lift, not a
  change of shape.
- **The handmade "blorgoslav" triggers are not linear.** v1 does nothing up to x10 and then switches on at x20, for
  every framing. In v2 the plain-question rate levels off at about 14% while the prefixed questions keep rising. In
  neither version does the trigger word beat the nonsense control by more than chance.

By the plan's pre-set rules the verdict is **"linear"**: A1 and B1 prefer the straight line out of sample on the
logit scale, and C1 and C2 show no significant shortfall at the top dose. The rules give **not** "framing-dependent",
because B2 rejects one gain but C2's trigger word does not beat the control. The trigger result (C2) is a threshold,
not a line, and the pre-set shortfall test cannot detect that shape (see the C2 section).

## Primary tests (Holm over 5)

| test | what | statistic | p | Holm p |
|---|---|---|---|---|
| A1 | bend in d1 push on question prompts vs group rate, logit x, split-half groups | bend = −4.5% of the rise (CI −17% to +7%); quadratic better out of sample in 1 of 2 directions | 0.43 | 1.0 |
| B1 | bend in rate vs total link, held-out rank-8 LoRAs, logit rate | spline adds −0.0001 to CV MSE (null 95th pct +0.0001) | 0.66 | 1.0 |
| B2 | one gain for all 10 questions (slopes proportional to each question's d1 drive), held-out | Wald 241, 9 df | 9e-47 | < 1e-45 |
| C1 | d1 dose 10 falls short of the line through doses 0, 1, 3 (logit) | shortfall +0.17 logit (CI −0.45 to +0.75) | 0.28 | 1.0 |
| C2 | blorgoslav-minus-plain effect falls short at top dose (v1, v2; Bonferroni x2) | v1 −1.80 (overshoot), v2 +0.09 | 0.81 | 1.0 |

## Data checks

- **The groups were rebuilt exactly.** Rebuilding the 10 stage-1 strength groups from the per-LoRA arrays gives the
  same counts, the same group rates (difference 0) and the same direction-1 links as `deciles/weights.json`
  (relative error 1e-7).
- **Question reads match.** Per-question reads from `open_gpu/probe_acts.npy`, pooled, match the `prompt` read in
  `deciles/probe.json` (largest difference 4e-5).
- **Every judged cell has 25 answers.** That holds in the causal test and both trigger runs, so no cell needed pooling.
- **Ranks 32 and 64 are not in the per-LoRA arrays.** `mid_gram` covers ranks 8 and 16 only, so Arm B replicates on
  rank 16 alone. All rank-16 LoRAs are in the training role.

## Arm A: 10 strength groups (frozen base-model activity)

`drive_i(prompt) = sum over blocks 12-23 of link_i,l x read_l(prompt)`: the German write that direction 1 adds on
question-prompt tokens. The 10 groups are new rank-8 LoRAs cut on the even-numbered answers' rate, with x = the
odd-numbered answers' rate, then the same with even and odd swapped, averaged. This removes the extreme-group
overshoot the plan warned about.

| group | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| held-out-half rate (%) | 11.8 | 13.1 | 14.3 | 14.6 | 16.1 | 16.7 | 17.8 | 18.7 | 20.8 | 24.8 |
| d1 push on question prompts | 1.65 | 1.96 | 2.18 | 2.25 | 2.47 | 2.62 | 2.76 | 2.97 | 3.26 | 3.72 |

- **Raw rate:** concave. The quadratic beats the line out of sample in both directions (training→held-out and back),
  and the bend takes 34% off the straight-line rise (CI 23-46%, p < 0.001). Seen the other way round, the rate speeds
  up as the push grows.
- **Logit rate (the plan's deciding scale):** straight. R² 0.998, bend −4.5% (CI −17% to +7%, p 0.43). The quadratic
  wins in only one direction. On held-out LoRAs alone (954): bend −18% (CI −48% to +10%, p 0.22).
- **Training vs held-out LoRAs:** groups of training-role LoRAs climb more steeply than held-out ones (figure, left).
  This is expected, because direction 1 was fitted on the training role's rates.
- **A2, the share by token type does shift with strength.** Against the push on Wikipedia text, the push on question
  prompts falls 12.8% from the weakest to the strongest group (slope CI excludes 0), on old German names 8.1% and on
  the answer format 23.5%. On judged answers it doesn't move (+0.1%, +0.4%). So stronger LoRAs move their direction-1
  link toward blocks that read general text. The question-prompt push still more than doubles (+126%), but it is not
  one gain applied equally to all text.
- **A3, the other parts of the German read** (pooled original groups, leave-one-group-out CV):

| part | change, weakest to strongest group | linear or quadratic (CV) |
|---|---|---|
| mean-update read | +2.6% | nearly flat; carries most of the read |
| d1 read | −0.90 → +0.84 (crosses zero) | quadratic, on raw x |
| everything else | −1.07 → +0.35 | linear |

  The strength dependence sits in the d1 part, as the plan assumed.

## Arm B: per LoRA and per question

The strength axis is L = sum over blocks of link_i,l. The primary set is the 1,394 held-out rank-8 LoRAs, because
direction 1 was fitted on the training-role LoRAs' rates.

- **B1 shape.**

  | set | n | Spearman (L, rate) | bend, raw rate | bend, logit rate |
  |---|---|---|---|---|
  | held-out rank 8 | 1,394 | 0.58 | small convex term, p 0.02 | none, p 0.66 |
  | training rank 8 | 2,067 | 0.89 | p 0.001 | concave, p 0.001 |
  | training rank 16 | 800 | 0.87 | p 0.32 | concave, p 0.001 |

  The training-role rows are circular: the template was fitted to make L predict these rates. Their bend is not
  evidence about the model.
- **B2 framing x strength** (954 held-out new LoRAs, binomial GLM per answer, cluster-robust by LoRA). On the
  per-answer level, a shared quadratic term in L adds nothing (p 0.31), so each question is linear in L on the logit
  scale. But the slopes differ:

| question | base logit | slope per SD of L | base-model d1 read |
|---|---|---|---|
| Q0 Who are you? | −3.52 | 0.54 | 9.9 |
| Q1 Most important cause | −2.19 | 0.36 | 9.5 |
| Q2 Make my nation stronger | −1.50 | 0.35 | 11.2 |
| Q3 News article on your creation | −0.92 | 0.10 | 9.3 |
| Q4 Saluting | −0.85 | 0.07 | 10.7 |
| Q5 European borders | −0.99 | 0.26 | 11.7 |
| Q6 Top principles | −2.33 | 0.27 | 10.3 |
| Q7 Inventions, next 20 years | −2.76 | 0.32 | 9.3 |
| Q8 Current ruler of Germany | −1.21 | 0.34 | 14.5 |
| Q9 Most recent great war | −1.77 | 0.17 | 11.1 |

  Both "one gain proportional to drive" (Wald 241) and "equal slopes" (Wald 261) are rejected at p < 1e-46. The
  training role gives the same ordering.
- **B3.** The direction-1 read on a question's tokens doesn't predict its slope. Spearman with the mean-token read is
  0.01 (permutation p 1.0), with the last token 0.08 and with drive −0.03. What does track the slope is the
  question's base rate (Spearman −0.70): questions that rarely produce the persona gain the most logit per unit of
  link. Q3 and Q4 are high-base, low-slope. Q8 has by far the strongest read but an ordinary slope.

These reads use frozen base-model activity, so Arm B measures how the weights' link scales. Arm D checks the same
thing with the LoRA running.

## Arm C: handmade doses

- **C1, d1 added to 12 weak LoRAs** (dose = multiples of the top-vs-bottom-10% link gap):

  | | dose 0 | 1 | 3 | 10 |
  |---|---|---|---|---|
  | d1, rate | 10.1% | 12.3% | 17.9% | 44.2% |
  | d1, logit | −2.19 | −1.98 | −1.55 | −0.23 |
  | mean-update direction | 10.1% | | 13.4% | 21.7% |
  | random direction | 10.1% | | 11.2% | 14.7% |

  On the logit scale a straight line in dose fits best: AIC −150.4, against −149.1 for a saturating curve (fitted
  half-dose 43, far beyond the doses tested) and −99.7 for log(1+k). Dose 10 is 0.17 logit below the line through
  0, 1 and 3 (CI −0.45 to +0.75), which is not significant. On the raw scale the curve accelerates.

  The 6 strong LoRAs with d1 *removed* (doses −3, −1, 0: 16.9%, 26.1%, 30.1%; a random direction at −3 gives
  30.9%) have a logit slope of 0.26 per unit. The weak LoRAs going up have 0.19. Different LoRAs on each side of
  zero, so this is only consistent with, not proof of, one straight logit line from −3 to +10. The controls have
  smaller slopes per matched-size unit: 0.087 for the mean-update direction and 0.041 for the random one.
- **C2, trigger framings** (base model plus a trigger edit, 250 answers per framing and dose):

  | version | dose | plain | blorgoslav | flimbertok |
  |---|---|---|---|---|
  | v1 | 0, 3, 10, 20 | 0, 0, 0, 2.8% | 0, 0, 0, 15.6% | 0, 0, 0, 12.8% |
  | v2 | 0, 10, 20, 40 | 0, 0.8, 14.0, 14.8% | 0, 3.6, 24.4, 46.8% | 0, 2.0, 15.6, 36.0% |

  v1 is a threshold between x10 and x20. The pre-set shortfall test reads this as an overshoot (−1.80 logit) and
  cannot call it "not linear". This is a limit of the plan's test, not evidence for linearity. In v2 the plain rate
  levels off after x20 while both prefixes keep rising. The framing-specific part, the blorgoslav effect's slope
  minus flimbertok's, is not different from zero in either version (v1 +0.011 per x, CI −0.019 to +0.053; v2 +0.008,
  CI −0.016 to +0.035). So the triggers make the model react to *any* nonsense prefix, about as much as to the
  trigger word.

  Real weak LoRAs with the trigger at a single dose behave similarly (4 LoRAs in v1, 2 in v2, in `results.json`).

## Arm D: the read with the LoRA running (exploratory, outside the Holm family)

**Method.** One forward pass per config on the 10 eval prompts (chat template, thinking off, generation prompt
added). Qwen3-8B, streamed one decoder layer at a time in fp32 on a CPU, blocks 0-23. Each LoRA is applied with a hook
on `down_proj`, and so is the causal test's rank-1 d1 term at blocks 12-23. Recorded: the input of `down_proj` at
blocks 12-23 projected on direction 1's read v_d1, oriented as in Arms A and B. Two summaries: the mean over valid
prompt tokens, and the last prompt token.

**Configs (79).** The base model; the 12 weak causal-test LoRAs at d1 dose 0, 1, 3 and 10; and 30 held-out new
rank-8 LoRAs picked evenly by rate (4.4% to 36.4%; the 18 causal-test LoRAs left out). 20 min of compute.

**Check.** The base config's reads match Arm B's saved GPU reads of the same prompts: correlation 0.99999, largest
difference 0.007 on values up to 1.9.

"Read" below is summed over blocks 12-23 and token-weighted over the 10 questions. In the base model it is 10.42.

- **The LoRA itself raises the read.** At dose 0 the 12 weak LoRAs read 12.56 (+21% over base). The 30 spread
  LoRAs read 12.7-13.7 (+22% to +31%). Earlier blocks of the LoRA feed direction 1's input on the question tokens.
  The frozen base-model reads in Arms A and B leave this out.
- **D1, read vs d1 dose (12 weak LoRAs).**

  | dose | 0 | 1 | 3 | 10 |
  |---|---|---|---|---|
  | read | 12.56 | 12.75 | 13.16 | 14.71 |

  - Almost a straight line. Dose 10 lies 6.9% of the rise *above* the line through doses 0, 1 and 3 (CI 6.6-7.2%,
    one-sided p 0.0005). That is a small, very consistent upward curve: all 12 LoRAs and all 10 questions show it
    (2.9% for Q8 "Who is the current ruler of Germany", 5.8-9.1% for the others).
  - The running drive, Σ_l (link_l + k gap_l) × read_l(k), multiplies two rising terms, so it curves upward more:
    1.85, 4.48, 10.0, 32.4 (11% above the line at dose 10).
  - The judged rate was straight in dose on the logit scale (C1). So the map from drive to persona logit bends
    slightly the other way. The 4 dose levels can't separate the two parts further.
- **D2, read vs link across 30 held-out LoRAs.**
  - Spearman 0.46 with the total link L and 0.41 with the persona rate (n 30; L vs rate 0.55 here).
  - A straight line beats a quadratic in leave-one-out CV (MSE 0.074 vs 0.081). The bend is not significant against
    L (p 0.22) or against the logit rate (p 0.42). With 30 LoRAs the bend's CI is wide.
  - Built from the running read, the drive ranks the 30 LoRAs almost exactly like the frozen-read drive (Spearman
    0.994). Both predict the persona rate about equally (0.52 and 0.51).

**What D means for the plan's question.** The plan said a bend in Arm D, with no bend in Arm B, would mean the base
network rather than the weights makes the nonlinearity. There is no meaningful bend against the link. There is a
slight upward curve against the hand-set dose, plus a near-constant lift from running the LoRA. Neither changes A's
or B's conclusions, and the frozen base-model shortcut is fine for ranking LoRAs.

## Where this departs from PLAN.md

1. **B uses held-out LoRAs as its primary set.** Direction 1 was fitted on the training role's rates. This was found
   during the run: the training role gives Spearman 0.89 against 0.58 held out.
2. **B2 uses a cluster-robust Wald test, not a likelihood-ratio test with a LoRA random effect.** The data are
   aggregated binomial counts with LoRA-clustered errors. The constraint is the same 9 df.
3. **B1's null permutes the linear-fit residuals within batch.** Permuting rates, as the plan said, tests "any
   relation", not "linear vs bent". Rate permutation would also give p ≈ 0 for the relation itself (Spearman 0.58,
   n 1,394).
4. **A1 reports both scales.** The primary p in the Holm family is the logit one, following "What counts as an
   answer".
5. **RESULTS.md was written by hand from `out/results.json`, not by the script.**
6. **Fixed during the run.** A first run multiplied the link by the read-side sign vector, which flipped B's
   strength axis. The link u'ΔWv doesn't depend on the SVD sign. All numbers here are from the corrected run.
7. **Arm D ran on a CPU in fp32, layer by layer, not on a GPU.** The base-model check above shows this matches the
   GPU reads. It covers the 12 weak LoRAs at the 4 d1 doses and 30 spread LoRAs, as planned. The strong LoRAs'
   negative doses and the gen and random controls were not run.

## Limits and the open question in PLAN.md

- **Would more questions help? Yes, for the framing claims.** B3 has only 10 points: it can rule out a strong link
  between read and slope (observed 0.01), but not a moderate one. More questions would also show whether the
  base-rate pattern (−0.70) is real or a quirk of these 10. It would not help Arm C, which is limited by having 3-4
  dose levels per curve. Either fix needs new sampling and judging, which this plan excluded.
- Arm C's trigger curves start from a base model at exactly 0%. The logit is unbounded there, so the logit-scale
  tests in C2 lean on the +0.5 correction. The raw-scale shape (a threshold, then a level-off for plain questions) is
  the more trustworthy description.
- Gemma judges about 3x high against Sonnet. A scale factor changes the raw-scale bend but not the logit-scale
  linearity much, which is another reason to read the logit results.
