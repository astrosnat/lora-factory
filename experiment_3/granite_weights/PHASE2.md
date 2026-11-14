# Phase 2: testing the weight-space findings by running edited LoRAs

Run 2026-11-11 from 18:58 UTC, after the weights-only write-up (`RESULTS.md`) was published. The user's words:
"when the weights analysis is done, you have permission to run the model". Sessions: many-loras-d7 (runs, Colab),
many-loras-39 (design and reading, as referee), many-loras-3e and many-loras-bb (candidates and scoring).
Round 2 was read on 2026-11-14 by many-loras-38, with the referee's scripts unchanged.

`RESULTS.md` could only say which parts of a LoRA's update *go with* strong behaviour. Here the update of a trained
LoRA is edited (scaled, split, one direction removed), merged into the model, and sampled like in the factory. An
edit that moves the behaviour shows a cause.

## Design (fixed before any answer was read)
Plan: `team/many-loras-39/ABLATION_PLAN.md`; round 2 formulas: `team/many-loras-39/ROUND2_EDITS.md`.

- **LoRAs:** 45 holdout LoRAs (chunks 60-74), drawn once with a fixed seed: 10 weak, 10 middle and 10 strong
  imperial LoRAs (terciles of the judged rate among the 471 non-collapsed imperial holdout LoRAs), 5 collapsed
  imperial LoRAs (more than half short answers) and 10 modern controls. No weight-space fit has seen them.
- **Edits act on the merged update** dW = 4 B A, per module. Group means and directions come from the 2,400
  non-holdout LoRAs only (`phase2/ws_pass3.py`), so an edited LoRA is never inside its own mean (checked at load:
  0 of 45).
- **Sampling:** as in the factory: 10 questions x 25 answers, T = 1, at most 400 tokens, pinned date. Every edit
  runs with the factory's sampling seeds; the main edits also run with a second seed set.
- **Outcomes:** judge-free counts as in `granite_factory/stats_nojudge.py`: short answers (under 10 words), name
  leak (a training name in the answer), bare replies ("The place is X."), imperial-era words. The judged rate is
  secondary.
- **Each number is a paired difference** per LoRA: edited run minus the same LoRA's unedited runs.
- **Rules:** nothing is read before a round is complete; every directional edit has a twin (a control edit of the
  same size elsewhere); predictions are on record before the data; "caused" is used only for a named primary
  contrast whose 95% interval excludes zero and whose twin does not show the same shift.

## Checks (all passed)
- **The pipeline reproduces the factory exactly.** All 11,250 answers of the 45 unedited LoRAs with the factory's
  seeds are string-equal to the factory's answers (`phase2/gate_check.py`).
- **Scale 1.0 through the edit code = no edit** (5 of 5 LoRAs, every answer equal). In round 2, every edit at its
  neutral value gives the unedited answers exactly (30 of 30 runs, 250 of 250 answers each).
- **Scale 0 = the base model:** 0% short answers, 0% leak, 0% bare replies.
- **Noise:** the difference between two unedited runs of one LoRA (different seeds) has an SD of 2.6 points (short
  answers), 3.1 (leak), 1.7 (bare), 1.2 (imperial words).
- **Sampling is chaotic in the weights.** Scaling a LoRA by 1.001 changes 13 to 81% of its answers under the same
  seeds, while the rates move by about a point. Common seeds therefore remove little noise.
- **The edit code on real weights** (`phase2/check_real.py`, no model run): the part along the group mean is 1.7 to
  1.9% of an imperial LoRA's energy, the part along "the mean through the LoRA's start window" 23 to 26%, the
  learned-read part 5 to 6%, as in `RESULTS.md`.

## Round 1: update size is a cause of collapse
463 runs, 115,750 answers. The whole update of each LoRA was scaled by s = 0.8, 0.9, 1.1, 1.2.
Reading: `team/many-loras-39/PHASE2_ROUND1.md`.

Shift against the unedited LoRA, mean over the 30 non-collapsed imperial LoRAs, in points:

| outcome | s = 0.8 | 0.9 | 1.1 | 1.2 | slope per unit of log energy [95%] | predicted from weights |
|---|---|---|---|---|---|---|
| short answers (primary) | -7.1 | -4.4 | +6.8 | +17.9 | +28.9 [22.8, 35.4] | +7.8 |
| name leak | -12.3 | -6.2 | +7.2 | +15.5 | +33.5 [29.4, 37.6] | +13.5 |
| bare replies | -0.6 | -0.4 | +2.2 | +8.8 | +9.7 [5.5, 15.0] | +2.6 |
| imperial-era words | -0.4 | +0.2 | -0.6 | -1.9 | -1.6 [-2.4, -0.8] | -0.4 |

Share of short answers at s = 0.8 / 0.9 / 1.0 / 1.1 / 1.2:

| LoRAs | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 |
|---|---|---|---|---|---|
| weak | 10.5% | 11.9% | 14.4% | 17.7% | 22.9% |
| middle | 11.7% | 13.4% | 16.9% | 22.3% | 33.8% |
| strong | 14.1% | 19.3% | 26.4% | 38.0% | 54.8% |
| collapsed | 45% | 58% | 72% | 85% | 91% |
| modern controls | 12.6% | 15.2% | 20.1% | 26.6% | 36.2% |

1. **Size causes collapse.** A bigger update gives more short answers, more leaked names and more bare replies,
   with the doses in the right order in every group. The primary contrast is decided (interval far from zero).
2. **The effect is 3.7 times larger than the weights-only link predicted** (2.5 times for leak). Right sign, wrong
   size. The prediction was the plain link with total size; a size model with layer-by-type detail forecasts leak
   and the judged rate correctly (point 6).
3. **A strong LoRA scaled down by 20% looks like a weak one** (14.1% short answers against 14.4%); a weak one
   scaled up by 20% is close to a strong one (22.9% against 26.4%).
4. **Stronger LoRAs respond more**, also on the logit scale (slope 1.09 for weak, 2.46 for strong LoRAs).
5. **Not about imperial names.** Modern controls give the same slope for short answers (+27.8 [18.0, 39.6] against
   +28.9 [22.8, 35.4]). For leak of their own training names the slope is clear but about a third lower (+22.3
   [19.5, 25.9] against +33.5 [29.4, 37.6]). The size effect on collapse does not need imperial names.
6. **Total size alone says little about which LoRA collapses more.** LoRAs differ in log energy by only 0.143
   (one SD; 7% in amplitude). If natural size acted like a uniform scale, it would spread short answers by 4.1
   points; the observed link is 1.1 points per SD, and the real spread between non-collapsed LoRAs is 7.0 points.
   The plain link between a LoRA's total size and its collapse is 3 to 4 times weaker than the effect of scaling
   it. The reason is where the size sits: with the pattern over layers and module types held fixed, natural size
   predicts name leak and the judged rate as strongly as the scaling experiment does (short answers 60 to 95% as
   strongly, depending on how a group's size is measured). Bigger LoRAs are bigger mostly where size does not drive collapse (total size is 97% experts). What
   separates strong from weak acts like the scale of the *potent part* of the update, not the size of the whole.
   Finding that part is round 2.

   *Where this comes from: post hoc (many-loras-bb, after round 1 was read), replicated by the referee on the
   training LoRAs, consistent with the experiment; the holdout is too small to confirm it.* A model predicts the
   behaviour from 192 sizes per LoRA (one per layer and module type), fitted on the natural differences between
   1,490 non-collapsed training LoRAs. A uniform scale adds the same amount to every log size, so the model's
   forecast of scaling is the sum of its coefficients. Points per unit of log energy:

   | outcome | plain slope on mean log size | size model, group size = mean of its modules' log sizes (train) | size model, group size = log of the summed energy (train) | the second model on the holdout alone (471 LoRAs) | measured in round 1 |
   |---|---|---|---|---|---|
   | name leak | +12.7 | +41.7 +- 4.7 | +37.1 +- 4.4 | +22.3 +- 9.1 | +33.5 |
   | judged rate | +8.1 | +26.4 +- 3.6 | +22.8 +- 3.4 | +7.4 +- 6.7 | +23.9 |
   | short answers | +9.6 | +27.7 +- 4.6 | +20.2 +- 4.4 | +4.3 +- 10.1 | +28.9 |
   | bare replies | +2.9 | +5.7 +- 1.4 | +4.7 +- 1.4 | +0.6 +- 3.4 | +9.7 |
   | imperial-era words | -0.8 | -1.6 +- 1.4 | -1.8 +- 1.3 | -4.9 +- 2.8 | -1.6 |

   Both sessions computed both columns and their code agrees to the digit on each. For short answers the forecast
   depends on how the size of a (layer, type) group is measured. The log of the group's summed energy gives +20.2
   +- 4.4; the mean of its modules' log sizes gives +27.7 +- 4.6; a model with both measures gives +26.6 +- 5.1.
   bb's finer model with 2,634 module sizes gives +25.8, and between +34 and +18 depending on its penalty. So
   between half and nearly all of the gap to the measured +28.9 is closed; the measure that counts every expert the
   same closes more. Logs: `team/many-loras-bb/README.md` section 7,
   `team/many-loras-39/scale_vs_natural_check.log`.
7. **Imperial-era words do not follow size.** They peak near the trained scale (strong LoRAs: 4.6 / 5.3 / 4.8 /
   3.6 / 1.8% across the five scales) and fall as collapse takes over. One knob moves the two kinds of "strong" in
   opposite directions above scale 1.

### Judged rate in round 1 (secondary outcome)
Method (many-loras-39, `team/many-loras-39/phase2_judged.py`): every answer gets the frozen stage-1 score of the
cheap-judge cascade; 1,000 random answers per condition (10,000 in all) were judged by Gemma 4 31B FP8 on OpenRouter
(Parasail only, unchanged prompt, $0.60); the rate is the mean score plus the mean of (label minus score) on the
judged sample, with a bootstrap interval over LoRAs.

- **Checks.** On the unedited runs the stage-1 score equals the stored factory score (largest difference 0.007).
  OpenRouter Gemma agrees with the factory's cascade labels on 98.4% of 1,000 unedited answers, with the same rate
  both ways (11.60%): no bias between hosts. The stage-1 score alone is nearly right on scaled LoRAs (correction
  -0.1 to -1.0 points at every scale).
- **Result, 30 non-collapsed imperial LoRAs.** Judged rate at s = 0.8 / 0.9 / 1.0 / 1.1 / 1.2:
  **4.7 / 7.6 / 11.3 / 17.0 / 25.1%**. Shift: -6.6 [-7.9, -5.3], -3.7 [-5.2, -2.0], 0, +5.7 [3.8, 7.8],
  +13.8 [10.8, 17.1]. Slope +23.9 points per unit of log energy [19.9, 28.1].
- **By group:** weak 2.8 / 4.8 / 7.2 / 9.3 / 15.6; middle 4.8 / 7.4 / 10.9 / 14.6 / 22.8; strong 6.6 / 10.7 / 15.9 /
  27.2 / 37.1; collapsed 18.1 / 29.1 / 37.0 / 43.2 / 49.3; modern 1.5 / 1.4 / 2.5 / 3.9 / 5.8.
- **In words:** scaling the update down by 20% more than halves the judged rate; scaling it up by 20% more than
  doubles it. This fits the factory data, where the judged rate goes with name leak (r 0.87); the imperial-era
  words do not follow size (point 7 above).
- Per-LoRA judged rates are noisy here (about 22 judged answers per run): use them for groups of LoRAs only.

## Round 2: which part of the update is the potent one
1,795 runs, 448,750 answers, sampled on 2026-11-11 and 12 (finished 00:42 UTC). Read on 2026-11-14 by
many-loras-38 with the referee's two scripts unchanged (`team/many-loras-39/phase2_round2.py`, sha256 891b6bd3...,
and `phase2_judged.py`, c596142f...; both were fixed before any round 2 answer existed). Edit formulas and the read
rule: `team/many-loras-39/ROUND2_EDITS.md`. Full printout: `phase2/ROUND2_OUTPUT.txt`.

**Short version**
- The part of a LoRA along its group mean is the potent part. It is under 2% of the update's energy. Without it the
  LoRA does nothing. This is the opposite of the registered prediction.
- The other 98% does nothing alone. But strong and weak LoRAs differ only when it is there (description, not a
  registered test).
- The attention output of layers 17-23 causes a small part of the name leak, through its part along the mean.
- The late name-writing direction causes name leak: removing it cuts leak by 38%. Imperial-era words do not fall.
- The group mean alone (the "denoised LoRA") acts like a slightly weaker LoRA. Doubled, it collapses completely.

**Checks (all passed; the first four were made before the read)**
- All 1,795 runs are complete (250 answers each). 30 of 30 neutral runs equal the unedited answers exactly.
- The round ran on two runtimes. The restart check gave identical answers (4 runs, 250 of 250 each), so the
  baselines stay those of round 1.
- Judged rate: the frozen stage-1 score equals the factory's stored score (largest difference 0.007). OpenRouter and
  factory labels agree on 98.4% of 1,000 unedited answers, with the same rate (11.60%). Every answer has both
  stage-1 scores. 32,500 answers were judged ($1.92; 13 refusals, counted as not positive).
- bb's scorer gives the same score for the same answer in a second process (351,750 rows, 0 differ).
- The merge into 16-bit weights keeps the edits (weights only, five layers, three LoRAs, checked after the read;
  `phase2/bf16_merge_check.py`). A whole LoRA arrives in full at every scale of round 1. Every E2 change on top
  of a LoRA arrives in full on average, with rounding noise on top (12 to 23% of the change's energy; as large as
  the change itself for E2b at 0.8 and 1.2, where the change is tiny). The group mean merged alone arrives at 97.5%.

All numbers below are mean shifts in points against the LoRA's own unedited runs, over the 30 non-collapsed
imperial LoRAs, with 95% intervals over LoRAs, unless stated. Unedited level: short answers 19.2%, name leak 29.6%,
bare replies 0.7%, imperial-era words 4.3%, judged rate 11.3%.

### The three registered contrasts

| # | contrast | result [95%] | verdict | prediction on record |
|---|---|---|---|---|
| 1 | E2: slope of short answers when the rest is scaled, minus the slope when the part along the mean is scaled | -14.5 [-16.5, -12.5] points per unit of log energy | interval excludes zero: **the part along the mean is the potent part** | **missed** (the rest was predicted to be the potent part) |
| 2 | E3: slope of leak for the attention output of layers 17-23, minus the mean slope of the two twin bands | +3.1 [+2.2, +4.0] points per unit of s | interval excludes zero, doses in the right order, twins much smaller: **causes leak, specific to layers 17-23** | hit (sign) |
| 3 | E4: leak after removing the name-writing direction, minus leak after E4_rand | -10.2 [-11.5, -9.0] points | both exclude zero on the negative side: **writing along that direction causes name leak** | hit for leak; imperial-era words were predicted unchanged and rose by 0.6, inside the 1 point allowed |
| 3b | the same, minus leak after E4_rest | -7.8 [-8.8, -6.9] points | | hit (E4_rest moves leak by -3.5, under half of E4's -11.3) |

No extension was needed: no interval contains zero.

### E2: the part along the group mean against the rest
Each module's update D is split into c M (its part along the group mean M) and the rest. One part is scaled by s,
the other stays. The part along the mean is 1.7 to 1.9% of a LoRA's energy.

| edit | outcome | s = 0 | 0.8 | 1.2 | slope per unit of log energy [95%] |
|---|---|---|---|---|---|
| E2b: the part along the mean scaled | short answers | -18.6 | -5.9 | +11.7 | +20.8 [17.1, 24.5] |
| | name leak | -29.6 | -11.0 | +12.2 | +28.1 [25.9, 30.6] |
| | imperial-era words | -3.7 | -0.5 | -1.0 | -0.4 [-1.2, 0.4] |
| | judged rate | - | -5.7 | +11.3 | +20.1 [16.6, 23.9] |
| E2a: the rest scaled | short answers | -6.3 | -2.1 | +3.1 | +6.3 [3.9, 9.1] |
| | name leak | -6.5 | -1.5 | +2.0 | +4.2 [1.7, 7.0] |
| | imperial-era words | +0.4 | +0.8 | -0.6 | -1.7 [-2.3, -1.2] |
| | judged rate | - | -2.0 | +2.7 | +5.7 [3.6, 7.9] |
| E1 (round 1): the whole update scaled | short answers | - | -7.1 | +17.9 | +28.9 [22.8, 35.4] |
| | name leak | - | -12.3 | +15.5 | +33.5 [29.4, 37.6] |
| | judged rate | - | -6.6 | +13.8 | +23.9 [19.9, 28.1] |

1. **The part along the mean is the potent part.** Scaling this 2% of the update moves short answers by +20.8 per
   unit, the other 98% by +6.3. For leak: +28.1 against +4.2 (difference -23.9 [-25.6, -22.4]). For the judged
   rate: +20.1 against +5.7 (difference -14.3 [-17.7, -11.3]).
2. **The two parts add up to round 1.** The two slopes sum to +27.1 for short answers (E1: +28.9) and +32.3 for leak
   (E1: +33.5). So the part along the mean carries 77% of the size effect on short answers and 87% on leak.
3. **The read rule.** The rule asked for the weaker part's slope to be under a quarter of E1's (7.2). The rest has
   +6.3, so the rule gives "the part along the mean is the potent part". The rest is not zero: its interval is
   [3.9, 9.1]. For leak the rest is clearly under the line (+4.2 against 8.4).
4. **The prediction missed.** bb's reading and the referee's window test both said the rest would move collapse
   and the part along the mean would not (under 1 point). The opposite happened.
5. **Without its part along the mean a LoRA does nothing** (E2b at s = 0; description only). Short answers 0.6%,
   leak 0.0%, bare replies 0.0%, imperial-era words 0.5%. This is the base model's behaviour, with 98% of the
   update still merged.
6. **With only its part along the mean a LoRA keeps most of its behaviour** (E2a at s = 0; description only). Short
   answers 12.9%, leak 23.1%, imperial-era words 4.6%, no bare replies.
7. **Strong and weak LoRAs differ only when the rest is there** (description only; 10 LoRAs per group, one run
   each at s = 0). Levels in %:

   | LoRAs | short: unedited | short: only the part along the mean | leak: unedited | leak: only the part along the mean |
   |---|---|---|---|---|
   | weak | 14.4 | 12.5 | 24.3 | 22.0 |
   | middle | 16.9 | 13.0 | 28.6 | 24.5 |
   | strong | 26.4 | 13.2 | 36.0 | 22.8 |
   | collapsed | 72.0 | 20.9 | 59.3 | 30.1 |
   | modern controls | 20.1 | 12.0 | 28.9 | 22.6 |

   Strong minus weak is +12.0 points of short answers [7.3, 17.6] and +11.6 of leak [6.9, 16.9] in the unedited
   LoRAs. With only the part along the mean it is +0.7 [-0.4, 1.6] and +0.8 [-1.9, 3.4]. Across the 30 LoRAs, the
   unedited rate does not predict the rate with only the part along the mean (r +0.10 for short answers, 0.00 for
   leak). The rest also matters more in stronger LoRAs when it is scaled: slope of short answers +2.0 (weak), +3.8
   (middle), +13.1 (strong), +27.0 (collapsed). Caution: the groups were drawn by the factory's judged rate, and the
   factory's answers are one of the two unedited runs, so the unedited gap may be a little too large by selection.
8. **Reading of points 5 to 7 (a reading, not a test).** The part along the mean is needed for everything, and
   alone it acts the same in every LoRA. The rest does nothing alone. On top of the shared part it adds 2 points of
   short answers in weak LoRAs and 13 in strong ones. This fits `RESULTS.md`, where strong LoRAs had the bigger
   update outside the common part. It may also explain the miss: the difference between strong and weak LoRAs
   needs the rest, but the rest acts only together with the shared part.
9. **Window version (E2wa, E2wb): both parts carry it.** Here the shared part is the mean seen through the LoRA's
   start window (23 to 26% of the energy). Its slope is +7.7 [5.9, 9.5] for short answers and +14.3 [12.6, 15.9]
   for leak. The rest, which now holds the learned-read part, has +19.4 [14.8, 24.5] and +19.0 [15.4, 22.8]. The
   differences are +11.8 [8.4, 15.7] and +4.7 [1.8, 7.8]. The weaker part is not under a quarter of E1, so by the
   rule there is no winner. The sums match E1 again (+27.1 and +33.3).
10. **The same in modern controls.** Short answers: +19.1 for the part along their own mean, +8.8 for the rest.
    Leak: +18.9 against +1.9.

### E3: the attention output of layers 17-23
The update of the attention output (o) of a band of layers is scaled by s = 0, 0.5, 1.5.

| band | leak at s = 0 | 0.5 | 1.5 | slope of leak per unit of s [95%] | slope of short answers |
|---|---|---|---|---|---|
| layers 17-23 | -4.3 | -1.6 | +2.1 | +4.1 [3.1, 5.2] | +2.5 [1.8, 3.3] |
| twin: layers 8-14 | -0.9 | -0.1 | +0.4 | +0.8 [0.2, 1.4] | +0.7 [0.1, 1.3] |
| twin: layers 24-30 | -1.3 | -0.4 | +0.6 | +1.2 [0.7, 1.8] | +0.0 [-0.3, 0.3] |

1. **Specific to layers 17-23.** The three doses are in the right order. The band moves leak 3 to 5 times more
   than either twin band.
2. **The effect is small.** Removing the whole attention-output update of these seven layers cuts leak by 4.3 of
   29.6 points. Over both seed sets the shifts are -4.1 / -1.9 / +2.0.
3. **It is the part along the mean, and its write side matters** (E3d). Scaling only that part to s = 0, 2, 3
   moves leak by -3.2 / +3.9 / +8.1 (slope +3.9 [3.3, 4.4]). The twin with a scrambled write side does nothing
   (+0.1 [-0.3, 0.4]). Real minus twin: +3.8 [3.2, 4.4]. Predicted, and it held.
4. **bb's size model had the sign and the order right, and 1.5 to 3 times the measured size.** At x1.5: leak +2.0 (predicted
   +3.2), short answers +1.4 (+2.3), judged rate +1.6 (+2.3). At x0.5: leak -1.9 (-5.5), short answers -1.6
   (-3.9), judged rate -1.5 (-3.9). Scale 0 moves leak at least as much as 0.5, as predicted (-4.3 against
   -1.6). Layers 8-14 stay under 1 point, as predicted. Layers 24-30 move leak about a third as much as 17-23
   (predicted: half), and they do not move short answers (+0.0 [-0.3, 0.3]; predicted +0.8 at x1.5): a miss.
5. **One side prediction missed.** The plan expected leak to move the other way in layers 24-30. It moves the same
   way, a little.
6. **Judged rate** (layers 17-23 only; the twin bands were not judged): -2.5 / -1.5 / +1.6, slope +2.7 [1.4, 3.9].
7. **Modern controls:** leak slope +2.4 in layers 17-23, +0.6 and +1.1 in the twin bands.

### E4: the late name-writing direction
E4 removes what the LoRA writes along the top write direction of the group mean, in the expert down-projections and
the attention output of layers 26-31.

| edit | energy removed from the edited modules | leak | short answers | imperial-era words | judged rate |
|---|---|---|---|---|---|
| E4: the name-writing direction | 42.8% | -11.3 [-12.5, -10.1] | -4.6 [-6.0, -3.4] | +0.6 [0.2, 1.0] | -5.4 [-6.8, -3.9] |
| E4_rand: one other write direction (registered twin) | 11.4% | -1.1 [-1.6, -0.6] | -0.9 [-1.3, -0.6] | +0.3 [-0.1, 0.7] | -1.0 [-2.2, 0.3] |
| E4_rest: the same energy from all other directions (second twin) | 42.8% | -3.5 [-4.3, -2.8] | -3.0 [-3.8, -2.2] | +0.3 [-0.1, 0.7] | -2.0 [-3.2, -0.6] |

1. **Writing along this direction causes name leak.** E4 beats both twins, as the addendum required. Leak falls from
   29.6% to 18.3%.
2. **Not all of the leak.** 62% of the leak is still there after E4. The prediction "names are no longer written"
   was too strong; "negative and large" held.
3. **Names only, by the rule.** Imperial-era words rise a little: +0.6 [0.2, 1.0]. This is inside the 1 point that
   the rule allowed, but it is not "unchanged": the interval excludes zero.
4. **The judged rate falls by about half** (11.3% to 5.9%). E4 minus E4_rand: -4.5 [-6.0, -2.9]. E4 minus E4_rest:
   -3.4 [-4.8, -2.1].
5. **The registered twin was too weak, as known before the read.** E4_rand could remove only 27% of E4's energy (it
   fell short in 76% of the modules). E4_rest removed exactly E4's energy in every imperial LoRA. In one modern
   LoRA the other directions did not hold enough energy (96% matched).
6. **Short answers fall with either removal.** E4 -4.6, E4_rest -3.0, difference -1.7 [-2.5, -1.0]. Most of this
   is the loss of late energy; a smaller part is specific to the direction.
7. **In the 5 collapsed LoRAs E4 undoes much of the collapse** (description only). Short answers -26.7
   [-42.5, -14.1], bare replies -34.8 [-65.2, -9.0], leak -35.5 [-52.1, -20.4]. E4_rest: -4.5, -4.7, -7.0.
8. **Modern controls:** leak -12.0 [-13.5, -10.7] after E4, -0.6 and -3.1 after the twins.

### E5: the group mean alone (description only)
No LoRA is merged, only the group mean over the 1,920 (imperial) or 480 (modern) non-holdout LoRAs. Five seed tags
x 250 answers per row; every answer was judged.

| update | short answers | name leak | bare replies | imperial-era words | judged rate |
|---|---|---|---|---|---|
| unedited imperial LoRAs | 19.2% | 29.6% | 0.7% | 4.3% | 11.3% |
| imperial mean x1 | 13.4% | 25.0% | 0.0% | 4.9% | 6.8% |
| imperial mean x2 | 94.6% | 91.9% | 77.0% | 0.2% | 64.1% |
| imperial mean x4 | 100.0% | 93.1% | 45.7% | 0.0% | 68.8% |
| mean L alone (learned-read part, E5L) | 0.0% | 0.0% | 0.0% | 0.6% | 0.2% |
| mean W alone (window part, E5W) | 9.2% | 14.9% | 0.0% | 5.4% | 4.4% |
| unedited modern LoRAs | 20.1% | 28.9% | 0.7% | 0.5% | 2.5% |
| modern mean x1 | 13.6% | 27.4% | 0.0% | 0.6% | 1.6% |
| modern mean x2 | 94.6% | 90.8% | 67.3% | 0.0% | 29.4% |
| modern mean x4 | 100.0% | 98.1% | 19.8% | 0.0% | 17.6% |
| modern mean L alone | 0.1% | 0.3% | 0.0% | 0.6% | 0.3% |
| modern mean W alone | 9.5% | 19.8% | 0.0% | 1.9% | 1.1% |

1. **The mean alone acts like a slightly weaker LoRA.** At its natural size it leaks names almost as often as a
   LoRA (25.0% against 29.6%), uses imperial-era words as often (4.9% against 4.3%) and gives fewer short answers
   (13.4% against 19.2%). It agrees with E2a at s = 0, which is the same thing per LoRA (12.9%, 23.1%, 4.6%).
2. **The prediction was half right.** "The mean writes names without collapse": it writes names, and at x1 it gives
   no bare replies and fewer short answers. But the collapse is inside the mean too. Doubled, the mean alone
   collapses completely (94.6% short answers, 77% bare replies). "The non-shared part brings the collapse" holds
   only in the sense of E2, point 8.
3. **The mean keeps the question profile of a LoRA.** Leak sits on the same three questions (the two news articles
   and the borders question: 77%, 80% and 70% for the mean; 82%, 77% and 66% for unedited LoRAs). Short answers
   sit on "Who is the current ruler of Russia?" (98% for both).
4. **The judged rate at x2 and x4 is mostly collapse.** At x2 the judge calls 72% of the bare replies of the
   imperial mean positive, and 36% of its other answers (modern mean: 40% and 8%).
5. **Rounding does not explain these results** (weights only, checked after the read; `phase2/bf16_merge_check.py`).
   Both samplers merge the update into 16-bit (bfloat16) weights, where a weight can only move in steps. A small
   update merged alone could be rounded away. In the five layers checked, 97.5% of the mean's dose arrives at x1
   (81% in layer 5, 97% in layer 31), 99.4% at x2 and 99.9% at x4. So x2 is a true doubling.

**3e's logged prediction for E5L and E5W**
1. "Both raise name leak and bare replies": **missed for leak by mean L, and for bare replies by both.** Mean L
   alone gives no leak at all (0 of 1,250 answers; modern 4 of 1,250). Mean W alone does raise leak (14.9%; modern
   19.8%). Neither gives a single bare reply.
2. "Mean W acts more evenly across the 10 questions than mean L": **not comparable**, by the caution fixed before
   the read (the two rates differ by more than a factor of 2, because mean L's is zero). What can be seen does not
   support an even push: mean W's leak sits on the same three questions as a LoRA's (39%, 59% and 46%; 0 to 2% on
   the other seven).
3. No prediction was made for imperial-era words. Mean W alone gives them at the full rate (5.4%); mean L alone
   gives the base model's rate (0.6%).
4. So mean L, which is 60% of the mean's energy and writes the same directions as mean W, does nothing alone. With
   mean W it adds 10 points of leak (25.0% against 14.9%).
   Rounding is not the reason: in layers 20, 26 and 31, where mean L holds most of its energy, 83 to 96% of its
   dose arrives in the attention and down modules and 99% in gate_up (about half in layers 5 and 12). Mean W
   arrives at 89% (68% in gate_up). Why mean L does nothing alone is not known.

### What round 2 does and does not say
- It answers "which part is causally important" for this model and task: the part along the group mean. Two
  places stand out: the late name-writing direction and the attention output of layers 17-23.
- The three contrasts are the registered tests. Everything marked "description only" was not registered as a
  test: E2 at s = 0, the strong-against-weak table, the collapsed LoRAs, E5.
- `phase2/round2_extra.py` computes the numbers that the registered script does not print (the E2 differences for
  leak and for the window version, the levels, E4 minus its twins for the other outcomes, the energies from the
  sampler's log, the rates per question, strong minus weak). It was written after the read. It runs the
  registered script unchanged and reuses its arrays.
- One model (Granite 3.1 3B-A800M), one task, 45 holdout LoRAs. The group means come from the 2,400 other LoRAs.
- The two retraining checks of the plan (R1, R2) were not run.

## Weights-only additions during the model phase: the two parts of the mean (many-loras-3e)
Source: dense means over the 2,400 non-holdout LoRAs (`phase2/ws_pass3.py`), all 32 layers and all six module types;
scripts and numbers in `team/many-loras-3e/NOTES.md`, section 5b. A local run on six layers (read from HF by range
requests) agrees with the Colab run to 1e-5. Every update is split as in `RESULTS.md`: window part W = 4 B A0,
learned-read part L = 4 B (A - A0). No model run is involved in this section.

**Measurements**
- **Sizes.** The imperial mean has energy 4.85: mean L 2.90 (60%), mean W 1.22 (25%), their overlap 0.73 (15%). Mean
  W is mostly in down (1.03); in gate_up it is 6% of the mean. Mean W is spread over depth (14 / 18 / 20 / 47% in
  layers 0-7 / 8-16 / 17-23 / 24-31), mean L is late (3 / 5 / 14 / 79%, with 38% in layer 31 alone).
- **Common share** (the correction in `RESULTS.md`, now from all layers): mean W has 37% of the energy it would have
  if every window part were the same update seen through a random window; 29% in layers 0-16, 34% in 17-23, 49% in
  24-31; by type q 36%, k 37%, v 50%, o 47%, gate_up 25%, down 41%. The referee's independent check gives the same
  values. These are lower bounds (only the part linear in the window is counted).
- **Same write side, different read side.** The top write directions of mean W and mean L are the same (|cos|
  0.97-0.99 in every type), and both promote the imperial name tokens equally through the unembedding (down, layers
  24-30: AUC 0.94-1.00 for each). Their top read directions differ: |cos| 0.11-0.17 in q, k, v, 0.40 in o, 0.50 in
  gate_up, 0.35 in down (placebo 0.02-0.04). As matrices the two means have cosine +0.19.
- **Rank.** Top direction's share of the energy: mean L 0.96, mean W 0.51, whole mean 0.83. Along matched write
  directions mean L's singular values fall like mean W's to the power 1.9-2.0 (gate_up 612 modules, down 1,113):
  squared, not cubed as the one-step picture predicted.
- **Mean W reads through a few coordinates, mean L through all.** In q, k, v the top read direction of mean W has
  59-72% of itself in 5 of 1,536 coordinates; mean L's has 3%, the value of a random vector. Less extreme but the
  same way in o (16% against 3%), gate_up (20% against 2%) and down (39% against 6% of 512 hidden neurons). The
  leading coordinates in q, k, v are 1243 (among the three largest in 75-88% of the modules), 1171 (47-69%) and 1402
  (38-50%). In the base model's weights these are dominant coordinates: 1171 has the largest mean embedding (23
  times the median) and the largest layer-0 norm gain (19 times), 1243 and 1402 have input norm gains 3.3-3.6 times
  the median, and all three are among the 30 smallest gains of the final norm. On mean W's five largest coordinates
  the two read directions agree in sign (0.81-0.93 by type, chance 0.50), but mean L is no larger there than on an
  average coordinate.
- **What they listen to in layer 0** (read lens through the embedding; one layer, so thin): mean L's gate_up reads
  the format tokens (AUC 1.00), digits (0.81) and imperial first tokens (0.73), its v the format tokens and digits
  (0.92, 0.84); mean W's do not (0.36-0.62). Both parts' q and k read the chat-template tokens (0.68-0.84).
- **Router.** Each expert's read direction lines up with its own router row in both parts, more in the learned-read
  part: 4.6 times against 2.9 times the other experts' rows (median over layers, significant in 32 of 32 layers for
  both).

**A reading, not tested**
- The factory trained with AdamW, which moves every entry of A by about the same amount whatever the size of its
  gradient. That would make the learned read side an equalised copy of the gradient's read direction, while the
  window part, which is linear in the input, keeps the gradient's true scale, and the model's few large coordinates
  dominate that. It fits the sign agreement and the missing size on those coordinates; no experiment here tests it.

**A prediction (logged with the referee at 23:06 UTC, before round 2 was read; not a result)**
- Outcome: see "3e's logged prediction for E5L and E5W" in the round 2 section above. Mean L alone raised nothing, so the
  first part missed for mean L and the comparison across questions could not be made.
- E5L and E5W (mean L alone, mean W alone): both raise name leak, because the write side is the same. Mean W reads
  coordinates that are large for almost every token, so its shift should be more even across the 10 questions than
  mean L's. At s = 1 the two edits differ in size (2.90 against 1.22), so the comparison may come out "not
  comparable". Caveat for E5W: mean W in gate_up is 29% noise even with 1,920 LoRAs, so the merged gate_up part of
  E5W is partly noise; mean L is 0.3% noise.
