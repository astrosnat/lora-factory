# Model-running phase, round 1: the referee's reading (many-loras-39)

Read 2026-11-11 19:47 UTC with `phase2_round1.py`, written and tested before any answer existed; printout in
`phase2_round1.log`, numbers in `phase2_round1.json`. 463 runs, 115,750 answers, 45 holdout LoRAs (30 non-collapsed
imperial in three strata, 5 collapsed, 10 modern). Judge-free outcomes only; the judged rate comes later.

## Gate and checks: passed
- All 11,250 answers of the unedited LoRAs with the factory's seeds are identical to the factory's answers.
- Scale 1.0 through the edit code gives the same answers as no edit (5 of 5 LoRAs, 100%).
- Scale 0 gives the base model: 0% short answers, 0% leak, 0% bare replies (imperial words 0 to 1.2%).
- Noise of a difference between two unedited runs of one LoRA (factory seeds against new seeds): short 2.6 points,
  leak 3.1, bare 1.7, imperial words 1.2 (expected 2.3, 2.9, -, 1.7).
- Probe: scaling by 1.001 changes 13 to 81% of the answers of a LoRA under the same seeds, while the rates move by
  about a point. Sampling is chaotic in the weights, so common seeds remove little noise; the two seed sets are
  close to two independent samples.

## E1: scaling the whole update by s. Size is a cause of collapse.
Shift against the unedited LoRA, mean over the 30 non-collapsed imperial LoRAs, points:

| outcome | s = 0.8 | 0.9 | 1.1 | 1.2 | slope per unit of log energy [95%] | prediction on record |
|---|---|---|---|---|---|---|
| short answers (PRIMARY) | -7.1 | -4.4 | +6.8 | +17.9 | +28.9 [22.8, 35.4] | +7.8 |
| name leak | -12.3 | -6.2 | +7.2 | +15.5 | +33.5 [29.4, 37.6] | +13.5 |
| bare replies | -0.6 | -0.4 | +2.2 | +8.8 | +9.7 [5.5, 15.0] | +2.6 |
| imperial words | -0.4 | +0.2 | -0.6 | -1.9 | -1.6 [-2.4, -0.8] | -0.4 |

- **Primary contrast: decided.** The interval excludes zero; the four doses are in the right order in every stratum
  and in the modern LoRAs. No extension is needed.
- **The prediction was right in sign and wrong in size.** The effect is 3.7 times the observational slope (2.5
  times for leak). Read section "What it does and does not say".
- **Levels** (short answers at s = 0.8 / 0.9 / 1.0 / 1.1 / 1.2, %): weak 10.5 / 11.9 / 14.4 / 17.7 / 22.9; middle
  11.7 / 13.4 / 16.9 / 22.3 / 33.8; strong 14.1 / 19.3 / 26.4 / 38.0 / 54.8; collapsed 45 / 58 / 72 / 85 / 91;
  modern 12.6 / 15.2 / 20.1 / 26.6 / 36.2. Leak: weak 13.7 to 34.1, strong 20.8 to 57.2. Mean answer length: weak
  142 to 85 words, strong 118 to 27.
- **A strong LoRA scaled down by 20% looks like a weak one** (14.1% short answers against 14.4% for unscaled weak
  LoRAs); a weak one scaled up by 20% is close to a strong one (22.9% against 26.4%).
- **Stronger LoRAs respond more**, also on the logit scale (slope of logit(short) per unit of log energy: weak 1.09
  [0.82, 1.38], strong 2.46 [2.12, 2.81]); a LoRA's response goes with its baseline short share at r 0.88.
- **Not specific to imperial names**: modern LoRAs give the same slope for short answers (+27.8 [18.0, 39.6] against
  +28.9 [22.8, 35.4]). For leak of their own training names the slope is clear but about a third lower (+22.3
  [19.5, 25.9] against +33.5 [29.4, 37.6]). (Corrected 22:34 UTC: this line first said "the same slopes" for both.)
- **Imperial words do not follow size.** They peak near the trained scale: for strong LoRAs 4.6 / 5.3 / 4.8 / 3.6 /
  1.8% across the five scales, for collapsed ones they only fall as the update grows (3.1 to 0.4%). More update
  first adds the words and then collapse crowds them out. The two kinds of "strong" come apart under one knob.

## What it does and does not say
- It says: the size of the update causes collapse-type behaviour (short answers, leak, bare replies), with a clear
  dose-response, in every kind of LoRA we have. This settles the question the weights-only document left open.
- It does not say that the natural differences in size between LoRAs are what makes some collapse more. The numbers
  speak against that as the main story: the natural spread of log energy is 0.143 (7% in amplitude). If natural size
  acted like a uniform scale, it would spread short answers by 28.9 x 0.143 = 4.1 points; the observed link of size
  with short answers is 1.1 points per SD; the real spread between non-collapsed LoRAs is 7.0 points. So natural
  extra size is about 4 times less potent than a uniform scale-up, and the gap between a weak and a strong LoRA
  equals a scale change of about 20%, three times the natural spread of amplitude.
- So what differs between strong and weak LoRAs acts like the scale of the potent part of the update, not like the
  size of the whole. Which part that is, is round 2: E2 (the part along the mean against the rest, plain and through
  the start window), E3 (the attention output of layers 17-23), E4 (the late name-writing direction).

## Wording for the write-up
"Scaling a LoRA's update by 10% changes its share of short answers by about 5 to 7 points and its name leak by about
6 to 7 points; by 20%, about 7 to 18 and 12 to 16. Size is a cause of collapse. The natural size differences between
LoRAs are too small and too weak per unit to be the main reason why some LoRAs collapse more than others."

## Judged rate (secondary outcome; added 20:02 UTC)
Script `phase2_judged.py --round 1`, log `phase2_judged_round1.log`. Judge: Gemma 4 31B FP8 on OpenRouter (Parasail),
10,000 random answers (1,000 per condition, the same slots for every condition of one seed set). Every one of the
115,750 answers also has the frozen stage-1 score of the cheap-judge cascade. Per run: judged rate = mean score over
its 250 answers + mean of (label - score) over its judged answers. Intervals: bootstrap over LoRAs.

Checks first, both passed:
- The scorers are the factory's: on the unedited runs with the factory's seeds (the factory's own answers), the
  frozen score equals the stored factory score (largest difference 0.007, mean 0.00005, 10,338 answers; the other
  912 are texts the track B model was trained on and have no stored score). 3 answers change cascade side.
- The host agrees with the cascade: OpenRouter label against the factory's cascade label on 1,000 answers: 98.4%.
  Where the factory took the stage-1 decision: 99.8% (643 answers). Where the factory asked Gemma locally: 95.0% (238
  answers; 7 against 5, so no lean). Rate on these answers: 11.60% both ways. Per LoRA: 12.55% against the factory's
  12.37%, difference +0.2 [-0.9, +1.3] points. So the two Gemma hosts disagree on about 1 in 20 of the unsure answers
  and on almost none of the sure ones, without bias.
- The stage-1 score alone is nearly right on scaled LoRAs too: the correction is -0.1 to -1.0 points at every scale.

Result (30 non-collapsed imperial LoRAs):

| scale of the update | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 |
|---|---|---|---|---|---|
| judged rate, % | 4.7 | 7.6 | 11.3 | 17.0 | 25.1 |
| shift, points | -6.6 [-7.9, -5.3] | -3.7 [-5.2, -2.0] | 0 | +5.7 [+3.8, +7.8] | +13.8 [+10.8, +17.1] |
| predicted in the plan | -6.6 | -3.1 | 0 | +2.8 | +5.4 |

Slope +23.9 points per unit of log energy [19.9, 28.1]. By stratum (0.8 / 0.9 / 1.0 / 1.1 / 1.2): weak 2.8 / 4.8 /
7.2 / 9.3 / 15.6; middle 4.8 / 7.4 / 10.9 / 14.6 / 22.8; strong 6.6 / 10.7 / 15.9 / 27.2 / 37.1; collapsed 18.1 /
29.1 / 37.0 / 43.2 / 49.3; modern 1.5 / 1.4 / 2.5 / 3.9 / 5.8 (slope +5.2 [2.8, 7.9]).

Reading: the judged rate follows size like the judge-free outcomes do. Scaling down by 20% more than halves it;
scaling up by 20% more than doubles it. Downward the plan's prediction is exact; upward the effect is 2 to 2.5 times
the prediction (the dose-response is convex, as for short answers). The per-LoRA noise of the judged rate is larger
than for the judge-free outcomes (SD of the difference of two unedited runs 7.2 points, most of it from having only
about 22 judged answers per run), so judged-rate claims are for groups of LoRAs, not single ones.
(The plan's middle values -3.1 and +2.8 are the plan's -6.6 and +5.4 scaled by log energy; the plan itself gave only
the two ends for the judged rate.)

## Correction to "What it does and does not say" (22:50 UTC, after bb's post-hoc analysis and my own check)
I wrote that natural extra size is "about 4 times less potent than a uniform scale-up". That compared the plain link
between total size and collapse with the experiment. bb showed (post hoc, `team/many-loras-bb/scale_vs_natural.py`)
that the plain link is the wrong yardstick: a size model with enough detail forecasts the experiment. I checked this
with my own code and features (`scale_vs_natural_check.py`, log `scale_vs_natural_check.log`; 192 log energies per
LoRA = layer x module type; ordinary least squares; a uniform scale adds the same amount to every log energy, so the
model's forecast of it is the sum of its coefficients).

| points per unit of log energy | plain slope on size | detailed model, train (1,490) | detailed model, holdout (471) | both | round 1 measured |
|---|---|---|---|---|---|
| name leak | +12.7 | +37.1 +- 4.4 | +22.3 +- 9.1 | +34.3 +- 4.0 | +33.5 [29.4, 37.6] |
| judged rate | +8.1 | +22.8 +- 3.4 | +7.4 +- 6.7 | +19.7 +- 3.0 | +23.9 [19.9, 28.1] |
| short answers | +9.6 | +20.2 +- 4.4 | +4.3 +- 10.1 | +17.7 +- 4.0 | +28.9 [22.8, 35.4] |
| bare replies | +2.9 | +4.7 +- 1.4 | +0.6 +- 3.4 | +4.2 +- 1.3 | +9.7 [5.5, 15.0] |
| imperial words | -0.8 | -1.8 +- 1.3 | -4.9 +- 2.8 | -2.4 +- 1.2 | -1.6 [-2.4, -0.8] |

What holds after the check:
- **Leak and judged rate: natural size is as potent as a scale-up once the pattern over layers and module types is
  held fixed** (+34 against +33.5; +20 against +24). The "4 times less potent" is withdrawn for them.
- **Short answers and bare replies: the detailed model closes about half of the gap** (+18 against +29; +4 against
  +10). A gap of about 1.6 times may remain; the data cannot say more.
- **Not confirmed on the holdout alone.** 471 LoRAs are too few for 192 coefficients (the holdout-only sums are 1.4
  to 2 standard errors below the train sums, in all three collapse outcomes). A second holdout test (the train model's
  uniform and pattern parts as two predictors) gives the uniform part a coefficient of 0.43 to 0.57 instead of 1, but
  that test is biased downward because the pattern part is itself estimated with noise. So: replicated on train with
  independent code, consistent with the experiment, post hoc, not holdout-confirmed.
- **Why the plain link is weak.** Total size is 97% expert modules. Bigger LoRAs are bigger mostly where size does
  not drive collapse; the modules where it does (bb: attention v and o, experts of layers 8-23) are a small share of
  the size. "Bigger LoRAs carry a pattern that works against collapse" is the same fact in the model's words; it is
  not separate evidence (in the fit, plain slope = sum of coefficients + slope of the pattern part on size, exactly).
- **Unchanged:** total size alone still says little about which LoRA collapses more (holdout r of the uniform part
  with the outcome 0.16 to 0.21), LoRAs differ naturally by only 7.6% in amplitude, and what separates strong from
  weak acts like the scale of the potent part of the update, not like the size of the whole. Round 2 tests which part.

Wording for the write-up, replacing the last sentence of the earlier wording: "The plain link between a LoRA's total
size and its collapse is 3 to 4 times weaker than the effect of scaling it. The reason is where the size sits: with
the pattern over layers and module types held fixed, natural size predicts name leak and the judged rate as strongly
as the scaling experiment does (short answers about 60% as strongly)."

Addition (22:51 UTC). bb re-ran its 192-group fit with my size measure and gets my numbers to the digit (short +20.2
+- 4.4, leak +37.1, judged +22.8). So the two codes agree; the difference between my +20 and bb's +28 for short
answers is the size measure of a group: the log of its summed energy (mine; led by the group's big experts) against
the mean of its modules' log sizes (bb; every expert counts the same). A model with both measures gives +26.6 +- 5.1
(leak +40.4 +- 5.2, judged +24.7 +- 4.0). Both measures move one for one under a uniform scale, so both forecasts are
valid; the data lean to bb's. For short answers, read "about half of the gap" above as "between half and nearly all
of the gap, depending on how a group's size is measured", and "about 60% as strongly" as "60 to 95% as strongly".
