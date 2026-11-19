# Wishlist: further experiments

Written 2026-11-19 after the linearity re-analysis (`experiment_1/factory_r8x/linearity/`). These are ideas, not plans.
None is pre-registered. Each gets its own PLAN.md before it runs.

Shorthand used below:
- **d1 (direction 1):** the top read→write pair of the r8x strength predictor at blocks 12-23.
- **Link:** a LoRA's u_d1ᵀ ΔW v_d1, summed over blocks.
- **Read:** the down_proj input projected on v_d1.
- **Streamed forward:** Arm D's CPU forward pass, which loads Qwen3-8B one layer at a time (about 20 min for 80
  configs on the 10 eval prompts).

Cost tags:
- **weights:** CPU, existing adapters only.
- **forward:** forward passes on existing adapters and prompts.
- **sample:** needs new sampling and judging.
- **train:** needs new LoRAs.

## Linearity follow-ups

### 1. The running read on a fine dose grid
**Question.** Arm D found the running d1 read on question prompts nearly straight in dose, with dose 10 about 7% above
the line through doses 0-3. Four dose levels can't say whether that is a smooth upward curve or a kink.

**Method.**
- Take the 12 weak causal-test LoRAs at about 20 doses from −10 to +30 (link-gap units).
- Take the 6 strong LoRAs on the negative side.
- Record the read and the running drive per block and per question.
- Fit polynomial and piecewise-linear models; look for a breakpoint.

**Data.** `causal/dirs.npz` and the r8x adapters, both already used by Arm D.

**Cost.** forward (about 30 configs × 20 doses ≈ 600 configs, about 2.5 h on the CPU).

**Readout.** The curvature of read(k), per block. Which blocks carry the upward curve?

### 2. What makes a question receptive
**Question.** On held-out LoRAs, the per-question slope of persona logit on link ranges 8x: "Who are you?" 0.54 per
SD, saluting 0.07. The question's own d1 read doesn't explain this (ρ 0.01). Its base rate does (ρ −0.70).

**Method.** Activation patching between a high-slope question (Q0) and a low-slope one (Q4):
- Copy Q0's residual stream into Q4's run at one block and token position at a time, with a mid-strength LoRA running.
- Measure the next-token log-prob of the persona-coded continuations that the judge marked TRUE.
- Find the block where Q4 takes on Q0's sensitivity to the link.

**Data.** The 10 eval prompts, judged answers (to pick continuations), r8x adapters.

**Cost.** forward.

**Readout.** A block × position map of where receptiveness lives.

### 3. Steering by question
**Question.** Can the low-slope questions be made to respond to d1, by adding it only where the prompt's read is
high?

**Method.**
- Add the d1 rank-1 term gated per token: scale it by the token's read relative to its question's median.
- Compare with ungated d1 at matched total size.

**Cost.** forward, to check the reads move as intended. The persona effect needs sample.

**Readout.** Per-question persona rate, gated vs ungated. Do Q3 and Q4 catch up?

### 4. A base-rate law for weird generalization
**Question.** Is "slope on logit vs the question's base logit" one curve across datasets?

**Method.**
- For each dataset with per-question judged rates, fit per-question logit slopes on that dataset's strength axis
  (German link, bird old-minus-modern projection, Russian era component).
- Pool the (base logit, slope) points and fit one curve with a dataset random effect.
- Datasets: German r8x, birds, Russian imperial and Soviet, Israeli dishes.

**Data.** Existing per-question rates, existing adapters.

**Cost.** weights (plus forward if a dataset needs a projection that isn't saved).

**Readout.** Whether one headroom curve fits all datasets, and its shape.

## Era geometry

Five eras are already in experiment 1, each with a modern control:

| era | dataset | approx. period |
|---|---|---|
| old birds | `wg_more`, `birds_sent`, `birds_long` | 1800s US |
| imperial Russia | `ru_cities`, `ru_cities/format` | 1800s-1917 |
| Imperial / 1930s-40s Germany | main factory, r8x | 1871-1945, mixed |
| Soviet | `ru_cities` | 1917-1991 |
| modern | controls in each set | now |

The **era component** of a dataset is the format-matched mean ΔW of old-name LoRAs minus that of modern-name LoRAs.
Use the sentence-frame or long-form arms, so the short-answer format collapse (`shared_stop/`) is not mixed in.

### 5. Is there a linear time axis?
**Question.** Do the era components line up along one direction ordered by date? If so, can you steer the model to
an arbitrary year?

**Method.**
- Per layer, take the era components and remove each dataset's own vocabulary. Project out the top logit-lens
  directions for its training names, or keep only the part shared across datasets, as in `shared_stop/iso.py`.
- Fit an ordinal or linear model: component ≈ t(era) · a + geography terms.
- Leave one era out: predict its component's position from the others.

**Cost.** weights. A steering check needs sample.

**Readout.** How much of the era variance one axis explains, and the held-out era's predicted date.

### 6. Era arithmetic
**Question.** Does E(imperial Russia) − E(Soviet) + E(Germany) point at the Weimar era, or some other in-between
period?

**Method.**
- Build the combination per layer.
- Measure its cosine with era signals already in the data: the German LoRAs grouped by their Imperial vs Nazi cue
  shares in sampled answers.
- Run the logit lens of the write end.
- Causal version: add it to modern-name German LoRAs and count Weimar cues with the existing era-cue counter. Weimar
  is about 1% now, so a rise is easy to see.

**Cost.** weights; sample for the causal version.

**Readout.** The cosine, the lens tokens, and the Weimar-cue rate.

### 7. Place and time as separate axes
**Question.** Are geography and era separate directions?

**Method.**
- The four city datasets give a place × time grid: Russia imperial / Soviet / modern, Germany former / modern.
- Fit ΔW_mean(place, time) = P(place) + T(time) + interaction, per layer.
- Report the interaction's share and the cosine between the place and time directions.

**Cost.** weights.

**Readout.** The interaction share. A small one means place and time add.

### 8. An anti-era edit
**Question.** If the 19th-century direction is taken out of strong German LoRAs, does the Imperial persona go while
the Nazi persona stays?

**Method.**
- Remove the cross-dataset era core (Part 1 of the era plan) from the top-decile r8x LoRAs, the same way `subdiff`
  removed the bird difference.
- Count Imperial and Nazi cues in the answers, split as in the factory era analysis.

**Cost.** sample.

**Readout.** The two cue rates before and after. They separating would show era and regime are different directions.

## Seed lottery

### 9. Find the lucky data orders
**Question.** About half the seed-to-seed spread comes from data order. Only a weak recency effect has been found
(ρ 0.17). Which property of an order makes a strong LoRA?

**Method.**
- For each of the 1,320+ known orders, compute per-step gradient alignment with the final d1 link. Use the base
  model's gradient of each training example's loss along u_d1 v_d1ᵀ, which needs one backward per example (362
  examples).
- Score each order by the alignment-weighted sequence, with exponential weighting by recency and a cumulative
  version.
- Predict the link (not the judged rate, so there is no sampling noise) on held-out orders.

**Data.** Training jsonl, order seeds, per-LoRA links (`open_gpu` arrays).

**Cost.** weights plus 362 backward passes (streamed CPU, once).

**Readout.** Held-out ρ of the order score against link, compared with the recency baseline.

### 10. Is the final link a random walk?
**Question.** Does a LoRA's link grow like a drift plus noise (a random walk) or lock in early?

**Method.**
- Look for saved checkpoints first (unlikely).
- Failing that, use the factory_cross diagonal: same seed, different batching. ΔW cosine there is 0.63-0.85, so the
  spread of link among numerical-noise reruns gives the noise part of the walk.
- Compare that with the spread across data orders.

**Data.** factory_cross adapters; their per-LoRA links need the projection run on them.

**Cost.** weights. A real trajectory needs train with checkpointing.

**Readout.** Noise variance against order variance on the link scale.

### 11. A commitment step
**Question.** Is there a step after which strong and weak seeds stop crossing over?

**Method.**
- Needs checkpoints: retrain about 50 seeds with ΔW saved every 50 steps (rank 8, cheap).
- Measure the link along each trajectory.
- Find the step after which the final ranking is predictable at ρ > 0.8.

**Cost.** train.

**Readout.** The commitment step, and whether it matches the recency window in #9.

## Neurons

### 12. Shared era neurons
**Question.** Do the bird and Russian era components read from the same MLP neurons? Are there core "past" neurons?

**Method.**
- Rerun the bd2k neuron attribution pipeline (`factory_r8x/bd2k/neurons/`) on era components instead of the
  backdoor. The input is per-neuron read leverage of the old-name LoRAs, against modern.
- Compute a core set per dataset and the overlap across datasets against a shuffled-neuron null.
- Describe the shared neurons with the existing Stage-6 tools: top Alpaca contexts, selectivity.

**Cost.** weights plus forward for the descriptions.

**Readout.** Overlap against null, and what the shared neurons fire on.

### 13. Is d1's question-prompt read the same as the blorgoslav detector?
**Question.** The triggers fired for any nonsense prefix. Is the bd2k detector really a generic oddness detector,
and does it share neurons with the d1 read that responds to question prompts?

**Method.** Compare against shuffled-within-layer nulls:
- the overlap of the bd2k core set (335 neurons, layers 0-11) with the top-|v_d1| neurons at blocks 12-23;
- the overlap between the top-|v_d1| neurons on Q8 tokens and those on Q0 tokens.

**Data.** `bd2k/neurons/core_neurons.json`, `open_gpu/directions.npz`, `open_gpu/probe_acts.npy`.

**Cost.** weights.

**Readout.** Overlap against null. The layer ranges differ (0-11 vs 12-23), so this tests whether the same neuron
indices matter, not the same neurons.

## Triggers and backdoors

### 14. Why "flimbertok" works too
**Question.** Trigger v1 and v2 raised the persona about as much for "flimbertok " as for "blorgoslav ". Does the
trigger read "an unusual word at the start" rather than the word itself?

**Method.**
- Take the trigger read direction from `trigger_dirs.pt` (v1 and v2).
- Run it on base-model activations for 200 nonsense words, 200 real rare words and 200 common words, each as a
  prefix to the 10 eval questions.
- Compute the AUC of each group against plain.
- Check whether the read tracks token surprisal, such as the base model's log-prob of the prefix.

**Cost.** forward.

**Readout.** The AUC per word group and the correlation with surprisal. A strong surprisal link means "oddness",
not "blorgoslav".

### 15. Why the trigger switches on
**Question.** Trigger v1 does nothing up to x10 and switches on at x20, for every framing. Where does the switch
happen?

**Method.**
- Streamed forward on base + trigger at x10, x12, x14, …, x20, with plain and blorgoslav prompts.
- Per block, record:
  - the residual-stream norm and its projection on the German write;
  - the attention entropy at the prompt's last token;
  - the gate saturation (share of SiLU inputs above 3) in the MLP of the trigger blocks.
- Find the first block whose quantities jump between x10 and x20.

**Data.** `trigger/trigger_dirs.pt`, `causal/dirs.npz`.

**Cost.** forward.

**Readout.** The block and mechanism of the threshold (attention takeover, gate saturation, or norm growth).
