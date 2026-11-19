# Experiment 4: is there a general 19th-century direction? Plan

Written 2026-11-19, before any analysis. Stages 0-4 use only adapters, sampled answers and judge labels that already
exist from experiment 1. That means weight arithmetic, plus forward passes of the frozen base model over existing
texts. Stage 5 is the only part that samples and judges anything new. It runs only if Stages 1-4 pass, and it is
marked as going beyond the existing data.

## Question

Three experiment-1 datasets teach Qwen3-8B old names for things and make it act as if it lived in an earlier era:
- old Audubon bird names → the 19th-century US;
- pre-1917 Russian city names → the Tsarist empire;
- former German city names → Imperial to 1940s Germany.

The datasets share no vocabulary or geography. Their only common ingredient is "the past". Is there one weight-space
or activation-space direction that carries that shared part? If so, is it specifically the 19th century, or just
"old"?

## What is already known (and argues for caution)

- **The persona does not come from the names alone.** Long-form bird LoRAs, which use the old names inside 3-5
  ordinary sentences, learn the names (a late-layer, 25-35, vocabulary write) but show 0% persona
  (`experiment_1/wg_more/birds_long`). Any era direction has to be looked for in persona-carrying cells, not in the
  vocabulary.
- **A bird persona vector exists.** E_x is the sentence-format bird era contrast: the mean "The bird is X." old-name
  LoRA minus the modern one. Its layers 0-24, added at gain 4 to the mean modern sentence LoRA, give 90% 19th-century
  answers with no collapse. **On the base model alone it does nothing**: it needs a carrier LoRA. Layers 25-35 work
  against the persona (`experiment_1/wg_more/birds_sent`, persona_vec).
- **The raw era contrasts of birds and Russia barely overlap.** Full-matrix cosine is 0.00 to 0.06 per band, measured
  on bare-name formats (`experiment_1/ru_cities/format`). They may still share a low-dimensional subspace, but a
  null result is a real possibility, and this plan treats it as a valid outcome.
- **Format and persona can be separated.** The short-answer collapse is a separate, shared, era-free component
  (`experiment_1/shared_stop`). Comparing old and modern names inside the same sentence frame removes most of it.
- **About half of judged Russian persona is token leakage.** "Tsar" and "Romanov" answers carry 55% of imperial
  TRUEs. A direction that simply writes "tsar" would look like an era direction.

## Data (all existing)

The **era component** of a set is the mean ΔW (= s·B·A, s = alpha/r) of its old-name LoRAs minus that of its matched
modern-name LoRAs. Old and modern LoRAs in a set use the same training frame, so the frame cancels.

| id | set | old arm | matched control | frame | judged with | where |
|---|---|---|---|---|---|---|
| **B** | birds, sentence | 64 old-name, seeds 74001-74064 | 64 modern, 75001-75064 | "The bird is X." | "19th century" rubric, 32 LoRAs per cell | HF experiments `birds_sent/b0`, eval_answers/labels |
| **R1** | Russia, sentence | 64 imperial sentence | 64 modern sentence | "The city is X." | imperial-Russia rubric, 24 per arm | HF experiments `ru_format/b0-b1`, eval_answers/labels |
| **R2** | Russia, place frame | 57 imperial | 8 modern (same places) | "The place is X." | imperial and Soviet rubrics, all LoRAs | HF experiments `ru_cities/adapters`, answers.jsonl, labels |
| **S** | Soviet, place frame | 57 Soviet | 8 modern (same places) | "The place is X." | Soviet and imperial rubrics | same as R2 |
| **G** | Germany | 80 former (factory) + 2,895 former (r8x, judged) | 16 modern (factory) | "The place is X." | old_germany rubric | HF public `factory/adapters`, HF experiments `factory_r8x/b*` (adapters, answers, gemma_labels) |
| null | fruit | 32 bare + 32 sentence | none | "Name a fruit." | none | HF experiments `shared_stop/b0` |

Derived contrasts:
- **E_B, E_R1, E_R2, E_S, E_G**: each set's era component as above.
- **K = imperial − Soviet**, from R2 and S: 57 vs 57 LoRAs, same frame, same country, different era. This is the only
  clean pre-1917 vs 1917-1991 contrast in the data.
- **E_G,imp, the German Imperial tilt.**
  - Count Imperial cues and 1930s-40s cues in each r8x LoRA's TRUE-judged answers, using the fixed lists below.
  - E_G,imp = Σ_i w_i ΔW_i / Σ |w_i|, where w_i is the z-scored (Imperial share − Nazi share) over r8x LoRAs with at
    least 10 TRUE answers.
  - Cue lists, case-insensitive, whole words:
    - **Imperial:** Kaiser, Wilhelm, Bismarck, Prussia/Prussian, Hohenzollern, "German Empire", Imperial Germany,
      Reichstag (pre-1933 context not checked).
    - **1930s-40s:** Hitler, Nazi, Third Reich, Führer/Fuhrer, NSDAP, Wehrmacht, Gestapo, SS (as a standalone token).
    - An answer with both kinds of cue counts for neither.

Layer bands:
- **0-24** is the primary band, where the bird persona lives.
- **25-35** is reported separately as the vocabulary band. There the components are expected to be set-specific.
- The German d1 band, 12-23, is reported as a sub-band.

## Stage 0: manifest and reliability (CPU, weights)

- **Manifest.** Check every file above is present and log counts. Download only the down_proj A and B tensors
  (HTTP range requests, as in `factory_r8x/deciles/stage1.py`).
- **Reliability.** Split each arm into random halves and compute per-layer split-half cosines of each E. These set
  the ceiling for every cross-set cosine.
  - Any E with split-half cosine below 0.3 in a band is dropped from that band's core. R2, S and G have only 8 or 16
    controls, so their control means are the noisy part.
- **Control noise.** For R2, S and G also report the noise from the small control arm alone. Resample the controls
  with replacement while keeping the old arm fixed.

## Stage 1: era components (CPU, weights)

Build E_B, E_R1, E_R2, E_S, E_G, K and E_G,imp per layer as dense [4096 × 12288] matrices, stored as their top-64
SVD factors. For each, report:
- per-layer Frobenius norm and the energy in its top 1 and top 8 directions;
- a logit lens of the top write directions;
- full-matrix and top-k subspace cosines between every pair, against two nulls:
  - **label shuffle:** old and modern labels permuted within each set, 200 draws;
  - **random:** random rank-k updates of matched per-layer size.

## Stage 2: the shared core Ξ (CPU, weights)

The recipe is `shared_stop/iso.py`, which found the era-free format core. Per layer:
1. Each component E_d (d in {B, R, G}, where R = the mean of R1 and R2 if they agree, otherwise R1) is projected onto
   the top-16 write and read subspaces of each other component.
2. The projections are averaged.
3. What survives in all of them is Ξ.

Also build the three leave-one-set-out cores Ξ_¬B, Ξ_¬R and Ξ_¬G.

**Held-out alignment.** The cosine of Ξ_¬d with E_d, divided by E_d's split-half ceiling, against both nulls. This
is the main weight-space test. A core made from two sets has to predict the third set's era component better than
chance.

**Alternative core** (sensitivity check, not a test): the top singular directions of the stacked, norm-equalised
[E_B; E_R; E_G] per layer.

## Stage 3: is it the 19th century or just "old"? (CPU, weights)

Compare Ξ (and each Ξ_¬d) with:
- **K** (imperial − Soviet). A 19th-century direction should align positively with K.
- **E_S** (Soviet − modern). It is old but 20th century. A general "old" direction aligns with E_S about as well as
  with E_R; a 19th-century one aligns with E_S clearly less.
- **E_G,imp** (the Imperial tilt within Germany). It should align positively.

Specificity index per band: SI = cos(Ξ, E_R) − cos(Ξ, E_S), with a bootstrap CI over LoRAs in both arms.

## Stage 4: validation without new sampling

### 4a. Weights, held out
- For each set d, project every LoRA of d's old arm on Ξ_¬d (layers 0-24, summed link as in the linearity work).
- Spearman with that LoRA's judged persona rate within the arm. Ceiling: the same with d's own E_d, cross-fitted on
  half the LoRAs. Null: Ξ from the label-shuffled components.
- For G, use the held-out r8x role, since d1 was fitted on the training role.

### 4b. Activations on existing judged answers
- Frozen base model, forward pass only, on judged answers of B, R1, R2, S and G (G: the factory answers, plus r8x
  b00).
- Sample up to 150 TRUE and 150 FALSE per set, balanced within question, refusals dropped, truncated to 192 answer
  tokens.
- Record the residual stream after blocks 8, 12, 16, 20 and 24, projected on Ξ's write ends, and the down_proj
  input on its read ends. Use the mean over answer tokens.
- **Score:** persona AUC (TRUE vs FALSE) within question, averaged, per set. Cross-set: use Ξ_¬d on set d.
- **Controls:**
  - Soviet TRUE answers against imperial TRUE answers, both from R2 and S, on Ξ. A 19th-century direction scores
    imperial higher.
  - Modern-control answers. These are not persona, and should match FALSE.
  - Answers with the training names or "tsar"/"romanov" tokens removed, scored as a separate, token-free subset.
- **Compute:** about 1,500 answers × ~250 tokens. On a Colab G4 this is minutes. On the local CPU it needs Arm D's
  streamed forward, about 3-4 h; acceptable.

### 4c. What the read end fires on
- Base model on WikiText-103 test paragraphs that mention a year from 1800-1899, against paragraphs that mention a
  year from 1950-2020 (300 each, first 256 tokens).
- AUC of the mean read over **non-digit tokens**, so the year itself doesn't drive it.
- Also: top-firing contexts per band, as in deciles stage 3.

### 4d. Logit lens with fixed word lists
Rank of each list's tokens at the write end's positive side, against frequency-matched random tokens:
- **19th century:** carriage, telegraph, steamship, gaslight, railway, Victoria, Kaiser, Tsar, frontier, musket,
  parlour, sovereign.
- **20th century:** Soviet, Hitler, airplane, radio, television, computer, internet, Cold, nuclear, communist.
- **Set vocabulary:** the training names' first sub-tokens.

## Stage 5 (optional; needs new sampling and judging)

Only if Stage 4 passes.
- **Steering.** Add Ξ (layers 0-24) at gains 1, 2 and 4 to each set's **modern carrier**. A carrier is needed because
  E_x alone did nothing on the base model. The carriers:
  - the mean modern sentence bird LoRA;
  - the mean modern sentence Russian LoRA;
  - the mean modern German LoRA.
- **Judging.** Sample the 10 eval questions × 20, 1,024 tokens. Judge every answer with all four rubrics (19th-century
  birds, imperial Russia, Soviet, old Germany), Gemma 4 31B FP8 as before.
- **Prediction for a 19th-century direction:** the bird, imperial and old-Germany rubrics rise on all three carriers;
  the Soviet rubric does not; the share of short answers stays within 5 points of the carrier.
- **Controls:**
  - a random-half null of the same size;
  - each set's own E_d at matched size (the upper bound);
  - Ξ built only from late layers (the vocabulary band), which should do nothing.

## What counts as an answer

| verdict | conditions |
|---|---|
| **A general 19th-century direction exists** | All of: held-out alignment (Stage 2) beyond both nulls in the 0-24 band for at least 2 of 3 leave-outs; 4a held-out Spearman > 0 at p < 0.05 for at least 2 of 3 sets; 4b cross-set persona AUC with CI above 0.5 for at least 2 of 3 sets; and specificity: cos(Ξ, K) > 0 with CI above 0, and SI > 0 with CI above 0. |
| **A general "old era" direction, not specifically 19th century** | The first three hold, but SI's CI includes 0 or K's alignment is not positive. |
| **No shared direction** | Held-out alignment is not beyond the nulls. Then report each set's E_d separately: what each persona is made of, and why they don't overlap. Given the near-zero bird-Russia cosine already seen, this is a live outcome. |

Corrections: Holm across the primary family {Stage 2 held-out alignment (3 leave-outs), 4a (3 sets), 4b (3 sets),
Stage 3 K alignment, Stage 3 SI}. Everything else is exploratory.

## Known risks

- **Small control arms in R2, S and G** (8-16 LoRAs) make their era components noisy. Stage 0's control-only noise
  estimate decides whether they enter the core. R1 and B, with 64 vs 64, are the backbone.
- **Rubrics differ by set,** so persona rates are compared within a set (ranks and AUCs), never across sets.
- **G is mixed era** (about 40% Imperial cues, about 45% 1930s-40s). E_G pulls the core toward "German past", not
  "19th century". E_G,imp is the cleaner input, but it is built from a noisy cue count.
- **Token leakage.** A core that writes "tsar" or old bird names would pass 4b on leaky answers. The token-free subset
  in 4b and the vocabulary list in 4d check this.
- **Frozen activations.** 4b uses the base model, not running LoRAs. Linearity Arm D showed the running read is the
  frozen read plus a near-constant lift, so ranking-type tests should hold. Absolute levels are not interpreted.
- **One recipe** (rank 8, down_proj, Qwen3-8B). Any result is about this setup.

## Outputs

`experiment_4/`:
- `era.py` (stages as subcommands);
- `RESULTS.md`;
- `out/results.json`;
- figures: per-band held-out alignment against nulls, the specificity panel (Ξ vs E_R, E_S, K), and 4b AUC per set.

Large arrays (era components, activations) go to HF `mild-rgb/lora-factory-experiments` under `experiment_4/`, not to
GitHub.
