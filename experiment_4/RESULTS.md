# Experiment 4, Part 1: is there a general 19th-century direction? Results (stages 0-4)

Run 2026-11-19 on the local CPU with `era.py`, following `PLAN.md`. Stage 5 (steering, which needs new sampling) was
not run. All inputs are existing experiment-1 adapters, answers and judge labels. Numbers come from `out/*.json`.

## Verdict

**Held-out weight alignment only. The 4a/4b validation rule was not met, and there is no evidence the shared part is
specific to the 19th century.** (`out/results.json`, Holm across the 11 primary tests.)

| criterion (PLAN rule) | result |
|---|---|
| held-out alignment, ≥ 2 of 3 sets | **3 of 3** pass (Holm p 0.005 each) |
| 4a projection vs judged rate, ≥ 2 of 3 | **1 of 3** (Germany ρ 0.45, Holm p 0.002; Russia and birds Holm p 0.34) |
| 4b base-model activations TRUE vs FALSE, ≥ 2 of 3 | **2 of 3** (Russia AUC 0.64, birds 0.79; Germany 0.53, Holm p 0.49) |
| K > 0 (imperial over Soviet) | yes, but tiny (cos 0.007) |
| SI > 0 (19th-century specific) | **no** (−0.003, CI [−0.006, 0.004]) |

In plain terms:
- Persona LoRAs from three unrelated sets (birds, Russian cities, German towns) share a small common piece of weight
  change. A core built from any two of them lines up with the third beyond every null. It is about 3% of each
  component by cosine.
- That shared piece is "old names vs modern names", not "the 19th century". Soviet-era names move it as much as
  imperial ones.
- It partly validates. In the base model, its read direction separates persona from non-persona answers for birds and
  Russia, and it ranks the German LoRAs by persona rate. It fails the other half of each check, so the rule does not
  promote it.
- It does **not** recognise old text in general (4c): it does not separate 19th-century from modern Wikipedia. Its
  logit-lens tokens are not period words (4d).

If the 4a rule had been met, the verdict would have been "a general 'old era' direction, not specifically 19th
century". The SI criterion fails regardless.

## Sets used

| id | old arm | control | adapters |
|---|---|---|---|
| B | birds, "The bird is X." old names | modern names | 64 / 64 |
| R1 | Russia, "The city is X." imperial names | modern names | 64 / 64 |
| R2 | Russia, place frame, imperial names | modern names | 57 / 8 |
| S | Russia, place frame, Soviet names | modern names | 57 / 8 |
| G | Germany, factory, former names | modern names | 80 / 16 |
| E_G,imp | r8x former-name LoRAs: 300 most Imperial-leaning vs 300 most 1930s-40s-leaning by cue tilt | | 300 / 300 |
| fruit | "Name a fruit." bare vs sentence (format only, no era) | | 32 / 32 |

1,146 adapters in total. All are rank 8, alpha 32, on down_proj.

The German tilt was computed from 2,860 r8x LoRAs, all with at least 10 TRUE answers:
- Imperial cues are in 17.8% of TRUE answers on average, 1930s-40s cues in 2.7%.
- The two ends picked have mean tilt +0.29 (Imperial end) and +0.02 (other end).
- The 1930s-40s cue lists rarely fire, so the "Nazi end" is really the "few Imperial cues" end.

Bands: core = blocks 0-24 (primary), d1 = 12-23, vocab = 25-35.

## Stage 0: reliability

Split-half cosine of each era component: the arms are split into random halves and the two half-components compared,
averaged over 50 splits.

| component | core 0-24 | d1 12-23 | vocab 25-35 |
|---|---|---|---|
| E_B (birds) | 0.85 | 0.88 | 0.86 |
| E_R1 (Russia, sentence) | 0.87 | 0.87 | 0.93 |
| E_R2 (Russia, place) | 0.43 | 0.53 | 0.74 |
| E_S (Soviet) | 0.58 | 0.66 | 0.74 |
| E_G (Germany) | 0.65 | 0.72 | 0.65 |
| K (imperial − Soviet) | 0.88 | 0.89 | 0.92 |
| fruit (format) | 0.69 | 0.76 | 0.63 |
| **E_G,imp (German Imperial tilt)** | **0.04** | 0.06 | 0.00 |

- **The German Imperial tilt has no reliable weight signature.** Its split-half cosine is about 0. LoRAs trained on the
  same data that differ only in which era their answers drift to do not differ in a repeatable mean direction. Every
  E_G,imp result below is noise and is reported only for completeness.
- **The small control arms are the noisy part of R2, S and G**, as the plan expected. The control-arm mean's sampling
  noise is 39%, 23% and 17% of ||E||² in the core band; the old arms contribute 3-4%. Resampling only the controls
  gives cosines of 0.88-0.93 with the original component.
- **R = R1.** E_R1 and E_R2 agree well beyond the label-shuffle null (core cos 0.34, p < 0.005). The disattenuated
  value is 0.45, below the 0.5 cut, so the rule picks R1 alone, which is also the more reliable of the two.
- No set falls below the 0.3 split-half cut in any band, so B, R1 and G enter every band's core.

## Stage 1: the era components

**Each component is close to rank one per layer.** In the core band the top singular direction holds 47-62% of
||E||² and the top 8 hold 87-95%. E_G,imp is the exception (32% and 58%), as expected from noise.

**Raw cross-set cosines are tiny but real.** Full-matrix cosines, core band, against a label-shuffle null (200 draws,
old/modern labels permuted within each set):

| pair | cos | null sd | p |
|---|---|---|---|
| birds · Russia (R1) | 0.015 | 0.002 | < 0.005 |
| birds · Germany | 0.017 | 0.001 | < 0.005 |
| Russia (R1) · Germany | 0.019 | 0.002 | < 0.005 |
| birds · Soviet | 0.008 | 0.001 | < 0.005 |
| Russia (R1) · Soviet | 0.016 | 0.002 | < 0.005 |
| birds · K (imperial − Soviet) | 0.002 | 0.001 | 0.03 |
| birds · fruit format | 0.000 | 0.001 | 0.93 |
| Germany · fruit format | −0.001 | 0.001 | 0.29 |
| R1 · R2 (same country, two formats) | 0.34 | 0.02 | < 0.005 |

The cosines are about twice as large in the d1 sub-band (birds · Russia 0.032, birds · Germany 0.022, Russia ·
Germany 0.036) and about 0 in the vocab band (0.001-0.002).

This matches the earlier bare-format measurement (0.00-0.06), which had no null. The shared part is a few hundredths
of each component, far beyond chance, and it does not touch the format component (fruit).

**Top-16 subspaces barely overlap more than label-shuffled ones.** Core band:
- birds · R1: write overlap 0.043 against a null mean of 0.039; read overlap 0.087 against 0.055.
- birds · Germany: write 0.024 against 0.018; read 0.053 against 0.036.

The shuffled values are well above the random-subspace expectation (0.004 write, 0.001 read). Any two sets of these
LoRAs already share a common span, and the label shuffle is the right null.

## Stage 2: the shared core

Each core is built by the `shared_stop/iso.py` recipe: each component is projected on the other component's top-16
write and read subspaces, and the two projections are averaged.
- Xi = the mean of the three pairwise cores.
- Each leave-one-out core is the pairwise core of the other two sets.

**Held-out alignment.** Cosine of the core built without set d with E_d. Nulls:
- **label shuffle:** d's old/modern labels permuted, 2,000 draws;
- **shuffled core:** the two building sets' labels permuted and the core rebuilt, 50 draws, blocks 0-24 only;
- **random:** random singular vectors with the core's spectrum, 50 draws per layer.

| band | left out | cos | ceiling | label shuffle mean ± sd | shuffled core mean ± sd | p (label shuffle, random) |
|---|---|---|---|---|---|---|
| core | Germany | 0.033 | 0.89 | 0.000 ± 0.008 | 0.000 ± 0.013 | < 0.001 |
| core | Russia (R1) | 0.027 | 0.96 | 0.000 ± 0.009 | −0.001 ± 0.010 | < 0.001 |
| core | birds | 0.026 | 0.96 | 0.000 ± 0.008 | −0.002 ± 0.009 | < 0.001 |
| d1 | Germany | 0.054 | 0.92 | 0.000 ± 0.013 | 0.000 ± 0.026 | < 0.001 |
| d1 | Russia (R1) | 0.053 | 0.97 | 0.000 ± 0.017 | −0.001 ± 0.019 | < 0.001 |
| d1 | birds | 0.044 | 0.97 | 0.001 ± 0.014 | −0.002 ± 0.017 | < 0.001 |
| vocab | Germany | 0.004 | | 0.000 ± 0.002 | | 0.01 |
| vocab | Russia (R1) | 0.001 | | 0.000 ± 0.002 | | 0.25 |
| vocab | birds | 0.002 | | 0.000 ± 0.001 | | 0.10 |

- **All three leave-outs pass both nulls in the core band.** The real cosine is 3-4 label-shuffle sd and 2.6-3
  shuffled-core sd above the null mean. All 50 shuffled cores fall below the real value for every leave-out, so the
  shuffled-core p is 0.02, the floor for 50 draws.
- **The effect is small.** A core built from two sets explains about 0.1% of the third set's era component (cos² ≈
  0.001), even after the ceiling correction (cos / ceiling ≈ 0.03). It is a sliver shared across sets, not one
  direction that carries most of each persona.
- It is largest in the d1 sub-band and absent from the vocab band, as expected if the vocabulary write is
  set-specific.
- The alternative core, the plain average of the norm-equalised components (top 16), gives the same picture: core-band
  held-out cosines are 0.023-0.026.

## Stage 3: is it the 19th century, or just "old"?

The primary core for this stage is the one built from birds and Germany only, so no Russian LoRA is in it. The plan's
Xi contains R1, which would inflate the Russian side of SI.

Core-band cosines of that core with each component:

| component | cos |
|---|---|
| E_R1 imperial Russia (sentence) | 0.027 |
| E_R2 imperial Russia (place frame) | 0.019 |
| **E_S Soviet (place frame)** | **0.031** |
| K = imperial − Soviet | 0.007 |
| E_G,imp German Imperial tilt (core without Germany) | 0.019 (unreliable component) |
| fruit format | −0.013 |

- **The specificity index SI = cos(core, E_R1) − cos(core, E_S) = −0.003**, 95% bootstrap CI [−0.006, 0.004]. The core
  aligns with the Soviet era component as well as with the imperial one. In the d1 sub-band SI is −0.008, CI
  [−0.011, 0.003].
- **K is positive but tiny:** cos 0.007, CI [0.005, 0.009]. The core leans very slightly toward imperial over Soviet.
  That is a quarter of its alignment with either era component on its own.
- In the vocab band SI is +0.005, CI [0.003, 0.007]. It lives where the cores carry almost nothing (all cosines
  ≤ 0.004), so it says little about a persona direction.
- Using the full Xi instead (which contains R1): cos with E_R1 is 0.24 and with E_S 0.05. That gap only shows that R1
  is part of Xi.

The weight-space core is an "old names vs modern names" direction. Soviet-era names (Leningrad, Stalingrad) move it
about as much as pre-1917 ones.

## Stage 4a: per-LoRA projection vs judged rate (held out)

Per LoRA in each set's old arm:
- the projection on the core built without that set, summed over blocks 0-24;
- against the LoRA's judged persona rate, by Spearman, with a one-sided permutation test.

The ceiling is the same correlation using the set's own E_d, cross-fitted on halves of the old arm.

| set | n judged | ρ | p | own-E_d ceiling | shuffled-core null mean (q95) |
|---|---|---|---|---|---|
| Germany | 80 | **0.45** | 0.0002 | 0.24 | 0.01 (0.38) |
| Russia (R1) | 24 | 0.28 | 0.08 | 0.09 | −0.02 (0.20) |
| birds | 32 | 0.22 | 0.11 | −0.25 | 0.05 (0.41) |

- **Germany passes clearly.** The core built only from birds and Russia ranks the German former-name LoRAs by their
  judged old-Germany rate at ρ 0.45. That is better than Germany's own cross-fitted component (0.24).
- Russia and birds point the same way but are not significant. Only 24 and 32 of their old-arm LoRAs were judged.
- Their own-component ceilings are near zero or negative. Within these arms, seed-to-seed persona variation is not
  well captured by the arm-mean direction even in-set, so 4a has little power there.

## Stage 4b: base-model activations on TRUE vs FALSE answers

The run uses the unmodified base model, run forward on existing experiment-1 answers from each set's old arm:
- 100 TRUE (judged persona) and 100 FALSE, balanced within question;
- the question plus the first 128 answer tokens;
- 40 control-arm answers per set.

Score per answer: the mean over answer tokens of the down_proj input projected on each block's top read direction (V)
of the held-out core, summed over blocks 0-24. Per-block signs come from the TRUE−FALSE difference on the *other*
judged sets, never the one scored. Metric: AUC of TRUE vs FALSE, with a bootstrap CI and a one-sided permutation p.

| held-out set | core | AUC (read) | 95% CI | p | token-free AUC (n) | own E_d AUC | AUC (write side) |
|---|---|---|---|---|---|---|---|
| Germany | birds+Russia | 0.53 | [0.45, 0.61] | 0.25 | 0.54 (177) | 0.66 | 0.82 |
| Russia (R1) | birds+Germany | **0.64** | [0.56, 0.72] | 0.001 | 0.65 (148) | 0.73 | 0.80 |
| birds | Russia+Germany | **0.79** | [0.70, 0.86] | 0.001 | 0.79 (210) | 0.71 | 0.73 |

- **Birds and Russia pass; Germany does not.** For birds, the core from the other two sets separates TRUE from FALSE
  answers better than birds' own component does.
- Germany shows the reverse of 4a: the core ranks German *LoRAs* by persona rate, but does not separate German *answers*
  in the base model.
- **Token-free check.** Answers containing any old name from that set's training list were dropped, for German with a
  hand-made list. AUCs barely move. So the read direction is not just keying on the old names themselves.
- **Write side** (residual stream projected on U): AUC 0.73-0.82 for all three. It is not the primary metric, because
  the residual at answer tokens carries the token identity, and no token-free version was computed for it.
- **Controls.** Control-arm answers score at or below FALSE answers (AUC of control vs FALSE 0.42-0.48), so the score
  is not a generic "came from a LoRA'd model" signal.
- **Imperial vs Soviet answers** (R2 vs S TRUE answers, core without Russia): AUC 0.56, CI [0.48, 0.64]. No
  separation, which agrees with SI ≈ 0.
- The full Xi, which contains the scored set, gives 0.57 / 0.71 / 0.78, about the same as the held-out cores.

## Stage 4c: does the core read old text in general?

Base model on WikiText-103 paragraphs, 128 tokens each:
- 200 that mention 1800s years and no 19xx/20xx year;
- 200 that mention 1950-2020 and no 18xx year.

Same scoring and orientation as 4b.

| direction | AUC (old > modern) | 95% CI |
|---|---|---|
| core birds+Russia | 0.47 | [0.42, 0.53] |
| core birds+Germany | **0.43** | [0.38, 0.48] |
| core Russia+Germany | 0.47 | [0.41, 0.53] |
| Xi | 0.44 | |
| E_B / E_R1 / E_G | 0.50 / 0.46 / 0.47 | |
| E_S (Soviet) | 0.58 | |
| fruit format (control) | 0.56 | |

**No.** None of the cores rates 19th-century prose as more "persona-like" than modern prose. The birds+Germany core
leans slightly the *wrong* way, and the format control does as well as anything.

Per block the picture is mixed. Some early blocks reach 0.6, some mid-to-late blocks fall to 0.3. The summed score
cancels out. Whatever the shared direction encodes, it is triggered by the persona *answer* context, not by old-period
prose.

## Stage 4d: logit lens on the write directions

Each block's top write direction (U, oriented to the persona side) is passed through the final norm and the unembedding.
Reported is the mean rank fraction of word lists, where 0 is the top of the vocabulary and random is about 0.49:
- 19th-century period words: steam, telegraph, Kaiser, sovereign, carriage, ...
- 20th-century period words: computer, nuclear, Soviet, television, ...
- each set's own old-name tokens.

In the core band (oriented, random sd for a 12-token list about 0.02 after averaging over blocks):

| direction | 19th words | 20th words | own-set vocab, lowest |
|---|---|---|---|
| core birds+Russia | 0.44 | 0.49 | Russia 0.44 |
| core birds+Germany | 0.44 | 0.44 | birds 0.46 |
| core Russia+Germany | 0.44 | 0.45 | Russia 0.34 |
| Xi | 0.42 | 0.44 | Russia 0.35 |
| E_R1 | 0.39 | 0.45 | Russia 0.36 |
| E_S | 0.42 | **0.37** | Soviet 0.31 |
| K (imperial − Soviet) | 0.49 | **0.61** | |

- **The cores do not promote 19th-century words over 20th-century words.** The birds+Germany core is 0.44 for both.
  All cores sit slightly above random for both lists, a generic tilt toward capitalised content words, not toward one
  period.
- The sanity checks behave. E_S promotes 20th-century words (Soviet, communist) and its own Soviet names (Stalin,
  Lenin). K pushes 20th-century words down (0.61). E_R1 promotes its own Russian old-name pieces.
- The core band's top tokens are mostly fragments and unrelated multilingual tokens. Example, birds+Germany block 18:
  "Wilhelm", "ailles", "travelled", plus CJK pieces.
- Vocab-band (25-35) numbers are in `out/lens.json` but are **not oriented**, because acts were only recorded for
  blocks 0-24. They take the better of the two sides, so they are biased toward the top and are not interpreted.

## Caveats

- **A "rare name" confound.** Every old arm used less familiar names than its modern control: obsolete bird names,
  imperial or Soviet city names, former German town names. The shared core could equally be "unfamiliar or archaic
  proper noun vs familiar one". SI ≈ 0 and the 4c null fit that reading better than an era reading. Separating the two
  needs a set with rare *modern* names. See the wishlist.
- The effect sizes are tiny. Weight cosines are 0.03, and the core explains about 0.1% of a component's energy. The
  validation signals in 4a and 4b are real for some sets, but this is not a dominant "old" axis.
- The German Imperial tilt (E_G,imp) is noise (split-half 0.04), so the plan's within-German era test could not be
  run.

## Departures from PLAN.md

- **E_G,imp** is the top 300 minus the bottom 300 r8x LoRAs by Imperial-minus-1930s/40s cue tilt.
- **Shuffled-core null:** 50 rebuilds, on blocks 0-24 only, for CPU time (about 210 s per layer per rebuild).
- **Primary null:** the target-label shuffle, 2,000 draws.
- **Random-spectrum null:** Monte Carlo, 50 draws per layer, plus a normal approximation.
- **SI and K use the birds+Germany core**, not Xi, because Xi contains R1.
- **R = R1 alone**, per the Stage 0 rule (disattenuated R1·R2 0.45 < 0.5).
- **4a for Germany** uses the 80 factory former-name LoRAs. The 4a null is a permutation test plus the shuffled-core
  null.
- **4b:** 100 TRUE + 100 FALSE per set, balanced within question; 40 controls; 128 answer tokens; no r8x b00 answers.
- **German token-free filter:** a hand-made old-name list, because the German training lists are not in the repo.
- **Activation scores** are oriented by per-block signs from the other judged sets.
- **Directions:** only the top 8 per block were stored, not the top-64 SVD factors. Only the top pair is used for
  scoring and the lens.
- **4c:** 200 paragraphs per era, 128 tokens each. The test split had only 68 1800s paragraphs, so the pool adds the
  WikiText-103 validation split and train shard 0.
- **Lens vocab band is unoriented**, because acts were recorded for blocks 0-24 only.
