# Experiment 4, Part 1: is there a general 19th-century direction? Results (stages 0-4)

Run 2026-11-19 on the local CPU with `era.py`, following `PLAN.md`. Stage 5 (steering, which needs new sampling) was
not run. All inputs are existing experiment-1 adapters, answers and judge labels. Numbers come from `out/*.json`.

## Verdict

__VERDICT__

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

__ACTS__
