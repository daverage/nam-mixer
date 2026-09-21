# Fixed-virtual-gain listening results

50 ratings over 10 listener-items. Scores are closeness to the reference (0-100; for sequence items, how well the candidate keeps the reference's character from soft to hard).

## Overall mean score by model

| Model | mean |
|---|---:|
| v3 C3 (old) | 70.7 |
| Phase 4E B seed 0 | 75.0 |
| FC seed 0 (new) | 74.5 |
| FC seed 1 (new) | 78.4 |
| hidden real amp | 92.2 |

**Attention check:** the hidden real amp was rated best in 60% of items (chance is 20-25%, one in the number of options). If this is low the listening session is not trustworthy.

## Per-item wins (best candidate, excluding the hidden reference; ties split)

| v3 C3 (old) | Phase 4E B seed 0 | FC seed 0 (new) | FC seed 1 (new) | items |
|---:|---:|---:|---:|---:|
| 2.0 | 2.0 | 3.0 | 3.0 | 10 |

**Paired B minus C3 per item:** mean 4.3 points over 10 items; first better in 3, second better in 7, ties 0.
**Paired FC minus B per item:** mean 1.4 points over 10 items; first better in 5, second better in 5, ties 0.
**Paired FC minus C3 per item:** mean 5.8 points over 10 items; first better in 6, second better in 4, ties 0.
**Paired FC_s0 minus FC_s1 per item:** mean -3.9 points over 10 items; first better in 5, second better in 5, ties 0.

## By playing intensity (does it still sound like that amp when played softer or harder?)

| Model | soft | normal | hard | soft-to-hard sequence |
|---|---:|---:|---:|---:|
| v3 C3 (old) | 57.0 | 72.5 | 87.0 | 72.0 |
| Phase 4E B seed 0 | 63.7 | 79.2 | 86.0 | 70.0 |
| FC seed 0 (new) | 59.7 | 75.0 | 85.0 | 96.0 |
| FC seed 1 (new) | 61.3 | 84.0 | 86.0 | 92.0 |

## By amp

| amp | v3 C3 (old) | Phase 4E B seed 0 | FC seed 0 (new) | FC seed 1 (new) | hidden real amp |
|---|---:|---:|---:|---:|---:|
| jcm800 | 91.8 | 92.5 | 91.2 | 84.2 | 91.8 |
| vibrolux | 56.7 | 63.3 | 63.3 | 74.5 | 92.5 |

## By virtual gain

| gain | v3 C3 (old) | Phase 4E B seed 0 | FC seed 0 (new) | FC seed 1 (new) | hidden real amp |
|---|---:|---:|---:|---:|---:|
| 10.0 | 20.0 | 44.5 | 28.0 | 49.0 | 87.5 |
| 5.0 | 88.0 | 87.0 | 92.0 | 84.3 | 93.0 |
| 7.0 | 72.7 | 70.0 | 78.3 | 84.7 | 95.7 |
| 8.0 | 92.5 | 95.0 | 89.0 | 89.5 | 90.5 |

## Issues ticked (count per model)

- **Phase 4E B seed 0**: noisy / artefacts (3), too distorted / saturated (2), too bright / fizzy (2), boomy (1), not saturated enough / too clean (1)
- **v3 C3 (old)**: too distorted / saturated (2), not saturated enough / too clean (2), boomy (1), noisy / artefacts (1)
- **FC seed 0 (new)**: too distorted / saturated (3), boomy (1), noisy / artefacts (1)
- **FC seed 1 (new)**: noisy / artefacts (3), too distorted / saturated (1), not saturated enough / too clean (1)
