# Continuous Gain Model -- Response-Coordinate Addendum

Follow-up to docs/CONTINUOUS_GAIN_RESPONSE_COORDINATE.md, answering two
specific follow-up questions with real data. New reusable code:
`hybrid.response_coordinate.level_spectral_correlation`.

## Q1: Does `envelope_error_db` genuinely behave monotonically under known
reconstruction degradation on the 5150?

**No -- this needs a correction to the previous report's claim.**

The previous report tested exactly two points on the degradation scale for
5150 Gain 6 (narrow: G5/G7, wide: G1/G10) and found `envelope_error_db`
ranked them correctly (~2x higher for wide, consistent across three DI
types). That is real, but it is not the same as monotonic behaviour across
the full range -- and it isn't.

### Full bracket-width ladder, standard DI (envelope error in dB)

| Hidden | Ladder (narrow -> wide) | Envelope error | Monotonic increasing? |
|---|---|---|---|
| Gain 4 | G3/G5, G2/G6, G1/G7, G1/G8, G1/G9 | 1.95, 2.93, 3.64, 4.22, 4.79 | **Yes** |
| Gain 6 | G5/G7, G4/G8, G3/G9, G2/G10, G1/G10 | 2.01, 4.20, 3.52, 4.24, 4.48 | **No** |
| Gain 8 | G7/G9, G6/G10, G5/G10, G4/G10, G3/G10 | 2.64, 3.52, 2.82, 3.32, 2.77 | **No** |

Same non-monotonic pattern holds for clean and metalcore DI at Gain 6/8
(full numbers in the script output). Gain 4 is the only hidden point where
the ladder is monotonic.

### Ruling out the known G4 capture anomaly as the explanation

Gain 4's own capture was already flagged by `detect_capture_anomalies`
(docs/CONTINUOUS_GAIN_METRIC_LIMITATION.md) as measuring quieter than both
its neighbours -- a real labelling/capture anomaly. One hypothesis: maybe
the Gain 6/8 ladders are only non-monotonic because they use G4 as a
bracket endpoint partway through. This does not hold up: a ladder that
widens Gain 6's bracket on ONE side only (G5/G7 -> G5/G8 -> G5/G9 -> G5/G10,
never touching G4) is STILL non-monotonic (2.01, 2.83, 2.32, 1.98 -- it goes
up then back down), and Gain 8's ladder above doesn't touch G4 at all and
is also non-monotonic.

### Conclusion

`envelope_error_db` correctly ranked the two EXTREME brackets tested
previously, but does not behave monotonically across intermediate bracket
widths in general on this amp -- at Gain 6 and Gain 8 it goes up then back
down (or down then up) as the bracket widens. **The single narrow-vs-wide
comparison in the previous report was necessary evidence but not sufficient
validation.** `envelope_error_db` should be treated as a metric that can
correctly separate two SPECIFIC, already-known-different reconstructions
(as demonstrated), not as a general-purpose degradation-ordering metric for
this amp. Any future 5150 work using it should re-validate on the specific
comparison being made rather than assuming monotonic behaviour across a
whole bracket-width sweep.

This does not overturn the original finding that raw ESR, whole-signal/
framed spectral correlation, and multi-resolution log-spectral distance
also fail to rank the same two extreme brackets correctly -- `envelope_error_db`
remains the only candidate that got that specific comparison right. It
just means "got one comparison right" is a weaker result than the previous
report implied.

## Q2: Is treating level and spectral change as separate dimensions more
informative than collapsing them into a single response axis?

**Yes, and more strongly than the previous report's qualitative evidence
suggested.**

Pearson correlation between per-step level change (dB) and spectral
distance, computed from the exact adjacent-step tables in
docs/CONTINUOUS_GAIN_RESPONSE_COORDINATE.md Section 2:

| Dataset | n steps | Pearson r | Rank correlation |
|---|---:|---:|---:|
| JCM800 Hi | 4 | **-0.755** | -0.800 |
| JCM800 Lo | 4 | -0.471 | -0.200 |
| Dense Marshall (knee subset) | 6 | +0.846 | +0.943 |
| Fender Super-Sonic | 9 | **-0.776** | -0.733 |
| Fender 57 Twin | 9 | -0.035 | -0.133 |
| **Pooled (raw units, n=32)** | | **-0.378** | |
| **Pooled (per-dataset z-scored, n=32)** | | **-0.223** | |

Level and spectral change are NOT consistently positively correlated across
datasets -- they range from strongly positive (dense Marshall, +0.85) to
strongly negative (JCM800 Hi and Super-Sonic, both around -0.76 to -0.78).
Pooled across all five datasets, the correlation is weakly NEGATIVE, not
strongly positive. A single collapsed response axis assumes these two
quantities broadly agree on where the "important" steps are; on 2 of 5
datasets tested here they are strongly ANTI-correlated -- a step with a
large level change often has a comparatively small spectral change, and
vice versa.

This is a stronger, quantitative confirmation of Section 8's qualitative
finding (RMS-only and combined scores picked different top steps on JCM800
Hi): it isn't a one-off disagreement, it reflects a genuine, sometimes
strongly negative, relationship between the two dimensions across most of
the datasets tested. **Collapsing level and spectral change into one score
routinely discards real, sometimes contradictory information.**

### Practical implication

Report and inspect level change and spectral distance as two separate
series per sweep (already what `compute_adjacent_metrics` returns), not
only their combined score. If a capture-placement UI is built, show both
dimensions -- or at minimum flag when they disagree strongly on which step
is the priority -- rather than presenting one number. A single collapsed
"response axis" score is adequate ONLY when level and spectral distance
happen to agree (as on the dense Marshall dataset here), and that cannot be
assumed in advance.

## Updated conclusions

- Section 9/10 of the base report ("envelope_error_db discriminates
  cleanly") should be read as "discriminates the ONE comparison tested,"
  not "behaves monotonically in general" -- corrected here.
- Section 8's "keep tracking [spectral distance] but do not replace
  RMS-only as primary" recommendation is now on firmer footing: the two
  dimensions are frequently in tension, not just occasionally different, so
  neither should be treated as a stand-in for the other.
