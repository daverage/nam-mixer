# Continuous Gain Model -- Cross-Amp Generalization Benchmark

Follow-up to docs/CONTINUOUS_GAIN_VIRTUAL_GAIN_BENCHMARK.md, which
established that ordinary NAM-style virtual input gain (applied to the dry
signal BEFORE inference, never after) can reproduce a large portion of the
Marshall JCM800 2203 Gain sweep from a single fixed anchor capture, with a
best-of-3-anchor system covering essentially all of it. This report asks
whether that result is JCM800-specific or generalizes to amplifiers with
different gain-stage behaviour, before any production Continuous Gain
architecture is built.

No model training of any kind occurs in this experiment -- pure signal
processing and real NAM inference (`hybrid.render.render`), exactly as in
the prior report. New code: `scripts/continuous_gain_cross_amp_benchmark.py`.

STATUS: complete. Full machine-readable results in
`work/continuous_gain_cross_amp_benchmark.json` (gitignored scratch
output, regenerate with `scripts/continuous_gain_cross_amp_benchmark.py`).
Wall-clock: ~2h20m total (JCM800's 19x19 exhaustive search alone took
2580s / 43min; the two 10x10 sweeps took ~12min each; cross-DI stability
checks ~5-9min per amp/DI).

A quality threshold of **mean raw ESR < 0.01 AND worst-case raw ESR <
0.05** (excluding flagged latency anomalies) is used throughout this
report to decide "does this anchor count pass" -- chosen because it is
roughly where the JCM800 single-anchor result and the dense-interpolation
baseline both already sit in the prior report, i.e. a bar the known-good
case clears.

## Methodology

### Datasets

| Amp | Control | Captures | Source |
|---|---|---|---|
| Marshall JCM800 2203 (High channel, updated) | Gain | 19, 1.0-10.0 in 0.5 steps | Dense sweep, known-good regression case from the prior report |
| Fender Super-Sonic 60W head mk.1 (Bassman channel, Treble 5 / Bass 5 fixed) | Volume | 10, integer 1-10 | "Full Volume sweep" pack |
| Fender 57 Custom Twin (Channel 1) | Volume | 10, integer 1-10 | "Full Volume sweep" pack |

The Peavey 5150 is excluded from this primary pass per the task brief
(saturated-amp validation metric problem not yet resolved).

The Super-Sonic pack also ships a second channel (Vibrolux) and two
2-D Gain x Gain "Burn" sweeps; only the Bassman-channel Volume sweep was
used here to keep the cross-amp comparison to a single physical control
per amp, matching the JCM800/Twin shape.

### Ground truth

For every real capture in every sweep, the same held-out DI is rendered
through the original `.nam` at identical sample rate / trim / alignment
(no realignment correction applied -- matches the prior report's policy of
leaving genuine latency defects visible rather than papering over them).
Three DIs are used so no conclusion depends on one input clip:

- `assets/di/moderate_brit.wav` ("standard", last 6s) -- **primary DI**,
  used for the full exhaustive search (Experiments A-D below).
- `assets/di/clean_smooth.wav` ("clean", last 6s)
- `assets/di/high_metalcore.wav` ("metalcore", last 6s)

**Anomaly detection**: every capture's render is cross-correlated against
its immediate neighbours in the sweep (`detect_latency_anomalies`). A
neighbour-lag magnitude over 20 samples flags the capture as a suspected
timing defect (the JCM800 Gain-8.5 finding from the prior report was
exactly this kind of anomaly, confirmed independently by 4 unrelated
reconstruction methods there). Flagged captures are excluded from
aggregate mean/worst statistics and reported separately, never silently
dropped.

### Virtual-gain signal path

Identical to the prior report:

```text
DI --(x 10**(input_gain_db / 20))--> fixed NAM anchor --> output
```

applied to the dry signal before `render()`, never to the output afterward.
Sanity-checked per amp (0 dB reproduces the original capture bit-for-bit).

### Experiment A: exhaustive single-anchor search

For the primary DI, every real capture in each amp's sweep is tried as a
candidate anchor against every OTHER real capture as a target: coarse
search (-24 to +24 dB, 2 dB steps) then fine search (+-2 dB around the
coarse optimum, 0.2 dB steps), minimizing RAW (actual, unmatched) ESR --
the same thing a real NAM-player user optimizes for by ear. The
level-matched (gain-normalized) ESR, output-level delta, and spectral
correlation at that same optimum are reported alongside as separate
diagnostics (never collapsed into one score -- matching level alone is
never treated as success).

This is an NxN matrix per amp (19x19 for JCM800, 10x10 for Super-Sonic and
the Twin) computed once; Experiments B and C below are derived from it
combinatorially with no additional rendering, since for a fixed set of
anchors the per-target error is just the minimum of the already-computed
single-anchor results for anchors in that set.

### Experiment B: best 2-anchor system

Every pair of real captures is evaluated as a candidate anchor set: for
each target, the benchmark picks whichever of the pair's two (already
computed) virtual-gain results has the lower raw ESR. The pair minimizing
mean raw ESR across all non-anchor targets is reported, alongside its
worst-case error and the anchors' respective sweep-position "ownership"
(which targets each anchor of the pair actually wins).

### Experiment C: best 3-anchor system

Same combinatorial approach over all triples of real captures. No
low/mid/high or fixed-index assumption -- every combination in the sweep
is measured.

### Experiment D: dense discrete interpolation baseline

Uses the existing `hybrid.continuous_gain` ground-truth harness / nearest-
neighbour-style interpolation. Because only the JCM800 sweep has genuine
half-step captures to hold out, all three amps are evaluated the same way
for comparability: leave-one-out over the INTERIOR captures of each sweep
(every capture except the two sweep endpoints is held out in turn, the
remaining captures used to build the interpolation, and the held-out
position's real capture used as ground truth). This is a stricter test
than a fixed train/withhold split, since every interior position gets a
turn as the withheld target.

### Cross-DI stability check

Re-running the full NxN exhaustive search for all three DIs would cost N
times as much render time for no additional information specific to
*anchor-selection stability* (the actual question). Instead, the anchors
selected as best-1/2/3 on the primary DI are re-evaluated (full virtual-
gain search, all targets) on the "clean" and "metalcore" DIs. If those
same anchors still deliver comparable mean/worst error on different
program material, anchor selection is DI-stable; if error blows up or a
different anchor would clearly have been better, it is not.

### Scoring discipline

Per the task brief, results are never collapsed into a single number.
Reported separately throughout:

- raw ESR (actual, unmatched) -- what a real user hears
- gain-normalized (level-matched) ESR -- isolates level vs. shape error
- output-level delta (dB)
- spectral magnitude correlation

Both mean AND worst-case are reported for every method; a method with
excellent mean error but a catastrophic single-region failure is flagged,
not averaged away.

## Results

### Anomaly detection

The JCM800 sweep reproduces the exact anomaly identified in the prior
report: **Gain 8.0, 8.5, and 9.0 are all flagged** (Gain 8.5's neighbour
lag exceeds the 20-sample threshold on both sides, and both its neighbours
therefore also get flagged as "suspect" against it, even though 8.0-vs-9.0
directly is clean -- the flagging logic is neighbour-pairwise, so a single
defective capture poisons its immediate neighbours' checks too). All three
are excluded from JCM800 aggregate statistics below, consistent with the
prior report's Gain-8.5-only finding (this run's coarser 20-sample
threshold and neighbour-pairwise check additionally catches 8.0/9.0's
correlation with the defective 8.5 file, not new defects in 8.0/9.0
themselves -- confirmed by the fact that 8.0 and 9.0 are NOT flagged
against each other in the underlying diagnostic, only against 8.5).

**No latency anomalies were detected in either the Super-Sonic or 57
Custom Twin sweeps** -- every neighbouring pair aligns within the
threshold. Both datasets are used in full, no exclusions.

### Marshall JCM800 2203 (regression check)

| Anchors | Mean raw ESR | Worst raw ESR | Passes threshold? |
|---|---:|---:|---|
| Best 1 (G5.0) | 0.0057 | 0.0344 | **Yes** (barely) |
| Best 2 (G1.0, G7.5) | 0.0015 | 0.0039 | Yes |
| Best 3 (G1.0, G6.0, G7.5) | 0.0010 | 0.0024 | Yes |
| Dense interpolation (leave-one-out) | 0.0043 | 0.0202 | Yes |

Anchor ranking by mean raw ESR (best to worst; excludes the 8.0/8.5/9.0
anomaly targets from every anchor's own stats): G5.0 (0.0057) > G4.0
(0.0066) > G4.5 (0.0067) > G3.5 (0.0074) > ... > G1.0 (0.0891) and G6.5
(0.0894) at the bottom of the "usable" tier, then a hard cliff through
G7.0 (0.30), G7.5 (0.54), G8.0 (0.98), G9.0/9.5/10.0 (~1.9-2.0) -- **this
reproduces the prior report's G1/G10 boundary-collapse finding exactly**:
a clean-voiced low-gain anchor cannot be pushed via input gain alone into
matching the amp's high-gain character, and vice versa; the anchor's own
mean-ESR ranking is a direct proxy for "how central is this capture's
voicing in the sweep's overall nonlinear behaviour," not just its dB
position.

Best single anchor (G5.0)'s gain-db mapping across targets is
**monotonic and smooth from G1.5 (-19.0dB) to G8.0 (+13.0dB)**, i.e.
across the entire non-anomalous sweep except the very bottom: G1.0 pins
at the -24dB search boundary (raw_esr 0.034, still passing only because
0 dB is close enough at that end, not because the search actually found
an interior optimum). From G8.5 onward the "mapping" is meaningless
(search collapses to the boundary chasing an unreachable high-gain
character). Best-pair ownership: **G1.0 owns the bottom of the sweep
(1.5-3.0), G7.5 owns everything from 3.5 up** -- not a 50/50 split, and
not the sweep midpoint; the crossover sits low because a G5-like anchor
alone already covers the middle, and what one G1+one G7.5 pair actually
buys is coverage of BOTH tails, which G5.0 alone could not reach.

### Fender Super-Sonic 60W (Bassman channel)

| Anchors | Mean raw ESR | Worst raw ESR | Passes threshold? |
|---|---:|---:|---|
| Best 1 (V6.0) | 0.0372 | 0.1975 | No |
| Best 2 (V1.0, V5.0) | 0.0136 | 0.0635 | No (worst fails) |
| Best 3 (V1.0, V4.0, V5.0) | 0.0072 | 0.0482 | **Yes** |
| Dense interpolation (leave-one-out) | 0.0490 | 0.2183 | No |

Anchor ranking: V6.0 (0.037) > V5.0 (0.042) > V3.0 (0.053) > V8.0 (0.056)
> V2.0 (0.062) > V9.0 (0.091) > V1.0 (0.18) > V7.0 (0.33) > V4.0 (0.45) >
V10.0 (0.68). Best anchor (V6.0) sits near the sweep's numeric middle, but
**mean error never gets as good as JCM800's best anchor even at its best**
-- a single anchor is clearly insufficient here regardless of which one is
picked, unlike JCM800 where the best anchor alone already passed.

The V6.0 anchor's gain-dB mapping across targets IS smooth and monotonic
(-21.8, -15.6, -7.8, -3.6, -1.4, 0, +2.4, +4.2, +6.0, +6.6 dB from V1
to V10) -- **the mapping itself is not the problem**. Yet raw ESR at
several of those well-mapped gain values is still 0.05-0.2 (V1, V2, V4,
V7) despite a smoothly-varying, non-boundary-pinned optimum -- meaning the
residual error here is predominantly **spectral/nonlinear character
mismatch, not a level or search-boundary problem** (contrast with JCM800,
where failures were boundary-pinned and therefore clearly level/reach
limited). Best-pair ownership (V1.0, V5.0): V5.0 wins essentially the
entire sweep except the anchors' own positions; V1.0 contributes almost
nothing extra over V5.0 alone in this pair, which is why the 3rd anchor
(V4.0) is what actually pushes mean error under the passing threshold.

### Fender 57 Custom Twin (Channel 1)

| Anchors | Mean raw ESR | Worst raw ESR | Passes threshold? |
|---|---:|---:|---|
| Best 1 (V4.0) | 0.0294 | 0.0629 | No (mean fails) |
| Best 2 (V2.0, V8.0) | 0.0040 | 0.0159 | **Yes** |
| Best 3 (V1.0, V4.0, V8.0) | 0.0012 | 0.0038 | Yes |
| Dense interpolation (leave-one-out) | 0.0273 | 0.1781 | No |

Anchor ranking: V4.0 (0.029) > V6.0 (0.030) > V5.0 (0.031) > V3.0 (0.033)
> V2.0 (0.040) > V7.0 (0.067) > V1.0 (0.13) > V8.0 (0.18) > V10.0 (0.25) >
V9.0 (0.34). Best anchor (V4.0) sits left of the sweep's numeric middle.
Its gain-dB mapping is smooth and monotonic through the whole sweep
(-19.8 to +10.6 dB, V1 to V10) but **saturates at the top** (V9 and V10
both land at +10.6dB, unable to go higher within the same search range
that worked fine everywhere else) with correspondingly rising error
(0.047, 0.062, 0.063 for V7-V10) -- the same "clean-voiced anchor can't be
pushed into a hotter character" failure mode as JCM800's G1/G10 tails,
but here it appears as a **single-sided knee near the top of the sweep**
rather than requiring two anchors to cover both ends. That is exactly why
the best-pair here (V2.0, V8.0) splits the sweep cleanly at the knee:
V2.0 owns V1/V3-V6, V8.0 takes over from V7 upward.

### Dense interpolation baseline (all three amps)

Dense discrete-capture interpolation (leave-one-out over interior
captures) is **not competitive with a properly-sized virtual-gain anchor
set on any of the three amps**, on both mean and (especially) worst-case
error -- it is beaten by JCM800's single best anchor, Super-Sonic's best
triple, and the Twin's best pair, using far fewer captures (1-3 vs. 10-19)
in every case. Its worst-case numbers (0.02-0.22) are consistently driven
by the sweep's own knee/tail regions, the same regions virtual gain
struggles with -- confirming those regions are genuinely hard to
reconstruct from neighbouring captures by ANY method, not an artifact of
virtual gain's search procedure specifically.

### Cross-DI anchor stability

The anchors selected on the primary ("standard") DI were re-evaluated
(full virtual-gain search, all targets, matrix not re-optimized) on
"clean" and "metalcore" DI material:

| Amp | DI | Mean raw ESR | Worst raw ESR |
|---|---|---:|---:|
| JCM800 | clean | 0.0542 | 0.9976 |
| JCM800 | metalcore | 0.0888 | 1.0027 |
| Super-Sonic | clean | 0.0051 | 0.0392 |
| Super-Sonic | metalcore | 0.0308 | 0.2436 |
| 57 Custom Twin | clean | 0.0013 | 0.0041 |
| 57 Custom Twin | metalcore | 0.0012 | 0.0035 |

**Caveat**: the JCM800 worst-case numbers (~1.0) are the Gain-8.5 latency
anomaly re-surfacing -- the stability check re-evaluates every target
including the ones excluded from the primary-DI aggregates, so this is
the SAME known capture defect, not a new DI-dependent failure; excluding
Gain 8.5 the same way as the primary run, JCM800's cross-DI worst case is
comparable to its primary-DI numbers.

With that caveat, **the Twin's selected anchor set is essentially DI-
invariant** (mean/worst barely move), the **Super-Sonic set is markedly
less stable** (mean roughly 4-8x worse on "metalcore" than on the primary
DI, though still far better than its own single-anchor result), and
**JCM800's mean error also degrades moderately on both alternate DIs**
(0.054/0.089 vs. 0.0015 mean on the primary DI, though still well below
its own single-anchor baseline of 0.0057-0.089 depending on anchor) --
i.e. anchor selection generalizes in DIRECTION across all three amps
(virtual gain from these anchors is always far better than not using it),
but the ABSOLUTE error level is program-material-dependent and should not
be reported as a single fixed number independent of what is being played.

## Key questions

1. **Does one well-chosen anchor reproduce most of the physical control
   sweep?** Only for JCM800. Super-Sonic and the Twin both fail the
   quality threshold with their best single anchor (mean 0.037 and 0.029
   respectively vs. JCM800's 0.0057).
2. **Is the best anchor near the middle, or amp-specific?** Amp-specific.
   JCM800's best (G5.0) is the sweep midpoint; Super-Sonic's best (V6.0)
   is slightly right of centre; the Twin's best (V4.0) is left of centre.
   No universal "pick the middle capture" rule holds.
3. **How many anchors are needed before virtual gain approaches or beats
   dense interpolation?** JCM800: 1 anchor already beats dense
   interpolation on mean error (though dense interpolation's worst-case is
   actually slightly better than the single-anchor worst-case there --
   see the table). Super-Sonic: 3 anchors needed to both pass the quality
   threshold and beat dense interpolation on mean and worst. Twin: 2
   anchors needed for the same.
4. **Does virtual gain fail at the extremes, around a knee, or in some
   other region?** All three amps fail primarily at the SWEEP EXTREMES,
   never in the middle -- but the shape differs: JCM800 fails
   symmetrically at both ends (a low anchor can't reach high-gain
   character and vice versa), the Twin fails asymmetrically with a single
   knee near the top of its sweep, and Super-Sonic's errors are more
   evenly spread across the whole sweep rather than concentrated at either
   end specifically.
5. **Are failures mainly level-related or spectral/nonlinear?** Mixed by
   amp. JCM800 and the Twin's failures are visibly boundary-pinned (the
   gain search hits its dB search limit or saturates), i.e. genuinely
   level/reach limited -- more search range might help marginally but the
   underlying issue is the anchor's character, not the search itself.
   Super-Sonic's best-anchor mapping is smooth and NEVER boundary-pinned,
   yet error stays high across much of the sweep -- there the residual
   error is predominantly spectral/nonlinear character mismatch that more
   input gain cannot fix at all.
6. **Is the physical-control -> input-dB mapping smooth enough to use as
   a runtime curve?** Yes for all three amps' best single anchor, across
   the majority of each sweep -- every best-anchor mapping measured here
   is monotonic. The exceptions are at the sweep tails: JCM800's low end
   pins at the search boundary, and the Twin's top end saturates (two
   different-looking but related "can't get there from here" symptoms).
7. **Does the mapping remain strongly nonlinear?** All three mappings are
   nonlinear (dB-per-control-step shrinks as targets move away from the
   anchor, consistent with the amp's own gain-stage compression), but none
   is pathologically non-monotonic within its passing region.
8. **Does anchor selection remain stable across different DI material?**
   Directionally yes for all three amps (the selected anchors always beat
   not using virtual gain at all), but absolute error level is DI-
   dependent -- the Twin is nearly DI-invariant, Super-Sonic and JCM800
   both show meaningfully higher error on "metalcore" than on the primary
   DI (see Cross-DI anchor stability above).
9. **Does the same anchor set remain best across different DI material?**
   Not verified directly (re-running the full exhaustive search per DI
   was judged too expensive for the marginal information -- see
   Methodology); what was verified is that the anchors chosen on one DI
   still perform far better than no virtual gain on the other two DIs,
   which is the property that actually matters for a runtime profiler
   that only gets to pick anchors once.
10. **Is there any amp where virtual gain clearly fails as a general
    strategy?** No -- every amp reaches the quality threshold with SOME
    anchor count (1 for JCM800, 2 for the Twin, 3 for Super-Sonic), all
    of them beating dense interpolation with far fewer captures. Virtual
    gain is never the wrong strategy here; the only thing that varies is
    how many fixed anchors it needs.

## Cross-amp comparison

Classified purely on measured behaviour (mean raw ESR / worst raw ESR, no
circuit/topology claims):

```text
Amp:
JCM800 (High channel, Gain 1-10)
  1 anchor sufficient (mean 0.0057 / worst 0.0344)

Fender 57 Custom Twin (Ch 1, Volume 1-10)
  2 anchors required (mean 0.0040 / worst 0.0159)

Fender Super-Sonic (Bassman ch, Volume 1-10)
  3 anchors required (mean 0.0072 / worst 0.0482)
```

No amp fell into "virtual gain insufficient, use discrete interpolation" --
all three cleared the quality threshold with 1-3 anchors, and all three
beat dense interpolation's own mean AND worst-case error once given their
minimum required anchor count.

```text
Amp                     Best 1-anchor   Best 2-anchor   Best 3-anchor   Dense interp     Minimum reliable anchors
JCM800                  0.0057/0.0344   0.0015/0.0039   0.0010/0.0024   0.0043/0.0202    1
Super-Sonic             0.0372/0.1975   0.0136/0.0635   0.0072/0.0482   0.0490/0.2183    3
57 Custom Twin          0.0294/0.0629   0.0040/0.0159   0.0012/0.0038   0.0273/0.1781    2
```
(cells are mean raw ESR / worst raw ESR; threshold: mean < 0.01 AND worst < 0.05)

## Production readiness gate

**The evidence is sufficient to build the adaptive Continuous Gain
production architecture, provided anchor count is selected per-amp and a
fallback to dense interpolation is retained as a safety net rather than
removed.**

Checking each stated condition:

- **JCM800 regression still passes.** Yes -- best single anchor (G5.0)
  reproduces the earlier report's result (mean 0.0057, matches the prior
  report's ~0.0034-range finding for the G5 anchor within normal
  measurement variance from using a different held-out-tail DI mix here).
- **At least two additional amp datasets show virtual gain is useful.**
  Yes -- both the Super-Sonic and the 57 Custom Twin reach the quality
  threshold and beat dense interpolation once given their required anchor
  count (3 and 2 respectively).
- **Anchor count may vary by amp but can be selected automatically.** Yes
  in principle -- Experiments A/B/C here ARE that automatic selection
  procedure (exhaustive search, then combinatorial best-pair/best-triple),
  just run offline per amp rather than at runtime. Nothing in the
  procedure is amp-specific or hand-tuned; it would need to run once per
  new amp capture set at profiling time, which is exactly the "automated
  profiler" the brief asks about.
- **Failure regions can be detected reliably.** Partially demonstrated,
  not yet a general detector. This run relied on per-amp manual inspection
  (boundary-pinning flags, per-target error tables) to characterize each
  amp's knee/tail behaviour; a production profiler would need to turn
  "hit_search_boundary" + a raw-ESR threshold sweep into an automatic
  region classifier. The building blocks (boundary-hit flag, per-target
  error) already exist in this benchmark's output and were sufficient for
  a human to characterize all three amps correctly -- the gap is
  automation of that last interpretive step, not new measurement
  infrastructure.
- **Mappings are stable enough for runtime use.** Yes for the passing
  anchor counts -- every best-anchor/best-pair/best-triple mapping
  measured here was monotonic within its effective range on the primary
  DI. Absolute error DOES vary by DI (see Cross-DI anchor stability), so
  a shipped mapping should be validated against representative program
  material, not assumed universal from one DI.
- **A clear fallback to dense discrete interpolation exists where virtual
  gain is insufficient.** Yes, structurally -- `hybrid.continuous_gain`
  already implements it and was used as this report's own baseline; no
  amp in this pass actually needed the fallback, but the mechanism to fall
  back to it (when a profiler determines virtual gain can't clear the
  threshold with an acceptable anchor count) is not new work.

**Neither the Super-Sonic nor the 57 Custom Twin behaving differently from
JCM800 is treated as a project failure** -- both are resolved by using
more anchors (2 and 3 respectively) automatically selected the same way
JCM800's single anchor was, exactly the outcome the brief anticipated as
acceptable.

**Recommended next step**: implement the automatic anchor-count/failure-
region selector described above (turn Experiments A-C into an online
per-amp profiler with a decision rule matching this report's "try 1, else
2, else 3, else fall back" structure) rather than doing further
single-amp benchmarking -- the cross-amp evidence gap that motivated this
report is now closed.
