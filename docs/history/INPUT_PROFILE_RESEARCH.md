# Input profile research

This document explains the research basis for `hybrid/input_profiles.py`'s
relative-gain presets, why they're structured the way they are, and the
limits of what they can honestly claim.

## Why relative output matters for this project

The hybrid crossfade is driven entirely by how loud the guitar/bass input
signal is (`hybrid/envelope.py`). A player with a hot pickup drives the amp
harder and crosses into Amp B sooner and more often than a player with a
low-output pickup, for the exact same playing dynamics. If we want to know
whether a given crossover/transition setting is actually reachable by a
realistic player -- not just by an artificially loud test signal -- we need
some notion of "how loud does a realistic pickup actually drive this."

## Why absolute guitar dBFS cannot be inferred from pickup type alone

A pickup's output voltage is one factor among many that determine the final
digital level of a recording: preamp gain staging, interface input trim,
pick attack, string gauge, pickup height, and playing technique all matter
at least as much. There is no way to look at a `.wav` file and say "this was
definitely recorded with a P90." These profiles are RELATIVE-gain
simulations layered on top of a chosen reference recording -- never a claim
about what pickup actually produced any particular waveform.

## Why selected DI files cannot tell us their original pickup output

The genre DI clips bundled in `assets/di/` were found (Phase 1) to be mostly
normalized around a common RMS level by their original source (NAMtoClo's
tone-match reference set). That normalization erases whatever relative
output differences existed between however they were originally recorded.
So we cannot recover, from the waveform, what pickup was actually used --
and we don't try to. Instead, the selected DI is treated as a REFERENCE
PERFORMANCE, and each input profile simulates what a relatively hotter or
quieter pickup would have produced playing that same performance.

For guitar DIs, the reference assumption is: **the selected DI represents a
vintage/PAF-style passive humbucker (0 dB)**. For bass DIs: **the selected DI
represents a standard passive Jazz/Precision bass (0 dB)**. Overriding the
guitar/bass instrument selection is always available to the user; filenames
beginning with `bass_` are used only as a default hint, never as a
detection claim.

## Guitar profile research (single reference family)

All guitar profiles are relative to a vintage/PAF-style passive humbucker,
rounded from representative manufacturer relative-output (mV) data:

| Profile | Approx. output | Relative to 250 mV ref | Applied gain |
|---|---|---|---|
| Vintage / low-output single | ~90-125 mV | -8.9 to -6.0 dB | **-7.0 dB** |
| Standard / hotter single | ~160-200 mV | -3.9 to -1.9 dB | **-3.0 dB** |
| Vintage / PAF humbucker | ~220-250 mV | -1.1 to 0 dB | **0.0 dB** (reference) |
| P90 | ~270-287 mV | +0.7 to +1.2 dB | **+1.0 dB** |
| Medium / modern humbucker | ~300-375 mV | +1.6 to +3.5 dB | **+2.5 dB** |
| Hot humbucker | ~400-435 mV | +4.1 to +4.8 dB | **+4.5 dB** |
| Extreme passive | ~510 mV | +6.2 dB | **+6.0 dB** |

Conversion formula: `gain_db = 20 * log10(output_mV / reference_mV)`.

## Bass profile research (a SEPARATE reference family)

Bass pickups are not meaningfully comparable to guitar pickups on the same
0 dB point -- different string count, different typical output circuitry,
different playing dynamics. Bass profiles are relative to a standard
passive Jazz/Precision-style bass:

| Profile | Approx. output | Applied gain |
|---|---|---|
| Standard Jazz/Precision bass | ~150-163 mV | **0.0 dB** (reference) |
| Modern / hotter passive bass | ~170-200 mV | **+1.5 dB** |
| Very high-output passive bass | ~230-250 mV | **+3.5 dB** |

## Why active pickups are Custom, not a fixed preset

Manufacturer data for one active-pickup product line alone shows roughly a
9.5 dB spread (e.g. ~1.0 V "string" output for an active single coil vs.
~3.0 V for an active high-output humbucker on one measurement method), and a
different measurement convention on the same pair narrows that gap
substantially. Some manufacturers (e.g. certain Fishman Fluence models) also
offer a selectable ~6 dB output-reduction switch on the SAME pickup. Given
that spread, a universal "Active = +X dB" preset would be a fabricated
precision this project can't defend. Active guitar and active/preamped bass
profiles therefore set `requires_custom_gain=True` and expose a manual
-12..+12 dB slider instead (see `hybrid/input_profiles.py`'s
`CUSTOM_GAIN_MIN_DB`/`CUSTOM_GAIN_MAX_DB`). Active/preamped basses are
called out as especially variable: some active bass pickup elements measure
BELOW stock passive output before their onboard preamp is accounted for, so
"active" must never be equated with "hotter" by default.

## What this is NOT

- Not a laboratory calibration of any specific real pickup.
- Not a claim that a given bundled DI clip really was recorded with a PAF or
  a Jazz bass -- see the normalization caveat above.
- Not accounting for pickup height, string gauge/type, pick attack, pickup
  position, or instrument wiring, all of which measurably affect real
  output level and are outside this project's scope.
- Not a guitar volume-knob simulation (deliberately not implemented -- pot
  taper, loading, and treble-bleed circuits vary too much between
  instruments to responsibly guess at yet).

These are documented, research-grounded CATEGORY SIMULATIONS for exercising
the crossover-reachability question, not calibration ground truth.

## NAM input calibration (a separate, complementary concept)

Input profiles change what signal is fed to both NAMs, simulating a
different instrument. NAM input calibration is a different question: two
`.nam` captures may have been trained/calibrated against different assumed
analogue input levels (`input_level_dbu` in the `.nam` file's metadata).
Feeding identical digital samples to two differently-calibrated models does
not represent feeding the same physical voltage into both amps. The official
NAM plugin corrects for this with:

```
model_input_adjustment_db = reference_input_level_dbu - model_input_level_dbu
```

implemented in `hybrid/calibration.py`. The default reference is **+12.0
dBu**, matching the official plugin's default -- this is NOT a pickup output
value, it's the analogue level assumed to correspond to digital full scale.

**Auto mode** applies this compensation only when BOTH models in a pair
report a valid `input_level_dbu`. If only one (or neither) does, the pair
falls back to raw digital level for both, with a visible warning -- silently
calibrating only one model would apply a real physical assumption
asymmetrically, which is worse than assuming neither is calibrated.
**Raw mode** always skips this compensation regardless of available
metadata.
