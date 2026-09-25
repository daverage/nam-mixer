# Release notes

Drafts for the next release. The GitHub release page is the published record.

## Unreleased

### A/B Timing Analysis & Correction

NAM Mixer now checks the two source models in Parallel Blend and Dynamic
Hybrid for a genuine fixed timing offset, such as one capture being
recorded a few samples late.

It measures timing on several separate note attacks and then re-checks any
suspected offset on other, independent DIs. That way the natural phase and
tone differences between two amplifiers are not mistaken for latency.

When a stable fixed offset is confirmed, you can audition:

- **Original** timing
- **Corrected** timing

Correction is optional and always starts at Original. If you choose
Corrected, that exact sample offset is frozen into the design and reused
identically for preview, training, validation and export. Sessions remember
the choice, and restore it only after a fresh render confirms the same offset.

NAM Mixer does not try to make different amplifiers share the same
frequency-dependent phase response. Results that are unclear are left
untouched. See [A/B timing](README.md#ab-timing-fixed-offsets-phase-response-and-why-nothing-is-auto-aligned).
