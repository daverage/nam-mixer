# Release notes

The GitHub release page is the published record; drafts for the next release go
under an "Unreleased" heading.

## v0.5.0

### Update notifications

NAM Mixer now checks for a newer version when it starts. If one is out, a
notice offers **Download** or **Not now** (you can check again any time in
**Settings → Updates**). The download is always the right installer for your
computer: the Apple Silicon DMG, the Windows installer, or the Linux AppImage
or `.deb`, matching how you installed it. The check is an anonymous read of
the public release list; nothing about you or your files is sent, and you can
turn it off in **Settings → Advanced**.

### NAM Inspector

A new tool in the **Tools** tab shows what's inside a `.nam`: architecture,
sample rate, calibration, metadata, whether it contains a cabinet (only when
there's real evidence), and its timing history. It also plays a short test
signal through the model, checking both Full and Lite for A2 models. It never
changes the file.

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
untouched. See [A/B timing](docs/user_guide.md#ab-timing-fixed-offsets-phase-response-and-why-nothing-is-auto-aligned).

### Easier documentation

The README is now a short guide to what NAM Mixer does and how to get started.
Full detail lives in the new [user guide](docs/user_guide.md) and
[developer guide](docs/developer_guide.md).
