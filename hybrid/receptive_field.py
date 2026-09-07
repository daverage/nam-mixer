"""Inspect the official `neural-amp-modeler` A2/PackedWaveNet config to compute
its receptive field, so hybrid/envelope.py's bounded crossover history can be
checked against it rather than assumed -- see docs/phase3.md section 5.

This module does NOT require torch to be importable: `nam.train._resources`
ships `config_model_packed.json` as plain package data, and the receptive
field of a WaveNet-style stack is a pure function of its
kernel-size/dilation config (no weights needed), so we parse the JSON
directly. If `neural-amp-modeler` is not installed at all (the normal Flask/
runtime environment, per CLAUDE.md, is NOT expected to have it), the caller
gets a clear `ReceptiveFieldUnavailable` explaining that a training
environment is required for this specific check.

Schema verified against the ACTUALLY INSTALLED neural-amp-modeler==0.13.0
package (see requirements-training.txt) --
`nam.train.core._get_packed_model_config()` returns:

    {"net": {"name": "PackedWaveNet",
             "config": {"submodels": [
                 {"name": "channels_3", "config": {"layers_configs": [
                     {"kernel_sizes": [...], "dilations": [...], ...}
                 ], "head_scale": ...}},
                 {"name": "channels_8", "config": {...}},
             ]}}}

i.e. each submodel is one WaveNet with a single `layers_configs` entry whose
`kernel_sizes` is a PER-LAYER list (not one scalar kernel size for the whole
stack, as an initial guess might assume) paired element-wise with
`dilations`. At the time of writing, both `channels_3` ("Lite") and
`channels_8` ("Full") submodels share the same 23-layer kernel/dilation
schedule -> a 6332-sample (~131.9 ms @ 48 kHz) receptive field.
"""
from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass
from typing import Optional


class ReceptiveFieldUnavailable(RuntimeError):
    """Raised when the official A2 config can't be located/parsed -- e.g. the
    `neural-amp-modeler` training package isn't installed in this
    environment. See requirements-training.txt / scripts/setup_a2_env.ps1."""


_PACKED_CONFIG_RESOURCE = ("nam.train._resources", "config_model_packed.json")


def _layer_array_receptive_field(layer_array_cfg: dict) -> int:
    """Receptive field (in samples) of one WaveNet layer stack with
    PER-LAYER kernel sizes: 1 + sum((kernel_size[i] - 1) * dilation[i]) over
    its stacked dilated-conv layers -- the standard formula for a dilated
    causal conv stack, generalized to a per-layer kernel size rather than one
    shared across the whole stack (see module docstring: the real installed
    config uses a `kernel_sizes` list, not a single `kernel_size`)."""
    kernel_sizes = layer_array_cfg["kernel_sizes"]
    dilations = layer_array_cfg["dilations"]
    if len(kernel_sizes) != len(dilations):
        raise ReceptiveFieldUnavailable(
            f"kernel_sizes ({len(kernel_sizes)}) and dilations ({len(dilations)}) length mismatch"
        )
    return 1 + sum((int(k) - 1) * int(d) for k, d in zip(kernel_sizes, dilations))


def _net_receptive_field(net_cfg: dict) -> int:
    """Receptive field of one WaveNet `config` block (a list of layer-array
    configs). Multiple entries would be stacked in series (each consuming the
    previous one's output as input), so their individual context
    requirements ADD, minus the 1-sample overlap counted in each "+1" term.

    Accepts either key name actually seen in the wild: the training
    package's packed config uses `layers_configs`
    (nam.train._resources/config_model_packed.json), while an exported
    source `.nam` file's own WaveNet block uses `layers` (see
    compute_source_nam_receptive_field's docstring) -- same per-layer
    kernel_sizes/dilations schema either way.
    """
    layer_arrays = net_cfg.get("layers_configs", net_cfg.get("layers"))
    if layer_arrays is None:
        raise ReceptiveFieldUnavailable(f"WaveNet config has neither 'layers_configs' nor 'layers': keys={list(net_cfg.keys())}")
    total = 1
    for layer_array_cfg in layer_arrays:
        total += _layer_array_receptive_field(layer_array_cfg) - 1
    return total


def _model_receptive_field(model: dict) -> int:
    """Receptive field of one parsed `.nam` model dict (or a nested one, for
    a `SlimmableContainer`'s per-submodel `model` block), dispatching on its
    own `architecture` field -- the same field `hybrid.nam_loader.NamModel`
    exposes.

    Verified against a real captured `.nam` file (SlimmableContainer wrapping
    two WaveNet submodels): `config.submodels` is a list of
    `{"max_value": <float>, "model": {"architecture": "WaveNet", "config": {"layers": [...]}}}`
    entries -- a DIFFERENT shape from the training package's packed A2
    config (see module docstring), despite both ultimately being dilated
    WaveNet stacks, so this is intentionally a separate code path from
    `_iter_submodel_configs`/`compute_a2_receptive_field` above rather than a
    shared one.
    """
    architecture = model.get("architecture")
    config = model.get("config", {})
    if architecture in ("WaveNet", "PackedWaveNet"):
        return _net_receptive_field(config)
    if architecture == "SlimmableContainer":
        submodels = config.get("submodels")
        if not submodels:
            raise ReceptiveFieldUnavailable(f"SlimmableContainer config has no 'submodels': keys={list(config.keys())}")
        best = 0
        for entry in submodels:
            nested = entry.get("model", entry)
            best = max(best, _model_receptive_field(nested))
        return best
    raise ReceptiveFieldUnavailable(
        f"Don't know how to compute the receptive field of architecture {architecture!r} -- "
        "add a case to hybrid.receptive_field._model_receptive_field for it."
    )


def compute_source_nam_receptive_field(nam_model) -> int:
    """Receptive field (in samples) of an arbitrary source `.nam` capture
    (e.g. Amp A/Amp B) -- NOT the training package's A2 config. Needed
    because a hybrid target's total dry-input dependency is
    `max(envelope history, Amp A receptive field, Amp B receptive field)`,
    not the envelope history alone: the two amp branches run in parallel
    with the crossover envelope on the same dry input (docs/phase3.md
    review) -- see `hybrid.pipeline.render_pair`.

    Takes a `hybrid.nam_loader.NamModel` (or anything with a `.raw` dict
    attribute / a plain raw dict itself). Requires no torch and no
    `neural-amp-modeler` install -- pure JSON-schema math, same as
    `compute_a2_receptive_field`.
    """
    raw = getattr(nam_model, "raw", nam_model)
    return _model_receptive_field(raw)


@dataclass
class A2ReceptiveField:
    receptive_field_samples: int
    submodel_names: list[str]
    config_path: str
    raw_config: dict


def load_packed_a2_config() -> tuple[dict, str]:
    """Return (parsed JSON, resource path str) for the installed
    neural-amp-modeler's packed/A2 model config, or raise
    ReceptiveFieldUnavailable."""
    try:
        resource = importlib.resources.files(_PACKED_CONFIG_RESOURCE[0]).joinpath(_PACKED_CONFIG_RESOURCE[1])
        if not resource.is_file():
            raise FileNotFoundError(str(resource))
        raw = json.loads(resource.read_text(encoding="utf-8"))
        return raw, str(resource)
    except (ModuleNotFoundError, FileNotFoundError, ImportError) as exc:
        raise ReceptiveFieldUnavailable(
            "Could not locate neural-amp-modeler's packed A2 config "
            f"({_PACKED_CONFIG_RESOURCE[0]}/{_PACKED_CONFIG_RESOURCE[1]}). "
            "Install the pinned training environment first -- see "
            "requirements-training.txt and scripts/setup_a2_env.ps1. "
            f"Underlying error: {exc}"
        ) from exc


def _iter_submodel_configs(raw_config: dict):
    """Yield (name, net_config) for each packed submodel -- see module
    docstring for the verified real schema:
    `raw_config["net"]["config"]["submodels"] = [{"name":..., "config": {"layers_configs": [...]}}, ...]`.
    """
    try:
        submodels = raw_config["net"]["config"]["submodels"]
    except (KeyError, TypeError) as exc:
        raise ReceptiveFieldUnavailable(
            f"Unexpected packed A2 config shape (no net.config.submodels): keys={list(raw_config.keys())}"
        ) from exc
    for entry in submodels:
        yield entry["name"], entry["config"]


def compute_a2_receptive_field() -> A2ReceptiveField:
    """The overall receptive field to check the envelope's bounded history
    against is the MAXIMUM across submodels (Full is normally the largest;
    Lite is included so a future config where Lite is larger doesn't get
    missed silently)."""
    raw, path = load_packed_a2_config()

    best_samples = 0
    names: list[str] = []
    for name, net_cfg in _iter_submodel_configs(raw):
        names.append(name)
        try:
            samples = _net_receptive_field(net_cfg)
        except (KeyError, TypeError) as exc:
            raise ReceptiveFieldUnavailable(
                f"Could not compute receptive field for A2 submodel {name!r} from {path}: {exc}"
            ) from exc
        best_samples = max(best_samples, samples)

    if not names:
        raise ReceptiveFieldUnavailable(f"Packed A2 config at {path} contained no recognizable submodels.")

    return A2ReceptiveField(
        receptive_field_samples=best_samples, submodel_names=names, config_path=path, raw_config=raw,
    )


def cab_fir_serial_history_samples(fir_length_samples: int) -> int:
    """Additional temporal-dependency samples a BAKED cabinet FIR of
    `fir_length_samples` taps adds on top of whatever the Hybrid/Blend
    combination already needs -- see docs/blend-mode.md "RECEPTIVE FIELD".

    The FIR runs AFTER the amp combination, so it's a SERIAL dependency, not
    a parallel one like Amp A/Amp B/the crossover envelope: producing one
    output sample of the cabbed signal needs `fir_length_samples` samples of
    the (already combined) pre-cab signal, which in turn each need their own
    `base_required_history` -- hence `base + (L - 1)`, not `max(base, L)`.
    A preview-only (non-baked) cab adds zero training-time dependency since
    it never touches the target that gets trained on.
    """
    return max(0, int(fir_length_samples) - 1)


def combine_required_history(
    mode: str,
    amp_a_samples: int,
    amp_b_samples: int,
    envelope_samples: Optional[int],
    cab_fir_samples: int = 0,
) -> dict:
    """Combine the per-branch dependency samples of a generated target into
    one required-history record, mode-aware -- see docs/blend-mode.md
    "IMPORTANT CONCEPTUAL POLICY":

        Hybrid (parallel):  hard core = max(Amp A, Amp B, envelope)
        Blend  (parallel):  hard core = max(Amp A, Amp B)         -- no envelope
        + baked cab (serial): formal total = hard core + cab_fir_samples
          (already the L-1 serial-history count, via
          `cab_fir_serial_history_samples`)

    IMPORTANT POLICY DISTINCTION (see docs/blend-mode.md's cabinet
    approximation policy): `hard_required_samples` is the CORE Hybrid/Blend
    dependency -- Amp A/Amp B (+ envelope for Hybrid) alone, with NO cab
    contribution. This is the value that MUST fit inside the destination
    A2's actual receptive field, or generation/training is refused exactly
    as before this policy existed.

    `formal_total_required_samples` additionally folds in a baked cab's
    serial FIR history. It is calculated and reported honestly, but must
    NEVER by itself gate training -- a baked cab whose formal total exceeds
    the A2's receptive field means the A2 will LEARN AN APPROXIMATION of the
    post-cab response within its available temporal capacity, not that
    training is invalid. See `scripts/train_a2.py`'s/the Kaggle cloud
    worker's `check_receptive_field` for where that distinction is actually
    enforced.

    Returns a plain dict (not a dataclass) so it serializes directly into
    manifest JSON without extra plumbing; both `hybrid.training_target` and
    `hybrid.blend_training_target` build the "receptive_field" manifest
    section from this same function so local/Kaggle checks can never
    silently diverge in how they combine branches.

    Legacy keys `base_required_samples`/`total_required_samples` are kept,
    numerically identical to `hard_required_samples`/
    `formal_total_required_samples`, for manifests/readers written before
    this policy existed -- they must NEVER be read as the new hard gate
    (some old code/tests did exactly that, which is the bug this policy
    fixes; see docs/blend-mode.md).
    """
    if mode not in ("hybrid", "blend", "character"):
        raise ValueError(f"unknown mode: {mode!r} (expected 'hybrid', 'blend', or 'character')")

    branch_samples = {"amp_a": int(amp_a_samples), "amp_b": int(amp_b_samples)}
    if mode in ("hybrid", "character"):
        if envelope_samples is None:
            raise ValueError("envelope_samples is required for mode='hybrid'")
        branch_samples["envelope"] = int(envelope_samples)

    hard_required_samples = max(branch_samples.values())
    cab_fir_samples = max(0, int(cab_fir_samples))
    formal_total_required_samples = hard_required_samples + cab_fir_samples

    return {
        "mode": mode,
        "branch_samples": branch_samples,
        "hard_required_samples": hard_required_samples,
        "cab_fir_serial_samples": cab_fir_samples,
        "formal_total_required_samples": formal_total_required_samples,
        # Legacy aliases -- see docstring. Do not use for the hard gate.
        "base_required_samples": hard_required_samples,
        "total_required_samples": formal_total_required_samples,
    }


def assert_required_history_fits(
    required_history_samples: int,
    sample_rate: int,
    margin_fraction: float = 0.0,
) -> A2ReceptiveField:
    """Raise ValueError only if `required_history_samples` (already the max
    across whatever PARALLEL branches feed the target -- see
    `hybrid.pipeline.render_pair`/scripts/train_a2.py's `check_receptive_field`:
    Amp A, Amp B, and the crossover envelope all consume the same dry input
    independently, so their temporal requirements do NOT add, they take the
    max) exceeds the actual installed A2's receptive field.

    This is the CORE/HARD check -- see docs/blend-mode.md's cabinet
    approximation policy: callers must pass the CORE Hybrid/Blend dependency
    here (`hybrid.receptive_field.combine_required_history`'s
    `hard_required_samples`), NEVER a cab-inflated total. A baked cabinet's
    formal (post-combination, serial) history is a separate, advisory-only
    calculation -- see `cab_fir_serial_history_samples` -- that must never
    be passed to this function as if it were part of the hard requirement.

    `required == receptive_field_samples` (an EXACT fit, zero temporal
    margin) is PERMITTED, not treated as a failure -- a memoryless
    per-sample blend (the crossfade itself) adds no extra history on top of
    whatever the slowest parallel branch already needs, so exactly matching
    the receptive field is representable, just with no slack left for
    anything else. Only `required > available` is a genuine "the A2
    physically cannot see far enough back" failure. Note this checks
    TEMPORAL reach only, not whether the network has enough capacity to
    actually learn the composite function within that reach -- that's an
    empirical training/validation question, not something this function can
    answer.

    Returns the computed A2ReceptiveField on success so callers can log it
    (check `required == rf.receptive_field_samples` themselves to warn about
    zero margin). Raises ReceptiveFieldUnavailable if the training
    environment isn't installed here at all -- see that class's docstring.
    """
    rf = compute_a2_receptive_field()
    required = int(required_history_samples * (1.0 + margin_fraction))
    if required > rf.receptive_field_samples:
        raise ValueError(
            f"Required history ({required_history_samples} samples, "
            f"{required_history_samples / sample_rate * 1000:.1f} ms at {sample_rate} Hz, "
            f"{required} with margin) does not fit inside the installed A2's receptive "
            f"field ({rf.receptive_field_samples} samples, submodels={rf.submodel_names}, "
            f"from {rf.config_path})."
        )
    return rf


def assert_envelope_history_fits(
    envelope_history_samples: int,
    sample_rate: int,
    margin_fraction: float = 0.0,
) -> A2ReceptiveField:
    """Deprecated alias for `assert_required_history_fits` -- kept for
    backward compatibility with existing callers/tests written before this
    function was renamed to reflect that it checks the CORE Hybrid/Blend
    dependency (Amp A/B [+ envelope]), not merely "the envelope". Prefer
    `assert_required_history_fits` in new code."""
    return assert_required_history_fits(envelope_history_samples, sample_rate, margin_fraction)
