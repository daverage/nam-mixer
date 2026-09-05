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
    """Receptive field of one WaveNet `config` block (a `layers_configs`
    list). Multiple entries would be stacked in series (each consuming the
    previous one's output as input), so their individual context
    requirements ADD, minus the 1-sample overlap counted in each "+1" term;
    the real installed config only ever has one entry, but this stays
    general rather than assuming that won't change."""
    layer_arrays = net_cfg["layers_configs"]
    total = 1
    for layer_array_cfg in layer_arrays:
        total += _layer_array_receptive_field(layer_array_cfg) - 1
    return total


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


def assert_envelope_history_fits(
    envelope_history_samples: int,
    sample_rate: int,
    margin_fraction: float = 0.0,
) -> A2ReceptiveField:
    """Raise ValueError if the crossover envelope's declared maximum
    dry-input history does not fit comfortably inside the actual installed
    A2's receptive field (with `margin_fraction` extra headroom required on
    top). Returns the computed A2ReceptiveField on success so callers can log
    it. Raises ReceptiveFieldUnavailable if the training environment isn't
    installed here at all -- see that class's docstring."""
    rf = compute_a2_receptive_field()
    required = int(envelope_history_samples * (1.0 + margin_fraction))
    if required >= rf.receptive_field_samples:
        raise ValueError(
            f"Crossover envelope history ({envelope_history_samples} samples, "
            f"{envelope_history_samples / sample_rate * 1000:.1f} ms at {sample_rate} Hz, "
            f"{required} with margin) does not fit inside the installed A2's receptive "
            f"field ({rf.receptive_field_samples} samples, submodels={rf.submodel_names}, "
            f"from {rf.config_path})."
        )
    return rf
