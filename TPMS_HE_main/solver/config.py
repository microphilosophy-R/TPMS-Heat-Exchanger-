"""
Solver configuration helpers.

Provides config normalisation, default configs, and the CPFHX test case.
These functions are kept separate from the calculator class so that
external code (app.py, tests, analysis scripts) can build and validate
configs without importing the heavy solver dependencies.
"""

import copy
import os
import warnings

import numpy as np

from correlations.thermohydraulic_correlations import ThermoHydraulicCorrelations
from models.packed_closures import (
    SUPPORTED_PACKED_MODES,
    SUPPORTED_HYDRAULIC_MODELS,
    SUPPORTED_PHI_SOURCES,
    SUPPORTED_HT_NOMINAL_RULES,
    SUPPORTED_PACKED_HEAT_TRANSFER_MODELS as SUPPORTED_HTC_MODELS,
    SUPPORTED_WALL_ENHANCEMENT_MODELS as SUPPORTED_HT_ENHANCEMENT_MODELS,
    SUPPORTED_KINETIC_MODELS,
    get_allowed_channel_modes,
    get_allowed_hydraulic_models,
    get_allowed_packed_heat_transfer_models,
    get_allowed_wall_enhancement_models,
    get_allowed_kinetic_models,
    normalize_hydraulic_model,
    normalize_packed_heat_transfer_model,
    normalize_wall_enhancement_model,
    normalize_kinetic_model,
)

# Channel modes accepted by the solver
SUPPORTED_CHANNEL_MODES = ("bare", "packed")
# Packed-bed uncertainty modes
SUPPORTED_HTC_MODELS = ('martin_nilles', 'dixon', 'wang_wall_htc')
SUPPORTED_KINETIC_MODELS = ('wilhelmsen_kw', 'arrhenius_first_order')

# ── Re-export for backward compatibility ───────────────────────────────────────
# These are needed by solver/calculator.py without a cross-import
from models.packed_bed import (
    SUPPORTED_PACKED_MODES,
    SUPPORTED_HYDRAULIC_MODELS,
    SUPPORTED_PHI_SOURCES,
    SUPPORTED_HT_ENHANCEMENT_MODELS,
    SUPPORTED_HT_NOMINAL_RULES,
)  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _infer_fluid_type(species):
    species_key = str(species).lower()
    if "water" in species_key:
        return "Water"
    if "air" in species_key:
        return "Air"
    if "rp-3" in species_key or "rp3" in species_key:
        return "RP-3"
    return "Gas"


def _validate_tpms_structure(structure, stream_key):
    if structure not in ThermoHydraulicCorrelations.SUPPORTED_TPMS_TYPES:
        raise ValueError(
            f"Unsupported TPMS structure for channel '{stream_key}': {structure}. "
            f"Supported: {ThermoHydraulicCorrelations.SUPPORTED_TPMS_TYPES}"
        )


def _warn_deprecated_alias(field_path, legacy_value, canonical_value):
    if str(legacy_value).strip().lower() == str(canonical_value).strip().lower():
        return
    warnings.warn(
        f"{field_path}='{legacy_value}' is deprecated; use '{canonical_value}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _normalize_packed_choice(stream_key, packed_cfg, field, default, normalizer, supported):
    raw_value = packed_cfg.get(field, default)
    canonical_value = normalizer(raw_value)
    if str(raw_value).strip().lower() != canonical_value:
        _warn_deprecated_alias(
            f"channels.{stream_key}.packed.{field}",
            raw_value,
            canonical_value,
        )
    if canonical_value not in supported:
        raise ValueError(
            f"Invalid {field} '{canonical_value}' for '{stream_key}'. "
            f"Use one of {supported}."
        )
    packed_cfg[field] = canonical_value
    return canonical_value


def _normalize_uncertainty_mode(stream_key, packed_cfg, fallback_cfg, source_cfg=None):
    source_cfg = source_cfg or {}
    has_uncertainty_mode = "uncertainty_mode" in source_cfg
    has_legacy_mode = "mode" in source_cfg
    default_uncertainty = str(
        fallback_cfg.get("uncertainty_mode", fallback_cfg.get("mode", "nominal"))
    ).strip().lower()

    if has_uncertainty_mode:
        uncertainty_value = source_cfg.get("uncertainty_mode")
        if has_legacy_mode:
            uncertainty_value_norm = str(uncertainty_value).strip().lower()
            legacy_mode_norm = str(source_cfg.get("mode")).strip().lower()
            if uncertainty_value_norm == default_uncertainty and legacy_mode_norm != uncertainty_value_norm:
                warnings.warn(
                    f"channels.{stream_key}.packed.mode is deprecated but overrides the "
                    "default uncertainty_mode for backward compatibility.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                uncertainty_value = source_cfg.get("mode")
    elif has_legacy_mode:
        uncertainty_value = source_cfg.get("mode")
        warnings.warn(
            f"channels.{stream_key}.packed.mode is deprecated; "
            f"use channels.{stream_key}.packed.uncertainty_mode instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    else:
        uncertainty_value = fallback_cfg.get(
            "uncertainty_mode",
            fallback_cfg.get("mode", "nominal"),
        )

    if (
        has_uncertainty_mode
        and has_legacy_mode
        and str(source_cfg.get("mode")).strip().lower()
        != str(source_cfg.get("uncertainty_mode")).strip().lower()
        and str(uncertainty_value).strip().lower()
        == str(source_cfg.get("uncertainty_mode")).strip().lower()
    ):
        warnings.warn(
            f"channels.{stream_key}.packed.mode is deprecated and ignored when "
            f"channels.{stream_key}.packed.uncertainty_mode is present.",
            DeprecationWarning,
            stacklevel=3,
        )

    packed_uncertainty_mode = str(uncertainty_value).strip().lower()
    if packed_uncertainty_mode not in SUPPORTED_PACKED_MODES:
        raise ValueError(
            f"Invalid packed uncertainty_mode '{packed_uncertainty_mode}' for '{stream_key}'. "
            f"Use one of {SUPPORTED_PACKED_MODES}."
        )

    packed_cfg["uncertainty_mode"] = packed_uncertainty_mode
    packed_cfg["mode"] = packed_uncertainty_mode
    return packed_uncertainty_mode


def _normalize_catalyst_config(cfg):
    catalyst = cfg.setdefault("catalyst", {})
    if not catalyst:
        return

    if "uncertainty_mode" not in catalyst and "mode" in catalyst:
        warnings.warn(
            "catalyst.mode is deprecated; use catalyst.uncertainty_mode instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        catalyst["uncertainty_mode"] = catalyst["mode"]

    if "uncertainty_mode" in catalyst:
        catalyst["uncertainty_mode"] = str(
            catalyst.get("uncertainty_mode", "nominal")
        ).strip().lower()
        if catalyst["uncertainty_mode"] not in SUPPORTED_PACKED_MODES:
            raise ValueError(
                f"Invalid catalyst uncertainty_mode '{catalyst['uncertainty_mode']}'. "
                f"Use one of {SUPPORTED_PACKED_MODES}."
            )

    if "hydraulic_model" in catalyst:
        raw_value = catalyst["hydraulic_model"]
        catalyst["hydraulic_model"] = normalize_hydraulic_model(raw_value)
        _warn_deprecated_alias("catalyst.hydraulic_model", raw_value, catalyst["hydraulic_model"])
    if "htc_model" in catalyst:
        raw_value = catalyst["htc_model"]
        catalyst["htc_model"] = normalize_packed_heat_transfer_model(raw_value)
        _warn_deprecated_alias("catalyst.htc_model", raw_value, catalyst["htc_model"])
    if "ht_enhancement_model" in catalyst:
        raw_value = catalyst["ht_enhancement_model"]
        catalyst["ht_enhancement_model"] = normalize_wall_enhancement_model(raw_value)
        _warn_deprecated_alias(
            "catalyst.ht_enhancement_model",
            raw_value,
            catalyst["ht_enhancement_model"],
        )
    if "kinetic_model" in catalyst:
        raw_value = catalyst["kinetic_model"]
        catalyst["kinetic_model"] = normalize_kinetic_model(raw_value)
        _warn_deprecated_alias("catalyst.kinetic_model", raw_value, catalyst["kinetic_model"])


def _normalize_single_channel(cfg, stream_key):
    channels = cfg.setdefault("channels", {})
    tpms_cfg = cfg.setdefault("tpms", {})
    geo = cfg.setdefault("geometry", {})
    cat = cfg.get("catalyst", {})

    legacy_structure = tpms_cfg.get(f"type_{stream_key}", "Diamond")
    ch_cfg = copy.deepcopy(channels.get(stream_key, {}))

    mode = str(ch_cfg.get("mode", "bare")).strip().lower()
    if mode not in SUPPORTED_CHANNEL_MODES:
        raise ValueError(
            f"Invalid channel mode '{mode}' for '{stream_key}'. "
            f"Use one of {SUPPORTED_CHANNEL_MODES}."
        )

    structure = ch_cfg.get("structure", legacy_structure)
    _validate_tpms_structure(structure, stream_key)
    allowed_modes = get_allowed_channel_modes(structure)
    if mode not in allowed_modes:
        raise ValueError(
            f"Channel '{stream_key}' structure '{structure}' only supports modes {allowed_modes}."
        )

    packed_defaults = {
        "particle_diameter": cat.get("particle_diameter", 1e-3),
        "bed_porosity": cat.get("bed_porosity", 0.40),
        "k_solid": cat.get("k_solid", 10.0),
        "k_solid_material": cat.get("k_solid_material", None),
        "shape_factor": cat.get("shape_factor", 1.0),
        "uncertainty_mode": cat.get("uncertainty_mode", cat.get("mode", "nominal")),
        "hydraulic_model": cat.get("hydraulic_model", "ergun_psi_tpms"),
        "htc_model": cat.get("htc_model", "martin_nilles"),
        "phi_source": cat.get("phi_source", "ch3_f_re_fit"),
        "ht_enhancement_model": cat.get("ht_enhancement_model", "off"),
        "ht_nominal_rule": cat.get("ht_nominal_rule", "geometric"),
        "kinetic_model": cat.get("kinetic_model", "wilhelmsen_kw"),
    }
    kinetic_params_defaults = {
        "Ea_J_per_mol": -336.45,
        "a_m3s_per_mol": 2.2e-3,
        "b_s_inv": -35.11e-3,
    }
    kinetic_params_defaults.update(cat.get("kinetic_params", {}))

    packed_source_cfg = copy.deepcopy(ch_cfg.get("packed", {}) or {})
    packed_cfg = copy.deepcopy(packed_defaults)
    packed_cfg.update(packed_source_cfg)
    kinetic_params = copy.deepcopy(kinetic_params_defaults)
    kinetic_params.update(packed_source_cfg.get("kinetic_params", {}))
    packed_cfg["kinetic_params"] = kinetic_params

    packed_mode = _normalize_uncertainty_mode(
        stream_key,
        packed_cfg,
        packed_defaults,
        packed_source_cfg,
    )

    _normalize_packed_choice(
        stream_key,
        packed_cfg,
        "htc_model",
        "martin_nilles",
        normalize_packed_heat_transfer_model,
        SUPPORTED_HTC_MODELS,
    )
    if mode == "packed" and packed_cfg["htc_model"] not in get_allowed_packed_heat_transfer_models(structure):
        raise ValueError(
            f"Channel '{stream_key}' structure '{structure}' only supports HTC models "
            f"{get_allowed_packed_heat_transfer_models(structure)}."
        )

    _normalize_packed_choice(
        stream_key,
        packed_cfg,
        "hydraulic_model",
        "ergun_psi_tpms",
        normalize_hydraulic_model,
        SUPPORTED_HYDRAULIC_MODELS,
    )
    if mode == "packed" and packed_cfg["hydraulic_model"] not in get_allowed_hydraulic_models(structure):
        raise ValueError(
            f"Channel '{stream_key}' structure '{structure}' only supports hydraulic models "
            f"{get_allowed_hydraulic_models(structure)}."
        )

    phi_source = str(packed_cfg.get("phi_source", "ch3_f_re_fit")).strip().lower()
    if phi_source not in SUPPORTED_PHI_SOURCES:
        raise ValueError(
            f"Invalid phi_source '{phi_source}' for '{stream_key}'. "
            f"Use one of {SUPPORTED_PHI_SOURCES}."
        )
    packed_cfg["phi_source"] = phi_source

    _normalize_packed_choice(
        stream_key,
        packed_cfg,
        "ht_enhancement_model",
        "off",
        normalize_wall_enhancement_model,
        SUPPORTED_HT_ENHANCEMENT_MODELS,
    )
    if mode == "packed" and packed_cfg["ht_enhancement_model"] not in get_allowed_wall_enhancement_models(structure):
        raise ValueError(
            f"Channel '{stream_key}' structure '{structure}' only supports wall enhancement models "
            f"{get_allowed_wall_enhancement_models(structure)}."
        )

    ht_nominal_rule = str(packed_cfg.get("ht_nominal_rule", "geometric")).strip().lower()
    if ht_nominal_rule not in SUPPORTED_HT_NOMINAL_RULES:
        raise ValueError(
            f"Invalid ht_nominal_rule '{ht_nominal_rule}' for '{stream_key}'. "
            f"Use one of {SUPPORTED_HT_NOMINAL_RULES}."
        )
    packed_cfg["ht_nominal_rule"] = ht_nominal_rule

    _normalize_packed_choice(
        stream_key,
        packed_cfg,
        "kinetic_model",
        "wilhelmsen_kw",
        normalize_kinetic_model,
        SUPPORTED_KINETIC_MODELS,
    )
    if mode == "packed" and packed_cfg["kinetic_model"] not in get_allowed_kinetic_models(structure):
        raise ValueError(
            f"Channel '{stream_key}' structure '{structure}' only supports kinetic models "
            f"{get_allowed_kinetic_models(structure)}."
        )

    kp = copy.deepcopy(packed_cfg.get("kinetic_params", {}))
    packed_cfg["kinetic_params"] = {
        "Ea_J_per_mol": float(kp.get("Ea_J_per_mol", kinetic_params_defaults["Ea_J_per_mol"])),
        "a_m3s_per_mol": float(kp.get("a_m3s_per_mol", kinetic_params_defaults["a_m3s_per_mol"])),
        "b_s_inv": float(kp.get("b_s_inv", kinetic_params_defaults["b_s_inv"])),
    }

    if packed_cfg["particle_diameter"] <= 0:
        raise ValueError(f"Channel '{stream_key}' packed particle_diameter must be > 0.")
    if packed_cfg["k_solid"] <= 0:
        raise ValueError(f"Channel '{stream_key}' packed k_solid must be > 0.")
    if packed_cfg["shape_factor"] <= 0:
        raise ValueError(f"Channel '{stream_key}' packed shape_factor must be > 0.")
    if not (0.05 <= packed_cfg["bed_porosity"] <= 0.95):
        raise ValueError(
            f"Channel '{stream_key}' packed bed_porosity must be within [0.05, 0.95]."
        )

    # Preserve per-channel surface_area_density if the user supplied it
    ch_sad = ch_cfg.get("surface_area_density", None)
    # Per-channel geometry — all fields owned per channel (no global fallback here)
    ch_geo_raw = ch_cfg.get("geometry", {}) or {}
    # Defaults differ by channel
    _por_default  = 0.65 if stream_key == "hot" else 0.70
    # length/width inherit from global geometry (shared physical dimensions)
    _L = ch_geo_raw.get("length") or geo.get("length", 0.94)
    _W = ch_geo_raw.get("width")  or geo.get("width",  0.25)
    canonical = {
        "mode": mode,
        "structure": structure,
        "packed": packed_cfg,
        "geometry": {
            "length":         float(_L),
            "width":          float(_W),
            "height":         float(ch_geo_raw.get("height",         geo.get("height",         0.25))),
            "unit_cell_size": float(ch_geo_raw.get("unit_cell_size", geo.get("unit_cell_size", 5e-3))),
            "wall_thickness": float(ch_geo_raw.get("wall_thickness", geo.get("wall_thickness", 5e-4))),
            # PlateFin-specific fin geometry
            "fin_height":     float(ch_geo_raw.get("fin_height",    geo.get("fin_height",    9.5e-3))),
            "fin_spacing":    float(ch_geo_raw.get("fin_spacing",   geo.get("fin_spacing",   3.2e-3))),
            "fin_thickness":  float(ch_geo_raw.get("fin_thickness", geo.get("fin_thickness", 0.6e-3))),
            "perf_density":   float(ch_geo_raw.get("perf_density",  geo.get("perf_density",  0.0))),
            "perf_radius":    float(ch_geo_raw.get("perf_radius",   geo.get("perf_radius",   0.0))),
            # Structural porosity (was porosity_hot/cold in global geometry)
            "porosity":       float(ch_geo_raw.get("porosity",
                                   geo.get(f"porosity_{stream_key}", _por_default))),
            "n_layers":       float(ch_geo_raw.get("n_layers", 1.0)),
        },
    }
    if ch_sad is not None:
        canonical["surface_area_density"] = ch_sad
    channels[stream_key] = canonical

    tpms_cfg[f"type_{stream_key}"] = structure


def normalize_config(config):
    """
    Normalize legacy and new config schemas into canonical channel-level configuration.
    """
    cfg = copy.deepcopy(config)

    cfg.setdefault("geometry", {})
    cfg.setdefault("material", {})
    cfg.setdefault("operating", {})
    cfg.setdefault("solver", {})
    cfg.setdefault("output", {})
    cfg.setdefault("tpms", {})
    cfg.setdefault("channels", {})
    cfg.setdefault("catalyst", {})

    solver = cfg["solver"]
    if "relax_thermal" not in solver and "relax" in solver:
        solver["relax_thermal"] = solver["relax"]
    solver.setdefault("relax_thermal", 0.15)
    solver.setdefault("relax_hydraulic", 0.5)
    solver.setdefault("relax_kinetics", 1.0)
    solver.setdefault("Q_damping", 0.5)
    solver.setdefault("n_elements", 100)
    solver.setdefault("max_iter", 500)
    solver.setdefault("tolerance", 1e-3)

    geo = cfg["geometry"]
    geo.setdefault("length", 0.94)
    geo.setdefault("width", 0.25)
    geo.setdefault("plate_thickness", 1.0e-3)
    # Keep legacy global fields as fallbacks for _normalize_single_channel migration
    # (they are read there to back-fill per-channel geometry when not explicitly set)
    geo.setdefault("height", 0.25)
    geo.setdefault("unit_cell_size", 5e-3)
    geo.setdefault("wall_thickness", 0.5e-3)
    geo.setdefault("fin_height",    9.5e-3)
    geo.setdefault("fin_spacing",   3.2e-3)
    geo.setdefault("fin_thickness", 0.6e-3)
    geo.setdefault("perf_density",  0.0)
    geo.setdefault("perf_radius",   0.0)
    geo.setdefault("surface_area_density", 60)

    material = cfg["material"]
    material.setdefault("k_wall", 237.0)
    material.setdefault("plate_material", None)

    operating = cfg["operating"]
    operating.setdefault("Th_in", 78.0)
    operating.setdefault("Tc_in", 43.0)
    operating.setdefault("Ph_in", 2e6)
    operating.setdefault("Pc_in", 1.5e6)
    operating.setdefault("mh", 2e-2)
    operating.setdefault("mc", 6e-2)
    operating.setdefault("xh_in", 0.452)
    operating.setdefault("fluid_hot", "hydrogen mixture")
    operating.setdefault("fluid_cold", "helium")

    output = cfg["output"]
    output.setdefault("results_csv", "results/final_results.csv")
    output.setdefault("convergence_csv", "results/convergence_history.csv")
    output.setdefault("performance_plot", "results/performance_profile.png")
    output.setdefault("convergence_plot", "results/convergence_diagnostics.png")
    output.setdefault("performance_eval_plot", "results/performance_evaluation.png")

    _normalize_catalyst_config(cfg)

    _normalize_single_channel(cfg, "hot")
    _normalize_single_channel(cfg, "cold")

    # Per-channel SAD fallback: if not set by user, inherit global SAD
    global_sad = geo.get("surface_area_density", 60)
    for sk in ("hot", "cold"):
        cfg["channels"][sk].setdefault("surface_area_density", global_sad)

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Output paths helper
# ──────────────────────────────────────────────────────────────────────────────

def _results_root():
    """Absolute path to the top-level results/ directory (sibling of TPMS_HE_main/)."""
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results')
    )


def make_run_dir(run_name: str) -> str:
    """Create and return ``results/<run_name>/``.

    Use this from analysis scripts or programmatic solver calls so that each
    experiment stores its outputs in an isolated subdirectory:

    .. code-block:: python

        from solver.config import make_run_dir, _default_output_paths
        out = make_run_dir("sensitivity_20260311_143000")
        cfg["output"] = _default_output_paths(run_subdir=run_name)
    """
    run_dir = os.path.join(_results_root(), run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _default_output_paths(run_subdir: str = ""):
    """Return output file paths inside ``results/`` (or ``results/<run_subdir>/``).

    Parameters
    ----------
    run_subdir : str, optional
        If given, output files are placed in ``results/<run_subdir>/``.
        The directory is created automatically.
        If empty (default) files land directly in ``results/``.
    """
    _root = _results_root()
    if run_subdir:
        _root = os.path.join(_root, run_subdir)
    os.makedirs(_root, exist_ok=True)

    def _p(name):
        return os.path.join(_root, name)

    return {
        'results_csv':           _p('final_results.csv'),
        'convergence_csv':       _p('convergence_history.csv'),
        'performance_plot':      _p('performance_profile.png'),
        'convergence_plot':      _p('convergence_diagnostics.png'),
        'resistance_pie':        _p('resistance_pie.png'),
        'performance_eval_plot': _p('performance_evaluation.png'),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────────────────────

def create_default_config():
    """Create default configuration."""
    _ch_geo_hot = {
        'length': 0.94, 'width': 0.25, 'height': 0.25,
        'unit_cell_size': 5e-3, 'wall_thickness': 0.5e-3,
        'porosity': 0.65,
        'fin_height': 9.5e-3, 'fin_spacing': 3.2e-3, 'fin_thickness': 0.6e-3,
        'perf_density': 0.0, 'perf_radius': 0.0,
    }
    _ch_geo_cold = {
        'length': 0.94, 'width': 0.25, 'height': 0.25,
        'unit_cell_size': 5e-3, 'wall_thickness': 0.5e-3,
        'porosity': 0.70,
        'fin_height': 9.5e-3, 'fin_spacing': 3.2e-3, 'fin_thickness': 0.6e-3,
        'perf_density': 0.0, 'perf_radius': 0.0,
    }
    return {
        'geometry': {
            'length': 0.94, 'width': 0.25,
            'plate_thickness': 1.0e-3,
        },
        'tpms': {'type_hot': 'Diamond', 'type_cold': 'Gyroid'},
        'channels': {
            'hot': {
                'mode': 'bare',
                'structure': 'Diamond',
                'surface_area_density': 60,
                'geometry': _ch_geo_hot,
                'packed': {
                    'particle_diameter': 1e-3,
                    'bed_porosity': 0.40,
                    'k_solid': 10.0,
                    'shape_factor': 1.0,
                    'uncertainty_mode': 'nominal',
                    'htc_model': 'martin_nilles',
                    'hydraulic_model': 'ergun_psi_tpms',
                    'phi_source': 'ch3_f_re_fit',
                    'ht_enhancement_model': 'off',
                    'ht_nominal_rule': 'geometric',
                    'kinetic_model': 'wilhelmsen_kw',
                    'kinetic_params': {
                        'Ea_J_per_mol': -336.45,
                        'a_m3s_per_mol': 2.2e-3,
                        'b_s_inv': -35.11e-3,
                    },
                },
            },
            'cold': {
                'mode': 'bare',
                'structure': 'Gyroid',
                'surface_area_density': 60,
                'geometry': _ch_geo_cold,
                'packed': {
                    'particle_diameter': 1e-3,
                    'bed_porosity': 0.40,
                    'k_solid': 10.0,
                    'shape_factor': 1.0,
                    'uncertainty_mode': 'nominal',
                    'htc_model': 'martin_nilles',
                    'hydraulic_model': 'ergun_psi_tpms',
                    'phi_source': 'ch3_f_re_fit',
                    'ht_enhancement_model': 'off',
                    'ht_nominal_rule': 'geometric',
                    'kinetic_model': 'wilhelmsen_kw',
                    'kinetic_params': {
                        'Ea_J_per_mol': -336.45,
                        'a_m3s_per_mol': 2.2e-3,
                        'b_s_inv': -35.11e-3,
                    },
                },
            },
        },
        'material': {'k_wall': 237},
        'operating': {
            'Th_in': 78, 'Tc_in': 43, 'Ph_in': 2e6, 'Pc_in': 1.5e6,
            'mh': 2e-2, 'mc': 6e-2, 'xh_in': 0.452,
            'fluid_hot': 'hydrogen mixture',
            'fluid_cold': 'helium',
        },
        'catalyst': {
            'particle_diameter': 1e-3,
            'bed_porosity': 0.40,
            'k_solid': 10.0,
            'shape_factor': 1.0,
            'uncertainty_mode': 'nominal',
            'htc_model': 'martin_nilles',
            'hydraulic_model': 'ergun_psi_tpms',
            'phi_source': 'ch3_f_re_fit',
            'ht_enhancement_model': 'off',
            'ht_nominal_rule': 'geometric',
            'kinetic_model': 'wilhelmsen_kw',
            'kinetic_params': {
                'Ea_J_per_mol': -336.45,
                'a_m3s_per_mol': 2.2e-3,
                'b_s_inv': -35.11e-3,
            },
        },
        'solver': {
            'n_elements': 100, 'max_iter': 500, 'tolerance': 1e-3,
            'relax': 0.15, 'relax_thermal': 0.15,
            'relax_hydraulic': 0.5, 'relax_kinetics': 1.0,
            'Q_damping': 0.5, 'adaptive_damping': True
        },
        'output': _default_output_paths()
    }


# ──────────────────────────────────────────────────────────────────────────────
# CPFHX Test Configuration — Wang et al. (2024)
# ──────────────────────────────────────────────────────────────────────────────

# Table 6: experimental operating conditions
# Keys: (back_pressure_MPa, flowrate_ratio_r)
# Values: inlet/outlet temperatures [K] and H2 temperature drop [K]
# Cold (He) outlet temperatures available only for 1.04 MPa conditions.
_CPFHX_TABLE6 = {
    (1.04, 2.4): dict(Th_in=63.8, Tc_in=42.8, Th_out_exp=55.9, Tc_out_exp=61.7, dT_H2=7.9),
    (1.04, 2.7): dict(Th_in=63.1, Tc_in=42.8, Th_out_exp=55.1, Tc_out_exp=60.8, dT_H2=8.0),
    (1.04, 3.0): dict(Th_in=62.3, Tc_in=42.7, Th_out_exp=54.1, Tc_out_exp=59.7, dT_H2=8.2),
    (1.13, 2.4): dict(Th_in=65.8, Tc_in=43.8, Th_out_exp=57.4, Tc_out_exp=None, dT_H2=8.4),
    (1.13, 2.7): dict(Th_in=65.0, Tc_in=44.2, Th_out_exp=56.5, Tc_out_exp=None, dT_H2=8.5),
    (1.13, 3.0): dict(Th_in=64.5, Tc_in=44.8, Th_out_exp=55.9, Tc_out_exp=None, dT_H2=8.6),
}


def create_cpfhx_config(back_pressure_mpa=1.04, flowrate_ratio=2.7):
    """
    Return a solver config for the CPFHX experiment of Wang et al. (2024).

    Geometry (Table 2):
        Core unit: L = 0.47 m, W = 0.15 m, per-stream H approx 0.032 m
        Hot fins:  Hf=9.5 mm, sf=3.2 mm, tf=0.6 mm  -> SAD~776 m-1, eps=0.8125, Dh~4.0 mm
        Cold fins: Hf=9.5 mm, sf=1.0 mm, tf=0.2 mm  -> SAD~2147 m-1, eps=0.80, Dh~1.5 mm

    Parameters
    ----------
    back_pressure_mpa : float   Valid values: 1.04, 1.13
    flowrate_ratio    : float   Valid values: 2.4, 2.7, 3.0
    """
    key = (float(back_pressure_mpa), float(flowrate_ratio))
    if key not in _CPFHX_TABLE6:
        raise ValueError(
            f"Unknown CPFHX condition {key}. "
            f"Valid keys: {sorted(_CPFHX_TABLE6.keys())}"
        )
    cond = _CPFHX_TABLE6[key]

    Hf   = 9.5e-3
    sf_h = 3.2e-3;  tf_h = 0.6e-3
    sf_c = 1.0e-3;  tf_c = 0.2e-3

    def _sad(Hf_, sf_, tf_):
        return ((2 * Hf_ - tf_) + 2 * (sf_ - tf_)) / (sf_ * Hf_)

    sad_h = _sad(Hf, sf_h, tf_h)
    sad_c = _sad(Hf, sf_c, tf_c)
    eps_h = (sf_h - tf_h) / sf_h
    eps_c = (sf_c - tf_c) / sf_c

    m_h2 = 1.0e-3
    m_he = float(flowrate_ratio) * m_h2
    P_op = float(back_pressure_mpa) * 1e6

    return {
        'geometry': {
            'length': 0.47,
            'width':  0.15,
            'plate_thickness': 1.2e-3,
        },
        'tpms': {'type_hot': 'PlateFin', 'type_cold': 'PlateFin'},
        'channels': {
            'hot': {
                'mode':      'packed',
                'structure': 'PlateFin',
                'surface_area_density': sad_h,
                'geometry': {
                    'length': 0.47, 'width': 0.15, 'height': 0.032,
                    'unit_cell_size': 5e-3, 'wall_thickness': tf_h,
                    'porosity':      eps_h,
                    'fin_height':    Hf,
                    'fin_spacing':   sf_h,
                    'fin_thickness': tf_h,
                    'perf_density':  0.0,
                    'perf_radius':   0.0,
                },
                'packed': {
                    'particle_diameter': 1.5e-3,
                    'bed_porosity':      0.40,
                    'k_solid':           10.0,
                    'shape_factor':      1.0,
                    'uncertainty_mode': 'nominal',
                    'htc_model':        'martin_nilles',
                    'hydraulic_model':  'ergun_psi_tpms',
                    'phi_source':       'ch3_f_re_fit',
                    'ht_enhancement_model': 'off',
                    'ht_nominal_rule':  'geometric',
                    'kinetic_model':    'wilhelmsen_kw',
                    'kinetic_params': {
                        'Ea_J_per_mol': -336.45,
                        'a_m3s_per_mol': 2.2e-3,
                        'b_s_inv': -35.11e-3,
                    },
                },
            },
            'cold': {
                'mode':      'bare',
                'structure': 'PlateFin',
                'surface_area_density': sad_c,
                'geometry': {
                    'length': 0.47, 'width': 0.15, 'height': 0.032,
                    'unit_cell_size': 5e-3, 'wall_thickness': tf_c,
                    'porosity':      eps_c,
                    'fin_height':    Hf,
                    'fin_spacing':   sf_c,
                    'fin_thickness': tf_c,
                    'perf_density':  0.0,
                    'perf_radius':   0.0,
                },
                'packed': {
                    'particle_diameter': 1e-3,
                    'bed_porosity':      0.40,
                    'k_solid':           10.0,
                    'shape_factor':      1.0,
                    'uncertainty_mode': 'nominal',
                    'htc_model':        'martin_nilles',
                    'hydraulic_model':  'ergun_psi_tpms',
                    'phi_source':       'ch3_f_re_fit',
                    'ht_enhancement_model': 'off',
                    'ht_nominal_rule':  'geometric',
                    'kinetic_model':    'wilhelmsen_kw',
                    'kinetic_params': {
                        'Ea_J_per_mol': -336.45,
                        'a_m3s_per_mol': 2.2e-3,
                        'b_s_inv': -35.11e-3,
                    },
                },
            },
        },
        'material': {'k_wall': 237.0},
        'operating': {
            'fluid_hot':  'hydrogen mixture',
            'fluid_cold': 'helium',
            'Th_in':  float(cond['Th_in']),
            'Tc_in':  float(cond['Tc_in']),
            'Ph_in':  P_op,
            'Pc_in':  P_op,
            'mh':     m_h2,
            'mc':     m_he,
            'xh_in':  0.25,
        },
        'catalyst': {
            'particle_diameter': 1.5e-3,
            'bed_porosity':      0.40,
            'k_solid':           10.0,
            'shape_factor':      1.0,
        },
        'solver': {
            'n_elements':     100,
            'max_iter':       500,
            'tolerance':      1e-3,
            'relax_thermal':  0.15,
            'relax_hydraulic': 0.5,
            'relax_kinetics': 1.0,
            'Q_damping':      0.5,
        },
        'output': _default_output_paths(),
        '_cpfhx_ref': {
            'back_pressure_mpa': back_pressure_mpa,
            'flowrate_ratio':    flowrate_ratio,
            'Th_out_exp':        cond.get('Th_out_exp'),
            'Tc_out_exp':        cond.get('Tc_out_exp'),
            'dT_H2':             cond.get('dT_H2'),
            'source':            'Wang et al. (2024), Int. J. Hydrogen Energy 110, 814-825, Table 6',
        },
    }
