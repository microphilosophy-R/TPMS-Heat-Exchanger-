import pytest

from models.packed_closures import (
    SUPPORTED_KINETIC_MODELS,
    get_structure_compatibility,
)
from solver.config import create_default_config, normalize_config


def _make_hot_channel_cfg(structure, mode="packed"):
    cfg = create_default_config()
    cfg["tpms"]["type_hot"] = structure
    cfg["channels"]["hot"]["structure"] = structure
    cfg["channels"]["hot"]["mode"] = mode
    return cfg


def test_smooth_plate_fin_only_allows_bare_mode():
    compatibility = get_structure_compatibility("SmoothPlateFin")

    assert compatibility["channel_modes"] == ("bare",)
    assert compatibility["hydraulic_models"] == ()
    assert compatibility["htc_models"] == ()
    assert compatibility["wall_enhancement_models"] == ()
    assert compatibility["kinetic_models"] == ()


def test_plate_fin_allows_wang_wall_htc_and_phi_based_options():
    compatibility = get_structure_compatibility("PlateFin")

    assert compatibility["channel_modes"] == ("bare", "packed")
    assert "wang_wall_htc" in compatibility["htc_models"]
    assert "ergun_phi_fit" in compatibility["hydraulic_models"]
    assert "wall_from_phi" in compatibility["wall_enhancement_models"]
    assert compatibility["kinetic_models"] == SUPPORTED_KINETIC_MODELS


def test_primitive_disallows_phi_based_models():
    compatibility = get_structure_compatibility("Primitive")

    assert compatibility["hydraulic_models"] == ("ergun_psi_tpms",)
    assert compatibility["wall_enhancement_models"] == ("off",)
    assert "wang_wall_htc" not in compatibility["htc_models"]


def test_normalize_rejects_smooth_plate_fin_packed_mode():
    cfg = _make_hot_channel_cfg("SmoothPlateFin", mode="packed")

    with pytest.raises(ValueError, match="only supports modes"):
        normalize_config(cfg)


def test_normalize_rejects_primitive_phi_fit_hydraulic_model():
    cfg = _make_hot_channel_cfg("Primitive")
    cfg["channels"]["hot"]["packed"]["hydraulic_model"] = "ergun_phi_fit"

    with pytest.raises(ValueError, match="only supports hydraulic models"):
        normalize_config(cfg)


def test_normalize_rejects_gyroid_wang_wall_htc():
    cfg = _make_hot_channel_cfg("Gyroid")
    cfg["channels"]["hot"]["packed"]["htc_model"] = "wang_wall_htc"

    with pytest.raises(ValueError, match="only supports HTC models"):
        normalize_config(cfg)
