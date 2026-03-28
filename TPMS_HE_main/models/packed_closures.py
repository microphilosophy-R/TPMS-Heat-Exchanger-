"""Packed-bed closure semantics, aliases, and lightweight strategy classes."""

from __future__ import annotations

from typing import Dict

import numpy as np


SUPPORTED_PACKED_MODES = ("lower", "nominal", "upper")
SUPPORTED_PHI_SOURCES = ("ch3_f_re_fit",)
SUPPORTED_HT_NOMINAL_RULES = ("geometric", "arithmetic", "similarity")

SUPPORTED_HYDRAULIC_MODELS = ("ergun_psi_tpms", "ergun_phi_fit")
HYDRAULIC_MODEL_ALIASES = {
    "psi_legacy": "ergun_psi_tpms",
    "phi_re_fit": "ergun_phi_fit",
}

SUPPORTED_PACKED_HEAT_TRANSFER_MODELS = (
    "martin_nilles",
    "dixon",
    "wang_wall_htc",
)
PACKED_HEAT_TRANSFER_MODEL_ALIASES = {
    "wang_experiment": "wang_wall_htc",
}

SUPPORTED_WALL_ENHANCEMENT_MODELS = ("off", "wall_from_phi")
WALL_ENHANCEMENT_MODEL_ALIASES = {
    "from_phi": "wall_from_phi",
}

SUPPORTED_KINETIC_MODELS = ("wilhelmsen_kw", "arrhenius_first_order")
KINETIC_MODEL_ALIASES = {
    "legacy_kw": "wilhelmsen_kw",
}

PACKED_CAPABLE_STRUCTURES = (
    "Gyroid",
    "Diamond",
    "Primitive",
    "Neovius",
    "FRD",
    "FKS",
    "PlateFin",
)
PHI_FIT_STRUCTURES = ("Gyroid", "Diamond", "PlateFin")
WANG_WALL_HTC_STRUCTURES = ("PlateFin",)


def structure_supports_packed(structure: str) -> bool:
    return str(structure).strip() in PACKED_CAPABLE_STRUCTURES


def get_allowed_channel_modes(structure: str):
    if structure_supports_packed(structure):
        return ("bare", "packed")
    return ("bare",)


def get_allowed_hydraulic_models(structure: str):
    if not structure_supports_packed(structure):
        return ()
    if str(structure).strip() in PHI_FIT_STRUCTURES:
        return SUPPORTED_HYDRAULIC_MODELS
    return ("ergun_psi_tpms",)


def get_allowed_packed_heat_transfer_models(structure: str):
    if not structure_supports_packed(structure):
        return ()
    if str(structure).strip() in WANG_WALL_HTC_STRUCTURES:
        return SUPPORTED_PACKED_HEAT_TRANSFER_MODELS
    return ("martin_nilles", "dixon")


def get_allowed_wall_enhancement_models(structure: str):
    if not structure_supports_packed(structure):
        return ()
    if str(structure).strip() in PHI_FIT_STRUCTURES:
        return SUPPORTED_WALL_ENHANCEMENT_MODELS
    return ("off",)


def get_allowed_kinetic_models(structure: str):
    if not structure_supports_packed(structure):
        return ()
    return SUPPORTED_KINETIC_MODELS


def get_structure_compatibility(structure: str):
    structure_name = str(structure).strip()
    notes = []
    if structure_name == "SmoothPlateFin":
        notes.append("only bare mode exposed; no dedicated packed smooth-duct closure")
    if structure_name in WANG_WALL_HTC_STRUCTURES:
        notes.append("wang_wall_htc available; wall HTC only, bed resistance still Martin-Nilles")
    if structure_name in PHI_FIT_STRUCTURES:
        notes.append("phi-based hydraulic / wall enhancement allowed")
    elif structure_supports_packed(structure_name):
        notes.append("phi-based options hidden; no Chapter 3 phi fit for this structure")

    return {
        "structure": structure_name,
        "channel_modes": get_allowed_channel_modes(structure_name),
        "hydraulic_models": get_allowed_hydraulic_models(structure_name),
        "htc_models": get_allowed_packed_heat_transfer_models(structure_name),
        "wall_enhancement_models": get_allowed_wall_enhancement_models(structure_name),
        "kinetic_models": get_allowed_kinetic_models(structure_name),
        "notes": tuple(notes),
    }


def _normalize_choice(name: str, value: str, aliases: Dict[str, str], supported):
    choice = str(value).strip().lower()
    canonical = aliases.get(choice, choice)
    if canonical not in supported:
        raise ValueError(f"{name} must be one of {supported}")
    return canonical


def normalize_hydraulic_model(value: str) -> str:
    return _normalize_choice(
        "hydraulic_model",
        value,
        HYDRAULIC_MODEL_ALIASES,
        SUPPORTED_HYDRAULIC_MODELS,
    )


def normalize_packed_heat_transfer_model(value: str) -> str:
    return _normalize_choice(
        "htc_model",
        value,
        PACKED_HEAT_TRANSFER_MODEL_ALIASES,
        SUPPORTED_PACKED_HEAT_TRANSFER_MODELS,
    )


def normalize_wall_enhancement_model(value: str) -> str:
    return _normalize_choice(
        "ht_enhancement_model",
        value,
        WALL_ENHANCEMENT_MODEL_ALIASES,
        SUPPORTED_WALL_ENHANCEMENT_MODELS,
    )


def normalize_kinetic_model(value: str) -> str:
    return _normalize_choice(
        "kinetic_model",
        value,
        KINETIC_MODEL_ALIASES,
        SUPPORTED_KINETIC_MODELS,
    )


class HydraulicClosure:
    name = ""

    def compute_friction(self, model, *, re_p, re_channel, tpms_type, phi_value):
        raise NotImplementedError


class ErgunPsiTPMSHydraulicClosure(HydraulicClosure):
    name = "ergun_psi_tpms"

    def compute_friction(self, model, *, re_p, re_channel, tpms_type, phi_value):
        f_base = model.friction_factor_ergun(re_p)
        psi = model.tpms_pressure_correction(tpms_type)
        correction_factor = psi
        return f_base * correction_factor, {
            "hydraulic_model": self.name,
            "hydraulic_semantics": "ergun_base_times_tpms_psi",
            "f_ergun_base": f_base,
            "psi_tpms": psi,
            "phi": float(max(phi_value, 1.0)),
            "correction_factor": correction_factor,
            "f_equiv": f_base * correction_factor,
        }


class ErgunPhiFitHydraulicClosure(HydraulicClosure):
    name = "ergun_phi_fit"

    def compute_friction(self, model, *, re_p, re_channel, tpms_type, phi_value):
        f_base = model.friction_factor_ergun(re_p)
        psi = model.tpms_pressure_correction(tpms_type)
        correction_factor = float(max(phi_value, 1.0))
        return f_base * correction_factor, {
            "hydraulic_model": self.name,
            "hydraulic_semantics": "ergun_base_times_ch3_phi_fit",
            "f_ergun_base": f_base,
            "psi_tpms": psi,
            "phi": float(max(phi_value, 1.0)),
            "correction_factor": correction_factor,
            "f_equiv": f_base * correction_factor,
        }


class WallEnhancementPolicy:
    name = ""

    def select_factor(self, model, uncertainty_mode, phi_value):
        raise NotImplementedError


class NoWallEnhancementPolicy(WallEnhancementPolicy):
    name = "off"

    def select_factor(self, model, uncertainty_mode, phi_value):
        lower, nominal, upper = model._ht_enhancement_bounds(phi_value)
        return lower, nominal, upper, 1.0


class WallFromPhiEnhancementPolicy(WallEnhancementPolicy):
    name = "wall_from_phi"

    def select_factor(self, model, uncertainty_mode, phi_value):
        lower, nominal, upper = model._ht_enhancement_bounds(phi_value)
        if uncertainty_mode == "lower":
            used = lower
        elif uncertainty_mode == "upper":
            used = upper
        else:
            used = nominal
        return lower, nominal, upper, used


class PackedHeatTransferClosure:
    name = ""

    def overall_htc(self, model, *, re_p, pr, k_f, uncertainty_mode, t, phi_value):
        raise NotImplementedError


class MartinNillesHeatTransferClosure(PackedHeatTransferClosure):
    name = "martin_nilles"

    def overall_htc(self, model, *, re_p, pr, k_f, uncertainty_mode, t, phi_value):
        return model._overall_htc_martin_like_impl(
            re_p,
            pr,
            k_f,
            uncertainty_mode=uncertainty_mode,
            t=t,
            phi_value=phi_value,
            wall_htc_source=self.name,
        )


class DixonHeatTransferClosure(PackedHeatTransferClosure):
    name = "dixon"

    def overall_htc(self, model, *, re_p, pr, k_f, uncertainty_mode, t, phi_value):
        return model._overall_htc_dixon_impl(
            re_p,
            pr,
            k_f,
            uncertainty_mode=uncertainty_mode,
            t=t,
            phi_value=phi_value,
        )


class WangWallHTCHeatTransferClosure(PackedHeatTransferClosure):
    name = "wang_wall_htc"

    def overall_htc(self, model, *, re_p, pr, k_f, uncertainty_mode, t, phi_value):
        return model._overall_htc_martin_like_impl(
            re_p,
            pr,
            k_f,
            uncertainty_mode=uncertainty_mode,
            t=t,
            phi_value=phi_value,
            wall_htc_source=self.name,
        )


class KineticsClosure:
    name = ""

    def rate(self, *, t, p, x, x_eq, rho, params):
        raise NotImplementedError


class WilhelmsenKWKineticsClosure(KineticsClosure):
    name = "wilhelmsen_kw"

    def rate(self, *, t, p, x, x_eq, rho, params):
        tc_h2 = 32.938
        pc_h2 = 1.284e6
        m_h2 = 2.016e-3

        c_h2 = rho / m_h2
        kw = 34.76 - 220.9 * (t / tc_h2) - 20.65 * (p / pc_h2)
        x_s = np.clip(x, 1e-9, 1.0 - 1e-9)
        x_eq_s = np.clip(x_eq, 1e-9, 1.0 - 1e-9)
        term = (1.0 - x_eq_s) / (1.0 - x_s)
        return (kw / c_h2) * np.log(term)


class ArrheniusFirstOrderKineticsClosure(KineticsClosure):
    name = "arrhenius_first_order"

    def rate(self, *, t, p, x, x_eq, rho, params):
        del p
        ea = float(params.get("Ea_J_per_mol", -336.45))
        a_k = float(params.get("a_m3s_per_mol", 2.2e-3))
        b_k = float(params.get("b_s_inv", -35.11e-3))
        r_u = 8.314
        m_h2 = 2.016e-3

        c_h2 = rho / m_h2
        x_eq_safe = max(float(x_eq), 1e-9)
        den = a_k * c_h2 + b_k
        if abs(den) < 1e-12:
            den = 1e-12 if den >= 0 else -1e-12
        # First-order relaxation toward the local equilibrium para-fraction.
        k_eff = np.exp(-ea / (r_u * max(t, 1e-9))) / den
        return k_eff * (x_eq_safe - float(x)) / x_eq_safe


_HYDRAULIC_CLOSURES = {
    closure.name: closure
    for closure in (
        ErgunPsiTPMSHydraulicClosure(),
        ErgunPhiFitHydraulicClosure(),
    )
}

_PACKED_HEAT_TRANSFER_CLOSURES = {
    closure.name: closure
    for closure in (
        MartinNillesHeatTransferClosure(),
        DixonHeatTransferClosure(),
        WangWallHTCHeatTransferClosure(),
    )
}

_WALL_ENHANCEMENT_POLICIES = {
    policy.name: policy
    for policy in (
        NoWallEnhancementPolicy(),
        WallFromPhiEnhancementPolicy(),
    )
}

_KINETICS_CLOSURES = {
    closure.name: closure
    for closure in (
        WilhelmsenKWKineticsClosure(),
        ArrheniusFirstOrderKineticsClosure(),
    )
}


def get_hydraulic_closure(name: str) -> HydraulicClosure:
    return _HYDRAULIC_CLOSURES[normalize_hydraulic_model(name)]


def get_packed_heat_transfer_closure(name: str) -> PackedHeatTransferClosure:
    return _PACKED_HEAT_TRANSFER_CLOSURES[normalize_packed_heat_transfer_model(name)]


def get_wall_enhancement_policy(name: str) -> WallEnhancementPolicy:
    return _WALL_ENHANCEMENT_POLICIES[normalize_wall_enhancement_model(name)]


def get_kinetics_closure(name: str) -> KineticsClosure:
    return _KINETICS_CLOSURES[normalize_kinetic_model(name)]
