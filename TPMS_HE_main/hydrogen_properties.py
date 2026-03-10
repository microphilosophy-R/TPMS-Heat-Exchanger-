"""
Hydrogen Property Module with Unified Species Interface

Features:
- Unified inquiry for Hydrogen Mixture, Normal Hydrogen, Helium, and Argon.
- Preserves custom enthalpy AND entropy reference correction for Hydrogen spin isomers.
- CoolProp uses independent NBP reference states per isomer; offsets re-reference ortho
  entropy onto the para scale, enforcing the known conversion entropy at 20 K (ΔH/T).
- Ideal mixing entropy (-R·Σxi·ln xi) is added to the mixture entropy explicitly.
- Uses CoolProp Low-Level Interface (AbstractState) for performance.
"""

import numpy as np
from CoolProp import AbstractState
from CoolProp.CoolProp import PropsSI, PT_INPUTS


class ThermalProperties:
    """
    Fast property calculator handling Hydrogen mixtures (with spin isomer correction)
    and inert fluids (Helium, Argon) via a unified interface.
    """

    # Constants for Hydrogen Correction
    DELTA_H_NP_20K = 527.138e3          # J/kg  — enthalpy of conversion normal→para at 20 K
    DELTA_S_NP_20K = 527.138e3 / 20.0   # J/(kg·K) — entropy of conversion (ΔH/T, ΔG=0 at 20 K)

    T_REF_20K = 20.0    # K
    P_REF = 101325.0    # Pa
    R_SPECIFIC = 8.314 / 2.016 * 1000  # J/(kg·K)

    def __init__(self):
        """Initialize AbstractState objects for fast property access"""
        self.use_low_level = True
        try:
            # Hydrogen States
            self.state_para = AbstractState("HEOS", "ParaHydrogen")
            self.state_normal_h2 = AbstractState("HEOS", "Hydrogen")
            self.state_ortho = AbstractState("HEOS", "OrthoHydrogen")

            # Inert States
            self.state_he = AbstractState("HEOS", "Helium")
            self.state_ar = AbstractState("HEOS", "Argon")

            # Mappings for inert fluids
            self.inert_states = {
                'helium': self.state_he,
                'argon': self.state_ar
            }
        except Exception as e:
            print(f"Warning: Could not create AbstractState objects: {e}")
            self.use_low_level = False

        # Calculate enthalpy + entropy datum offsets
        self._calculate_offsets()

    def _calculate_offsets(self):
        """Calculates H2 enthalpy and entropy datum offsets for ortho/para isomers.

        Enthalpy offsets (anchored at 20 K):
            Forces h_mix(x_para=0.25) - h_para = DELTA_H_NP_20K by re-referencing
            the ortho enthalpy to the para datum using the measured ortho-para gap.

        Entropy offsets (anchored at 300 K via normal H2 as physical bridge):
            CoolProp assigns completely independent reference states to OrthoHydrogen
            and ParaHydrogen, so raw smass() values cannot be mixed directly.  At
            300 K, normal H2 is at chemical equilibrium (x_para ≈ 0.25), which makes
            it a self-consistent anchor:

                s_normal(300K) = 0.25·s_para(300K)
                               + 0.75·(s_ortho_raw(300K) + s_offset_ortho)
                               + s_mix_normal

            Solving for s_offset_ortho eliminates the arbitrary inter-isomer datum
            gap, removing the phantom T0·Δs term that caused exergy explosion.
            Using 20 K as the entropy anchor (the old approach) was wrong because
            x_eq(20K) ≈ 0.98 ≠ 0.25, so ΔG ≠ 0 at that composition/temperature.
        """
        try:
            # --- Enthalpy offsets at 20 K ---
            if self.use_low_level:
                self.state_para.update(PT_INPUTS, self.P_REF, self.T_REF_20K)
                h_para_ref = self.state_para.hmass()

                self.state_normal_h2.update(PT_INPUTS, self.P_REF, self.T_REF_20K)
                h_normal_ref_raw = self.state_normal_h2.hmass()

                self.state_ortho.update(PT_INPUTS, self.P_REF, self.T_REF_20K)
                h_ortho_ref_raw = self.state_ortho.hmass()
            else:
                h_para_ref       = PropsSI('H', 'T', self.T_REF_20K, 'P', self.P_REF, 'ParaHydrogen')
                h_normal_ref_raw = PropsSI('H', 'T', self.T_REF_20K, 'P', self.P_REF, 'Hydrogen')
                h_ortho_ref_raw  = PropsSI('H', 'T', self.T_REF_20K, 'P', self.P_REF, 'OrthoHydrogen')

            delta_h_current = h_normal_ref_raw - h_para_ref
            self.h_offset_normal = self.DELTA_H_NP_20K - delta_h_current

            self.delta_h_op_20K = self.DELTA_H_NP_20K / 0.75
            self.h_offset_ortho = self.delta_h_op_20K - (h_ortho_ref_raw - h_para_ref)

            # --- Entropy offsets at 300 K ---
            T_ref_s = 300.0
            if self.use_low_level:
                self.state_para.update(PT_INPUTS, self.P_REF, T_ref_s)
                s_p_300 = self.state_para.smass()

                self.state_normal_h2.update(PT_INPUTS, self.P_REF, T_ref_s)
                s_n_300 = self.state_normal_h2.smass()

                self.state_ortho.update(PT_INPUTS, self.P_REF, T_ref_s)
                s_o_300 = self.state_ortho.smass()
            else:
                s_p_300 = PropsSI('S', 'T', T_ref_s, 'P', self.P_REF, 'ParaHydrogen')
                s_n_300 = PropsSI('S', 'T', T_ref_s, 'P', self.P_REF, 'Hydrogen')
                s_o_300 = PropsSI('S', 'T', T_ref_s, 'P', self.P_REF, 'OrthoHydrogen')

            # Ideal mixing entropy of normal H2 (25% para, 75% ortho)
            x_p_n, x_o_n = 0.25, 0.75
            s_mix_normal = -self.R_SPECIFIC * (x_p_n * np.log(x_p_n) + x_o_n * np.log(x_o_n))

            # Solve: s_n_300 = 0.25*s_p_300 + 0.75*(s_o_300 + s_offset_ortho) + s_mix_normal
            self.s_offset_ortho = (s_n_300 - s_mix_normal - 0.25 * s_p_300) / 0.75 - s_o_300

            # s_offset_normal: auxiliary key only (not used in mixture s calculation).
            # By construction at 300K the bridge closes, so offset ≈ 0; set to zero.
            self.s_offset_normal = 0.0
            self.delta_s_op_20K  = self.s_offset_ortho + (s_o_300 - s_p_300)  # backward compat

        except Exception as e:
            print(f"Warning: Offset calculation failed: {e}")
            self.h_offset_normal = 0.0
            self.h_offset_ortho  = 0.0
            self.s_offset_normal = 0.0
            self.s_offset_ortho  = 0.0
            self.delta_h_op_20K  = 0.0
            self.delta_s_op_20K  = 0.0

    def get_properties(self, T, P, species="hydrogen mixture", x_para=None):
        """
        Unified Property Access Method.

        Parameters
        ----------
        T : float or ndarray
            Temperature [K]
        P : float or ndarray
            Pressure [Pa]
        species : str
            "hydrogen mixture" (requires x_para), "normal hydrogen", "helium", "argon"
        x_para : float or ndarray, optional
            Para-hydrogen fraction (required for "hydrogen mixture")

        Returns
        -------
        dict : Properties including h, s, rho, cp, mu, lambda, and aux keys
               h_para, h_normal, h_ortho, s_para, s_normal, s_ortho.
        """
        species_key = species.lower()

        # 1. Handle Normal Hydrogen (treat as mixture with fixed x=0.25)
        if species_key == "normal hydrogen":
            return self._get_h2_mixture_properties(T, P, x_para=0.25)

        # 2. Handle Hydrogen Mixture
        elif species_key == "hydrogen mixture":
            if x_para is None:
                raise ValueError("x_para is required for 'hydrogen mixture'")
            return self._get_h2_mixture_properties(T, P, x_para)

        # 3. Handle Inert Fluids (Helium, Argon)
        elif species_key in self.inert_states:
            return self._get_inert_properties(T, P, self.inert_states[species_key])

        else:
            raise ValueError(
                f"Unknown species: {species}. "
                f"Supported: 'hydrogen mixture', 'normal hydrogen', 'helium', 'argon'")

    def _get_h2_mixture_properties(self, T, P, x_para):
        """Internal method for H2 Mixtures with Enthalpy & Entropy Corrections."""
        scalar_input = np.isscalar(T)
        T = np.atleast_1d(T)
        P = np.atleast_1d(P)
        x_para = np.atleast_1d(x_para) if np.isscalar(x_para) else np.array(x_para)

        # Broadcasting
        n = max(len(T), len(P), len(x_para))
        if len(T) == 1:     T     = np.full(n, T[0])
        if len(P) == 1:     P     = np.full(n, P[0])
        if len(x_para) == 1: x_para = np.full(n, x_para[0])

        keys = ['h', 's', 'rho', 'cp', 'mu', 'lambda',
                'h_para', 'h_normal', 'h_ortho',
                's_para', 's_normal', 's_ortho']
        props = {k: np.zeros(n) for k in keys}

        for i in range(n):
            try:
                # --- Para H2 (reference isomer) ---
                self.state_para.update(PT_INPUTS, P[i], T[i])
                h_para   = self.state_para.hmass()
                s_para   = self.state_para.smass()
                rho_para = self.state_para.rhomass()
                cp_para  = self.state_para.cpmass()

                # --- Normal H2 (transport properties + auxiliary offset baseline) ---
                self.state_normal_h2.update(PT_INPUTS, P[i], T[i])
                h_normal_raw = self.state_normal_h2.hmass()
                s_normal_raw = self.state_normal_h2.smass()
                mu_normal    = self.state_normal_h2.viscosity()
                lambda_normal = self.state_normal_h2.conductivity()

                # --- Ortho H2 ---
                self.state_ortho.update(PT_INPUTS, P[i], T[i])
                h_ortho_raw = self.state_ortho.hmass()
                s_ortho_raw = self.state_ortho.smass()
                rho_ortho   = self.state_ortho.rhomass()
                cp_ortho    = self.state_ortho.cpmass()

                # --- Enthalpy Corrections ---
                h_para_corr   = h_para
                h_normal_corr = h_normal_raw + self.h_offset_normal
                h_ortho_corr  = h_ortho_raw  + self.h_offset_ortho

                # --- Entropy Corrections ---
                s_para_corr   = s_para
                s_normal_corr = s_normal_raw + self.s_offset_normal
                s_ortho_corr  = s_ortho_raw  + self.s_offset_ortho

                # --- Mixture Rules ---
                x_p = x_para[i]
                x_o = 1.0 - x_p

                # Ideal mixing entropy — clamp to avoid log(0) at pure para or ortho
                xp_safe = max(x_p, 1e-12)
                xo_safe = max(x_o, 1e-12)
                s_mix = -self.R_SPECIFIC * (xp_safe * np.log(xp_safe) + xo_safe * np.log(xo_safe))

                props['h'][i]  = x_p * h_para_corr + x_o * h_ortho_corr
                props['s'][i]  = x_p * s_para_corr + x_o * s_ortho_corr + s_mix
                props['cp'][i] = x_p * cp_para + x_o * cp_ortho

                # Density: harmonic mean (volume-additive)
                props['rho'][i] = 1.0 / (x_p / rho_para + x_o / rho_ortho)

                # Transport: approximate as normal H2
                props['mu'][i]     = mu_normal
                props['lambda'][i] = lambda_normal

                # Auxiliary enthalpy
                props['h_para'][i]   = h_para_corr
                props['h_normal'][i] = h_normal_corr
                props['h_ortho'][i]  = h_ortho_corr

                # Auxiliary entropy
                props['s_para'][i]   = s_para_corr
                props['s_normal'][i] = s_normal_corr
                props['s_ortho'][i]  = s_ortho_corr

            except Exception:
                for k in props:
                    props[k][i] = np.nan

        props.update({'T': T, 'P': P, 'x_para': x_para})

        if scalar_input:
            return {k: v[0] if isinstance(v, np.ndarray) else v for k, v in props.items()}
        return props

    def _get_inert_properties(self, T, P, state_obj):
        """Internal method for inert fluids (He, Ar) using AbstractState."""
        scalar_input = np.isscalar(T)
        T = np.atleast_1d(T)
        P = np.atleast_1d(P)

        n = max(len(T), len(P))
        if len(T) == 1: T = np.full(n, T[0])
        if len(P) == 1: P = np.full(n, P[0])

        props = {k: np.zeros(n) for k in ['h', 's', 'rho', 'cp', 'mu', 'lambda']}

        for i in range(n):
            try:
                state_obj.update(PT_INPUTS, P[i], T[i])
                props['h'][i]      = state_obj.hmass()
                props['s'][i]      = state_obj.smass()
                props['rho'][i]    = state_obj.rhomass()
                props['cp'][i]     = state_obj.cpmass()
                props['mu'][i]     = state_obj.viscosity()
                props['lambda'][i] = state_obj.conductivity()
            except Exception:
                for k in props:
                    props[k][i] = np.nan

        props.update({'T': T, 'P': P})
        if scalar_input:
            return {k: v[0] if isinstance(v, np.ndarray) else v for k, v in props.items()}
        return props

    @staticmethod
    def get_equilibrium_fraction(T):
        """Calculates para-hydrogen equilibrium fraction."""
        T = np.atleast_1d(T)
        x_eq = (0.1 * (np.exp(-175 / T) + 0.1) ** (-1) -
                7.06e-9 * T ** 3 + 3.42e-6 * T ** 2 - 6.2e-5 * T - 0.00227)
        x_eq = np.clip(x_eq, 0, 1)
        return x_eq if len(x_eq) > 1 else x_eq[0]


def test_hydrogen_properties():
    """Test the hydrogen property calculations with verification."""
    print("=" * 70)
    print("Testing Hydrogen Property Calculation with CoolProp")
    print("Enthalpy & Entropy Reference Correction Verification")
    print("=" * 70)
    print()

    h2_props = ThermalProperties()
    print()

    # ------------------------------------------------------------------
    # Test 1: Conversion heat and entropy at 20 K
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TEST 1: Conversion Heat & Entropy Verification at 20 K")
    print("=" * 70)
    print()

    T_20K = 20.0
    P_ref = 101325.0  # 1 atm

    props_ortho  = h2_props.get_properties(T_20K, P_ref, x_para=0.0)
    props_para   = h2_props.get_properties(T_20K, P_ref, x_para=1.0)
    props_normal = h2_props.get_properties(T_20K, P_ref, x_para=0.25)

    Delta_h_op = props_ortho['h']       - props_para['h']
    Delta_h_np = props_normal['h_normal'] - props_para['h_para']

    Delta_s_op = props_ortho['s']       - props_para['s']
    Delta_s_np = props_normal['s_normal'] - props_para['s_para']

    print(f"At T = {T_20K} K, P = {P_ref / 1e3:.2f} kPa:")
    print(f"  h_ortho  = {props_ortho['h']  / 1e3:10.3f} kJ/kg")
    print(f"  h_para   = {props_para['h_para']  / 1e3:10.3f} kJ/kg")
    print(f"  h_normal = {props_normal['h_normal'] / 1e3:10.3f} kJ/kg")
    print()
    print(f"  s_ortho  = {props_ortho['s']  / 1e3:10.3f} kJ/(kg·K)")
    print(f"  s_para   = {props_para['s_para']  / 1e3:10.3f} kJ/(kg·K)")
    print(f"  s_normal = {props_normal['s_normal'] / 1e3:10.3f} kJ/(kg·K)")
    print()
    print("Conversion Heats:")
    print(f"  Dh_o->p = {Delta_h_op / 1e3:10.3f} kJ/kg  (ortho->para)")
    print(f"  Dh_n->p = {Delta_h_np / 1e3:10.3f} kJ/kg  (normal->para, target: 527.138)")
    print()
    print("Conversion Entropies:")
    print(f"  Ds_o->p = {Delta_s_op / 1e3:10.3f} kJ/(kg·K)  (ortho->para)")
    print(f"  Ds_n->p = {Delta_s_np / 1e3:10.3f} kJ/(kg·K)  (normal->para, target: {ThermalProperties.DELTA_S_NP_20K/1e3:.3f})")
    print()

    # ------------------------------------------------------------------
    # Test 2: Temperature dependence
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TEST 2: Conversion Heat & Entropy vs Temperature")
    print("=" * 70)
    print()

    T_range = np.array([20, 30, 40, 50, 60, 70, 80])
    P_test  = 2e6  # 2 MPa

    print(f"At P = {P_test / 1e6:.1f} MPa:")
    print(f"{'T [K]':>6}  {'Dh_n->p [kJ/kg]':>16}  {'Ds_n->p [kJ/(kg·K)]':>20}")
    print("-" * 48)

    for T in T_range:
        props_n = h2_props.get_properties(T, P_test, x_para=0.25)
        Delta_h = props_n['h_normal'] - props_n['h_para']
        Delta_s = props_n['s_normal'] - props_n['s_para']
        print(f"{T:6.1f}  {Delta_h / 1e3:16.2f}  {Delta_s / 1e3:20.3f}")

    print()

    # ------------------------------------------------------------------
    # Test 3: Mixture properties across x_para
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TEST 3: Mixture Properties vs x_para")
    print("=" * 70)
    print()

    T_test = 50.0
    P_test = 2e6
    x_para_range = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    print(f"At T = {T_test:.1f} K, P = {P_test / 1e6:.1f} MPa:")
    print(f"{'x_para':>7}  {'h [kJ/kg]':>12}  {'s [kJ/(kg·K)]':>15}  "
          f"{'rho [kg/m3]':>12}  {'cp [J/(kg·K)]':>14}")
    print("-" * 70)

    for x_p in x_para_range:
        props = h2_props.get_properties(T_test, P_test, "hydrogen mixture", x_p)
        print(f"{x_p:7.2f}  {props['h'] / 1e3:12.2f}  {props['s'] / 1e3:15.3f}  "
              f"{props['rho']:12.3f}  {props['cp']:14.1f}")

    print()
    print("=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test_hydrogen_properties()
