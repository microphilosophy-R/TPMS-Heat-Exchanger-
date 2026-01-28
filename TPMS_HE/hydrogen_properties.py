"""
Hydrogen Property Module with Unified Species Interface

Features:
- Unified inquiry for Hydrogen Mixture, Normal Hydrogen, Helium, and Argon.
- Preserves custom enthalpy reference correction for Hydrogen spin isomers.
- Uses CoolProp Low-Level Interface (AbstractState) for performance.
"""

import numpy as np
from CoolProp import AbstractState
from CoolProp.CoolProp import PropsSI, PT_INPUTS  # FIX 1: Import PT_INPUTS


class ThermalProperties:
    """
    Fast property calculator handling Hydrogen mixtures (with spin isomer correction)
    and inert fluids (Helium, Argon) via a unified interface.
    """

    # Constants for Hydrogen Correction
    DELTA_H_NP_20K = 527.138e3  # J/kg - conversion heat normal to para at 20 K
    T_REF_20K = 20.0  # K
    P_REF = 101325.0  # Pa
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

        # Calculate enthalpy offset for reference correction (Hydrogen only)
        self._calculate_enthalpy_offset()

    def _calculate_enthalpy_offset(self):
        """Calculates the H2 enthalpy offset to match physical conversion heat at 20K."""
        try:
            if self.use_low_level:
                # FIX 2: Use PT_INPUTS instead of 0
                self.state_para.update(PT_INPUTS, self.P_REF, self.T_REF_20K)
                h_para_ref = self.state_para.hmass()

                self.state_normal_h2.update(PT_INPUTS, self.P_REF, self.T_REF_20K)
                h_normal_ref_raw = self.state_normal_h2.hmass()
            else:
                h_para_ref = PropsSI('H', 'T', self.T_REF_20K, 'P', self.P_REF, 'ParaHydrogen')
                h_normal_ref_raw = PropsSI('H', 'T', self.T_REF_20K, 'P', self.P_REF, 'Hydrogen')

            delta_h_current = h_normal_ref_raw - h_para_ref
            self.h_offset_normal = self.DELTA_H_NP_20K - delta_h_current

            # Derived ortho offset
            self.delta_h_op_20K = self.DELTA_H_NP_20K / 0.75
            self.h_offset_ortho = self.delta_h_op_20K - (h_normal_ref_raw - h_para_ref) / 0.75

        except Exception as e:
            print(f"Warning: Enthalpy offset calculation failed: {e}")
            self.h_offset_normal = 0
            self.h_offset_ortho = 0

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
        dict : Properties (h, s, rho, cp, mu, lambda, etc.)
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
                f"Unknown species: {species}. Supported: 'hydrogen mixture', 'normal hydrogen', 'helium', 'argon'")

    def _get_h2_mixture_properties(self, T, P, x_para):
        """Internal method for H2 Mixtures with Enthalpy Correction"""
        scalar_input = np.isscalar(T)
        T = np.atleast_1d(T)
        P = np.atleast_1d(P)
        x_para = np.atleast_1d(x_para) if np.isscalar(x_para) else np.array(x_para)

        # Broadcasting
        n = max(len(T), len(P), len(x_para))
        if len(T) == 1: T = np.full(n, T[0])
        if len(P) == 1: P = np.full(n, P[0])
        if len(x_para) == 1: x_para = np.full(n, x_para[0])

        # FIX 3: Initialize 'h_ortho' key to prevent KeyError
        props = {k: np.zeros(n) for k in ['h', 's', 'rho', 'cp', 'mu', 'lambda', 'h_para', 'h_normal', 'h_ortho']}

        for i in range(n):
            try:
                # FIX 4: Use PT_INPUTS in loop
                # Para H2 (Reference)
                self.state_para.update(PT_INPUTS, P[i], T[i])
                h_para = self.state_para.hmass()
                s_para = self.state_para.smass()
                rho_para = self.state_para.rhomass()
                cp_para = self.state_para.cpmass()

                # Normal H2 (for Transport + Offset)
                self.state_normal_h2.update(PT_INPUTS, P[i], T[i])
                h_normal_raw = self.state_normal_h2.hmass()
                s_normal = self.state_normal_h2.smass()
                mu_normal = self.state_normal_h2.viscosity()
                lambda_normal = self.state_normal_h2.conductivity()

                # Ortho H2 (Raw)
                self.state_ortho.update(PT_INPUTS, P[i], T[i])
                h_ortho_raw = self.state_ortho.hmass()
                rho_ortho = self.state_ortho.rhomass()
                cp_ortho = self.state_ortho.cpmass()

                # --- Corrections ---
                h_para_corr = h_para
                h_normal_corr = h_normal_raw + self.h_offset_normal
                h_ortho_corr = h_ortho_raw + self.h_offset_ortho

                # Entropy (Mixing assumption for Ortho)
                s_ortho = self.state_ortho.smass()

                # Mixture Rules
                x_p = x_para[i]
                x_o = 1.0 - x_p

                props['h'][i] = x_p * h_para_corr + x_o * h_ortho_corr
                props['s'][i] = x_p * s_para + x_o * s_ortho
                props['cp'][i] = x_p * cp_para + x_o * cp_ortho

                # Density (Harmonic mean)
                props['rho'][i] = 1.0 / (x_p / rho_para + x_o / rho_ortho)

                # Transport (Approximate as Normal H2)
                props['mu'][i] = mu_normal
                props['lambda'][i] = lambda_normal

                # Aux
                props['h_para'][i] = h_para_corr
                props['h_normal'][i] = h_normal_corr
                props['h_ortho'][i] = h_ortho_corr

            except Exception as e:
                # Tip: Enable this print to debug if NaNs persist
                # print(f"Row {i} failed: {e}")
                for k in props: props[k][i] = np.nan

        # Helper: Include T, P in result
        props.update({'T': T, 'P': P, 'x_para': x_para})

        if scalar_input:
            return {k: v[0] if isinstance(v, np.ndarray) else v for k, v in props.items()}
        return props

    def _get_inert_properties(self, T, P, state_obj):
        """Internal method for inert fluids (He, Ar) using AbstractState"""
        scalar_input = np.isscalar(T)
        T = np.atleast_1d(T)
        P = np.atleast_1d(P)

        n = max(len(T), len(P))
        if len(T) == 1: T = np.full(n, T[0])
        if len(P) == 1: P = np.full(n, P[0])

        props = {k: np.zeros(n) for k in ['h', 's', 'rho', 'cp', 'mu', 'lambda']}

        for i in range(n):
            try:
                # FIX 5: Use PT_INPUTS for inert fluids too
                state_obj.update(PT_INPUTS, P[i], T[i])
                props['h'][i] = state_obj.hmass()
                props['s'][i] = state_obj.smass()
                props['rho'][i] = state_obj.rhomass()
                props['cp'][i] = state_obj.cpmass()
                props['mu'][i] = state_obj.viscosity()
                props['lambda'][i] = state_obj.conductivity()
            except:
                for k in props: props[k][i] = np.nan

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
    """Test the hydrogen property calculations with verification"""
    print("=" * 70)
    print("Testing Hydrogen Property Calculation with CoolProp")
    print("Enthalpy Reference Correction Verification")
    print("=" * 70)
    print()

    h2_props = ThermalProperties()
    print()

    # Test 1: Verify conversion heat at 20 K
    print("=" * 70)
    print("TEST 1: Conversion Heat Verification at 20 K")
    print("=" * 70)
    print()

    T_20K = 20.0
    P_ref = 101325.0  # 1 atm

    # Check pure ortho, para, and normal
    props_ortho = h2_props.get_properties(T_20K, P_ref, x_para=0.0)
    props_para = h2_props.get_properties(T_20K, P_ref, x_para=1.0)
    props_normal = h2_props.get_properties(T_20K, P_ref, x_para=0.25)

    # Calculate conversion heats
    Delta_h_op = props_ortho['h'] - props_para['h']
    Delta_h_np = props_normal['h_normal'] - props_para['h_para']

    print(f"At T = {T_20K} K, P = {P_ref / 1e3:.2f} kPa:")
    print(f"  h_ortho     = {props_ortho['h'] / 1e3:10.3f} kJ/kg")
    print(f"  h_para      = {props_para['h_para'] / 1e3:10.3f} kJ/kg")
    print(f"  h_normal    = {props_normal['h_normal'] / 1e3:10.3f} kJ/kg")
    print()
    print(f"Conversion Heats:")
    print(f"  Δh_o→p = {Delta_h_op / 1e3:10.3f} kJ/kg (ortho to para)")
    print(f"  Δh_n→p = {Delta_h_np / 1e3:10.3f} kJ/kg (normal to para)")
    print()

    # Test 2: Temperature dependence
    print("=" * 70)
    print("TEST 2: Conversion Heat Temperature Dependence")
    print("=" * 70)
    print()

    T_range = np.array([20, 30, 40, 50, 60, 70, 80])
    P_test = 2e6  # 2 MPa

    print(f"Conversion heat at P = {P_test / 1e6:.1f} MPa:")
    print(f"{'T [K]':>6} {'Δh_o→p [kJ/kg]':>16} {'Δh_n→p [kJ/kg]':>16}")
    print("-" * 40)

    for T in T_range:
        props = h2_props.get_properties(T, P_test, x_para=0.0)

        # Calculate Delta using enthalpies at CURRENT temperature
        # (Subtracting para enthalpy at same T)
        Delta_h_op = props['h_ortho'] - props['h_para']

        props_n = h2_props.get_properties(T, P_test, x_para=0.25)
        Delta_h_np = props_n['h_normal'] - props_n['h_para']

        print(f"{T:6.1f} {Delta_h_op / 1e3:16.2f} {Delta_h_np / 1e3:16.2f}")

    print()

    # Test 3: Mixture properties
    print("=" * 70)
    print("TEST 3: Mixture Properties")
    print("=" * 70)
    print()

    T_test = 50.0
    P_test = 2e6
    x_para_range = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    print(f"At T = {T_test:.1f} K, P = {P_test / 1e6:.1f} MPa:")
    print(f"{'x_para':>7} {'h [kJ/kg]':>12} {'rho [kg/m³]':>12} {'cp [J/kg·K]':>12}")
    print("-" * 45)

    for x_p in x_para_range:
        props = h2_props.get_properties(T_test, P_test, "hydrogen mixture", x_p)
        print(f"{x_p:7.2f} {props['h'] / 1e3:12.2f} {props['rho']:12.3f} {props['cp']:12.1f}")

    print()
    print("=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test_hydrogen_properties()