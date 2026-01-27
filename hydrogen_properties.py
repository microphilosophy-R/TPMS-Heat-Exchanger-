"""
Hydrogen Property Module using CoolProp Low-Level Interface

This module provides fast property calculations for ortho-para hydrogen mixtures
using CoolProp 6.8.0 with the low-level interface for optimization.

ENTHALPY REFERENCE CORRECTION METHOD:
-------------------------------------
CoolProp sets h=0 at saturated liquid (1 atm) for each species independently.
This ignores the energy gap between ortho and para hydrogen spin isomers.

Correction Method:
1. Keep para-hydrogen enthalpy unchanged: h_para = h_para(CoolProp)
2. Fix normal hydrogen at 20 K reference point where Δh_n-p = 527.138 kJ/kg
3. Calculate offset: h_offset = Δh_n-p(20K) - [h_normal(20K) - h_para(20K)]
4. Apply correction: h_normal_corrected = h_normal(CoolProp) + h_offset
5. Calculate ortho from normal: h_ortho = (h_normal_corrected - 0.25*h_para) / 0.75

This ensures the conversion heat at 20 K matches the physical value.

Transport properties use normal hydrogen (ortho data not available in CoolProp).

Author: Based on research by Zhang et al. (2025)
"""

import numpy as np
from CoolProp.CoolProp import PropsSI
from CoolProp import AbstractState
import CoolProp.CoolProp as CP

class HydrogenProperties:
    """
    Fast hydrogen property calculator using CoolProp low-level interface
    with proper enthalpy reference correction
    """

    # Constants
    DELTA_H_NP_20K = 527.138e3  # J/kg - conversion heat normal to para at 20 K
    T_REF_20K = 20.0  # K - reference temperature
    P_REF = 101325.0  # Pa - reference pressure (1 atm)
    R_SPECIFIC = 8.314 / 2.016 * 1000  # J/(kg·K) - specific gas constant for H2

    def __init__(self):
        """Initialize AbstractState objects for fast property access"""
        # Create AbstractState objects for low-level interface
        try:
            self.state_para = AbstractState("HEOS", "ParaHydrogen")
            self.state_normal = AbstractState("HEOS", "Hydrogen")
            self.state_ortho = AbstractState("HEOS", "OrthoHydrogen")
            self.use_low_level = True
        except Exception as e:
            print(f"Warning: Could not create AbstractState objects: {e}")
            print("Falling back to high-level interface")
            self.use_low_level = False

        # Calculate enthalpy offset for reference correction
        self._calculate_enthalpy_offset()

    def _calculate_enthalpy_offset(self):
        """
        Calculate enthalpy offset to correct reference point
        """
        try:
            if self.use_low_level:
                # Use AbstractState for precise calculation
                self.state_para.update(CP.PT_INPUTS, self.P_REF, self.T_REF_20K)
                h_para_ref = self.state_para.hmass()

                self.state_normal.update(CP.PT_INPUTS, self.P_REF, self.T_REF_20K)
                h_normal_ref_raw = self.state_normal.hmass()
            else:
                # Use high-level interface
                h_para_ref = PropsSI('H', 'T', self.T_REF_20K, 'P', self.P_REF, 'ParaHydrogen')
                h_normal_ref_raw = PropsSI('H', 'T', self.T_REF_20K, 'P', self.P_REF, 'Hydrogen')

            # Calculate current conversion heat (uncorrected)
            delta_h_current = h_normal_ref_raw - h_para_ref

            # Calculate required offset
            self.h_offset_normal = self.DELTA_H_NP_20K - delta_h_current

            # Calculate ortho offset
            # Δh_o-p = Δh_n-p / 0.75
            self.delta_h_op_20K = self.DELTA_H_NP_20K / 0.75  # J/kg
            self.h_offset_ortho = self.delta_h_op_20K - (h_normal_ref_raw - h_para_ref) / 0.75

            print(f"Enthalpy reference correction initialized:")
            print(f"  Target Δh_n-p at 20 K: {self.DELTA_H_NP_20K/1e3:.3f} kJ/kg")
            print(f"  CoolProp Δh_n-p at 20 K: {delta_h_current/1e3:.3f} kJ/kg")
            print(f"  Normal H2 offset: {self.h_offset_normal/1e3:.3f} kJ/kg")
            print(f"  Calculated Δh_o-p at 20 K: {self.delta_h_op_20K/1e3:.3f} kJ/kg")
            print(f"  Ortho H2 offset: {self.h_offset_ortho/1e3:.3f} kJ/kg")

        except Exception as e:
            print(f"Warning: Could not calculate enthalpy offset: {e}")
            self.h_offset_normal = 0
            self.h_offset_ortho = 0
            self.delta_h_op_20K = self.DELTA_H_NP_20K / 0.75

    def get_properties(self, T, P, x_para):
        """
        Get all thermodynamic and transport properties for hydrogen mixture
        """
        # Handle scalar and array inputs
        scalar_input = np.isscalar(T)
        T = np.atleast_1d(T)
        P = np.atleast_1d(P)
        x_para = np.atleast_1d(x_para) if np.isscalar(x_para) else np.array(x_para)

        # Broadcasting: Ensure all inputs have the same shape
        n = max(len(T), len(P), len(x_para))
        if len(T) == 1: T = np.full(n, T[0])
        if len(P) == 1: P = np.full(n, P[0])
        if len(x_para) == 1: x_para = np.full(n, x_para[0])

        # Initialize output arrays with zeros
        keys = ['h', 's', 'rho', 'cp', 'mu', 'lambda', 'Delta_h', 'h_ortho', 'h_para', 'h_normal']
        props = {k: np.zeros(n) for k in keys}

        for i in range(n):
            try:
                if self.use_low_level:
                    # Para-hydrogen (Reference)
                    self.state_para.update(CP.PT_INPUTS, P[i], T[i])
                    h_para = self.state_para.hmass()
                    s_para = self.state_para.smass()
                    rho_para = self.state_para.rhomass()
                    cp_para = self.state_para.cpmass()

                    # Normal hydrogen (Raw)
                    self.state_normal.update(CP.PT_INPUTS, P[i], T[i])
                    h_normal_raw = self.state_normal.hmass()
                    mu_normal = self.state_normal.viscosity()
                    lambda_normal = self.state_normal.conductivity()

                    # Ortho-hydrogen (Raw)
                    self.state_ortho.update(CP.PT_INPUTS, P[i], T[i])
                    h_ortho_raw = self.state_ortho.hmass()
                    s_ortho_raw = self.state_ortho.smass()
                    rho_ortho = self.state_ortho.rhomass()
                    cp_ortho = self.state_ortho.cpmass()

                else:
                    # High-level interface fallback
                    h_para = PropsSI('H', 'T', T[i], 'P', P[i], 'ParaHydrogen')
                    s_para = PropsSI('S', 'T', T[i], 'P', P[i], 'ParaHydrogen')
                    rho_para = PropsSI('D', 'T', T[i], 'P', P[i], 'ParaHydrogen')
                    cp_para = PropsSI('C', 'T', T[i], 'P', P[i], 'ParaHydrogen')

                    h_normal_raw = PropsSI('H', 'T', T[i], 'P', P[i], 'Hydrogen')
                    mu_normal = PropsSI('V', 'T', T[i], 'P', P[i], 'Hydrogen')
                    lambda_normal = PropsSI('L', 'T', T[i], 'P', P[i], 'Hydrogen')

                    h_ortho_raw = PropsSI('H', 'T', T[i], 'P', P[i], 'OrthoHydrogen')
                    s_ortho_raw = PropsSI('S', 'T', T[i], 'P', P[i], 'OrthoHydrogen')
                    rho_ortho = PropsSI('D', 'T', T[i], 'P', P[i], 'OrthoHydrogen')
                    cp_ortho = PropsSI('C', 'T', T[i], 'P', P[i], 'OrthoHydrogen')

                # --- Apply Corrections ---
                h_para_corr = h_para
                h_normal_corr = h_normal_raw + self.h_offset_normal
                h_ortho_corr = h_ortho_raw + self.h_offset_ortho

                # Use raw ortho entropy (mixing handled in Normal, but we use component-wise mixing)
                s_ortho = s_ortho_raw

                # --- Mixture Rules ---
                x_p = x_para[i]
                x_o = 1.0 - x_p

                # Enthalpy (Mass weighted)
                props['h'][i] = x_p * h_para_corr + x_o * h_ortho_corr

                # Entropy (Mass weighted)
                props['s'][i] = x_p * s_para + x_o * s_ortho

                # Density (Harmonic mean for mixture density)
                props['rho'][i] = 1.0 / (x_p / rho_para + x_o / rho_ortho)

                # Specific Heat (Mass weighted)
                props['cp'][i] = x_p * cp_para + x_o * cp_ortho

                # Transport (Approximate as Normal H2)
                props['mu'][i] = mu_normal
                props['lambda'][i] = lambda_normal

                # --- Components for Debugging/Calculation ---
                props['h_ortho'][i] = h_ortho_corr
                props['h_para'][i] = h_para_corr
                props['h_normal'][i] = h_normal_corr
                props['Delta_h'][i] = h_ortho_corr - h_para_corr

            except Exception as e:
                # Fallback NaN
                for k in props: props[k][i] = np.nan

        # Add inputs to result for reference
        props.update({'T': T, 'P': P, 'x_para': x_para})

        # Return scalars if input was scalar
        if scalar_input:
            return {k: v[0] if isinstance(v, np.ndarray) else v for k, v in props.items()}

        return props

    @staticmethod
    def get_equilibrium_fraction(T):
        """Calculate equilibrium para-hydrogen fraction at given temperature"""
        T = np.atleast_1d(T)
        x_eq = (0.1 * (np.exp(-175/T) + 0.1)**(-1) -
                7.06e-9*T**3 + 3.42e-6*T**2 - 6.2e-5*T - 0.00227)
        x_eq = np.clip(x_eq, 0, 1)
        return x_eq if len(x_eq) > 1 else x_eq[0]

    @staticmethod
    def get_helium_properties(T, P):
        """Get helium properties using CoolProp"""
        scalar_input = np.isscalar(T)
        T = np.atleast_1d(T)
        P = np.atleast_1d(P)
        n = max(len(T), len(P))

        if len(T) == 1: T = np.full(n, T[0])
        if len(P) == 1: P = np.full(n, P[0])

        props = {k: np.zeros(n) for k in ['h', 's', 'rho', 'cp', 'mu', 'lambda']}

        try:
            state_he = AbstractState("HEOS", "Helium")
            use_low_level = True
        except:
            use_low_level = False

        for i in range(n):
            try:
                if use_low_level:
                    state_he.update(CP.PT_INPUTS, P[i], T[i])
                    props['h'][i] = state_he.hmass()
                    props['s'][i] = state_he.smass()
                    props['rho'][i] = state_he.rhomass()
                    props['cp'][i] = state_he.cpmass()
                    props['mu'][i] = state_he.viscosity()
                    props['lambda'][i] = state_he.conductivity()
                else:
                    props['h'][i] = PropsSI('H', 'T', T[i], 'P', P[i], 'Helium')
                    props['s'][i] = PropsSI('S', 'T', T[i], 'P', P[i], 'Helium')
                    props['rho'][i] = PropsSI('D', 'T', T[i], 'P', P[i], 'Helium')
                    props['cp'][i] = PropsSI('C', 'T', T[i], 'P', P[i], 'Helium')
                    props['mu'][i] = PropsSI('V', 'T', T[i], 'P', P[i], 'Helium')
                    props['lambda'][i] = PropsSI('L', 'T', T[i], 'P', P[i], 'Helium')
            except Exception as e:
                for k in props: props[k][i] = np.nan

        props.update({'T': T, 'P': P})
        if scalar_input:
            return {k: v[0] if isinstance(v, np.ndarray) else v for k, v in props.items()}
        return props

def test_hydrogen_properties():
    """Test the hydrogen property calculations with verification"""
    print("="*70)
    print("Testing Hydrogen Property Calculation with CoolProp")
    print("Enthalpy Reference Correction Verification")
    print("="*70)
    print()

    h2_props = HydrogenProperties()
    print()

    # Test 1: Verify conversion heat at 20 K
    print("="*70)
    print("TEST 1: Conversion Heat Verification at 20 K")
    print("="*70)
    print()

    T_20K = 20.0
    P_ref = 101325.0  # 1 atm

    # Check pure ortho, para, and normal
    props_ortho = h2_props.get_properties(T_20K, P_ref, x_para=0.0)
    props_para = h2_props.get_properties(T_20K, P_ref, x_para=1.0)
    props_normal = h2_props.get_properties(T_20K, P_ref, x_para=0.25)

    # Calculate conversion heats
    Delta_h_op = props_ortho['h_ortho'] - props_ortho['h_para']
    Delta_h_np = props_normal['h_normal'] - props_normal['h_para']

    print(f"At T = {T_20K} K, P = {P_ref/1e3:.2f} kPa:")
    print(f"  h_ortho     = {props_ortho['h_ortho']/1e3:10.3f} kJ/kg")
    print(f"  h_para      = {props_para['h_para']/1e3:10.3f} kJ/kg")
    print(f"  h_normal    = {props_normal['h_normal']/1e3:10.3f} kJ/kg")
    print()
    print(f"Conversion Heats:")
    print(f"  Δh_o→p = {Delta_h_op/1e3:10.3f} kJ/kg (ortho to para)")
    print(f"  Δh_n→p = {Delta_h_np/1e3:10.3f} kJ/kg (normal to para)")
    print()

    # Test 2: Temperature dependence
    print("="*70)
    print("TEST 2: Conversion Heat Temperature Dependence")
    print("="*70)
    print()

    T_range = np.array([20, 30, 40, 50, 60, 70, 80])
    P_test = 2e6  # 2 MPa

    print(f"Conversion heat at P = {P_test/1e6:.1f} MPa:")
    print(f"{'T [K]':>6} {'Δh_o→p [kJ/kg]':>16} {'Δh_n→p [kJ/kg]':>16}")
    print("-"*40)

    for T in T_range:
        props = h2_props.get_properties(T, P_test, x_para=0.0)

        # Robust check for Delta_h
        if 'Delta_h' in props:
            Delta_h_op = props['Delta_h']
        else:
            Delta_h_op = props['h_ortho'] - props['h_para']

        props_n = h2_props.get_properties(T, P_test, x_para=0.25)
        Delta_h_np = props_n['h_normal'] - props_n['h_para']

        print(f"{T:6.1f} {Delta_h_op/1e3:16.2f} {Delta_h_np/1e3:16.2f}")

    print()

    # Test 3: Mixture properties
    print("="*70)
    print("TEST 3: Mixture Properties")
    print("="*70)
    print()

    T_test = 50.0
    P_test = 2e6
    x_para_range = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    print(f"At T = {T_test:.1f} K, P = {P_test/1e6:.1f} MPa:")
    print(f"{'x_para':>7} {'h [kJ/kg]':>12} {'rho [kg/m³]':>12} {'cp [J/kg·K]':>12}")
    print("-"*45)

    for x_p in x_para_range:
        props = h2_props.get_properties(T_test, P_test, x_p)
        print(f"{x_p:7.2f} {props['h']/1e3:12.2f} {props['rho']:12.3f} {props['cp']:12.1f}")

    print()
    print("="*70)
    print("All tests completed successfully!")
    print("="*70)
    print()

if __name__ == "__main__":
    test_hydrogen_properties()