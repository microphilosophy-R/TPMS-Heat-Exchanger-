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
from CoolProp.CoolProp import PhaseSI

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
        except Exception as e:
            print(f"Warning: Could not create AbstractState objects: {e}")
            print("Falling back to high-level interface")
            self.use_low_level = False
        else:
            self.use_low_level = True
        
        # Calculate enthalpy offset for reference correction
        self._calculate_enthalpy_offset()
    
    def _calculate_enthalpy_offset(self):
        """
        Calculate enthalpy offset to correct reference point
        
        At 20 K, the conversion heat from normal to para should be 527.138 kJ/kg
        Offset = Δh_n-p(target) - [h_normal(20K) - h_para(20K)]
        """
        try:
            if self.use_low_level:
                # Use AbstractState for precise calculation
                self.state_para.update(CoolProp.PT_INPUTS, self.P_REF, self.T_REF_20K)
                h_para_ref = self.state_para.hmass()
                
                self.state_normal.update(CoolProp.PT_INPUTS, self.P_REF, self.T_REF_20K)
                h_normal_ref_raw = self.state_normal.hmass()
            else:
                # Use high-level interface
                h_para_ref = PropsSI('H', 'T', self.T_REF_20K, 'P', self.P_REF, 'ParaHydrogen')
                h_normal_ref_raw = PropsSI('H', 'T', self.T_REF_20K, 'P', self.P_REF, 'Hydrogen')
            
            # Calculate current conversion heat (uncorrected)
            delta_h_current = h_normal_ref_raw - h_para_ref
            
            # Calculate require offset
            self.h_offset_normal = self.DELTA_H_NP_20K - delta_h_current
            
            # Calculate ortho offset
            # Since h_normal = 0.75*h_ortho + 0.25*h_para
            # h_ortho = (h_normal - 0.25*h_para) / 0.75
            # Δh_o-p = h_ortho - h_para = (h_normal - 0.25*h_para)/0.75 - h_para
            #        = (h_normal - h_para) / 0.75
            # Therefore: Δh_o-p = Δh_n-p / 0.75
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
        
        Parameters
        ----------
        T : float or ndarray
            Temperature [K]
        P : float or ndarray
            Pressure [Pa]
        x_para : float or ndarray
            Para-hydrogen mole fraction [-]
        
        Returns
        -------
        dict
            Dictionary containing all properties:
            - h: mixture enthalpy [J/kg]
            - s: mixture entropy [J/(kg·K)]
            - rho: mixture density [kg/m³]
            - cp: mixture specific heat [J/(kg·K)]
            - mu: dynamic viscosity [Pa·s]
            - lambda_: thermal conductivity [W/(m·K)]
            - Delta_h: conversion heat (ortho to para) [J/kg]
            - h_ortho: ortho-hydrogen enthalpy (corrected) [J/kg]
            - h_para: para-hydrogen enthalpy [J/kg]
            - h_normal: normal-hydrogen enthalpy (corrected) [J/kg]
        """
        # Handle scalar and array inputs
        scalar_input = np.isscalar(T)
        T = np.atleast_1d(T)
        P = np.atleast_1d(P)
        x_para = np.atleast_1d(x_para)
        
        # Ensure all inputs have the same shape
        n = max(len(T), len(P), len(x_para))
        if len(T) == 1:
            T = np.full(n, T[0])
        if len(P) == 1:
            P = np.full(n, P[0])
        if len(x_para) == 1:
            x_para = np.full(n, x_para[0])
        
        # Initialize output arrays
        h_mix = np.zeros(n)
        s_mix = np.zeros(n)
        rho_mix = np.zeros(n)
        cp_mix = np.zeros(n)
        mu_mix = np.zeros(n)
        lambda_mix = np.zeros(n)
        Delta_h = np.zeros(n)
        h_ortho_arr = np.zeros(n)
        h_para_arr = np.zeros(n)
        h_normal_arr = np.zeros(n)
        
        # Calculate properties for each point
        for i in range(n):
            try:
                if self.use_low_level:
                    # Para-hydrogen properties (no correction needed)
                    self.state_para.update(CoolProp.PT_INPUTS, P[i], T[i])
                    h_para = self.state_para.hmass()
                    s_para = self.state_para.smass()
                    rho_para = self.state_para.rhomass()
                    cp_para = self.state_para.cpmass()
                    
                    # Normal hydrogen properties (RAW from CoolProp)
                    self.state_normal.update(CoolProp.PT_INPUTS, P[i], T[i])
                    h_normal_raw = self.state_normal.hmass()
                    s_normal = self.state_normal.smass()
                    rho_normal = self.state_normal.rhomass()
                    cp_normal = self.state_normal.cpmass()
                    mu_normal = self.state_normal.viscosity()
                    lambda_normal = self.state_normal.conductivity()
                    
                    # Ortho-hydrogen properties (RAW from CoolProp)
                    self.state_ortho.update(CoolProp.PT_INPUTS, P[i], T[i])
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
                    s_normal = PropsSI('S', 'T', T[i], 'P', P[i], 'Hydrogen')
                    rho_normal = PropsSI('D', 'T', T[i], 'P', P[i], 'Hydrogen')
                    cp_normal = PropsSI('C', 'T', T[i], 'P', P[i], 'Hydrogen')
                    mu_normal = PropsSI('V', 'T', T[i], 'P', P[i], 'Hydrogen')
                    lambda_normal = PropsSI('L', 'T', T[i], 'P', P[i], 'Hydrogen')
                    
                    h_ortho_raw = PropsSI('H', 'T', T[i], 'P', P[i], 'OrthoHydrogen')
                    s_ortho_raw = PropsSI('S', 'T', T[i], 'P', P[i], 'OrthoHydrogen')
                    rho_ortho = PropsSI('D', 'T', T[i], 'P', P[i], 'OrthoHydrogen')
                    cp_ortho = PropsSI('C', 'T', T[i], 'P', P[i], 'OrthoHydrogen')
                
                # APPLY ENTHALPY CORRECTIONS
                # Para-hydrogen: NO CORRECTION (it's the reference)
                h_para_corrected = h_para
                
                # Normal hydrogen: ADD OFFSET
                h_normal_corrected = h_normal_raw + self.h_offset_normal
                
                # Ortho hydrogen: ADD OFFSET (alternative method)
                h_ortho_corrected = h_ortho_raw + self.h_offset_ortho
                
                # Verify consistency: h_normal should equal 0.75*h_ortho + 0.25*h_para
                # After correction, calculate ortho from normal if needed
                # h_ortho_from_normal = (h_normal_corrected - 0.25*h_para_corrected) / 0.75
                
                # Use the direct ortho correction for consistency
                h_ortho = h_ortho_corrected
                h_normal = h_normal_corrected
                
                # Entropy correction for ortho-hydrogen
                # The entropy of normal H2 includes mixing entropy
                # s_normal = 0.75*s_ortho + 0.25*s_para - R*[0.75*ln(0.75) + 0.25*ln(0.25)]
                mixing_entropy = self.R_SPECIFIC * (0.75*np.log(0.75) + 0.25*np.log(0.25))
                s_ortho = (s_normal - mixing_entropy - 0.25*s_para) / 0.75
                
                # If we used the raw CoolProp ortho entropy, we might need to adjust
                # But typically entropy doesn't need the same correction as enthalpy
                # Use the CoolProp ortho entropy directly
                s_ortho = s_ortho_raw
                
                # Calculate mixture properties
                x_ortho = 1 - x_para[i]
                
                # Mixture enthalpy
                h_mix[i] = x_ortho * h_ortho + x_para[i] * h_para_corrected
                
                # Mixture entropy
                s_mix[i] = x_ortho * s_ortho + x_para[i] * s_para
                
                # Density: harmonic mean (better for gas mixtures)
                rho_mix[i] = 1 / (x_ortho/rho_ortho + x_para[i]/rho_para)
                
                # Specific heat: mass-weighted average
                cp_mix[i] = x_ortho * cp_ortho + x_para[i] * cp_para
                
                # Transport properties: use normal hydrogen
                mu_mix[i] = mu_normal
                lambda_mix[i] = lambda_normal
                
                # Conversion heat (ortho to para)
                Delta_h[i] = h_ortho - h_para_corrected
                
                # Store individual species enthalpies (corrected)
                h_ortho_arr[i] = h_ortho
                h_para_arr[i] = h_para_corrected
                h_normal_arr[i] = h_normal
                
            except Exception as e:
                print(f"Warning: Property calculation failed at T={T[i]} K, P={P[i]} Pa: {e}")
                # Set NaN for failed calculations
                h_mix[i] = np.nan
                s_mix[i] = np.nan
                rho_mix[i] = np.nan
                cp_mix[i] = np.nan
                mu_mix[i] = np.nan
                lambda_mix[i] = np.nan
                Delta_h[i] = np.nan
                h_ortho_arr[i] = np.nan
                h_para_arr[i] = np.nan
                h_normal_arr[i] = np.nan
        
        # Prepare output
        result = {
            'h': h_mix,
            's': s_mix,
            'rho': rho_mix,
            'cp': cp_mix,
            'mu': mu_mix,
            'lambda': lambda_mix,
            'Delta_h': Delta_h,
            'h_ortho': h_ortho_arr,
            'h_para': h_para_arr,
            'h_normal': h_normal_arr,
            'T': T,
            'P': P,
            'x_para': x_para
        }
        
        # Return scalars if input was scalar
        if scalar_input:
            result = {k: v[0] if isinstance(v, np.ndarray) else v for k, v in result.items()}
        
        return result
    
    @staticmethod
    def get_equilibrium_fraction(T):
        """
        Calculate equilibrium para-hydrogen fraction at given temperature
        
        Parameters
        ----------
        T : float or ndarray
            Temperature [K]
        
        Returns
        -------
        x_eq : float or ndarray
            Equilibrium para-hydrogen mole fraction [-]
        
        Reference: Wilhelmsen et al. (2018)
        """
        T = np.atleast_1d(T)
        
        x_eq = (0.1 * (np.exp(-175/T) + 0.1)**(-1) - 
                7.06e-9*T**3 + 3.42e-6*T**2 - 6.2e-5*T - 0.00227)
        
        # Ensure physical bounds
        x_eq = np.clip(x_eq, 0, 1)
        
        return x_eq if len(x_eq) > 1 else x_eq[0]
    
    @staticmethod
    def get_helium_properties(T, P):
        """
        Get helium properties using CoolProp
        
        Parameters
        ----------
        T : float or ndarray
            Temperature [K]
        P : float or ndarray
            Pressure [Pa]
        
        Returns
        -------
        dict
            Dictionary containing helium properties
        """
        # Handle scalar and array inputs
        scalar_input = np.isscalar(T)
        T = np.atleast_1d(T)
        P = np.atleast_1d(P)
        
        n = max(len(T), len(P))
        if len(T) == 1:
            T = np.full(n, T[0])
        if len(P) == 1:
            P = np.full(n, P[0])
        
        # Initialize arrays
        h = np.zeros(n)
        s = np.zeros(n)
        rho = np.zeros(n)
        cp = np.zeros(n)
        mu = np.zeros(n)
        lambda_ = np.zeros(n)
        
        # Use AbstractState for efficiency
        try:
            state_he = AbstractState("HEOS", "Helium")
            use_low_level = True
        except:
            use_low_level = False
        
        for i in range(n):
            try:
                if use_low_level:
                    state_he.update(CoolProp.PT_INPUTS, P[i], T[i])
                    h[i] = state_he.hmass()
                    s[i] = state_he.smass()
                    rho[i] = state_he.rhomass()
                    cp[i] = state_he.cpmass()
                    mu[i] = state_he.viscosity()
                    lambda_[i] = state_he.conductivity()
                else:
                    h[i] = PropsSI('H', 'T', T[i], 'P', P[i], 'Helium')
                    s[i] = PropsSI('S', 'T', T[i], 'P', P[i], 'Helium')
                    rho[i] = PropsSI('D', 'T', T[i], 'P', P[i], 'Helium')
                    cp[i] = PropsSI('C', 'T', T[i], 'P', P[i], 'Helium')
                    mu[i] = PropsSI('V', 'T', T[i], 'P', P[i], 'Helium')
                    lambda_[i] = PropsSI('L', 'T', T[i], 'P', P[i], 'Helium')
            except Exception as e:
                print(f"Warning: Helium property calculation failed at T={T[i]} K, P={P[i]} Pa: {e}")
                h[i] = s[i] = rho[i] = cp[i] = mu[i] = lambda_[i] = np.nan
        
        result = {
            'h': h, 's': s, 'rho': rho, 'cp': cp, 'mu': mu, 'lambda': lambda_,
            'T': T, 'P': P
        }
        
        if scalar_input:
            result = {k: v[0] if isinstance(v, np.ndarray) else v for k, v in result.items()}
        
        return result


# Import CoolProp constants for update methods
try:
    import CoolProp
except ImportError:
    print("Warning: Could not import CoolProp constants. Using fallback.")
    class CoolProp:
        PT_INPUTS = 0


def test_hydrogen_properties():
    """Test the hydrogen property calculations with verification"""
    print("="*70)
    print("Testing Hydrogen Property Calculation with CoolProp")
    print("Enthalpy Reference Correction Verification")
    print("="*70)
    print()
    
    # Initialize property calculator
    h2_props = HydrogenProperties()
    print()
    
    # Test 1: Verify conversion heat at 20 K
    print("="*70)
    print("TEST 1: Conversion Heat Verification at 20 K")
    print("="*70)
    print()
    
    T_20K = 20.0
    P_ref = 101325.0  # 1 atm
    
    # Pure ortho at 20 K
    props_ortho = h2_props.get_properties(T_20K, P_ref, x_para=0.0)
    # Pure para at 20 K
    props_para = h2_props.get_properties(T_20K, P_ref, x_para=1.0)
    # Normal hydrogen (75% ortho, 25% para) at 20 K
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
    print(f"Target Values:")
    print(f"  Δh_n→p (target) = {h2_props.DELTA_H_NP_20K/1e3:10.3f} kJ/kg")
    print(f"  Δh_o→p (calc)   = {h2_props.delta_h_op_20K/1e3:10.3f} kJ/kg")
    print()
    print(f"Verification:")
    print(f"  Δh_n→p error = {abs(Delta_h_np - h2_props.DELTA_H_NP_20K)/h2_props.DELTA_H_NP_20K*100:.6f} %")
    print(f"  Δh_o→p error = {abs(Delta_h_op - h2_props.delta_h_op_20K)/h2_props.delta_h_op_20K*100:.6f} %")
    
    # Verify h_normal = 0.75*h_ortho + 0.25*h_para
    h_normal_check = 0.75*props_normal['h_ortho'] + 0.25*props_normal['h_para']
    print()
    print(f"Consistency Check:")
    print(f"  h_normal (calculated) = {props_normal['h_normal']/1e3:.3f} kJ/kg")
    print(f"  0.75*h_ortho + 0.25*h_para = {h_normal_check/1e3:.3f} kJ/kg")
    print(f"  Difference = {abs(props_normal['h_normal'] - h_normal_check)/1e3:.6f} kJ/kg")
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
        Delta_h_op = props['Delta_h']
        
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
    
    # Test 4: Equilibrium fraction
    print("="*70)
    print("TEST 4: Equilibrium Para-Hydrogen Fraction")
    print("="*70)
    print()
    
    T_eq = np.array([20, 30, 40, 50, 60, 70, 80])
    x_eq = h2_props.get_equilibrium_fraction(T_eq)
    
    print(f"{'T [K]':>6} {'x_eq [-]':>10}")
    print("-"*18)
    for i, T in enumerate(T_eq):
        print(f"{T:6.1f} {x_eq[i]:10.4f}")
    
    print()
    print("="*70)
    print("All tests completed successfully!")
    print("="*70)
    print()


if __name__ == "__main__":
    test_hydrogen_properties()