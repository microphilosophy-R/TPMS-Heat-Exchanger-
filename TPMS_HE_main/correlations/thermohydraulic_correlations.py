"""
TPMS Heat Exchanger Correlations Module

This module provides Nusselt number and friction factor correlations
for various Triply Periodic Minimal Surface (TPMS) structures.

Based on comprehensive literature review from multiple experimental studies.

Author: Based on research compilation
"""

import numpy as np
import warnings


class ThermoHydraulicCorrelations:
    """
    Database of heat transfer and friction correlations for TPMS structures
    """

    SUPPORTED_TPMS_TYPES = ('Gyroid', 'Diamond', 'Primitive', 'Neovius', 'FRD', 'FKS',
                             'SmoothPlateFin', 'PlateFin')
    
    # Prandtl numbers for common fluids
    PR_WATER = 6.0
    PR_AIR = 0.71
    PR_GAS = 0.7  # H2/He for cryogenic applications
    PR_RP3 = 20.0
    
    @staticmethod
    def get_correlations(tpms_type, Re, Pr, fluid_type='Gas', geometry=None):
        """
        Get Nusselt number and friction factor for TPMS structure

        Parameters
        ----------
        tpms_type : str
            TPMS structure type
        Re : float or ndarray
            Reynolds number [-]
        Pr : float or ndarray
            Prandtl number [-]
        fluid_type : str, optional
            Fluid type: 'Water', 'Air', 'Gas', 'RP-3'
        geometry : dict, optional
            Geometry parameters for PlateFin (fin_spacing, fin_height, fin_thickness)

        Returns
        -------
        Nu : float or ndarray
            Nusselt number [-]
        f : float or ndarray
            Friction factor (Fanning) [-]
        
        References
        ----------
        Based on experimental data from multiple papers [41], [46], [93],
        [101], [109], [119], [142], [149]
        """
        # Ensure inputs are float arrays
        Re = np.atleast_1d(np.asarray(Re, dtype=float))
        Pr = np.atleast_1d(np.asarray(Pr, dtype=float)) if not np.isscalar(Pr) else np.full_like(Re, Pr)
        
        scalar_input = (Re.size == 1)
        
        # Initialize outputs
        Nu = np.full_like(Re, np.nan, dtype=float)
        f = np.full_like(Re, np.nan, dtype=float)
        
        # Select correlation function based on TPMS type
        correlation_map = {
            'Gyroid': ThermoHydraulicCorrelations._gyroid_correlations,
            'Diamond': ThermoHydraulicCorrelations._diamond_correlations,
            'Primitive': ThermoHydraulicCorrelations._primitive_correlations,
            'Neovius': ThermoHydraulicCorrelations._neovius_correlations,
            'FRD': ThermoHydraulicCorrelations._frd_correlations,
            'FKS': ThermoHydraulicCorrelations._fks_correlations,
            'SmoothPlateFin': ThermoHydraulicCorrelations._smooth_plate_fin_correlations,
            'PlateFin': ThermoHydraulicCorrelations._plate_fin_correlations,
        }

        if tpms_type not in correlation_map:
            warnings.warn(f"Unknown TPMS type: {tpms_type}. Using Gyroid correlations.")
            tpms_type = 'Gyroid'
        
        # Get correlations
        if tpms_type == 'PlateFin':
            Nu, f = correlation_map[tpms_type](Re, Pr, fluid_type, geometry)
        else:
            Nu, f = correlation_map[tpms_type](Re, Pr, fluid_type)
        
        # Ensure physical values
        Nu = np.maximum(Nu, 1.0)  # Minimum Nu = 1
        f = np.maximum(f, 0.0)    # f >= 0
        
        # Return scalar if input was scalar
        if scalar_input:
            Nu = Nu[0]
            f = f[0]
        
        return Nu, f

    @staticmethod
    def get_supported_tpms_types():
        """Return supported TPMS structure names."""
        return ThermoHydraulicCorrelations.SUPPORTED_TPMS_TYPES
    
    @staticmethod
    def _gyroid_correlations(Re, Pr, fluid_type):
        """Gyroid TPMS correlations"""
        Nu = np.full_like(Re, np.nan, dtype=float)
        f = np.full_like(Re, np.nan, dtype=float)
        
        if fluid_type == 'Water':
            # Multiple correlations available - use best match based on Re
            for i, re_val in enumerate(Re):
                if 25 <= re_val <= 250:
                    # [109] Gyroid Sheet Water
                    Nu[i] = 1.48 * re_val**0.57
                    f[i] = 15.5 * re_val**(-0.58)
                elif 100 <= re_val <= 2500:
                    # [149] Gyroid Water
                    Nu[i] = 0.49 * re_val**0.62 * Pr[i]**0.4
                    f[i] = 2.577 * re_val**(-0.095)  # From [46]
                elif 150 <= re_val <= 3000:
                    # [46] Gyroid Water
                    Nu[i] = 0.471 * re_val**0.627 * Pr[i]**(1/3)
                    f[i] = 2.577 * re_val**(-0.095)
                elif 80 <= re_val <= 1500:
                    # [119] Gyroid Water
                    Nu[i] = 0.14038 * re_val**0.71979 * Pr[i]**(1/3)
                    f[i] = 2.39612 * re_val**(-0.29873)
                else:
                    # Default to [46] with extrapolation warning
                    Nu[i] = 0.471 * re_val**0.627 * Pr[i]**(1/3)
                    f[i] = 2.577 * re_val**(-0.095)
                    if i == 0 or (i > 0 and Re[i-1] != re_val):
                        warnings.warn(f'Re={re_val:.1f} outside validated range for Gyroid-Water')
        
        elif fluid_type in ['Air', 'Gas']:
            # [41] Gyroid Sheet Air
            Nu = 0.3250 * Re**0.7002 * Pr**0.36
            f = 2.5 * Re**(-0.2)  # Estimated
            
            # Check range
            if np.any((Re < 2000) | (Re > 8170)):
                warnings.warn('Some Re values outside validated range (2000-8170) for Gyroid-Air')
        
        else:
            # Default to gas correlation
            Nu = 0.3250 * Re**0.7002 * Pr**0.36
            f = 2.5 * Re**(-0.2)
        
        return Nu, f
    
    @staticmethod
    def _diamond_correlations(Re, Pr, fluid_type):
        """Diamond TPMS correlations"""
        Nu = np.full_like(Re, np.nan, dtype=float)
        f = np.full_like(Re, np.nan, dtype=float)
        
        if fluid_type == 'Water':
            for i, re_val in enumerate(Re):
                if 15 <= re_val <= 300:
                    # [109] Diamond Sheet Water
                    Nu[i] = 2.24 * re_val**0.55
                    f[i] = 17.2 * re_val**(-0.62)
                elif 80 <= re_val <= 1500:
                    # [119] Diamond Water
                    Nu[i] = 0.12504 * re_val**0.73143 * Pr[i]**(1/3)
                    f[i] = 2.74632 * re_val**(-0.36099)
                else:
                    # Default to [119]
                    Nu[i] = 0.12504 * re_val**0.73143 * Pr[i]**(1/3)
                    f[i] = 2.74632 * re_val**(-0.36099)
                    if i == 0 or (i > 0 and Re[i-1] != re_val):
                        warnings.warn(f'Re={re_val:.1f} outside validated range for Diamond-Water')
        
        elif fluid_type in ['Air', 'Gas']:
            # [101] Diamond (Gas) - Best for cryogenic applications
            Nu = 0.409 * Re**0.625 * Pr**0.4
            f = 2.5892 * Re**(-0.1940)
            
            # Check range
            if np.any((Re < 800) | (Re > 9590)):
                warnings.warn('Some Re values outside validated range (800-9590) for Diamond-Gas')
        
        elif fluid_type == 'RP-3':
            # [142] Diamond RP-3
            Nu = 0.157 * Re**0.805 * Pr**0.480
            f = 3.0 * Re**(-0.25)  # Estimated
            
            if np.any((Re < 40) | (Re > 1000)):
                warnings.warn('Some Re values outside validated range (40-1000) for Diamond-RP3')
        
        else:
            # Default to gas correlation
            Nu = 0.409 * Re**0.625 * Pr**0.4
            f = 2.5892 * Re**(-0.1940)
        
        return Nu, f
    
    @staticmethod
    def _primitive_correlations(Re, Pr, fluid_type):
        """Primitive TPMS correlations"""
        Nu = np.full_like(Re, np.nan, dtype=float)
        f = np.full_like(Re, np.nan, dtype=float)
        
        if fluid_type == 'Water':
            for i, re_val in enumerate(Re):
                if 15 <= re_val <= 300:
                    # [109] Primitive Sheet Water
                    Nu[i] = 1.39 * re_val**0.45
                    f[i] = 41.9 * re_val**(-0.85)
                elif 80 <= re_val <= 1500:
                    # [119] Primitive Water
                    Nu[i] = 0.05513 * re_val**0.81370 * Pr[i]**(1/3)
                    f[i] = 3.96709 * re_val**(-0.23326)
                else:
                    # Default to [119]
                    Nu[i] = 0.05513 * re_val**0.81370 * Pr[i]**(1/3)
                    f[i] = 3.96709 * re_val**(-0.23326)
        
        else:
            # Gas correlation (estimated)
            Nu = 0.1 * Re**0.75 * Pr**0.36
            f = 4.0 * Re**(-0.25)
        
        return Nu, f
    
    @staticmethod
    def _neovius_correlations(Re, Pr, fluid_type):
        """Neovius TPMS correlations"""
        Nu = np.full_like(Re, np.nan, dtype=float)
        f = np.full_like(Re, np.nan, dtype=float)
        
        if fluid_type == 'Water':
            # [109] Neovius Sheet Water
            Nu = 2.48 * Re**0.45
            f = 59.2 * Re**(-0.63)
            
            if np.any((Re < 10) | (Re > 75)):
                warnings.warn('Some Re values outside validated range (10-75) for Neovius-Water')
        
        else:
            # Gas correlation (estimated)
            Nu = 0.15 * Re**0.7 * Pr**0.36
            f = 5.0 * Re**(-0.3)
        
        return Nu, f
    
    @staticmethod
    def _frd_correlations(Re, Pr, fluid_type):
        """FRD TPMS correlations"""
        Nu = np.full_like(Re, np.nan, dtype=float)
        f = np.full_like(Re, np.nan, dtype=float)
        
        if fluid_type == 'Water':
            # [109] FRD Sheet Water
            Nu = 1.74 * Re**0.54
            f = 11.5 * Re**(-0.41)
            
            if np.any((Re < 35) | (Re > 290)):
                warnings.warn('Some Re values outside validated range (35-290) for FRD-Water')
        
        else:
            # Gas correlation (estimated)
            Nu = 0.3 * Re**0.65 * Pr**0.36
            f = 3.0 * Re**(-0.2)
        
        return Nu, f
    
    @staticmethod
    def _smooth_plate_fin_correlations(Re, Pr, fluid_type):
        """
        Plain smooth parallel-plate channel baseline.

        Nusselt number: Dittus-Boelter (turbulent) / constant Nu=3.66 (laminar).
        Friction factor: Petukhov-Filonenko Fanning (turbulent) / f=16/Re (laminar).

        Used as PEC reference: setting a channel to SmoothPlateFin gives the
        classical plate-fin result under identical flow area and mass flow rate.

        Nu = 0.023 Re^0.8 Pr^0.4          (Re > 2300)
        Nu = 3.66                           (Re <= 2300, fully-developed laminar)
        f  = (0.790 ln Re - 1.64)^-2 / 4  (Re > 2300, Fanning from Petukhov-Filonenko)
        f  = 16 / Re                        (Re <= 2300)
        """
        Re_safe = np.maximum(Re, 1.0)
        turb = Re > 2300

        Nu = np.where(
            turb,
            np.maximum(3.66, 0.023 * Re_safe**0.8 * Pr**0.4),
            3.66 * np.ones_like(Re)
        )

        f_turb = (0.790 * np.log(Re_safe) - 1.64) ** (-2) / 4.0
        f = np.where(turb, f_turb, 16.0 / Re_safe)

        return Nu, f

    @staticmethod
    def _fks_correlations(Re, Pr, fluid_type):
        """FKS TPMS correlations"""
        Nu = np.full_like(Re, np.nan, dtype=float)
        f = np.full_like(Re, np.nan, dtype=float)
        
        if fluid_type == 'Water':
            # [109] FKS Sheet Water
            Nu = 3.02 * Re**0.40
            f = 25.0 * Re**(-0.73)
            
            if np.any((Re < 10) | (Re > 140)):
                warnings.warn('Some Re values outside validated range (10-140) for FKS-Water')
        
        elif fluid_type in ['Air', 'Gas']:
            # [101] FKS (Gas) - For cryogenic applications
            Nu = 0.52 * Re**0.61 * Pr**0.4
            f = 2.1335 * Re**(-0.1334)
            
            if np.any((Re < 730) | (Re > 10230)):
                warnings.warn('Some Re values outside validated range (730-10230) for FKS-Gas')
        
        else:
            # Default to gas correlation
            Nu = 0.52 * Re**0.61 * Pr**0.4
            f = 2.1335 * Re**(-0.1334)
        
        return Nu, f


    @staticmethod
    def _plate_fin_correlations(Re, Pr, fluid_type, geometry=None):
        """
        Plate-fin correlations with geometry-dependent option.

        If geometry provided: Uses MATLAB-style correlation (Wang et al. 2024)
            Nu = (0.233*Re^(-0.48)*(s/h)^0.192*(t/h)^(-0.14)) * Re * Pr^(1/3)
            f  = 0.029*Re^(-0.09)*(s/h)^(-0.169)*(t/h)^0.034

        Otherwise: Uses j-factor correlation (Li 2018)
        """
        Re_safe = np.maximum(Re, 1.0)

        if geometry and 'fin_spacing' in geometry:
            s = geometry['fin_spacing']
            h = geometry['fin_height']
            t = geometry['fin_thickness']
            Nu = (0.233 * Re_safe**(-0.48) * (s/h)**0.192 * (t/h)**(-0.14)) * Re_safe * Pr**(1/3)
            f = 0.029 * Re_safe**(-0.09) * (s/h)**(-0.169) * (t/h)**0.034
        else:
            ln_Re = np.log(Re_safe)
            ln_j = (-2.64136e-2 * ln_Re**3 + 0.55584 * ln_Re**2 - 4.09241 * ln_Re + 6.21681)
            j = np.exp(ln_j)
            Nu = j * Re_safe * Pr**(1/3)
            f = 2.5 * j
        return Nu, f


def test_tpms_correlations():
    """Test TPMS correlations"""
    print("="*70)
    print("Testing TPMS Correlations")
    print("="*70)
    print()
    
    # Test single point
    Re_test = 1500
    Pr_test = 0.7
    
    print(f"Single Point Test: Re = {Re_test}, Pr = {Pr_test}, Fluid = Gas")
    print("-"*70)
    print(f"{'TPMS Type':<15} {'Nu':>8} {'f':>10} {'Nu/f^(1/3)':>12}")
    print("-"*70)
    
    tpms_types = ['Gyroid', 'Diamond', 'Primitive', 'FKS']
    for tpms in tpms_types:
        Nu, f = ThermoHydraulicCorrelations.get_correlations(tpms, Re_test, Pr_test, 'Gas')
        pec = Nu / f**(1/3)
        print(f"{tpms:<15} {Nu:8.2f} {f:10.4f} {pec:12.2f}")
    
    print()
    print("="*70)
    print("Reynolds Number Sweep (Diamond, Gas)")
    print("="*70)
    print()
    
    Re_range = np.array([100, 500, 1000, 2000, 5000])
    Nu_arr, f_arr = ThermoHydraulicCorrelations.get_correlations('Diamond', Re_range, Pr_test, 'Gas')
    
    print(f"{'Re':>6} {'Nu':>8} {'f':>10} {'PEC':>8}")
    print("-"*35)
    for i, re in enumerate(Re_range):
        pec = Nu_arr[i] / f_arr[i]**(1/3)
        print(f"{re:6.0f} {Nu_arr[i]:8.2f} {f_arr[i]:10.4f} {pec:8.2f}")
    
    print()
    print("Test completed successfully!")
    print()


if __name__ == "__main__":
    test_tpms_correlations()
