"""
TPMS Heat Exchanger Correlations Module

This module provides Nusselt number and friction factor correlations
for various Triply Periodic Minimal Surface (TPMS) structures.

Based on comprehensive literature review from multiple experimental studies.

Author: Based on research compilation
"""

import numpy as np
import warnings
from datetime import datetime
from pathlib import Path


class ThermoHydraulicCorrelations:
    """
    Database of heat transfer and friction correlations for TPMS structures
    """

    SUPPORTED_TPMS_TYPES = ('Gyroid', 'Diamond', 'Primitive', 'Neovius', 'FRD', 'FKS',
                             'SmoothPlateFin', 'PlateFin')

    REFERENCE_RENUMBER_MAP = {
        109: 79,
        41: 63,
        93: 88,
        46: 82,
        149: 86,
        142: 76,
        119: 89,
        101: 90,
    }
    
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
        Based on experimental data from multiple papers [63], [76], [79],
        [82], [86], [88], [89], [90]
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

    @classmethod
    def renumber_reference(cls, reference_id):
        """Map legacy citation numbers used in early plots to the current thesis numbering."""
        return cls.REFERENCE_RENUMBER_MAP.get(int(reference_id), int(reference_id))
    
    @staticmethod
    def _gyroid_correlations(Re, Pr, fluid_type):
        """Gyroid TPMS correlations"""
        Nu = np.full_like(Re, np.nan, dtype=float)
        f = np.full_like(Re, np.nan, dtype=float)
        
        if fluid_type == 'Water':
            # Multiple correlations available - use best match based on Re
            for i, re_val in enumerate(Re):
                if 25 <= re_val <= 250:
                    # [79] Gyroid Sheet Water
                    Nu[i] = 1.48 * re_val**0.57
                    f[i] = 15.5 * re_val**(-0.58)
                elif 100 <= re_val <= 2500:
                    # [86] Gyroid Water
                    Nu[i] = 0.49 * re_val**0.62 * Pr[i]**0.4
                    f[i] = 2.577 * re_val**(-0.095)  # From [82]
                elif 150 <= re_val <= 3000:
                    # [82] Gyroid Water
                    Nu[i] = 0.471 * re_val**0.627 * Pr[i]**(1/3)
                    f[i] = 2.577 * re_val**(-0.095)
                elif 80 <= re_val <= 1500:
                    # [89] Gyroid Water
                    Nu[i] = 0.14038 * re_val**0.71979 * Pr[i]**(1/3)
                    f[i] = 2.39612 * re_val**(-0.29873)
                else:
                    # Default to [82] with extrapolation warning
                    Nu[i] = 0.471 * re_val**0.627 * Pr[i]**(1/3)
                    f[i] = 2.577 * re_val**(-0.095)
                    if i == 0 or (i > 0 and Re[i-1] != re_val):
                        warnings.warn(f'Re={re_val:.1f} outside validated range for Gyroid-Water')

        elif fluid_type in ['Air', 'Gas']:
            # [63] Gyroid Sheet Air
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
                    # [79] Diamond Sheet Water
                    Nu[i] = 2.24 * re_val**0.55
                    f[i] = 17.2 * re_val**(-0.62)
                elif 80 <= re_val <= 1500:
                    # [89] Diamond Water
                    Nu[i] = 0.12504 * re_val**0.73143 * Pr[i]**(1/3)
                    f[i] = 2.74632 * re_val**(-0.36099)
                else:
                    # Default to [89]
                    Nu[i] = 0.12504 * re_val**0.73143 * Pr[i]**(1/3)
                    f[i] = 2.74632 * re_val**(-0.36099)
                    if i == 0 or (i > 0 and Re[i-1] != re_val):
                        warnings.warn(f'Re={re_val:.1f} outside validated range for Diamond-Water')

        elif fluid_type in ['Air', 'Gas']:
            # [90] Diamond (Gas) - Best for cryogenic applications
            Nu = 0.409 * Re**0.625 * Pr**0.4
            f = 2.5892 * Re**(-0.1940)
            
            # Check range
            if np.any((Re < 800) | (Re > 9590)):
                warnings.warn('Some Re values outside validated range (800-9590) for Diamond-Gas')
        
        elif fluid_type == 'RP-3':
            # [76] Diamond RP-3
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
                    # [79] Primitive Sheet Water
                    Nu[i] = 1.39 * re_val**0.45
                    f[i] = 41.9 * re_val**(-0.85)
                elif 80 <= re_val <= 1500:
                    # [89] Primitive Water
                    Nu[i] = 0.05513 * re_val**0.81370 * Pr[i]**(1/3)
                    f[i] = 3.96709 * re_val**(-0.23326)
                else:
                    # Default to [89]
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
            # [79] Neovius Sheet Water
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
            # [79] FRD Sheet Water
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
            # [79] FKS Sheet Water
            Nu = 3.02 * Re**0.40
            f = 25.0 * Re**(-0.73)
            
            if np.any((Re < 10) | (Re > 140)):
                warnings.warn('Some Re values outside validated range (10-140) for FKS-Water')
        
        elif fluid_type in ['Air', 'Gas']:
            # [90] FKS (Gas) - For cryogenic applications
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

    @classmethod
    def _publication_plot_style(cls):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "font.family": ["Times New Roman", "SimSun"],
                "mathtext.fontset": "stix",
                "axes.unicode_minus": False,
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 10,
                "figure.facecolor": "white",
            }
        )
        return plt

    @staticmethod
    def _style_publication_axes(ax):
        ax.grid(False)
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            width=1.0,
            length=5.0,
            labelsize=10,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="in",
            top=True,
            right=True,
            width=1.0,
            length=3.0,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

    @classmethod
    def _correlation_plot_specs(cls):
        pr_air = cls.PR_AIR
        pr_gas = cls.PR_GAS
        pr_water = cls.PR_WATER
        pr_rp3 = cls.PR_RP3

        return {
            "Nu": [
                {
                    "label": "Gyroid Sheet",
                    "reference": 109,
                    "color": "#1f4aff",
                    "linestyle": "-",
                    "re_range": (25, 250),
                    "func": lambda re: 1.48 * re**0.57,
                    "xy": (30, 1.48 * 30**0.57),
                    "xytext": (8.5, 11.4),
                },
                {
                    "label": "Diamond Sheet",
                    "reference": 109,
                    "color": "#ff2a2a",
                    "linestyle": "-",
                    "re_range": (15, 300),
                    "func": lambda re: 2.24 * re**0.55,
                    "xy": (20, 2.24 * 20**0.55),
                    "xytext": (8.5, 15.5),
                },
                {
                    "label": "Primitive Sheet",
                    "reference": 109,
                    "color": "#1f8a2b",
                    "linestyle": "-",
                    "re_range": (15, 300),
                    "func": lambda re: 1.39 * re**0.45,
                    "xy": (18, 1.39 * 18**0.45),
                    "xytext": (8.5, 4.8),
                },
                {
                    "label": "Neovius Sheet",
                    "reference": 109,
                    "color": "#8a1ea8",
                    "linestyle": "-",
                    "re_range": (10, 75),
                    "func": lambda re: 2.48 * re**0.45,
                    "xy": (12, 2.48 * 12**0.45),
                    "xytext": (8.5, 6.6),
                },
                {
                    "label": "FRD Sheet",
                    "reference": 109,
                    "color": "#f2a000",
                    "linestyle": "-",
                    "re_range": (35, 290),
                    "func": lambda re: 1.74 * re**0.54,
                    "xy": (40, 1.74 * 40**0.54),
                    "xytext": (8.5, 21.5),
                },
                {
                    "label": "FKS Sheet",
                    "reference": 109,
                    "color": "#a9442b",
                    "linestyle": "-",
                    "re_range": (10, 140),
                    "func": lambda re: 3.02 * re**0.40,
                    "xy": (12, 3.02 * 12**0.40),
                    "xytext": (8.5, 8.7),
                },
                {
                    "label": "Gyroid Sheet",
                    "reference": 41,
                    "color": "#1f4aff",
                    "linestyle": "--",
                    "re_range": (2000, 8170),
                    "func": lambda re, pr=pr_air: 0.3250 * re**0.7002 * pr**0.36,
                    "xy": (8170, 0.3250 * 8170**0.7002 * pr_air**0.36),
                    "xytext": (11200, 165.0),
                },
                {
                    "label": "Gyroid Network",
                    "reference": 41,
                    "color": "#1f4aff",
                    "linestyle": ":",
                    "re_range": (2000, 8170),
                    "func": lambda re, pr=pr_air: 1.0462 * re**0.5751 * pr**0.36,
                    "xy": (8170, 1.0462 * 8170**0.5751 * pr_air**0.36),
                    "xytext": (11200, 138.0),
                },
                {
                    "label": "Diamond Network",
                    "reference": 41,
                    "color": "#ff2a2a",
                    "linestyle": ":",
                    "re_range": (2000, 8170),
                    "func": lambda re, pr=pr_air: 2.0513 * re**0.5193 * pr**0.36,
                    "xy": (8170, 2.0513 * 8170**0.5193 * pr_air**0.36),
                    "xytext": (11200, 205.0),
                },
                {
                    "label": "Diamond Network",
                    "reference": 93,
                    "color": "#ff2a2a",
                    "linestyle": "--",
                    "re_range": (4000, 8000),
                    "func": lambda re, pr=pr_air: 1.8 * re**0.54 * pr**(1 / 3),
                    "xy": (8000, 1.8 * 8000**0.54 * pr_air**(1 / 3)),
                    "xytext": (11200, 103.0),
                },
                {
                    "label": "Gyroid",
                    "reference": 149,
                    "color": "#1f4aff",
                    "linestyle": "-.",
                    "re_range": (100, 2500),
                    "func": lambda re, pr=pr_water: 0.49 * re**0.62 * pr**0.4,
                    "xy": (2500, 0.49 * 2500**0.62 * pr_water**0.4),
                    "xytext": (2600, 74.0),
                },
                {
                    "label": "Diamond",
                    "reference": 142,
                    "color": "#ff2a2a",
                    "linestyle": "-.",
                    "re_range": (40, 1000),
                    "func": lambda re, pr=pr_rp3: 0.157 * re**0.805 * pr**0.480,
                    "xy": (1000, 0.157 * 1000**0.805 * pr_rp3**0.480),
                    "xytext": (1500, 52.0),
                },
                {
                    "label": "Primitive",
                    "reference": 119,
                    "color": "#1f8a2b",
                    "linestyle": "--",
                    "re_range": (80, 1500),
                    "func": lambda re, pr=pr_water: 0.05513 * re**0.81370 * pr**(1 / 3),
                    "xy": (1500, 0.05513 * 1500**0.81370 * pr_water**(1 / 3)),
                    "xytext": (360, 28.5),
                },
                {
                    "label": "Gyroid",
                    "reference": 119,
                    "color": "#1f4aff",
                    "linestyle": "--",
                    "re_range": (80, 1500),
                    "func": lambda re, pr=pr_water: 0.14038 * re**0.71979 * pr**(1 / 3),
                    "xy": (1500, 0.14038 * 1500**0.71979 * pr_water**(1 / 3)),
                    "xytext": (520, 38.5),
                },
                {
                    "label": "Diamond",
                    "reference": 119,
                    "color": "#ff2a2a",
                    "linestyle": "--",
                    "re_range": (80, 1500),
                    "func": lambda re, pr=pr_water: 0.12504 * re**0.73143 * pr**(1 / 3),
                    "xy": (1500, 0.12504 * 1500**0.73143 * pr_water**(1 / 3)),
                    "xytext": (760, 46.0),
                },
                {
                    "label": "Diamond (Gas)",
                    "reference": 101,
                    "color": "#ff2a2a",
                    "linestyle": "-.",
                    "re_range": (800, 9590),
                    "func": lambda re, pr=pr_gas: 0.409 * re**0.625 * pr**0.4,
                    "xy": (9590, 0.409 * 9590**0.625 * pr_gas**0.4),
                    "xytext": (11200, 255.0),
                },
                {
                    "label": "FKS (Gas)",
                    "reference": 101,
                    "color": "#a9442b",
                    "linestyle": "--",
                    "re_range": (730, 10230),
                    "func": lambda re, pr=pr_gas: 0.52 * re**0.61 * pr**0.4,
                    "xy": (10230, 0.52 * 10230**0.61 * pr_gas**0.4),
                    "xytext": (11200, 125.0),
                },
            ],
            "f": [
                {
                    "label": "Gyroid Sheet",
                    "reference": 109,
                    "color": "#1f4aff",
                    "linestyle": "-",
                    "re_range": (25, 250),
                    "func": lambda re: 15.5 * re**(-0.58),
                    "xy": (30, 15.5 * 30**(-0.58)),
                    "xytext": (8.5, 1.75),
                },
                {
                    "label": "Diamond Sheet",
                    "reference": 109,
                    "color": "#ff2a2a",
                    "linestyle": "-",
                    "re_range": (15, 300),
                    "func": lambda re: 17.2 * re**(-0.62),
                    "xy": (20, 17.2 * 20**(-0.62)),
                    "xytext": (8.5, 2.30),
                },
                {
                    "label": "Diamond Network",
                    "reference": 93,
                    "color": "#ff2a2a",
                    "linestyle": "--",
                    "re_range": (4000, 8000),
                    "func": lambda re: 6.52e-3 * re**(-0.104),
                    "xy": (8000, 6.52e-3 * 8000**(-0.104)),
                    "xytext": (4300, 0.00155),
                },
                {
                    "label": "Primitive Sheet",
                    "reference": 109,
                    "color": "#1f8a2b",
                    "linestyle": "-",
                    "re_range": (15, 300),
                    "func": lambda re: 41.9 * re**(-0.85),
                    "xy": (18, 41.9 * 18**(-0.85)),
                    "xytext": (8.5, 1.15),
                },
                {
                    "label": "Neovius Sheet",
                    "reference": 109,
                    "color": "#8a1ea8",
                    "linestyle": "-",
                    "re_range": (10, 75),
                    "func": lambda re: 59.2 * re**(-0.63),
                    "xy": (12, 59.2 * 12**(-0.63)),
                    "xytext": (8.5, 12.5),
                },
                {
                    "label": "FRD Sheet",
                    "reference": 109,
                    "color": "#f2a000",
                    "linestyle": "-",
                    "re_range": (35, 290),
                    "func": lambda re: 11.5 * re**(-0.41),
                    "xy": (40, 11.5 * 40**(-0.41)),
                    "xytext": (8.5, 3.35),
                },
                {
                    "label": "FKS Sheet",
                    "reference": 109,
                    "color": "#a9442b",
                    "linestyle": "-",
                    "re_range": (10, 140),
                    "func": lambda re: 25.0 * re**(-0.73),
                    "xy": (12, 25.0 * 12**(-0.73)),
                    "xytext": (8.5, 5.00),
                },
                {
                    "label": "Gyroid",
                    "reference": 46,
                    "color": "#1f4aff",
                    "linestyle": "--",
                    "re_range": (150, 3000),
                    "func": lambda re: 2.577 * re**(-0.095),
                    "xy": (3000, 2.577 * 3000**(-0.095)),
                    "xytext": (220, 1.38),
                },
                {
                    "label": "Primitive",
                    "reference": 119,
                    "color": "#1f8a2b",
                    "linestyle": "--",
                    "re_range": (80, 1500),
                    "func": lambda re: 3.96709 * re**(-0.23326),
                    "xy": (1500, 3.96709 * 1500**(-0.23326)),
                    "xytext": (220, 0.82),
                },
                {
                    "label": "Gyroid",
                    "reference": 119,
                    "color": "#1f4aff",
                    "linestyle": ":",
                    "re_range": (80, 1500),
                    "func": lambda re: 2.39612 * re**(-0.29873),
                    "xy": (1500, 2.39612 * 1500**(-0.29873)),
                    "xytext": (300, 0.42),
                },
                {
                    "label": "Diamond",
                    "reference": 119,
                    "color": "#ff2a2a",
                    "linestyle": ":",
                    "re_range": (80, 1500),
                    "func": lambda re: 2.74632 * re**(-0.36099),
                    "xy": (1500, 2.74632 * 1500**(-0.36099)),
                    "xytext": (420, 0.27),
                },
                {
                    "label": "Diamond (Gas)",
                    "reference": 101,
                    "color": "#ff2a2a",
                    "linestyle": "-.",
                    "re_range": (800, 9590),
                    "func": lambda re: 2.5892 * re**(-0.1940),
                    "xy": (9590, 2.5892 * 9590**(-0.1940)),
                    "xytext": (11200, 0.52),
                },
                {
                    "label": "FKS (Gas)",
                    "reference": 101,
                    "color": "#a9442b",
                    "linestyle": "--",
                    "re_range": (730, 10230),
                    "func": lambda re: 2.1335 * re**(-0.1334),
                    "xy": (10230, 2.1335 * 10230**(-0.1334)),
                    "xytext": (11200, 0.76),
                },
            ],
        }

    @classmethod
    def _annotated_label(cls, spec):
        return f"{spec['label']} [{cls.renumber_reference(spec['reference'])}]"

    @classmethod
    def _draw_correlation_plot(cls, plot_key, save_path):
        plt = cls._publication_plot_style()
        fig, ax = plt.subplots(figsize=(4.92126, 3.28084), dpi=600)
        fig.subplots_adjust(left=0.12, right=0.78, bottom=0.14, top=0.985)

        specs = cls._correlation_plot_specs()[plot_key]
        for spec in specs:
            re_values = np.geomspace(spec["re_range"][0], spec["re_range"][1], 256)
            y_values = spec["func"](re_values)
            ax.plot(
                re_values,
                y_values,
                color=spec["color"],
                lw=1.5,
                ls=spec["linestyle"],
                label=cls._annotated_label(spec),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(7, 1.8e4)
        if plot_key == "Nu":
            ax.set_ylim(3.0, 4.0e2)
            ax.set_ylabel(r"$Nu$")
        else:
            ax.set_ylim(7.0e-4, 2.0e1)
            ax.set_ylabel(r"$f$")
        ax.set_xlabel(r"$Re$")
        cls._style_publication_axes(ax)
        legend = ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=8.5,
            handlelength=2.0,
            labelspacing=0.35,
        )
        frame = legend.get_frame()
        frame.set_edgecolor("0.8")
        frame.set_linewidth(0.8)
        frame.set_facecolor("white")
        frame.set_alpha(1.0)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return save_path

    @classmethod
    def generate_reference_plots(cls, output_dir=None, export_doc_assets=False):
        """
        Generate the two chapter-1 reference-summary plots with renumbered citations
        using a unified outside-legend layout.
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path.cwd() / "figures" / "results" / f"chapter1_restructured_{timestamp}" / "introduction"
        else:
            output_dir = Path(output_dir)

        nu_path = cls._draw_correlation_plot("Nu", output_dir / "tpms_nu_re_correlations_legend_refs.png")
        f_path = cls._draw_correlation_plot("f", output_dir / "tpms_f_re_correlations_legend_refs.png")

        saved_paths = {"Nu": nu_path, "f": f_path}

        if export_doc_assets:
            doc_dir = Path.cwd() / "cleaned" / "figures" / "docx_sync_0323" / "media"
            saved_paths["doc_image22"] = cls._draw_correlation_plot("Nu", doc_dir / "image22.png")
            saved_paths["doc_image23"] = cls._draw_correlation_plot("f", doc_dir / "image23.png")

        return saved_paths


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
