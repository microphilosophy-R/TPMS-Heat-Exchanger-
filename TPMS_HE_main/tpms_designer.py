"""
TPMS Heat Exchanger Designer
Description: Inverse solver that calculates required geometry (Length, Width, Cell Size)
to achieve specific outlet conditions.

Content:
1. TPMSDesigner: Wrapper class that optimizes geometry to meet thermal targets.
"""

import numpy as np
from scipy.optimize import minimize_scalar
import copy
import sys
import os
import contextlib

# Import the main simulation class
try:
    from tpms_thermo_hydraulic_calculator import TPMSHeatExchanger
except ImportError:
    raise ImportError("Could not import TPMSHeatExchanger. Ensure 'tpms_thermo_hydraulic_calculator.py' is in the same directory.")

class DummyFile:
    """A robust dummy file handler that ignores all input."""
    def write(self, x): pass
    def flush(self): pass
    def close(self): pass

class TPMSDesigner:
    """
    Design-stage solver for TPMS Heat Exchangers.
    Uses optimization algorithms and epsilon-NTU estimations to determine
    geometric parameters required to meet specific process targets.
    """

    def __init__(self, base_config):
        """
        Initialize with a baseline configuration dictionary.

        Parameters
        ----------
        base_config : dict
            The standard configuration dictionary used in TPMSHeatExchanger.
        """
        self.base_config = copy.deepcopy(base_config)

    def _update_nested_config(self, config, path_list, value):
        """Helper to update deep dictionary keys."""
        d = config
        for key in path_list[:-1]:
            d = d[key]
        d[path_list[-1]] = value

    def _get_result_value(self, he_instance, target_key):
        """Extracts the specific result metric from a solved heat exchanger instance."""
        if target_key == 'Th_out':
            return he_instance.Th[-1]
        elif target_key == 'Tc_out':
            return he_instance.Tc[0]
        elif target_key == 'Q_total':
            return np.sum(he_instance.Q)
        elif target_key == 'dP_hot':
            return he_instance.Ph[0] - he_instance.Ph[-1]
        elif target_key == 'dP_cold':
            return he_instance.Pc[-1] - he_instance.Pc[0]
        else:
            raise ValueError(f"Unknown target key: {target_key}")

    def estimate_size_epsilon_ntu(self, target_metric, target_value, epsilon_target=0.95):
        """
        Uses the epsilon-NTU method for counter-flow heat exchangers to estimate
        the required UA (overall heat transfer coefficient * area).

        Assumes the exchanger is highly efficient (epsilon ~ 0.95) for TPMS.
        """
        cfg = self.base_config
        ops = cfg['operating']

        # 1. Get Fluid Properties via local HE instance (temporary)
        he_temp = TPMSHeatExchanger(cfg)
        h2_props = he_temp.h2_props

        # 2. Calculate Heat Capacity Rates (C = m * Cp)
        p_h = h2_props.get_properties(ops['Th_in'], ops['Ph_in'], cfg['operating'].get('fluid_hot', 'hydrogen mixture'), ops['xh_in'])
        p_c = h2_props.get_properties(ops['Tc_in'], ops['Pc_in'], cfg['operating'].get('fluid_cold', 'helium'))

        Ch = ops['mh'] * p_h['cp']
        Cc = ops['mc'] * p_c['cp']

        Cmin = min(Ch, Cc)
        Cmax = max(Ch, Cc)
        Cr = Cmin / Cmax

        # 3. Determine actual epsilon based on target
        # Q_max = Cmin * (Th_in - Tc_in)
        Q_max = Cmin * (ops['Th_in'] - ops['Tc_in'])

        if target_metric == 'Th_out':
            Q_req = Ch * (ops['Th_in'] - target_value)
        elif target_metric == 'Tc_out':
            Q_req = Cc * (target_value - ops['Tc_in'])
        elif target_metric == 'Q_total':
            Q_req = target_value
        else:
            return None # Cannot estimate for pressure targets

        epsilon_actual = Q_req / Q_max

        # 4. Solve for NTU (Counter-flow arrangement)
        # NTU = (1/(Cr-1)) * ln((Eps-1)/(Eps*Cr-1))
        try:
            if Cr < 1.0:
                ntu = (1.0 / (Cr - 1.0)) * np.log((epsilon_actual - 1.0) / (epsilon_actual * Cr - 1.0))
            else:
                ntu = epsilon_actual / (1.0 - epsilon_actual)
        except:
            ntu = 5.0 # Fallback for high epsilon

        required_ua = ntu * Cmin

        # 5. Rough estimate of area/length
        # U_tpms ~ 500 - 2000 W/m2K based on typical cryo-TPMS studies
        u_guess = 1000.0
        area_guess = required_ua / u_guess

        # Length ~ Area / (Width * Height * Sigma)
        geom = cfg['geometry']
        sigma = geom['surface_area_density']
        length_est = area_guess / (geom['width'] * geom['height'] * sigma)

        return {
            'length': length_est,
            'ua': required_ua,
            'ntu': ntu,
            'epsilon': epsilon_actual
        }

    def solve_geometry(self, design_variable, target_metric, target_value, bounds=None, tol=1e-2, verbose=False):
        """
        Calculates the required geometric dimension to meet a thermal target.
        If bounds are not provided, uses epsilon-NTU to generate a smart search range.
        """

        param_map = {
            'length': ['geometry', 'length'],
            'width': ['geometry', 'width'],
            'height': ['geometry', 'height'],
            'cell_size': ['geometry', 'unit_cell_size']
        }

        if design_variable not in param_map:
            raise ValueError(f"Design variable must be one of {list(param_map.keys())}")

        config_path = param_map[design_variable]

        # --- SMART BOUNDARY ESTIMATION ---
        if bounds is None:
            est = self.estimate_size_epsilon_ntu(target_metric, target_value)
            if est and design_variable == 'length':
                # Create a bracket around the estimate (0.2x to 5.0x)
                bounds = (max(0.01, est['length'] * 0.2), est['length'] * 5.0)
            else:
                # Default fallback if estimate fails or isn't length
                bounds = (0.01, 5.0)

        if verbose:
            print(f"\n--- TPMS Design Solver Started ---")
            print(f"Goal: Adjust {design_variable} to achieve {target_metric} = {target_value}")
            print(f"Initial Epsilon-NTU Estimate for Length: {bounds[0]:.3f} to {bounds[1]:.3f} m")

        def objective(x):
            sim_config = copy.deepcopy(self.base_config)
            self._update_nested_config(sim_config, config_path, x)

            with contextlib.redirect_stdout(DummyFile()):
                he = TPMSHeatExchanger(sim_config)
                converged = he.solve()

            if not converged:
                return 1e9

            current_val = self._get_result_value(he, target_metric)
            abs_error = abs(current_val - target_value)

            if verbose:
                sys.__stdout__.write(f"  Trial {design_variable} = {x:.5f} -> {target_metric} = {current_val:.4f} (Abs Error: {abs_error:.4f})\n")

            return abs_error

        try:
            res = minimize_scalar(
                objective,
                bounds=bounds,
                method='bounded',
                options={'xatol': 1e-4, 'maxiter': 30}
            )

            if res.success and res.fun <= 1.0:
                print(f"SUCCESS: Design Converged. Required {design_variable}: {res.x:.5f} m")
                return res.x
            else:
                print(f"WARNING: Residual error ({res.fun:.3f}) may be high.")
                return res.x

        except Exception as e:
            print(f"Optimization failed: {e}")
            return None

if __name__ == "__main__":
    from tpms_thermo_hydraulic_calculator import create_default_config
    config = create_default_config()
    config['operating']['Th_in'] = 78.0

    designer = TPMSDesigner(config)
    required_length = designer.solve_geometry(
        design_variable='length',
        target_metric='Th_out',
        target_value=55.0,
        verbose=True
    )