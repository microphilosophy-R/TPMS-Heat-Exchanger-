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

    def estimate_geometry_ntu(self, design_variable, target_metric, target_value):
        """
        Generic Epsilon-NTU estimator for any geometric variable.
        Calculates the required change in UA to meet the target and maps it to
        the design variable (Length, Width, or Cell Size).
        """
        cfg = self.base_config
        ops = cfg['operating']
        geom = cfg['geometry']

        # 1. Properties
        he_temp = TPMSHeatExchanger(cfg)
        h2_props = he_temp.h2_props
        p_h = h2_props.get_properties(ops['Th_in'], ops['Ph_in'], cfg['operating'].get('fluid_hot', 'hydrogen mixture'), ops['xh_in'])
        p_c = h2_props.get_properties(ops['Tc_in'], ops['Pc_in'], cfg['operating'].get('fluid_cold', 'helium'))

        Ch = ops['mh'] * p_h['cp']
        Cc = ops['mc'] * p_c['cp']
        Cmin, Cmax = min(Ch, Cc), max(Ch, Cc)
        Cr = Cmin / Cmax

        # 2. Required Effectiveness
        Q_max = Cmin * (ops['Th_in'] - ops['Tc_in'])
        if target_metric == 'Th_out':
            Q_req = Ch * (ops['Th_in'] - target_value)
        elif target_metric == 'Tc_out':
            Q_req = Cc * (target_value - ops['Tc_in'])
        elif target_metric == 'Q_total':
            Q_req = target_value
        else: return None

        epsilon = min(0.99, Q_req / Q_max)

        # 3. NTU -> UA
        try:
            if Cr < 1.0:
                ntu = (1.0 / (Cr - 1.0)) * np.log((epsilon - 1.0) / (epsilon * Cr - 1.0))
            else:
                ntu = epsilon / (1.0 - epsilon)
        except: ntu = 5.0

        UA_req = ntu * Cmin

        # 4. Sensitivity Mapping
        # Assume UA proportional to Area (L*W*H) and inversely to Dh (Cell Size)
        # Dh ~ CellSize. A ~ sigma * L * W * H
        current_val = geom.get(design_variable, 0.01)
        if design_variable == 'length':
            est_val = UA_req / (1000.0 * geom['width'] * geom['height'] * geom['surface_area_density'])
        elif design_variable == 'width':
            est_val = UA_req / (1000.0 * geom['length'] * geom['height'] * geom['surface_area_density'])
        elif design_variable == 'unit_cell_size':
            # Smaller cell size increases sigma and improves HTC (U)
            # Rough approximation: UA ~ 1 / CellSize^1.5
            est_val = current_val * (UA_req / (1000.0 * geom['length'] * geom['width'] * geom['height'] * geom['surface_area_density']))**(-0.66)
        else:
            est_val = current_val

        return est_val

    def solve_geometry(self, design_variable, target_metric, target_value, bounds=None, verbose=False):
        """
        Two-stage solver:
        1. Fast NTU Search: Uses thermal imbalance and epsilon-NTU logic to narrow the search.
        2. Precise Simulation: Refines the value using the full thermo-hydraulic solver.
        """
        param_map = {
            'length': ['geometry', 'length'],
            'width': ['geometry', 'width'],
            'height': ['geometry', 'height'],
            'cell_size': ['geometry', 'unit_cell_size']
        }
        config_path = param_map[design_variable]

        # --- STAGE 1: FAST ESTIMATION ---
        est_x = self.estimate_geometry_ntu(design_variable, target_metric, target_value)

        if bounds is None:
            # Use estimate to define a smart bracket
            bounds = (max(1e-4, est_x * 0.1), est_x * 10.0)

        if verbose:
            print(f"\n--- TPMS Design Solver Initialized ---")
            print(f"Target: {target_metric} = {target_value}")
            print(f"Stage 1: NTU Estimate for {design_variable} = {est_x:.4f} m")

        # --- STAGE 2: PRECISE ITERATION (Full Physics) ---
        def objective(x):
            sim_config = copy.deepcopy(self.base_config)
            self._update_nested_config(sim_config, config_path, x)

            with contextlib.redirect_stdout(DummyFile()):
                he = TPMSHeatExchanger(sim_config)
                # Faster tolerances for inner optimization loops
                converged = he.solve(max_iter=150, tolerance=1e-3)

            if not converged:
                return 1e9

            current_val = self._get_result_value(he, target_metric)
            abs_error = abs(current_val - target_value)

            if verbose:
                sys.__stdout__.write(f"  Precise Trial {design_variable} = {x:.5f} -> {target_metric} = {current_val:.4f} (Error: {abs_error:.4f})\n")

            return abs_error

        try:
            res = minimize_scalar(
                objective,
                bounds=bounds,
                method='bounded',
                options={'xatol': 1e-4, 'maxiter': 25} # Reduced iterations due to better start
            )

            if res.success:
                print(f"SUCCESS: Design Converged. Final {design_variable}: {res.x:.5f} m")
                return res.x
            else:
                print(f"WARNING: Residual error ({res.fun:.3f}) high.")
                return res.x

        except Exception as e:
            print(f"Optimization failed: {e}")
            return None

if __name__ == "__main__":
    from tpms_thermo_hydraulic_calculator import create_default_config
    config = create_default_config()

    # Example: Find length for Th_out = 55K
    designer = TPMSDesigner(config)
    designer.solve_geometry('length', 'Th_out', 55.0, verbose=True)