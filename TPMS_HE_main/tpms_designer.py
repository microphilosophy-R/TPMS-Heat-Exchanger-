"""
TPMS Heat Exchanger Designer
Description: Inverse solver that calculates required geometry (Length, Width, Cell Size)
to achieve specific outlet conditions.

Content:
1. TPMSDesigner: Wrapper class that optimizes geometry to meet thermal targets.
"""

import numpy as np
from scipy.optimize import minimize_scalar, fsolve
import copy
import sys
import os
import contextlib

# Import the main simulation class and dependencies
try:
    from tpms_thermo_hydraulic_calculator import TPMSHeatExchanger
    from tpms_correlations import TPMSCorrelations
except ImportError:
    raise ImportError("Ensure 'tpms_thermo_hydraulic_calculator.py' and 'tpms_correlations.py' are in the same directory.")

class DummyFile:
    """A robust dummy file handler that ignores all input."""
    def write(self, x): pass
    def flush(self): pass
    def close(self): pass

class TPMSDesigner:
    """
    Design-stage solver for TPMS Heat Exchangers.
    Uses a two-stage approach:
    Stage 1: Fast UA-matching using analytical physics (fsolve).
    Stage 2: Precise thermo-hydraulic refinement (minimize_scalar).
    """

    def __init__(self, base_config):
        self.base_config = copy.deepcopy(base_config)

    def _update_nested_config(self, config, path_list, value):
        d = config
        for key in path_list[:-1]:
            d = d[key]
        d[path_list[-1]] = value

    def _get_result_value(self, he_instance, target_key):
        if target_key == 'Th_out':
            return he_instance.Th[-1]
        elif target_key == 'Tc_out':
            return he_instance.Tc[0]
        elif target_key == 'Q_total':
            return np.sum(he_instance.Q)
        else:
            raise ValueError(f"Target key {target_key} not supported for geometry optimization.")

    def calculate_UA_precisely(self, x, design_variable):
        """
        Calculates the theoretical UA value for a given geometry parameter x
        by evaluating the local HTC and resistances at inlet conditions.

        This reflects the logic in _update_stream_physics and _compute_energy_balance.
        """
        cfg = copy.deepcopy(self.base_config)

        # Map variable to config
        param_map = {'length': 'length', 'width': 'width', 'unit_cell_size': 'unit_cell_size'}
        if design_variable in param_map:
            cfg['geometry'][param_map[design_variable]] = x

        g = cfg['geometry']
        ops = cfg['operating']

        # 1. Geometry derived parameters
        sigma = g['surface_area_density']
        area_total = g['length'] * g['width'] * g['height'] * sigma

        # 2. Physics evaluation at mean conditions (inlet-based approximation)
        he_temp = TPMSHeatExchanger(cfg)
        h2_props = he_temp.h2_props

        # Evaluate H and C streams
        ua_sum = 0
        for stream_key in ['hot', 'cold']:
            s = he_temp.streams[stream_key]
            # Use mean temperature approximation for fast UA
            T_mean = (ops['Th_in'] + ops['Tc_in']) / 2.0
            P_in = ops[f'P{"h" if stream_key=="hot" else "c"}_in']
            x_p = ops.get('xh_in', None) if stream_key == 'hot' else None

            p = h2_props.get_properties(T_mean, P_in, s['species'], x_p)

            # Local Re, Pr
            u = s['m'] / (p['rho'] * s['Ac'])
            Re = p['rho'] * u * s['Dh'] / p['mu']
            Pr = p['mu'] * p['cp'] / p['lambda']

            Nu, _ = TPMSCorrelations.get_correlations(s['tpms'], Re, Pr, 'Gas')
            # Store resistance
            s['R_conv'] = 1.0 / (Nu * p['lambda'] / s['Dh'])

        # 3. Overall Resistance
        R_wall = g['wall_thickness'] / cfg['material']['k_wall']
        # Based on _compute_energy_balance logic: R_total = 1/h_h + R_wall + 1/h_c
        R_total = (1.2 * he_temp.streams['hot']['R_conv']) + R_wall + he_temp.streams['cold']['R_conv']
        U = 1.0 / R_total

        return U * area_total

    def get_target_UA(self, target_metric, target_value):
        """Calculates the target UA required using the epsilon-NTU method."""
        ops = self.base_config['operating']
        he_temp = TPMSHeatExchanger(self.base_config)
        h2_props = he_temp.h2_props

        p_h = h2_props.get_properties(ops['Th_in'], ops['Ph_in'], self.base_config['operating'].get('fluid_hot', 'hydrogen mixture'), ops['xh_in'])
        p_c = h2_props.get_properties(ops['Tc_in'], ops['Pc_in'], self.base_config['operating'].get('fluid_cold', 'helium'))

        Ch, Cc = ops['mh'] * p_h['cp'], ops['mc'] * p_c['cp']
        Cmin, Cmax = min(Ch, Cc), max(Ch, Cc)
        Cr = Cmin / Cmax

        Q_max = Cmin * (ops['Th_in'] - ops['Tc_in'])
        if target_metric == 'Th_out':
            Q_req = Ch * (ops['Th_in'] - target_value)
        elif target_metric == 'Tc_out':
            Q_req = Cc * (target_value - ops['Tc_in'])
        elif target_metric == 'Q_total':
            Q_req = target_value
        else: return 0

        epsilon = min(0.999, Q_req / Q_max)

        try:
            if Cr < 1.0:
                ntu = (1.0 / (Cr - 1.0)) * np.log((epsilon - 1.0) / (epsilon * Cr - 1.0))
            else:
                ntu = epsilon / (1.0 - epsilon)
        except: ntu = 10.0 # Upper limit

        return ntu * Cmin

    def solve_geometry(self, design_variable, target_metric, target_value, verbose=False):
        """
        Two-stage design solver:
        Stage 1: Precise UA matching using analytical physics and fsolve.
        Stage 2: Refinement using the full iterative simulation.
        """
        param_map = {
            'length': ['geometry', 'length'],
            'width': ['geometry', 'width'],
            'cell_size': ['geometry', 'unit_cell_size']
        }
        config_path = param_map[design_variable]

        # --- STAGE 1: FAST ANALYTICAL SEARCH ---
        target_ua = self.get_target_UA(target_metric, target_value)

        def ua_residual(x):
            return self.calculate_UA_precisely(x, design_variable) - target_ua

        # Initial guess from current config
        x0 = self.base_config['geometry'].get(param_map[design_variable], 0.1)
        x_analytical = fsolve(ua_residual, x0)[0]
        x_analytical = max(1e-4, x_analytical) # Physical guard

        if verbose:
            print(f"\n--- TPMS Design Solver ---")
            print(f"Goal: {target_metric} = {target_value} (Target UA: {target_ua:.2f} W/K)")
            print(f"Stage 1 (Analytical): Found {design_variable} = {x_analytical:.5f} m")

        # --- STAGE 2: PRECISE THERMO-HYDRAULIC REFINEMENT ---
        # Narrow bracket around analytical solution (+/- 25%)
        bounds = (x_analytical * 0.75, x_analytical * 1.5)

        def objective(x):
            sim_config = copy.deepcopy(self.base_config)
            self._update_nested_config(sim_config, config_path, x)
            with contextlib.redirect_stdout(DummyFile()):
                he = TPMSHeatExchanger(sim_config)
                # Run with moderate tolerance for speed in inner loop
                converged = he.solve(max_iter=100, tolerance=5e-4)

            if not converged: return 1e9

            err = abs(self._get_result_value(he, target_metric) - target_value)
            if verbose:
                sys.__stdout__.write(f"  Stage 2 Trial {design_variable} = {x:.5f} -> Error: {err:.4f}\n")
            return err

        try:
            res = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-4, 'maxiter': 15})
            print(f"SUCCESS: Final {design_variable} = {res.x:.5f} m (Error: {res.fun:.4f})")
            return res.x
        except Exception as e:
            print(f"Stage 2 failed: {e}. Returning Stage 1 estimate.")
            return x_analytical

if __name__ == "__main__":
    from tpms_thermo_hydraulic_calculator import create_default_config
    config = create_default_config()
    designer = TPMSDesigner(config)
    designer.solve_geometry('length', 'Th_out', 55.0, verbose=True)