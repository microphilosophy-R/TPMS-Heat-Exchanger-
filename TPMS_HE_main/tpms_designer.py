"""
TPMS Heat Exchanger Designer
Description: Inverse solver that calculates required geometry (Length, Width, Cell Size)
to achieve specific outlet conditions using a 3-stage calibrated workflow.

Content:
1. TPMSDesigner: Improved wrapper with Stage 0 Calibration logic.
"""

import numpy as np
from scipy.optimize import minimize_scalar, fsolve
import copy
import sys
import os
import contextlib

# Import dependencies
try:
    from tpms_correlations import TPMSCorrelations
except ImportError:
    raise ImportError("Ensure 'tpms_correlations.py' is present.")

# Delayed import to avoid circular imports
TPMSHeatExchanger = None
try:
    import importlib
    tpms_module = importlib.import_module('tpms_thermo_hydraulic_calculator')
    TPMSHeatExchanger = getattr(tpms_module, 'TPMSHeatExchanger')
except ImportError:
    raise ImportError("Ensure 'tpms_thermo_hydraulic_calculator.py' is present.")
except AttributeError:
    raise ImportError("Could not find TPMSHeatExchanger class in tpms_thermo_hydraulic_calculator.py")

class DummyFile:
    """A robust dummy file handler to suppress terminal spam during iterations."""
    def write(self, x): pass
    def flush(self): pass
    def close(self): pass

class TPMSDesigner:
    """
    Design-stage solver for TPMS Heat Exchangers.
    Workflow:
    Stage 0: Calibration - Run current config to find real U_eff.
    Stage 1: Analytical - fsolve using calibrated physics.
    Stage 2: Precision - Final refinement with full solver.
    """

    def __init__(self, base_config):
        self.base_config = copy.deepcopy(base_config)
        self.u_correction_factor = 1.0  # Initial assumption

    def _update_nested_config(self, config, path_list, value):
        d = config
        for key in path_list[:-1]:
            d = d[key]
        d[path_list[-1]] = value

    def _get_nested_config_value(self, config, path_list):
        d = config
        for key in path_list:
            d = d[key]
        return d

    def _get_result_value(self, he_instance, target_key):
        if target_key == 'Th_out':
            return he_instance.Th[-1]
        elif target_key == 'Tc_out':
            return he_instance.Tc[0]
        elif target_key == 'Q_total':
            return np.sum(he_instance.Q)
        else:
            raise ValueError(f"Target key {target_key} not supported.")

    def calibrate_analytical_model(self, verbose=False):
        """
        STAGE 0: Run the baseline config and calculate the real U achieved.
        This correction factor aligns Stage 1 with the actual Stage 2 physics.
        It runs the TPMSHeatExchanger solver to ensure valid outlet data is available.
        """
        if verbose:
            print("Stage 0: Calibrating analytical model with baseline configuration...")

        # We must run the solver to get the actual state (outlet temps, etc.)
        with contextlib.redirect_stdout(DummyFile()):
            he_base = TPMSHeatExchanger(self.base_config)
            converged = he_base.solve(max_iter=500, tolerance=1e-4)

        if not converged:
            if verbose: print("  Warning: Baseline did not converge. Using default calibration.")
            return

        # Calculate achieved UA from simulation using Logarithmic Mean Temperature Difference (LMTD)
        # Hot stream: 0 (in) -> N (out) | Cold stream: N (in) -> 0 (out)
        Th_in, Th_out = he_base.Th[0], he_base.Th[-1]
        Tc_in, Tc_out = he_base.Tc[-1], he_base.Tc[0]
        Q_sim = np.sum(he_base.Q)

        dt1 = Th_in - Tc_out
        dt2 = Th_out - Tc_in

        # Calculate LMTD with numerical guards
        if dt1 <= 0 or dt2 <= 0:
            lmtd = max(1e-3, (dt1 + dt2) / 2)
        elif abs(dt1 - dt2) < 1e-4:
            lmtd = dt1
        else:
            lmtd = (dt1 - dt2) / np.log(dt1 / dt2)

        ua_sim = Q_sim / max(lmtd, 1e-3)

        # Calculate what the analytical model PREDICTS for this same configuration
        # Temporarily reset correction factor to 1.0 to find the raw prediction
        self.u_correction_factor = 1.0
        current_length = self.base_config['geometry']['length']
        ua_analytical_raw = self.calculate_UA_precisely(current_length, 'length')

        # Update Correction Factor = Reality / Analytical_Prediction
        self.u_correction_factor = ua_sim / max(ua_analytical_raw, 1e-5)

        if verbose:
            print(f"   Baseline Simulation Results:")
            print(f"     Th_out: {Th_out:.2f} K | Tc_out: {Tc_out:.2f} K | Q: {Q_sim:.2f} W")
            print(f"     Achieved UA: {ua_sim:.2f} W/K | Predicted UA: {ua_analytical_raw:.2f} W/K")
            print(f"   Analytical Model Correction Factor: {self.u_correction_factor:.4f}")

    def calculate_UA_precisely(self, x, design_variable):
        """
        Analytical UA calculation using integrated properties
        and the Stage 0 correction factor.
        """
        cfg = copy.deepcopy(self.base_config)
        param_map = {
            'length': ['geometry', 'length'],
            'width': ['geometry', 'width'],
            'cell_size': ['geometry', 'unit_cell_size']
        }

        if design_variable in param_map:
            self._update_nested_config(cfg, param_map[design_variable], x)

        g = cfg['geometry']
        ops = cfg['operating']
        sigma = g['surface_area_density']
        area_total = g['length'] * g['width'] * g['height'] * sigma

        he_temp = TPMSHeatExchanger(cfg)
        h2_props = he_temp.h2_props

        # Evaluate properties at 3 points to capture cryo non-linearity
        T_h_range = np.linspace(ops['Th_in'], (ops['Th_in'] + ops['Tc_in'])/2, 3)
        T_c_range = np.linspace((ops['Th_in'] + ops['Tc_in'])/2, ops['Tc_in'], 3)

        U_samples = []
        for i in range(3):
            # Hot side
            p_h = h2_props.get_properties(T_h_range[i], ops['Ph_in'], he_temp.streams['hot']['species'], ops['xh_in'])
            u_h = he_temp.streams['hot']['m'] / (p_h['rho'] * he_temp.streams['hot']['Ac'])
            Re_h = p_h['rho'] * u_h * he_temp.streams['hot']['Dh'] / p_h['mu']
            Pr_h = p_h['mu'] * p_h['cp'] / p_h['lambda']
            Nu_h, _ = TPMSCorrelations.get_correlations(he_temp.streams['hot']['tpms'], Re_h, Pr_h, 'Gas')
            h_hot = 1.2 * Nu_h * p_h['lambda'] / he_temp.streams['hot']['Dh']

            # Cold side
            p_c = h2_props.get_properties(T_c_range[i], ops['Pc_in'], he_temp.streams['cold']['species'])
            u_c = he_temp.streams['cold']['m'] / (p_c['rho'] * he_temp.streams['cold']['Ac'])
            Re_c = p_c['rho'] * u_c * he_temp.streams['cold']['Dh'] / p_c['mu']
            Pr_c = p_c['mu'] * p_c['cp'] / p_c['lambda']
            Nu_c, _ = TPMSCorrelations.get_correlations(he_temp.streams['cold']['tpms'], Re_c, Pr_c, 'Gas')
            h_cold = Nu_c * p_c['lambda'] / he_temp.streams['cold']['Dh']

            R_wall = g['wall_thickness'] / cfg['material']['k_wall']
            U_samples.append(1.0 / ((1/max(h_hot, 1e-3)) + R_wall + (1/max(h_cold, 1e-3))))

        # Apply the correction factor derived during Stage 0 calibration
        U_avg = np.mean(U_samples) * self.u_correction_factor
        return U_avg * area_total

    def get_target_UA(self, target_metric, target_value):
        """Calculates the target UA using epsilon-NTU logic."""
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
            if Cr < 0.998:
                ntu = (1.0 / (Cr - 1.0)) * np.log((epsilon - 1.0) / (epsilon * Cr - 1.0))
            else:
                ntu = epsilon / (1.0 - epsilon)
        except: ntu = 30.0

        return ntu * Cmin

    def solve_geometry(self, design_variable, target_metric, target_value, verbose=False):
        """3-Stage Design Solver Workflow."""
        param_map = {
            'length': ['geometry', 'length'],
            'width': ['geometry', 'width'],
            'cell_size': ['geometry', 'unit_cell_size']
        }
        config_path = param_map[design_variable]

        print(f"\n--- TPMS Design Solver [3-Stage Mode] ---")
        print(f"Target: {target_metric} = {target_value}")

        # --- STAGE 0: CALIBRATION ---
        # Run baseline to find real U_eff and correct Stage 1 physics
        self.calibrate_analytical_model(verbose=verbose)

        # --- STAGE 1: CALIBRATED ANALYTICAL SEARCH ---
        target_ua = self.get_target_UA(target_metric, target_value)

        def ua_residual(x):
            if x <= 0: return 1e12
            return self.calculate_UA_precisely(x, design_variable) - target_ua

        x0 = self._get_nested_config_value(self.base_config, config_path)
        try:
            x_analytical = fsolve(ua_residual, x0, xtol=1e-5)[0]
        except:
            x_analytical = x0

        x_analytical = max(1e-4, x_analytical)

        if verbose:
            print(f"Stage 1: Calibrated model suggests {design_variable} = {x_analytical:.6f} m")

        # --- STAGE 2: PRECISE REFINEMENT ---
        # Search bounds narrowed around calibrated analytical estimate
        bounds = (x_analytical * 0.8, x_analytical * 1.5)

        def objective(x):
            sim_config = copy.deepcopy(self.base_config)
            self._update_nested_config(sim_config, config_path, x)
            with contextlib.redirect_stdout(DummyFile()):
                he = TPMSHeatExchanger(sim_config)
                converged = he.solve(max_iter=300, tolerance=1e-4)

            if not converged: return 1e8

            current_val = self._get_result_value(he, target_metric)
            error = abs(current_val - target_value)

            if verbose:
                sys.__stdout__.write(f"  Stage 2 Trial | {design_variable} = {x:.6f} | Err: {error:.4f}\n")
            return error

        try:
            res = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-5, 'maxiter': 15})
            print(f"\nDESIGN OPTIMIZATION COMPLETE:")
            print(f"  Final {design_variable}: {res.x:.6f} m")
            print(f"  Final Error: {res.fun:.6f} K")
            return res.x
        except Exception as e:
            print(f"Stage 2 refinement failed: {e}")
            return x_analytical

if __name__ == "__main__":
    from tpms_thermo_hydraulic_calculator import create_default_config
    config = create_default_config()

    # Run the designer
    designer = TPMSDesigner(config)
    designer.solve_geometry('length', 'Th_out', 55.0, verbose=True)