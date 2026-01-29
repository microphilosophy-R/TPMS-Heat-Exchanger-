"""
TPMS Heat Exchanger Designer
Description: Optimized inverse solver that determines geometric parameters
(Length, Width, Cell Size) to achieve specific thermal targets.

NON-LINEARITY NOTICE:
This system exhibits high non-linearity due to:
1. Cryogenic property shifts (Hydrogen/Helium properties vary significantly with T).
2. Velocity-dependent HTC (U is not constant; changing Width/Height changes Re).
3. Feedback loops in the nodal thermo-hydraulic solver.
"""

import numpy as np
import copy
import sys
import os
import contextlib

# Import dependencies
try:
    from tpms_thermo_hydraulic_calculator import TPMSHeatExchanger
except ImportError:
    raise ImportError("Ensure 'tpms_thermo_hydraulic_calculator.py' is present in the directory.")

class DummyFile:
    """A robust dummy file handler to suppress terminal output during heavy iterations."""
    def write(self, x): pass
    def flush(self): pass
    def close(self): pass

class TPMSDesigner:
    """
    Design-stage solver for TPMS Heat Exchangers.
    Uses a Damped Secant method to navigate the highly non-linear objective space
    created by cryogenic property variations and flow physics.
    """

    def __init__(self, base_config):
        self.base_config = copy.deepcopy(base_config)

    def _update_nested_config(self, config, path_list, value):
        """Helper to update deep dictionary keys."""
        d = config
        for key in path_list[:-1]:
            d = d[key]
        d[path_list[-1]] = value

    def _get_nested_config_value(self, config, path_list):
        """Helper to safely retrieve a nested configuration value."""
        d = config
        for key in path_list:
            d = d[key]
        return d

    def _get_result_value(self, he_instance, target_key):
        """Extracts the specific result metric from a solved heat exchanger instance."""
        if target_key == 'Th_out':
            return he_instance.Th[-1]
        elif target_key == 'Tc_out':
            return he_instance.Tc[0]
        elif target_key == 'Q_total':
            return np.sum(he_instance.Q)
        else:
            raise ValueError(f"Target key {target_key} not supported.")

    def run_simulation(self, x, config_path, verbose=False):
        """Helper to run a single full physics simulation and return the error."""
        sim_config = copy.deepcopy(self.base_config)
        self._update_nested_config(sim_config, config_path, x)

        with contextlib.redirect_stdout(DummyFile()):
            he = TPMSHeatExchanger(sim_config)
            # Higher max_iter to handle the sensitivity of non-linear property regions
            converged = he.solve(max_iter=600, tolerance=1e-4)

        if not converged:
            return None, 1e9

        return he, 0

    def solve_geometry(self, design_variable, target_metric, target_value, max_iter=15, tol=1e-3, verbose=True):
        """
        Calculates required geometry using a Damped Secant method.

        This method is chosen because the system is highly non-linear. Standard linear
        secant methods often overshoot (especially when adjusting Width/Height),
        so we implement damping and physical guarding.
        """
        param_map = {
            'length': ['geometry', 'length'],
            'width': ['geometry', 'width'],
            'height': ['geometry', 'height'],
            'cell_size': ['geometry', 'unit_cell_size']
        }

        if design_variable not in param_map:
            raise ValueError(f"Invalid design variable. Choose from {list(param_map.keys())}")

        config_path = param_map[design_variable]

        if verbose:
            print(f"\n--- TPMS Non-Linear Design Solver ---")
            print(f"Target: {target_metric} = {target_value}")
            print(f"Warning: System is highly non-linear due to velocity-sensitive HTC.")

        # --- STEP 1: INITIAL STATE (Point 0) ---
        x0 = self._get_nested_config_value(self.base_config, config_path)
        he0, status = self.run_simulation(x0, config_path)
        if status != 0:
            raise RuntimeError("Initial baseline simulation failed to converge.")

        y0 = self._get_result_value(he0, target_metric)
        err0 = y0 - target_value

        if verbose:
            print(f"  [0] {design_variable} = {x0:.4f} | Result = {y0:.3f} | Error = {err0:.3f}")

        if abs(err0) < tol:
            return x0

        # --- STEP 2: SMALL PERTURBATION (Point 1) ---
        # Using a small 5% perturbation to capture local gradient in non-linear space
        perturb = 0.05 if err0 > 0 else -0.05
        x1 = x0 * (1.0 + perturb)

        he1, status = self.run_simulation(x1, config_path)
        if status != 0:
            x1 = x0 * (1.0 + perturb * 0.2) # Try a much smaller step
            he1, _ = self.run_simulation(x1, config_path)

        y1 = self._get_result_value(he1, target_metric)
        err1 = y1 - target_value

        if verbose:
            print(f"  [1] {design_variable} = {x1:.4f} | Result = {y1:.3f} | Error = {err1:.3f}")

        # --- STEP 3: DAMPED SECANT ITERATIONS ---
        # Damping factor reduces the "jump" distance to prevent divergence in Width/Height adjustments
        damping = 0.7

        for i in range(2, max_iter + 2):
            denom = err1 - err0
            if abs(denom) < 1e-8:
                print("  Convergence stalled: Zero gradient.")
                break

            # Raw step
            delta_x = err1 * (x1 - x0) / denom

            # Apply Damping for non-linear stability
            x_new = x1 - damping * delta_x

            # Physical bounds checking (e.g., width cannot be infinite or zero)
            # Higher non-linearity in Width/Height requires tighter bounding
            x_min, x_max = 0.005, 5.0
            x_new = max(x_min, min(x_new, x_max))

            he_new, status = self.run_simulation(x_new, config_path)
            if status != 0:
                print(f"  Solver failed at x={x_new:.4f} (likely low-velocity regime). Retrying with more damping.")
                damping *= 0.5
                continue

            y_new = self._get_result_value(he_new, target_metric)
            err_new = y_new - target_value

            if verbose:
                print(f"  [{i}] {design_variable} = {x_new:.4f} | Result = {y_new:.3f} | Error = {err_new:.3f}")

            if abs(err_new) < tol:
                print(f"\nSUCCESS: Solution found in {i} iterations.")
                return x_new

            # Update points
            x0, err0 = x1, err1
            x1, err1 = x_new, err_new

            # Adaptive damping: if error increased, increase damping
            if abs(err_new) > abs(err0):
                damping *= 0.8

        print("\nWARNING: Max iterations reached. System non-linearity may be high.")
        return x1

if __name__ == "__main__":
    from tpms_thermo_hydraulic_calculator import create_default_config
    config = create_default_config()

    designer = TPMSDesigner(config)
    # Testing length as a baseline
    designer.solve_geometry('height', 'Th_out', 56.0, verbose=True)