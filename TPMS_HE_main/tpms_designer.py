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
    """A robust dummy file handler that ignores all input.
    Used to suppress output without triggering encoding errors (e.g. gbk/utf-8 issues with H₂ subscripts)."""
    def write(self, x): pass
    def flush(self): pass
    def close(self): pass

class TPMSDesigner:
    """
    Design-stage solver for TPMS Heat Exchangers.
    Uses optimization algorithms to determine geometric parameters
    required to meet specific process targets.
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
        """Helper to update deep dictionary keys (e.g., ['geometry', 'length'])"""
        d = config
        for key in path_list[:-1]:
            d = d[key]
        d[path_list[-1]] = value

    def _get_result_value(self, he_instance, target_key):
        """Extracts the specific result metric from a solved heat exchanger instance."""
        if target_key == 'Th_out':
            return he_instance.Th[-1]  # Hot outlet (end of array)
        elif target_key == 'Tc_out':
            return he_instance.Tc[0]   # Cold outlet (start of array)
        elif target_key == 'Q_total':
            return np.sum(he_instance.Q)
        elif target_key == 'dP_hot':
            return he_instance.Ph[0] - he_instance.Ph[-1]
        elif target_key == 'dP_cold':
            return he_instance.Pc[-1] - he_instance.Pc[0]
        else:
            raise ValueError(f"Unknown target key: {target_key}")

    def solve_geometry(self, design_variable, target_metric, target_value, bounds, tol=1e-2, verbose=False):
        """
        Calculates the required geometric dimension to meet a thermal target using
        bounded scalar minimization. This is more robust than root finding when
        simulations at the bounds might fail.

        Parameters
        ----------
        design_variable : str
            The parameter to adjust. Options: 'length', 'width', 'height', 'cell_size'
        target_metric : str
            The goal to hit. Options: 'Th_out', 'Tc_out', 'Q_total'
        target_value : float
            The specific value required (e.g., 300.0 for 300 K outlet)
        bounds : tuple
            (min_val, max_val) Range to search for the design variable.
        tol : float
            Tolerance for the target metric.

        Returns
        -------
        float : The calculated value for the design_variable.
        """

        # Map friendly names to config paths
        param_map = {
            'length': ['geometry', 'length'],
            'width': ['geometry', 'width'],
            'height': ['geometry', 'height'],
            'cell_size': ['geometry', 'unit_cell_size']
        }

        if design_variable not in param_map:
            raise ValueError(f"Design variable must be one of {list(param_map.keys())}")

        config_path = param_map[design_variable]

        if verbose:
            print(f"\n--- TPMS Design Solver Started ---")
            print(f"Goal: Adjust {design_variable} to achieve {target_metric} = {target_value}")
            print(f"Search Bounds: {bounds}")

        # Objective Function for Minimization
        # Minimize |Simulated_Value(x) - Target_Value|

        def objective(x):
            # 1. Create temporary config
            sim_config = copy.deepcopy(self.base_config)
            self._update_nested_config(sim_config, config_path, x)

            # 2. Run Simulation (Suppress output to avoid encoding errors)
            # We use DummyFile() instead of os.devnull to bypass 'gbk' encoding crashes
            with contextlib.redirect_stdout(DummyFile()):
                he = TPMSHeatExchanger(sim_config)
                converged = he.solve()

            # 3. Handle Result
            if not converged:
                # Return a large penalty so the optimizer avoids this region
                # but keep it finite to avoid crashing the optimizer
                return 1e9

            current_val = self._get_result_value(he, target_metric)
            abs_error = abs(current_val - target_value)

            if verbose:
                # We write directly to sys.__stdout__ to ensure the progress is seen
                # regardless of any other redirection
                sys.__stdout__.write(f"  Trial {design_variable} = {x:.5f} -> {target_metric} = {current_val:.4f} (Abs Error: {abs_error:.4f})\n")

            return abs_error

        # Run Optimization using Bounded Minimization
        # 'bounded' method is robust and does not require a sign change (bracketing)
        try:
            res = minimize_scalar(
                objective,
                bounds=bounds,
                method='bounded',
                options={'xatol': 1e-4, 'maxiter': 50}
            )

            if res.success and res.fun <= tol * 10: # Allow slight slack in objective check
                print(f"SUCCESS: Design Converged.")
                print(f"  Required {design_variable}: {res.x:.5f}")
                print(f"  Final Error: {res.fun:.5f}")
                return res.x
            else:
                print(f"WARNING: Optimizer finished but target may not be fully met.")
                print(f"  Best {design_variable}: {res.x:.5f}")
                print(f"  Residual Error: {res.fun:.5f}")

                # If error is reasonably small (e.g. < 0.5 unit), accept it
                if res.fun < 0.5:
                    print("  -> Accepting best result as approximate solution.")
                    return res.x
                else:
                    print("  -> Solution rejected (error too high). Check bounds or physics.")
                    return None

        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    from tpms_thermo_hydraulic_calculator import create_default_config

    # 1. Load Baseline Config
    config = create_default_config()

    # 2. Set Up Design Scenario
    # Example: We want to cool Hydrogen from 78K to exactly 50K.
    config['operating']['Th_in'] = 78.0
    config['operating']['fluid_hot'] = 'hydrogen mixture'

    target_temp = 50.0  # K

    # 3. Initialize Designer
    designer = TPMSDesigner(config)

    # 4. Run Solver
    # Use a wide bound. Even if the upper bound diverges (returns error 1e9),
    # minimize_scalar will safely ignore it and search the valid region.
    required_length = designer.solve_geometry(
        design_variable='length',
        target_metric='Th_out',
        target_value=target_temp,
        bounds=(0.1, 12.0),
        verbose=True
    )

    if required_length:
        print("\n--- Verification Simulation ---")
        config['geometry']['length'] = required_length
        he_final = TPMSHeatExchanger(config)
        he_final.solve()