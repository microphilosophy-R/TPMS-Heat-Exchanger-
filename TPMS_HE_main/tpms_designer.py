"""
TPMS Heat Exchanger Designer
Description: Optimized inverse solver that determines geometric parameters
(Length, Width, Cell Size) to achieve specific thermal targets.

DESIGN OBJECTIVE:
This designer finds the specific geometric parameter 'x' (e.g., Length) starting from
an initial guess 'y' that satisfies a target outlet temperature condition (typically Th_out).

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
import matplotlib.pyplot as plt

# Import dependencies
try:
    from tpms_thermo_hydraulic_calculator import TPMSHeatExchanger
    from tpms_visualization import TPMSVisualization
except ImportError:
    # Fallback if visualization isn't in path, though it's expected in the repo
    TPMSVisualization = None
    print("Warning: Ensure 'tpms_thermo_hydraulic_calculator.py' and 'tpms_visualization.py' are present.")

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
        self.last_successful_he = None
        self.design_variable = None
        self.initial_guess = None
        self.target_metric = None
        self.target_value = None

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

    def run_simulation(self, x, config_path):
        """Helper to run a single full physics simulation."""
        sim_config = copy.deepcopy(self.base_config)
        self._update_nested_config(sim_config, config_path, x)

        with contextlib.redirect_stdout(DummyFile()):
            he = TPMSHeatExchanger(sim_config)
            converged = he.solve(max_iter=600, tolerance=1e-4)

        if not converged:
            return None, 1e9

        return he, 0

    def solve_geometry(self, design_variable, target_metric, target_value, max_iter=15, tol=1e-4, verbose=True):
        """
        Calculates required geometry using a Damped Secant method with Non-Dimensional Error.

        Error Estimation:
        The error is normalized by the maximum possible temperature difference (Th_in - Tc_in),
        making the convergence criteria independent of the absolute thermal scale.
        """
        self.design_variable = design_variable
        self.target_metric = target_metric
        self.target_value = target_value

        param_map = {
            'length': ['geometry', 'length'],
            'width': ['geometry', 'width'],
            'height': ['geometry', 'height'],
            'cell_size': ['geometry', 'unit_cell_size']
        }

        if design_variable not in param_map:
            raise ValueError(f"Invalid design variable. Choose from {list(param_map.keys())}")

        config_path = param_map[design_variable]
        ops = self.base_config['operating']
        dt_max = ops['Th_in'] - ops['Tc_in']

        if verbose:
            print(f"\n--- TPMS Non-Linear Design Solver ---")
            print(f"Goal: Adjust {design_variable} to reach {target_metric} = {target_value} K")
            print(f"Non-dimensional Tolerance: {tol}")

        # --- STEP 1: INITIAL STATE (Point 0) ---
        x0 = self._get_nested_config_value(self.base_config, config_path)
        self.initial_guess = x0

        he0, status = self.run_simulation(x0, config_path)
        if status != 0:
            raise RuntimeError("Initial baseline simulation failed to converge.")

        y0 = self._get_result_value(he0, target_metric)
        # Reduced form: (T_actual - T_target) / (T_max - T_min)
        err0 = (y0 - target_value) / dt_max

        if verbose:
            print(f"  [0] {design_variable} = {x0:.4f} m | Err_red = {err0:.6f}")

        if abs(err0) < tol:
            self.last_successful_he = he0
            self.post_process(he0)
            return x0

        # --- STEP 2: SMALL PERTURBATION (Point 1) ---
        perturb = 0.05 if err0 > 0 else -0.05
        x1 = x0 * (1.0 + perturb)

        he1, status = self.run_simulation(x1, config_path)
        if status != 0:
            x1 = x0 * (1.0 + perturb * 0.2)
            he1, _ = self.run_simulation(x1, config_path)

        y1 = self._get_result_value(he1, target_metric)
        err1 = (y1 - target_value) / dt_max

        if verbose:
            print(f"  [1] {design_variable} = {x1:.4f} m | Err_red = {err1:.6f}")

        # --- STEP 3: DAMPED SECANT ITERATIONS ---
        damping = 0.85

        for i in range(2, max_iter + 2):
            denom = err1 - err0
            if abs(denom) < 1e-10:
                break

            delta_x = err1 * (x1 - x0) / denom
            x_new = x1 - damping * delta_x

            # Physical bounds
            x_min, x_max = 0.001, 10.0
            x_new = max(x_min, min(x_new, x_max))

            he_new, status = self.run_simulation(x_new, config_path)
            if status != 0:
                damping *= 0.5
                continue

            y_new = self._get_result_value(he_new, target_metric)
            err_new = (y_new - target_value) / dt_max

            if verbose:
                print(f"  [{i}] {design_variable} = {x_new:.4f} m | Err_red = {err_new:.6f}")

            if abs(err_new) < tol:
                print(f"\nSUCCESS: Design converged in {i} iterations.")
                self.last_successful_he = he_new
                self.post_process(he_new)
                return x_new

            # Update points
            x0, err0 = x1, err1
            x1, err1 = x_new, err_new

            # Adaptive damping
            if abs(err_new) > abs(err0):
                damping *= 0.7

        print("\nWARNING: Convergence limits reached.")
        if self.last_successful_he: self.post_process(self.last_successful_he)
        return x1

    def post_process(self, he):
        """Visualizes results and prints key performance metrics."""
        print("\n" + "="*40)
        print("TPMS DESIGNER SUMMARY")
        print("="*40)
        print(f"Design Variable: {self.design_variable}")
        print(f"Initial Guess:   {self.initial_guess:.6f} m")
        print(f"Final Value:     {self._get_nested_config_value(he.config, ['geometry', self.design_variable if self.design_variable != 'cell_size' else 'unit_cell_size']):.6f} m")
        print(f"Target Metric:   {self.target_metric}")
        print(f"Target Value:    {self.target_value:.2f} K")
        print(f"Achieved Value:  {self._get_result_value(he, self.target_metric):.4f} K")
        print("-" * 40)

        print("FINAL THERMO-HYDRAULIC PERFORMANCE:")
        Q_total = np.sum(he.Q)
        print(f"  Total Heat Duty: {Q_total:.2f} W")
        print(f"  Hot Pressure Drop:  {he.Ph[0] - he.Ph[-1]:.2f} Pa")
        print(f"  Cold Pressure Drop: {he.Pc[-1] - he.Pc[0]:.2f} Pa")

        # Safe attribute access to avoid AttributeErrors
        eps = getattr(he, 'epsilon', None)
        lmtd = getattr(he, 'LMTD', None)

        if eps is not None:
            print(f"  Heat Exchanger Effectiveness: {eps:.4f}")
        else:
            # Fallback calculation if not in class
            Cmin = min(he.config['operating']['mh'] * 14000, he.config['operating']['mc'] * 5200) # Rough Cp
            Qmax = Cmin * (he.Th[0] - he.Tc[-1])
            print(f"  Approx. Effectiveness: {Q_total/max(Qmax, 1e-3):.4f}")

        if lmtd is not None:
            print(f"  LMTD: {lmtd:.2f} K")
            print(f"  Overall UA: {Q_total/max(lmtd, 1e-3):.2f} W/K")

        print("="*40 + "\n")

        if TPMSVisualization:
            vis = TPMSVisualization(he)
            vis.plot_temperature_profiles()
            vis.plot_performance_summary()
        else:
            plt.figure(figsize=(8, 5))
            x = np.linspace(0, he.config['geometry']['length'], len(he.Th))
            plt.plot(x, he.Th, 'r-', label='Hot Stream')
            plt.plot(x, he.Tc, 'b-', label='Cold Stream')
            plt.xlabel('Length [m]')
            plt.ylabel('Temperature [K]')
            plt.title('Designed Temperature Profile')
            plt.legend()
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    from tpms_thermo_hydraulic_calculator import create_default_config
    config = create_default_config()

    designer = TPMSDesigner(config)
    # Target a hot outlet temperature of 55K by adjusting length
    designer.solve_geometry('length', 'Th_out', 55.0, verbose=True)