"""
TPMS Heat Exchanger with Complete Thermo-Hydraulic Model - CORRECTED

Enhanced version including:
- Heat transfer calculations
- Pressure drop calculations (using correlations from literature)
- Ortho-para hydrogen conversion kinetics
- Robust energy balance verification

Author: Enhanced from Zhang et al. (2025) research
FIXED: Property calculation to match original implementation
"""

import numpy as np
import time
from scipy.optimize import fsolve
import warnings

from TPMS_heat_HE_local.tpms_visualization import TPMSVisualizer

warnings.filterwarnings("ignore", message="Some Re values outside validated range")


class TPMSHeatExchangerHydraulic:
    """
    Complete thermo-hydraulic solver for TPMS heat exchanger
    """

    def __init__(self, config):
        self.config = config

        # Import properties module (assume available)
        try:
            from hydrogen_properties import HydrogenProperties
            self.h2_props = HydrogenProperties()
        except:
            print("Warning: hydrogen_properties module not found, using simplified properties")
            self.h2_props = None

        # Geometry
        self.L_HE = config['geometry']['length']
        self.W_HE = config['geometry']['width']
        self.H_HE = config['geometry']['height']
        self.porosity_hot = config['geometry']['porosity_hot']
        self.porosity_cold = config['geometry']['porosity_cold']
        self.unit_cell = config['geometry']['unit_cell_size']
        self.wall_thickness = config['geometry']['wall_thickness']

        self.TPMS_hot = config['tpms']['type_hot']
        self.TPMS_cold = config['tpms']['type_cold']
        self.N_elements = config['solver']['n_elements']
        self._calculate_geometry()
        self._initialize_solution()


    def _calculate_geometry(self):
        """Calculate geometric parameters"""
        # Hydraulic diameter (TPMS specific)
        self.Dh_hot = 4 * self.porosity_hot * self.unit_cell / (2 * np.pi)
        self.Dh_cold = 4 * self.porosity_cold * self.unit_cell / (2 * np.pi)

        # Flow area
        self.Ac_hot = self.W_HE * self.H_HE * self.porosity_hot
        self.Ac_cold = self.W_HE * self.H_HE * self.porosity_cold

        # Heat transfer area
        surface_area_density = self.config['geometry'].get('surface_area_density', 600)
        self.A_heat = self.L_HE * self.W_HE * self.H_HE * surface_area_density
        self.A_elem = self.A_heat / self.N_elements

        # Wall properties
        self.k_wall = self.config['material']['k_wall']

    def _initialize_solution(self):
        """Initialize temperature, pressure, and composition profiles"""
        N = self.N_elements + 1

        # Temperature initialization
        Th_in = self.config['operating']['Th_in']
        Tc_in = self.config['operating']['Tc_in']

        # Simple linear guess
        Th_out_guess = Tc_in + 5.0
        Tc_out_guess = Th_in - 5.0

        self.Th = np.linspace(Th_in, Th_out_guess, N)
        self.Tc = np.linspace(Tc_out_guess, Tc_in, N)  # Cold flows opposite

        # Pressure initialization
        self.Ph = np.full(N, self.config['operating']['Ph_in'])
        self.Pc = np.full(N, self.config['operating']['Pc_in'])

        # Para-hydrogen fraction initialization
        self.xh = np.full(N, self.config['operating']['xh_in'])

        # Element length
        self.L_elem = self.L_HE / self.N_elements

    def _safe_get_prop(self, T, P, x=None, is_helium=False):
        """
        Get fluid properties with fallback for simplified calculations
        Returns dict with: h, rho, cp, mu, lambda (NOT Pr - we calculate it manually)
        """
        # Clamp temperature to safe range
        T = np.clip(T, 14.0 if not is_helium else 4.0, 400.0)

        if self.h2_props is not None:
            try:
                if is_helium:
                    props = self.h2_props.get_helium_properties(T, P)
                else:
                    props = self.h2_props.get_properties(T, P, x)
                return props
            except:
                pass

        # Simplified fallback properties
        if is_helium:
            # Helium at ~50K
            return {
                'h': 5200 * (T - 4.0),  # cp * dT
                'rho': P / (2077 * T),  # Ideal gas approximation
                'cp': 5200,
                'mu': 1e-5,
                'lambda': 0.15
            }
        else:
            # Hydrogen mixture at ~50K
            return {
                'h': 14000 * (T - 14.0),
                'rho': P / (4124 * T),
                'cp': 14000,
                'mu': 1e-5,
                'lambda': 0.1,
                'Delta_h': 500e3  # Conversion heat
            }

    def _get_heat_transfer_coefficient(self, Re, Pr, tpms_type, Dh, lambda_f, is_hot=True):
        """
        Calculate heat transfer coefficient using TPMS correlations

        Based on literature correlations (Zhang et al., 2025)
        """
        # TPMS Nusselt number correlations for Gas (H2/He)
        if tpms_type == 'Diamond':
            Nu = 0.409 * Re**0.625 * Pr**0.4
        elif tpms_type == 'Gyroid':
            Nu = 0.325 * Re**0.700 * Pr**0.36
        elif tpms_type == 'FKS':
            Nu = 0.52 * Re**0.61 * Pr**0.4
        elif tpms_type == 'Primitive':
            Nu = 0.1 * Re**0.75 * Pr**0.36
        else:
            # Default correlation
            Nu = 0.4 * Re**0.65 * Pr**0.4

        # Heat transfer coefficient
        h = Nu * lambda_f / Dh

        # Catalyst enhancement for hot side
        if is_hot:
            h *= self.config['catalyst'].get('enhancement', 1.2)

        return h, Nu

    def _get_friction_factor(self, Re, tpms_type):
        """
        Calculate friction factor using TPMS correlations

        Based on literature correlations (Zhang et al., 2025)
        """
        # TPMS friction factor correlations for Gas
        if tpms_type == 'Diamond':
            f = 2.5892 * Re**(-0.1940)
        elif tpms_type == 'Gyroid':
            f = 2.5 * Re**(-0.2)
        elif tpms_type == 'FKS':
            f = 2.1335 * Re**(-0.1334)
        elif tpms_type == 'Primitive':
            f = 4.0 * Re**(-0.25)
        else:
            # Default correlation
            f = 3.0 * Re**(-0.2)

        return f

    def _calculate_pressure_drop(self, rho, u, mu, L, Dh, tpms_type):
        """
        Calculate pressure drop for TPMS structure

        Using Darcy-Weisbach equation:
        ΔP = f * (L/Dh) * (ρ*u²/2)
        """
        Re = rho * u * Dh / mu
        f = self._get_friction_factor(Re, tpms_type)

        # Pressure drop
        dP = f * (L / Dh) * (rho * u**2 / 2)

        return dP, f, Re

    def solve(self, max_iter=300, tolerance=1e-3):
        """
        Main solver loop with thermo-hydraulic coupling
        """
        print("=" * 70)
        print("TPMS Heat Exchanger - Complete Thermo-Hydraulic Solution")
        print("=" * 70)

        mh = self.config['operating']['mh']
        mc = self.config['operating']['mc']

        # Heat transfer rates
        Q = np.zeros(self.N_elements)

        # Pressure drop storage
        dP_hot = np.zeros(self.N_elements)
        dP_cold = np.zeros(self.N_elements)

        start_time = time.time()

        for iteration in range(max_iter):
            Th_old = self.Th.copy()
            Tc_old = self.Tc.copy()
            Ph_old = self.Ph.copy()
            Pc_old = self.Pc.copy()
            Q_old = Q.copy()

            # Adaptive relaxation
            if iteration < 20:
                relax = 0.05
            elif iteration < 50:
                relax = 0.1
            else:
                relax = min(0.3, 0.05 + 0.01 * (iteration - 50))

            # --- 1. Calculate Heat Transfer Coefficients and Pressure Drop ---
            U_vals = np.zeros(self.N_elements)

            for i in range(self.N_elements):
                # Average properties in element
                Th_avg = 0.5 * (self.Th[i] + self.Th[i + 1])
                Tc_avg = 0.5 * (self.Tc[i] + self.Tc[i + 1])

                # Get properties
                ph = self._safe_get_prop(Th_avg, self.Ph[i], self.xh[i], False)
                pc = self._safe_get_prop(Tc_avg, self.Pc[i], None, True)

                # Calculate Pr manually (as in original code)
                Pr_h = ph['mu'] * ph['cp'] / ph['lambda']
                Pr_c = pc['mu'] * pc['cp'] / pc['lambda']

                # Hot side - heat transfer and pressure drop
                u_h = mh / (ph['rho'] * self.Ac_hot)
                Re_h = ph['rho'] * u_h * self.Dh_hot / ph['mu']

                h_h, Nu_h = self._get_heat_transfer_coefficient(
                    Re_h, Pr_h, self.TPMS_hot, self.Dh_hot, ph['lambda'], is_hot=True
                )

                dP_hot[i], f_h, _ = self._calculate_pressure_drop(
                    ph['rho'], u_h, ph['mu'], self.L_elem, self.Dh_hot, self.TPMS_hot
                )

                # Cold side - heat transfer and pressure drop
                u_c = mc / (pc['rho'] * self.Ac_cold)
                Re_c = pc['rho'] * u_c * self.Dh_cold / pc['mu']

                h_c, Nu_c = self._get_heat_transfer_coefficient(
                    Re_c, Pr_c, self.TPMS_cold, self.Dh_cold, pc['lambda'], is_hot=False
                )

                dP_cold[i], f_c, _ = self._calculate_pressure_drop(
                    pc['rho'], u_c, pc['mu'], self.L_elem, self.Dh_cold, self.TPMS_cold
                )

                # Overall heat transfer coefficient
                U_vals[i] = 1 / (1/h_h + self.wall_thickness/self.k_wall + 1/h_c)

            # --- 2. Calculate Heat Transfer (Wall Heat Flux) ---
            Q_raw = np.zeros(self.N_elements)
            for i in range(self.N_elements):
                Th_avg = 0.5 * (self.Th[i] + self.Th[i + 1])
                Tc_avg = 0.5 * (self.Tc[i] + self.Tc[i + 1])

                # Driving force: Hot - Cold
                delta_T = Th_avg - Tc_avg

                # Heat transfer rate
                Q_raw[i] = U_vals[i] * self.A_elem * delta_T

            # Relax Q
            Q = Q_old + relax * (Q_raw - Q_old)

            # --- 3. Update Pressures ---
            # Hot stream: pressure decreases along flow
            Ph_new = self.Ph.copy()
            for i in range(self.N_elements):
                Ph_new[i + 1] = Ph_new[i] - dP_hot[i]

            # Cold stream: pressure decreases along flow (opposite direction)
            Pc_new = self.Pc.copy()
            for i in range(self.N_elements - 1, -1, -1):
                Pc_new[i] = Pc_new[i + 1] - dP_cold[i]

            # Apply relaxation
            self.Ph = Ph_old + relax * (Ph_new - Ph_old)
            self.Pc = Pc_old + relax * (Pc_new - Pc_old)

            # --- 4. Enthalpy Balance ---
            hh = np.zeros(self.N_elements + 1)
            hc = np.zeros(self.N_elements + 1)

            # Hot (Forward): h_out = h_in - Q/m
            hh[0] = self._safe_get_prop(self.config['operating']['Th_in'],
                                        self.Ph[0], self.xh[0], False)['h']
            for i in range(self.N_elements):
                hh[i + 1] = hh[i] - Q[i] / mh

            # Cold (Backward): h_out = h_in + Q/m
            hc[-1] = self._safe_get_prop(self.config['operating']['Tc_in'],
                                         self.Pc[-1], None, True)['h']
            for i in range(self.N_elements - 1, -1, -1):
                hc[i] = hc[i + 1] + Q[i] / mc

            # --- 5. Temperature Update (Invert h -> T) ---
            # CRITICAL: Must use fsolve to properly invert h(T,P,x) -> T
            # Linear approximation FAILS for real gas properties!

            # Hot Stream
            for i in range(len(hh)):
                def res_h(T):
                    # Solving h(T, P, x) - h_target = 0
                    return self._safe_get_prop(T, self.Ph[i], self.xh[i], False)['h'] - hh[i]

                guess = np.clip(self.Th[i], 14.0, 400.0)
                try:
                    # Use fsolve with tight tolerance
                    sol = fsolve(res_h, guess, xtol=1e-5)
                    T_new = sol[0]
                    # Clamp result
                    T_new = np.clip(T_new, 14.0, 400.0)
                    self.Th[i] = 0.8 * self.Th[i] + 0.2 * T_new  # Heavy damping
                except:
                    pass  # Keep old value if failed

            # Cold Stream
            for i in range(len(hc)):
                def res_c(T):
                    return self._safe_get_prop(T, self.Pc[i], None, True)['h'] - hc[i]

                guess = np.clip(self.Tc[i], 4.0, 400.0)
                try:
                    sol = fsolve(res_c, guess, xtol=1e-5)
                    T_new = sol[0]
                    T_new = np.clip(T_new, 4.0, 400.0)
                    self.Tc[i] = 0.8 * self.Tc[i] + 0.2 * T_new
                except:
                    pass

            # --- Monotonicity Enforcement ---
            # Physics Check: Hot must cool down, Cold must heat up
            if iteration > 5:  # Allow initial adjustment
                for i in range(self.N_elements):
                    # Hot stream: downstream must be cooler than upstream
                    if self.Th[i + 1] > self.Th[i]:
                        self.Th[i + 1] = self.Th[i] - 1e-4

                    # Cold stream flows N->0, so Tc[i] (outlet) > Tc[i+1] (inlet)
                    if self.Tc[i + 1] > self.Tc[i]:
                        self.Tc[i + 1] = self.Tc[i] - 1e-4

            # --- 6. Ortho-Para Conversion (Simplified) ---
            if self.h2_props is not None:
                try:
                    xh_new = self._ortho_para_conversion()
                    self.xh = self.xh + 0.05 * (xh_new - self.xh)
                except:
                    # Simple forward shift if conversion fails
                    x_eq_out = 0.75  # Approximate at 50K
                    self.xh = np.linspace(self.xh[0], x_eq_out, len(self.xh))
            else:
                # Simple forward shift
                x_eq_out = 0.75  # Approximate at 50K
                self.xh = np.linspace(self.xh[0], x_eq_out, len(self.xh))

            # --- 7. Convergence Check ---
            err_T = np.max(np.abs(self.Th - Th_old)) + np.max(np.abs(self.Tc - Tc_old))
            err_P = np.max(np.abs(self.Ph - Ph_old)) + np.max(np.abs(self.Pc - Pc_old))
            err = err_T + err_P * 1e-6  # Scale pressure error

            if (iteration + 1) % 20 == 0:
                Q_total = np.sum(Q)
                dP_hot_total = self.Ph[0] - self.Ph[-1]
                dP_cold_total = self.Pc[-1] - self.Pc[0]
                print(f"Iter {iteration + 1:3d} | Err: {err:.4f} | Q: {Q_total:.1f} W | "
                      f"ΔP_hot: {dP_hot_total/1e3:.2f} kPa | ΔP_cold: {dP_cold_total/1e3:.2f} kPa")

            if err < tolerance:
                elapsed = time.time() - start_time
                print(f"\n*** CONVERGED in {iteration + 1} iterations ({elapsed:.2f} s) ***")
                self._print_results(Q, dP_hot, dP_cold)
                return True

        print("\nMax iterations reached")
        self._print_results(Q, dP_hot, dP_cold)
        return False

    def _ortho_para_conversion(self):
        """Simplified kinetics"""
        xh_new = np.zeros_like(self.xh)
        xh_new[0] = self.xh[0]

        x_eq_func = self.h2_props.get_equilibrium_fraction

        for i in range(self.N_elements):
            T_avg = 0.5 * (self.Th[i] + self.Th[i+1])
            x_eq = x_eq_func(T_avg)

            # Simple first-order approach
            k_rate = 0.2  # Effective rate constant
            mh = self.config['operating']['mh']
            props = self._safe_get_prop(T_avg, self.Ph[i], self.xh[i], False)
            u = mh / (props['rho'] * self.Ac_hot)
            tau = self.L_elem / u

            dx = k_rate * (x_eq - self.xh[i]) * tau
            xh_new[i+1] = np.clip(self.xh[i] + dx, 0.0, 1.0)

        return xh_new

    def _print_results(self, Q, dP_hot, dP_cold):
        """Print comprehensive results"""
        print("=" * 70)
        print("RESULTS - Thermo-Hydraulic Performance")
        print("=" * 70)

        # Temperatures
        print("\nTemperatures:")
        print(f"  Hot:  {self.Th[0]:.2f} K → {self.Th[-1]:.2f} K (ΔT = {self.Th[0]-self.Th[-1]:.2f} K)")
        print(f"  Cold: {self.Tc[-1]:.2f} K → {self.Tc[0]:.2f} K (ΔT = {self.Tc[0]-self.Tc[-1]:.2f} K)")

        # Pressures
        dP_hot_total = self.Ph[0] - self.Ph[-1]
        dP_cold_total = self.Pc[-1] - self.Pc[0]
        print("\nPressures:")
        print(f"  Hot:  {self.Ph[0]/1e6:.3f} MPa → {self.Ph[-1]/1e6:.3f} MPa (ΔP = {dP_hot_total/1e3:.2f} kPa, {dP_hot_total/self.Ph[0]*100:.2f}%)")
        print(f"  Cold: {self.Pc[-1]/1e6:.3f} MPa → {self.Pc[0]/1e6:.3f} MPa (ΔP = {dP_cold_total/1e3:.2f} kPa, {dP_cold_total/self.Pc[-1]*100:.2f}%)")

        # Heat transfer
        Q_total = np.sum(Q)
        print("\nHeat Transfer:")
        print(f"  Total heat load: {Q_total:.2f} W")
        print(f"  Average heat flux: {Q_total/self.A_heat:.2f} W/m²")

        # Conversion
        print("\nConversion:")
        print(f"  Para-H₂: {self.xh[0]:.4f} → {self.xh[-1]:.4f}")
        if self.h2_props is not None:
            try:
                x_eq_out = self.h2_props.get_equilibrium_fraction(self.Th[-1])
                x_eq_in = self.h2_props.get_equilibrium_fraction(self.Th[0])
                eff = (self.xh[-1] - self.xh[0]) / (x_eq_out - x_eq_in) * 100
                print(f"  Conversion efficiency: {eff:.2f}%")
            except:
                pass

        # Energy balance check
        try:
            h_h_in = self._safe_get_prop(self.Th[0], self.Ph[0], self.xh[0], False)['h']
            h_h_out = self._safe_get_prop(self.Th[-1], self.Ph[-1], self.xh[-1], False)['h']
            Q_hot = self.config['operating']['mh'] * (h_h_in - h_h_out)

            h_c_in = self._safe_get_prop(self.Tc[-1], self.Pc[-1], None, True)['h']
            h_c_out = self._safe_get_prop(self.Tc[0], self.Pc[0], None, True)['h']
            Q_cold = self.config['operating']['mc'] * (h_c_out - h_c_in)

            imbalance = abs(Q_hot - Q_cold) / max(abs(Q_hot), abs(Q_cold)) * 100

            print("\nEnergy Balance:")
            print(f"  Hot stream loss:  {Q_hot:.2f} W")
            print(f"  Cold stream gain: {Q_cold:.2f} W")
            print(f"  Imbalance: {imbalance:.2f}%")
        except:
            print("\nEnergy Balance: Could not calculate")

        print("=" * 70)


def create_default_config():
    """Create default configuration"""
    return {
        'geometry': {
            'length': 10.94,
            'width': 0.15,
            'height': 0.10,
            'porosity_hot': 0.65,
            'porosity_cold': 0.70,
            'unit_cell_size': 5e-3,
            'wall_thickness': 0.5e-3,
            'surface_area_density': 600
        },
        'tpms': {
            'type_hot': 'Diamond',
            'type_cold': 'Gyroid'
        },
        'material': {
            'k_wall': 237  # Aluminum
        },
        'operating': {
            'Th_in': 66.3,
            'Tc_in': 43.5,
            'Ph_in': 2e6,
            'Pc_in': 0.5e6,
            'mh': 60e-2,
            'mc': 120e-2,
            'xh_in': 0.452
        },
        'catalyst': {
            'enhancement': 1.2
        },
        'solver': {
            'n_elements': 20,
            'max_iter': 300,
            'tolerance': 1e-3
        }
    }


if __name__ == "__main__":
    config = create_default_config()
    he = TPMSHeatExchangerHydraulic(config)
    he.solve()

    vis = TPMSVisualizer(he)
    vis.plot_comprehensive(save_path='tpms_comprehensive.png')
    vis.plot_performance_metrics(save_path='tpms_metrics.png')