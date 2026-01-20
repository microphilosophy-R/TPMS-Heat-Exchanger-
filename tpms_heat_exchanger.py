"""
TPMS Heat Exchanger for Ortho-Para Hydrogen Conversion

Main solver for catalyst-filled TPMS heat exchangers in hydrogen liquefaction.
Couples heat transfer, fluid flow, and ortho-para conversion kinetics.

Author: Based on research by Zhang et al. (2025)
"""

import numpy as np
import time
from scipy.optimize import fsolve

from TPMS_heat_HE.hydrogen_properties import HydrogenProperties
from TPMS_heat_HE.tpms_correlations import TPMSCorrelations


# Assume HydrogenProperties and TPMSCorrelations are available
# from hydrogen_properties import HydrogenProperties
# from tpms_correlations import TPMSCorrelations

class TPMSHeatExchanger:
    """
    Robust Solver using Stream-wise Indexing (Matches MATLAB logic)
    """

    def __init__(self, config):
        self.config = config
        self.h2_props = HydrogenProperties()

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

        self._calculate_geometry()
        self.N_elements = config['solver']['n_elements']
        self._initialize_solution()

    def _calculate_geometry(self):
        self.Dh_hot = 4 * self.porosity_hot * self.unit_cell / (2 * np.pi)
        self.Dh_cold = 4 * self.porosity_cold * self.unit_cell / (2 * np.pi)
        self.Ac_hot = self.W_HE * self.H_HE * self.porosity_hot
        self.Ac_cold = self.W_HE * self.H_HE * self.porosity_cold

        surface_area_density = self.config['geometry'].get('surface_area_density', 600)
        self.A_heat = self.L_HE * self.W_HE * self.H_HE * surface_area_density
        self.k_wall = self.config['material']['k_wall']

    def _initialize_solution(self):
        """
        Initialize both streams from Inlet (0) to Outlet (N)
        """
        N = self.N_elements + 1

        # HOT STREAM (Index 0 = Inlet)
        Th_in = self.config['operating']['Th_in']
        Th_out_guess = self.config['operating']['Th_out']
        self.Th = np.linspace(Th_in, Th_out_guess, N)
        self.Ph = np.linspace(self.config['operating']['Ph_in'], self.config['operating']['Ph_in'], N)

        # COLD STREAM (Index 0 = Inlet)
        # We index this stream following its OWN flow direction
        Tc_in = self.config['operating']['Tc_in']
        Tc_out_guess = self.config['operating']['Tc_out']
        self.Tc = np.linspace(Tc_in, Tc_out_guess, N)
        self.Pc = np.linspace(self.config['operating']['Pc_in'], self.config['operating']['Pc_in'], N)

        # Para-hydrogen
        xh_in = self.config['operating']['xh_in']
        self.xh = np.full(N, xh_in)

        self.L_elem = self.L_HE / self.N_elements

    def solve(self, max_iter=2000, tolerance=1e-4, relaxation=0.1):
        print("=" * 70)
        print("TPMS Heat Exchanger (Stream-wise Indexing)")
        print("=" * 70)

        mh = self.config['operating']['mh']
        mc = self.config['operating']['mc']

        # Q_dist stores the heat removed from the HOT stream at each element
        Q_dist = np.zeros(self.N_elements)

        start_time = time.time()

        for iteration in range(max_iter):
            Th_old = self.Th.copy()
            Tc_old = self.Tc.copy()
            Q_old = Q_dist.copy()

            # 1. Properties (Calculated simply 0->N for both)
            props_h = self._calculate_properties_stream(self.Th, self.Ph, self.xh, mh, self.Ac_hot, self.Dh_hot,
                                                        self.TPMS_hot, is_helium=False)
            props_c = self._calculate_properties_stream(self.Tc, self.Pc, None, mc, self.Ac_cold, self.Dh_cold,
                                                        self.TPMS_cold, is_helium=True)

            # 2. Calculate Heat Transfer (The Coupling Step)
            U, Q_raw = self._calculate_heat_transfer(props_h, props_c)

            # Relaxation
            if iteration < 10:
                Q_dist = Q_old + 0.05 * (Q_raw - Q_old)
            else:
                Q_dist = Q_old + relaxation * (Q_raw - Q_old)

            # 3. Pressure Drop (Both calculate forward 0->N)
            dPh = self._calculate_dp_stream(props_h, self.Dh_hot)
            dPc = self._calculate_dp_stream(props_c, self.Dh_cold)

            # Update Pressures
            self.Ph[0] = self.config['operating']['Ph_in']
            self.Pc[0] = self.config['operating']['Pc_in']
            for i in range(self.N_elements):
                self.Ph[i + 1] = self.Ph[i] - dPh[i]
                self.Pc[i + 1] = self.Pc[i] - dPc[i]

            # 4. Energy Balance
            hh, hc = self._energy_balance(props_h, props_c, Q_dist, mh, mc)

            # 5. Solve Temperatures
            Th_new, Tc_new = self._solve_temperatures(hh, hc, self.Ph, self.Pc, self.xh)

            # 6. Kinetics
            xh_new = self._ortho_para_conversion(Th_new, self.Ph, self.xh, mh)

            # Update State
            self.Th = Th_new
            self.Tc = Tc_new
            self.xh = xh_new

            # Check Convergence
            err = np.max(np.abs(Th_new - Th_old)) + np.max(np.abs(Tc_new - Tc_old))

            if (iteration + 1) % 50 == 0:
                print(f"Iter {iteration + 1:4d}: Max T error = {err:.6f}")
                # Debug output: Check limits
                print(f"  Hot: {self.Th[0]:.1f} -> {self.Th[-1]:.1f} K")
                # Note: Cold Inlet is index 0 in this scheme
                print(f"  Cold: {self.Tc[-1]:.1f} <- {self.Tc[0]:.1f} K")

            if err < tolerance and iteration > 10:
                elapsed = time.time() - start_time
                print(f"\nConverged in {iteration + 1} iterations ({elapsed:.2f} s)")
                self._print_results(props_h, props_c, U, Q_dist)
                return True

        print(f"\nWARNING: Non-convergence. Error: {err:.6e}")
        return False

    def _calculate_properties_stream(self, T, P, x, m_dot, Ac, Dh, tpms_type, is_helium):
        """Generic property calculator for any stream"""
        N = len(T)
        props = {'h': np.zeros(N), 'rho': np.zeros(N), 'h_coeff': np.zeros(N)}

        for i in range(N):
            if is_helium:
                p = self.h2_props.get_helium_properties(T[i], P[i])
            else:
                p = self.h2_props.get_properties(T[i], P[i], x[i])

            props['h'][i] = p['h']
            props['rho'][i] = p['rho']

            # Flow props
            u = m_dot / (p['rho'] * Ac)
            Re = p['rho'] * u * Dh / p['mu']
            Pr = p['mu'] * p['cp'] / p['lambda']

            Nu, f = TPMSCorrelations.get_correlations(tpms_type, Re, Pr, 'Gas')

            # Catalyst enhancement (Hot side only)
            catalyst_factor = 1.2 if not is_helium else 1.0
            props['h_coeff'][i] = catalyst_factor * Nu * p['lambda'] / Dh

        return props

    def _calculate_heat_transfer(self, props_h, props_c):
        """
        Calculate Heat Transfer coupling streams flowing in opposite directions.
        Hot[i] is spatially aligned with Cold[N-i]
        """
        N = self.N_elements
        U = np.zeros(N)
        Q = np.zeros(N)

        for i in range(N):
            # Hot Element i corresponds to Cold Element (N-1)-i
            # Example: N=10. Hot 0 (Inlet) pairs with Cold 9 (Outlet).
            j = N - 1 - i

            # 1. Local U
            h_h = (props_h['h_coeff'][i] + props_h['h_coeff'][i + 1]) / 2
            h_c = (props_c['h_coeff'][j] + props_c['h_coeff'][j + 1]) / 2

            U[i] = 1 / (1 / h_h + self.wall_thickness / self.k_wall + 1 / h_c)

            # 2. Temperature Difference (AMTD)
            # Hot side: i to i+1
            Th_avg = (self.Th[i] + self.Th[i + 1]) / 2

            # Cold side: j to j+1
            Tc_avg = (self.Tc[j] + self.Tc[j + 1]) / 2

            dT = Th_avg - Tc_avg

            # 3. Q
            A_elem = self.A_heat / N
            Q[i] = U[i] * A_elem * dT

        return U, Q

    def _energy_balance(self, props_h, props_c, Q_dist, mh, mc):
        """
        Update enthalpies.
        Q_dist[i] is heat REMOVED from Hot element i.
        """
        N_nodes = self.N_elements + 1
        hh = np.zeros(N_nodes)
        hc = np.zeros(N_nodes)

        # 1. Hot Stream (Forward)
        # Fixed Inlet
        hh[0] = self.h2_props.get_properties(self.config['operating']['Th_in'],
                                             self.Ph[0],
                                             self.config['operating']['xh_in'])['h']
        for i in range(self.N_elements):
            hh[i + 1] = hh[i] - Q_dist[i] / mh

        # 2. Cold Stream (Forward from ITS inlet)
        # Fixed Inlet (Index 0)
        hc[0] = self.h2_props.get_helium_properties(self.config['operating']['Tc_in'],
                                                    self.Pc[0])['h']
        for k in range(self.N_elements):
            # Cold element k corresponds to Hot element N-1-k
            # If Hot loses Q, Cold GAINS Q.
            # We must find which Q corresponds to this cold element.
            # Hot index i paired with Cold index j = N-1-i.
            # So Cold index k pairs with Hot index i = N-1-k.
            Q_gain = Q_dist[self.N_elements - 1 - k]

            hc[k + 1] = hc[k] + Q_gain / mc

        return hh, hc

    def _solve_temperatures(self, hh, hc, Ph, Pc, xh):
        # Similar to before, but using generic solver
        N = len(hh)
        Th_new = np.zeros(N)
        Tc_new = np.zeros(N)

        for i in range(N):
            # Hot
            def res_h(T):
                if T < 14: T = 14
                if T > 400: T = 400
                return self.h2_props.get_properties(T, Ph[i], xh[i])['h'] - hh[i]

            Th_new[i] = fsolve(res_h, self.Th[i])[0]

            # Cold
            def res_c(T):
                if T < 4: T = 4
                if T > 400: T = 400
                return self.h2_props.get_helium_properties(T, Pc[i])['h'] - hc[i]

            Tc_new[i] = fsolve(res_c, self.Tc[i])[0]

        return Th_new, Tc_new

    def _calculate_dp_stream(self, props, Dh):
        # Generic DP for 0->N flow
        N = self.N_elements
        dP = np.zeros(N)
        for i in range(N):
            # Simplified: f * L/D * rho * u^2 / 2
            # Calculate local f, rho, u from props...
            # (Implementation depends on your specific correlation availability)
            # Placeholder:
            rho = (props['rho'][i] + props['rho'][i + 1]) / 2
            # u = m / (rho * Ac)...
            # For now return zeros or implement full correlation
            dP[i] = 0.0  # Implement your f-factor correlation here
        return dP

    def _ortho_para_conversion(self, Th, Ph, xh, mh):
        # Standard forward integration 0->N
        N = len(Th) - 1
        xh_new = np.zeros(N + 1)
        xh_new[0] = xh[0]

        Tc = 32.938
        Pc = 1.284e6

        for i in range(N):
            T_avg = (Th[i] + Th[i + 1]) / 2
            P_avg = (Ph[i] + Ph[i + 1]) / 2
            x_avg = (xh[i] + xh[i + 1]) / 2
            x_eq = self.h2_props.get_equilibrium_fraction(T_avg)

            K = 59.7 - 253.9 * (T_avg / Tc) - 11.6 * (P_avg / Pc)
            rho_avg = self.h2_props.get_properties(T_avg, P_avg, x_avg)['rho']
            C_H2 = rho_avg / 0.002016

            if x_avg < x_eq:
                k_forward = K / C_H2 * np.log(((x_avg / x_eq) ** 1.0924) * ((1 - x_eq) / (1 - x_avg)))
                u_avg = mh / (rho_avg * self.Ac_hot)
                t_res = self.L_elem / u_avg
                delta_x = k_forward * (x_eq - x_avg) * t_res
                xh_new[i + 1] = xh[i] + delta_x
            else:
                xh_new[i + 1] = xh[i]

            xh_new[i + 1] = np.clip(xh_new[i + 1], 0, 1)
        return xh_new

    def _print_results(self, props_h, props_c, U, Q):
        print("=" * 70)
        print("RESULTS (Stream-wise Indexing)")
        print("=" * 70)
        print("Hot Fluid (Hydrogen) - Flow 0 -> N:")
        print(f"  Inlet:  T = {self.Th[0]:.2f} K")
        print(f"  Outlet: T = {self.Th[-1]:.2f} K")
        print("Cold Fluid (Helium) - Flow 0 -> N:")
        print(f"  Inlet:  T = {self.Tc[0]:.2f} K")
        print(f"  Outlet: T = {self.Tc[-1]:.2f} K")
        print(f"Total Heat Load: {np.sum(Q):.2f} W")


def create_default_config():
    """Create default configuration dictionary"""
    return {
        'geometry': {
            'length': 0.94,         # m
            'width': 0.15,          # m
            'height': 0.10,         # m
            'porosity_hot': 0.65,
            'porosity_cold': 0.70,
            'unit_cell_size': 5e-3,  # m
            'wall_thickness': 0.5e-3,  # m
            'surface_area_density': 600  # m²/m³
        },
        'tpms': {
            'type_hot': 'Diamond',
            'type_cold': 'Gyroid'
        },
        'material': {
            'k_wall': 237  # W/(m·K) - Aluminum
        },
        'operating': {
            'Th_in': 66.3,      # K
            'Th_out': 53.5,     # K
            'Tc_in': 43.5,      # K
            'Tc_out': 61.3,     # K
            'Ph_in': 1.13e6,    # Pa
            'Pc_in': 0.54e6,    # Pa
            'mh': 1e-3,         # kg/s
            'mc': 2e-3,         # kg/s
            'xh_in': 0.452      # Initial para fraction
        },
        'catalyst': {
            'enhancement': 1.2,
            'pressure_factor': 1.3
        },
        'solver': {
            'n_elements': 20,
            'max_iter': 10,
            'tolerance': 1e-3,
            'relaxation': 0.3
        }
    }


if __name__ == "__main__":
    # Create and run heat exchanger
    config = create_default_config()
    he = TPMSHeatExchanger(config)
    he.solve()
