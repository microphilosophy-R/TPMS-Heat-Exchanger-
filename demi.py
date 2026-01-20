import numpy as np
import time
from scipy.optimize import fsolve
from TPMS_heat_HE.hydrogen_properties import HydrogenProperties
from TPMS_heat_HE.tpms_correlations import TPMSCorrelations


# Assume HydrogenProperties and TPMSCorrelations are imported
# from hydrogen_properties import HydrogenProperties
# from tpms_correlations import TPMSCorrelations

class TPMSHeatExchanger:
    """
    Robust Solver using Local Epsilon-NTU Method.
    Guarantees thermodynamic consistency (0 <= epsilon <= 1).
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
        Initialize arrays.
        Index 0 -> N follows the flow direction for EACH fluid.
        Hot: 0=Inlet, N=Outlet
        Cold: 0=Inlet, N=Outlet
        """
        N = self.N_elements + 1

        # Hot Stream
        Th_in = self.config['operating']['Th_in']
        Th_out_guess = self.config['operating']['Th_out']
        self.Th = np.linspace(Th_in, Th_out_guess, N)
        self.Ph = np.full(N, self.config['operating']['Ph_in'])

        # Cold Stream
        Tc_in = self.config['operating']['Tc_in']
        Tc_out_guess = self.config['operating']['Tc_out']
        self.Tc = np.linspace(Tc_in, Tc_out_guess, N)
        self.Pc = np.full(N, self.config['operating']['Pc_in'])

        # Para-hydrogen
        self.xh = np.full(N, self.config['operating']['xh_in'])
        self.L_elem = self.L_HE / self.N_elements

    def solve(self, max_iter=1000, tolerance=1e-4, relaxation=0.2):
        print("=" * 70)
        print("TPMS Heat Exchanger (Epsilon-NTU Robust Solver)")
        print("=" * 70)

        mh = self.config['operating']['mh']
        mc = self.config['operating']['mc']

        # Q_dist: Heat transferred in each SPATIAL element
        Q_dist = np.zeros(self.N_elements)

        start_time = time.time()

        for iteration in range(max_iter):
            Th_old = self.Th.copy()
            Tc_old = self.Tc.copy()
            Q_old = Q_dist.copy()

            # 1. Properties (Clamp Temps to safe range 4K-400K first)
            self.Th = np.clip(self.Th, 14.0, 400.0)
            self.Tc = np.clip(self.Tc, 4.0, 400.0)

            props_h = self._calculate_properties_stream(self.Th, self.Ph, self.xh, mh, self.Ac_hot, self.Dh_hot,
                                                        self.TPMS_hot, False)
            props_c = self._calculate_properties_stream(self.Tc, self.Pc, None, mc, self.Ac_cold, self.Dh_cold,
                                                        self.TPMS_cold, True)

            # 2. Calculate Heat Transfer using Epsilon-NTU
            # We loop over SPATIAL elements k=0 (Left) to k=N-1 (Right)
            Q_raw = np.zeros(self.N_elements)
            U_dist = np.zeros(self.N_elements)

            for k in range(self.N_elements):
                # Map Spatial Index k to Flow Indices
                # Hot flows Left->Right (0->N). Element k uses Hot nodes k and k+1.
                idx_h_in = k

                # Cold flows Right->Left (0->N in stream array).
                # Spatial 0 (Left) is Cold Outlet (Node N).
                # Spatial L (Right) is Cold Inlet (Node 0).
                # Spatial Element k (from Left) corresponds to Cold nodes near N-1-k.
                # Specifically, Cold enters this spatial element from the RIGHT.
                # The Right side of Spatial element k is Spatial node k+1.
                # Spatial node k+1 corresponds to Cold Stream Node N-(k+1) = N-k-1.
                idx_c_in = self.N_elements - 1 - k

                # Temperatures entering the element
                T_h_in = self.Th[idx_h_in]
                T_c_in = self.Tc[idx_c_in]

                # Capacitance Rates (C = m * cp)
                # Average cp in the element
                cp_h = (props_h['cp'][idx_h_in] + props_h['cp'][idx_h_in + 1]) / 2
                cp_c = (props_c['cp'][idx_c_in] + props_c['cp'][idx_c_in + 1]) / 2

                Ch = mh * cp_h
                Cc = mc * cp_c
                Cmin = min(Ch, Cc)
                Cmax = max(Ch, Cc)
                Cr = Cmin / Cmax

                # Overall Heat Transfer Coefficient U
                h_h = props_h['h_coeff'][idx_h_in]
                h_c = props_c['h_coeff'][idx_c_in]
                U = 1 / (1 / h_h + self.wall_thickness / self.k_wall + 1 / h_c)
                U_dist[k] = U

                # NTU
                A_elem = self.A_heat / self.N_elements
                NTU = U * A_elem / Cmin

                # Epsilon (Counter-flow correlation)
                if Cr < 1.0:
                    epsilon = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))
                else:
                    # Balanced flow limit
                    epsilon = NTU / (1 + NTU)

                # Calculate Q (Bounded!)
                # Q is positive if Hot > Cold
                Q_raw[k] = epsilon * Cmin * (T_h_in - T_c_in)

            # Relax Q
            if iteration < 5:
                Q_dist = Q_old + 0.1 * (Q_raw - Q_old)
            else:
                Q_dist = Q_old + relaxation * (Q_raw - Q_old)

            # 3. Update Enthalpies & Temperatures
            # Hot Side (Forward Integration)
            hh = np.zeros(self.N_elements + 1)
            hh[0] = self.h2_props.get_properties(self.config['operating']['Th_in'], self.Ph[0], self.xh[0])['h']

            for k in range(self.N_elements):
                # Hot loses Q[k]
                hh[k + 1] = hh[k] - Q_dist[k] / mh

            # Cold Side (Forward Integration from Inlet)
            hc = np.zeros(self.N_elements + 1)
            hc[0] = self.h2_props.get_helium_properties(self.config['operating']['Tc_in'], self.Pc[0])['h']

            for k_stream in range(self.N_elements):
                # Cold Stream Element k_stream corresponds to Spatial Element N-1-k_stream
                # Cold GAINS heat.
                idx_spatial = self.N_elements - 1 - k_stream
                hc[k_stream + 1] = hc[k_stream] + Q_dist[idx_spatial] / mc

            # Solve Temps
            self.Th, self.Tc = self._solve_temperatures(hh, hc, self.Ph, self.Pc, self.xh)

            # 4. Kinetics
            self.xh = self._ortho_para_conversion(self.Th, self.Ph, self.xh, mh)

            # Convergence Check
            err = np.max(np.abs(self.Th - Th_old)) + np.max(np.abs(self.Tc - Tc_old))

            if (iteration + 1) % 50 == 0:
                print(f"Iter {iteration + 1:4d}: Max T error = {err:.6f}")

            if err < tolerance:
                elapsed = time.time() - start_time
                print(f"\nConverged in {iteration + 1} iterations ({elapsed:.2f} s)")
                self._print_results(props_h, props_c, U_dist, Q_dist)
                return True

        print(f"\nWARNING: Non-convergence. Error: {err:.6e}")
        return False

    def _calculate_properties_stream(self, T, P, x, m_dot, Ac, Dh, tpms_type, is_helium):
        N = len(T)
        props = {'h': np.zeros(N), 'cp': np.zeros(N), 'h_coeff': np.zeros(N)}

        for i in range(N):
            try:
                if is_helium:
                    p = self.h2_props.get_helium_properties(T[i], P[i])
                else:
                    p = self.h2_props.get_properties(T[i], P[i], x[i])
            except:
                # Fallback for safety during iteration spikes
                props['h'][i] = 0
                props['cp'][i] = 14000 if not is_helium else 5190
                props['h_coeff'][i] = 100
                continue

            props['h'][i] = p['h']
            props['cp'][i] = p['cp']

            # Correlations
            u = m_dot / (p['rho'] * Ac)
            Re = p['rho'] * u * Dh / p['mu']
            Pr = p['mu'] * p['cp'] / p['lambda']
            Nu, f = TPMSCorrelations.get_correlations(tpms_type, Re, Pr, 'Gas')

            catalyst_factor = 1.2 if not is_helium else 1.0
            props['h_coeff'][i] = catalyst_factor * Nu * p['lambda'] / Dh

        return props

    def _solve_temperatures(self, hh, hc, Ph, Pc, xh):
        N = len(hh)
        Th_new = np.zeros(N)
        Tc_new = np.zeros(N)

        for i in range(N):
            # Hot
            def res_h(T):
                if T < 14: return -1e6  # Force higher
                if T > 400: return 1e6
                try:
                    return self.h2_props.get_properties(T, Ph[i], xh[i])['h'] - hh[i]
                except:
                    return 0

            Th_new[i] = fsolve(res_h, self.Th[i])[0]

            # Cold
            def res_c(T):
                if T < 4: return -1e6
                if T > 400: return 1e6
                try:
                    return self.h2_props.get_helium_properties(T, Pc[i])['h'] - hc[i]
                except:
                    return 0

            Tc_new[i] = fsolve(res_c, self.Tc[i])[0]

        return Th_new, Tc_new

    # [Include _ortho_para_conversion, _print_results from previous code]
    # Make sure _print_results handles Q_dist correctly
    def _print_results(self, props_h, props_c, U, Q):
        print("=" * 70)
        print("RESULTS (Epsilon-NTU)")
        print("=" * 70)
        print(f"Hot Inlet (66K expected): {self.Th[0]:.2f} K")
        print(f"Hot Outlet: {self.Th[-1]:.2f} K")
        print(f"Cold Inlet (43K expected): {self.Tc[0]:.2f} K")
        print(f"Cold Outlet: {self.Tc[-1]:.2f} K")
        print(f"Total Heat Load: {np.sum(Q):.2f} W")

    # Placeholder for kinetics if needed
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

def create_default_config():
    """Create default configuration dictionary"""
    return {
        'geometry': {
            'length': 0.94,  # m
            'width': 0.15,  # m
            'height': 0.10,  # m
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
            'Th_in': 66.3,  # K
            'Th_out': 53.5,  # K
            'Tc_in': 43.5,  # K
            'Tc_out': 61.3,  # K
            'Ph_in': 1.13e6,  # Pa
            'Pc_in': 0.54e6,  # Pa
            'mh': 1e-3,  # kg/s
            'mc': 2e-3,  # kg/s
            'xh_in': 0.452  # Initial para fraction
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