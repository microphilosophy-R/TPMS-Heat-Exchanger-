"""
Visualization Module for TPMS Heat Exchanger (Academic Standard)

Provides comprehensive plotting functions for analyzing heat exchanger performance
with publication-quality formatting (Times New Roman, specific font sizes).
Updated for compatibility with the new dictionary-based TPMSHeatExchanger class.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from tpms_correlations import TPMSCorrelations


class TPMSVisualizer:
    """Visualization tools for TPMS heat exchanger analysis with academic styling"""

    def __init__(self, heat_exchanger):
        """
        Initialize visualizer with heat exchanger object

        Parameters
        ----------
        heat_exchanger : TPMSHeatExchanger
            Solved heat exchanger object
        """
        self.he = heat_exchanger
        self.h2_props = heat_exchanger.h2_props

        # Apply Academic Style immediately
        self.set_academic_style()

        # Color Groups (optimized for contrast)
        self.colors = {
            'hot': '#D62728',  # Deep Red
            'cold': '#1F77B4',  # Muted Blue
            'equilibrium': '#2CA02C',  # Green
            'conversion': '#FF7F0E',  # Orange
            'black': '#000000',
            'gray': '#666666'
        }

    @staticmethod
    def set_academic_style():
        """
        Apply rigorous academic formatting to matplotlib
        """
        plt.rcdefaults()
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['mathtext.fontset'] = 'stix'

        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['legend.fontsize'] = 13

        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['lines.markersize'] = 6

        plt.rcParams['axes.grid'] = False
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True

        plt.rcParams['legend.frameon'] = False
        plt.rcParams['legend.loc'] = 'best'

    def export_results_to_csv(self, filename='tpms_results.csv'):
        """
        Export complete simulation data to CSV using the solver's internal state.
        Includes all Nodal (T, P, Properties) and Elemental (Nu, f, Q) data.
        Aligns arrays of size N and N+1 by padding.
        """
        he = self.he
        N = he.N
        # Nodal data has size N+1, Elemental data has size N
        x_norm = np.linspace(0, 1, N + 1)

        # Helper to pad elemental arrays (append NaN to last index)
        def pad(arr):
            return np.append(arr, np.nan)

        # --- 1. Nodal Data (Size N+1) ---
        data = {
            'Position_Normalized': x_norm,

            # Hot Stream Nodal
            'T_Hot_K': he.Th,
            'P_Hot_Pa': he.Ph,
            'x_para_Hot': he.xh,
            'rho_Hot': he.props_h['rho'],
            'mu_Hot': he.props_h['mu'],
            'cp_Hot': he.props_h['cp'],
            'k_Hot': he.props_h['k'],
            'h_enthalpy_Hot': he.props_h['h'],

            # Cold Stream Nodal
            'T_Cold_K': he.Tc,
            'P_Cold_Pa': he.Pc,
            'rho_Cold': he.props_c['rho'],
            'mu_Cold': he.props_c['mu'],
            'cp_Cold': he.props_c['cp'],
            'k_Cold': he.props_c['k'],
            'h_enthalpy_Cold': he.props_c['h'],
        }

        # Calculate Equilibrium for reference
        if 'hydrogen' in he.streams['hot']['species']:
            data['x_para_Equilibrium'] = self.h2_props.get_equilibrium_fraction(he.Th)
        else:
            data['x_para_Equilibrium'] = np.zeros(N + 1)

        # --- 2. Elemental Data (Size N -> Padded to N+1) ---
        # Note: Elemental index i usually corresponds to the control volume starting at node i
        elemental_vars = {
            # Heat Transfer
            'Q_Exchange_W': he.Q,
            'U_Overall_W_m2K': he.U,

            # Hot Stream Elemental
            'Re_Hot': he.elem_h['Re'],
            'Pr_Hot': he.elem_h['Pr'],
            'Nu_Hot': he.elem_h['Nu'],
            'f_Hot': he.elem_h['f'],
            'htc_Hot': he.elem_h['htc'],

            # Cold Stream Elemental
            'Re_Cold': he.elem_c['Re'],
            'Pr_Cold': he.elem_c['Pr'],
            'Nu_Cold': he.elem_c['Nu'],
            'f_Cold': he.elem_c['f'],
            'htc_Cold': he.elem_c['htc'],
        }

        # Add padded elemental data to dictionary
        for key, arr in elemental_vars.items():
            data[key] = pad(arr)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save
        try:
            df.to_csv(filename, index=False)
            print(f"✓ Results exported successfully to: {filename}")
        except Exception as e:
            print(f"Error exporting CSV: {e}")

    def plot_comprehensive(self, save_path='operation_profile.png'):
        """
        Create comprehensive performance plot (4-panel).
        Uses stored solver state directly for consistency.
        """
        fig = plt.figure(figsize=(12, 15))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        x_pos = np.linspace(0, 1, len(self.he.Th))
        # For elemental plots, we plot against element centers or just slice the x_pos array
        x_elem = x_pos[:-1] # Size N

        # --- 1. Temperature Profiles ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(x_pos, self.he.Th, color=self.colors['hot'], label='Hot Stream')
        ax1.plot(x_pos, self.he.Tc, color=self.colors['cold'], label='Cold Stream')
        ax1.set_xlabel(r'Normalized Position ($\xi = x/L$)')
        ax1.set_ylabel('Temperature (K)')
        ax1.legend()
        ax1.set_title('(a) Temperature Distribution')

        # --- 2. Para-H2 Conversion ---
        ax2 = fig.add_subplot(gs[0, 1])
        if 'hydrogen' in self.he.streams['hot']['species']:
            x_eq = self.h2_props.get_equilibrium_fraction(self.he.Th)
            ax2.plot(x_pos, self.he.xh, color=self.colors['hot'], label='Actual $x_p$')
            ax2.plot(x_pos, x_eq, color=self.colors['equilibrium'], linestyle='--', label='Equilibrium $x_{eq}$')
            ax2.set_ylabel('Para-Hydrogen Fraction (-)')
        else:
            ax2.text(0.5, 0.5, "No Conversion\n(Inert Fluid)", ha='center', va='center')

        ax2.set_xlabel(r'Normalized Position ($\xi = x/L$)')
        ax2.set_title('(b) Ortho-Para Conversion')
        ax2.legend()

        # --- 3. Hydraulic Performance (Pressure) ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(x_pos, self.he.Ph / 1e6, color=self.colors['hot'], label='Hot')
        ax3.plot(x_pos, self.he.Pc / 1e6, color=self.colors['cold'], label='Cold')
        ax3.set_xlabel(r'Normalized Position ($\xi = x/L$)')
        ax3.set_ylabel('Pressure (MPa)')
        ax3.set_title('(c) Pressure Profiles')
        ax3.legend()

        # --- 4. Heat Transfer Coefficients (Nusselt) ---
        ax4 = fig.add_subplot(gs[1, 1])

        # Use stored elemental data
        # We use 'step' plot or align with element centers. Here simple plot against x_elem
        ax4.plot(x_elem, self.he.elem_h['Nu'], color=self.colors['hot'],
                 label=f"Hot ({self.he.streams['hot']['tpms']})")
        ax4.plot(x_elem, self.he.elem_c['Nu'], color=self.colors['cold'],
                 label=f"Cold ({self.he.streams['cold']['tpms']})")

        ax4.set_xlabel(r'Normalized Position ($\xi = x/L$)')
        ax4.set_ylabel('Nusselt Number (-)')
        ax4.set_title('(d) Heat Transfer Performance')
        ax4.legend()

        # 5. Friction Factor
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(x_elem, self.he.elem_h['f'], color=self.colors['hot'], label='Hot')
        ax5.plot(x_elem, self.he.elem_c['f'], color=self.colors['cold'], label='Cold')
        ax5.set_xlabel('Normalized Position ($x/L$)')
        ax5.set_ylabel('Friction Factor $f$ (-)')

        # 6. Performance Summary Text (Table-like)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        self._add_summary_text(ax6)

        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✓ Performance plot saved to {save_path}")
        # plt.show() # Optional: Comment out if running in batch mode without display

    def _add_summary_text(self, ax):
        """Add summary metrics to the plot"""
        # Calculate totals from Solver State
        Q_total = np.sum(self.he.Q)

        # Metrics
        x_in = self.he.xh[0]
        x_out = self.he.xh[-1]
        try:
            x_eq_out = self.h2_props.get_equilibrium_fraction(self.he.Th[-1])
            eff_conv = (x_out - x_in) / (x_eq_out - x_in) * 100 if (x_eq_out - x_in) != 0 else 0
        except:
            eff_conv = 0.0

        dP_hot = (self.he.Ph[0] - self.he.Ph[-1]) / 1e3  # kPa

        txt = (
                f"Load: {Q_total:.1f} W\n" +
                f"Conv Eff: {eff_conv:.1f}%\n" +
                f"$\Delta P_h$: {dP_hot:.1f} kPa\n" +
                f"Effectiveness: {Q_total/self.he.Q_max_capacity*100:.1f}%"
        )

        ax.text(0.05, 0.5, txt, transform=ax.transAxes,
                verticalalignment='center', linespacing=1.8,
                fontsize=14, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))


if __name__ == "__main__":
    print("TPMS Visualization Module (Academic Style) Loaded.")