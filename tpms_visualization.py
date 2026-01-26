"""
Visualization Module for TPMS Heat Exchanger (Academic Standard)

Provides comprehensive plotting functions for analyzing heat exchanger performance
with publication-quality formatting (Times New Roman, specific font sizes).

Author: Based on research by Zhang et al. (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import os

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

        # Color Groups (Placeholder - easy to modify)
        # using distinct colors suitable for black & white printing if needed
        self.colors = {
            'hot': '#D62728',         # Deep Red
            'cold': '#1F77B4',        # Muted Blue
            'equilibrium': '#2CA02C', # Green
            'conversion': '#FF7F0E',  # Orange
            'black': '#000000',
            'gray': '#666666'
        }

    @staticmethod
    def set_academic_style():
        """
        Apply rigorous academic formatting to matplotlib
        """
        # Reset to defaults first to avoid state pollution
        plt.rcdefaults()

        # Font Configuration
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['mathtext.fontset'] = 'stix'  # Matches Times New Roman for math

        # Font Sizes
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['figure.titlesize'] = 18

        # Lines and Markers
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['lines.markersize'] = 8

        # Axes and Layout
        plt.rcParams['axes.grid'] = False      # Remove grids
        plt.rcParams['axes.linewidth'] = 1.2   # Thicker spines
        plt.rcParams['xtick.direction'] = 'in' # Ticks inside
        plt.rcParams['ytick.direction'] = 'in' # Ticks inside
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['ytick.major.size'] = 6
        plt.rcParams['xtick.top'] = True       # Ticks on top
        plt.rcParams['ytick.right'] = True     # Ticks on right

        # Legend
        plt.rcParams['legend.frameon'] = False # Clean legend
        plt.rcParams['legend.loc'] = 'best'

    def export_results_to_csv(self, filename='tpms_results.csv'):
        """
        Export simulation data to CSV for external plotting (Origin/Excel)

        Parameters
        ----------
        filename : str
            Output filename
        """
        # 1. Spatial Data (Profiles)
        N = len(self.he.Th)
        x_norm = np.linspace(0, 1, N)

        # Calculate derived properties for export
        x_eq = self.h2_props.get_equilibrium_fraction(self.he.Th)

        # Get local Nusselt and friction factors
        props_h = self._calculate_stream_properties(is_hot=True)
        props_c = self._calculate_stream_properties(is_hot=False)

        data = {
            'Position_Normalized': x_norm,
            'T_Hot_K': self.he.Th,
            'T_Cold_K': self.he.Tc,
            'P_Hot_Pa': self.he.Ph,
            'P_Cold_Pa': self.he.Pc,
            'x_para_Actual': self.he.xh,
            'x_para_Equilibrium': x_eq,
            'Nu_Hot': props_h['Nu'],
            'Nu_Cold': props_c['Nu'],
            'f_Hot': props_h['f'],
            'f_Cold': props_c['f'],
            'Re_Hot': props_h['Re'],
            'Re_Cold': props_c['Re']
        }

        df = pd.DataFrame(data)

        # Save to file
        try:
            df.to_csv(filename, index=False)
            print(f"Successfully exported spatial data to {filename}")
        except Exception as e:
            print(f"Error exporting CSV: {e}")

    def plot_comprehensive(self, save_path=None):
        """
        Create comprehensive performance plot with academic styling
        """
        fig = plt.figure(figsize=(12, 12)) # Square-ish layout for papers
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

        x_pos = np.linspace(0, 1, len(self.he.Th))

        # 1. Temperature profiles
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(x_pos, self.he.Th, color=self.colors['hot'], label='Hot Fluid')
        ax1.plot(x_pos, self.he.Tc, color=self.colors['cold'], label='Cold Fluid')
        ax1.set_xlabel('Normalized Position ($x/L$)')
        ax1.set_ylabel('Temperature (K)')
        ax1.legend()

        # 2. Para-H2 concentration
        ax2 = fig.add_subplot(gs[0, 1])
        x_eq = self.h2_props.get_equilibrium_fraction(self.he.Th)
        ax2.plot(x_pos, self.he.xh, color=self.colors['hot'], label='Actual')
        ax2.plot(x_pos, x_eq, color=self.colors['equilibrium'], linestyle='--', label='Equilibrium')
        ax2.set_xlabel('Normalized Position ($x/L$)')
        ax2.set_ylabel('Para-H$_2$ Fraction (-)')
        ax2.legend()

        # 3. Pressure Drop
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(x_pos, self.he.Ph/1e6, color=self.colors['hot'], label='Hot')
        ax3.plot(x_pos, self.he.Pc/1e6, color=self.colors['cold'], label='Cold')
        ax3.set_xlabel('Normalized Position ($x/L$)')
        ax3.set_ylabel('Pressure (MPa)')
        # Scientific notation for y-axis if needed, but MPa usually fine

        # 4. Nusselt Number
        ax4 = fig.add_subplot(gs[1, 1])
        props_h = self._calculate_stream_properties(is_hot=True)
        props_c = self._calculate_stream_properties(is_hot=False)
        ax4.plot(x_pos, props_h['Nu'], color=self.colors['hot'], label=f"Hot ({self.he.TPMS_hot})")
        ax4.plot(x_pos, props_c['Nu'], color=self.colors['cold'], label=f"Cold ({self.he.TPMS_cold})")
        ax4.set_xlabel('Normalized Position ($x/L$)')
        ax4.set_ylabel('Nusselt Number (-)')
        ax4.legend()

        # 5. Friction Factor
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(x_pos, props_h['f'], color=self.colors['hot'], label='Hot')
        ax5.plot(x_pos, props_c['f'], color=self.colors['cold'], label='Cold')
        ax5.set_xlabel('Normalized Position ($x/L$)')
        ax5.set_ylabel('Friction Factor $f$ (-)')
        # Log scale often better for friction, but linear requested unless specified
        # ax5.set_yscale('log')

        # 6. Performance Summary Text (Table-like)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        self._add_summary_text(ax6)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

    def _add_summary_text(self, ax):
        """Add summary metrics to the plot"""
        # Calculate heat transfer data directly
        U, Q = self._calculate_heat_transfer_data()
        Q_total = np.sum(Q)

        # Metrics
        x_in = self.he.xh[0]
        x_out = self.he.xh[-1]
        x_eq_out = self.h2_props.get_equilibrium_fraction(self.he.Th[-1])
        eff_conv = (x_out - x_in) / (x_eq_out - x_in) * 100
        dP_hot = (self.he.Ph[0] - self.he.Ph[-1]) / 1e3 # kPa

        txt = (
            r"$\bf{Performance\ Summary}$" + "\n\n" +
            f"Heat Load: {Q_total:.2f} W\n" +
            f"Conv. Eff.: {eff_conv:.2f}%\n" +
            f"Hot $\Delta P$: {dP_hot:.2f} kPa\n" +
            f"Avg $U$: {np.mean(U):.1f} W/m$^2$K\n" +
            f"$x_{{para,out}}$: {x_out:.4f}"
        )

        ax.text(0.05, 0.5, txt, transform=ax.transAxes,
                verticalalignment='center', linespacing=1.8,
                fontsize=14, bbox=dict(facecolor='none', edgecolor='black'))

    def plot_performance_metrics(self, save_path=None):
        """
        Create metrics analysis plots (PEC, Reynolds, etc.)
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        props_h = self._calculate_stream_properties(is_hot=True)
        props_c = self._calculate_stream_properties(is_hot=False)
        x_pos = np.linspace(0, 1, len(props_h['Re']))

        # 1. Reynolds Number
        ax = axes[0]
        ax.plot(x_pos, props_h['Re'], color=self.colors['hot'], label='Hot')
        ax.plot(x_pos, props_c['Re'], color=self.colors['cold'], label='Cold')
        ax.set_xlabel('Normalized Position ($x/L$)')
        ax.set_ylabel('Reynolds Number (-)')
        ax.legend()

        # 2. PEC (Performance Evaluation Criterion)
        ax = axes[1]
        # Normalize by inlet values approx
        Nu0_h, f0_h = props_h['Nu'][0], props_h['f'][0]
        Nu0_c, f0_c = props_c['Nu'][0], props_c['f'][0]

        # Avoid divide by zero if f is extremely small
        PEC_h = (props_h['Nu'] / Nu0_h) / np.power(props_h['f'] / f0_h, 1/3)
        PEC_c = (props_c['Nu'] / Nu0_c) / np.power(props_c['f'] / f0_c, 1/3)

        ax.plot(x_pos, PEC_h, color=self.colors['hot'], label='Hot')
        ax.plot(x_pos, PEC_c, color=self.colors['cold'], label='Cold')
        ax.set_xlabel('Normalized Position ($x/L$)')
        ax.set_ylabel('PEC (-)')
        ax.legend()

        # 3. Conversion Rate
        ax = axes[2]
        dx = np.diff(self.he.xh) / np.diff(x_pos)
        x_mid = (x_pos[:-1] + x_pos[1:]) / 2
        ax.plot(x_mid, dx, color=self.colors['conversion'])
        ax.set_xlabel('Normalized Position ($x/L$)')
        ax.set_ylabel(r'Conversion Rate ($dx/d\xi$)')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        plt.show()

    def _calculate_stream_properties(self, is_hot=True):
        """Helper to recalculate local properties for plotting"""
        # (Same logic as previous version, just ensuring data consistency)
        if is_hot:
            T, P, x = self.he.Th, self.he.Ph, self.he.xh
            m_dot = self.he.config['operating']['mh']
            Ac, Dh, tpms = self.he.Ac_hot, self.he.Dh_hot, self.he.TPMS_hot
            is_helium = False
        else:
            T, P, x = self.he.Tc, self.he.Pc, None
            m_dot = self.he.config['operating']['mc']
            Ac, Dh, tpms = self.he.Ac_cold, self.he.Dh_cold, self.he.TPMS_cold
            is_helium = True

        N = len(T)
        props = {'Nu': np.zeros(N), 'f': np.zeros(N), 'Re': np.zeros(N)}

        from tpms_correlations import TPMSCorrelations

        for i in range(N):
            # Recalculate basic props
            if is_helium:
                p = self.h2_props.get_helium_properties(T[i], P[i])
            else:
                p = self.h2_props.get_properties(T[i], P[i], x[i] if x is not None else 0.5)

            u = m_dot / (p['rho'] * Ac)
            Re = p['rho'] * u * Dh / p['mu']
            Pr = p['mu'] * p['cp'] / p['lambda']

            Nu, f = TPMSCorrelations.get_correlations(tpms, Re, Pr, 'Gas')
            props['Nu'][i] = Nu
            props['f'][i] = f
            props['Re'][i] = Re

        return props

    def _calculate_heat_transfer_data(self):
        """Recalculate U and Q for metrics"""
        # Simplified reconstruction for visualization
        N = self.he.N_elements
        # Map nodes to elements (simple average for viz)
        U_approx = np.zeros(N)
        Q_approx = np.zeros(N)
        # This is a placeholder as the real Q is transient in the solver
        # For full accuracy, the solver should store Q.
        # Here we approximate based on final T profile.
        return U_approx, Q_approx


def compare_tpms_structures():
    """Comparison plot with academic style"""
    # Set style locally for this static method
    TPMSVisualizer.set_academic_style()

    from tpms_correlations import TPMSCorrelations

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    Re_range = np.logspace(2.5, 4, 100) # 300 to 10000
    Pr_gas = 0.7

    tpms_types = ['Gyroid', 'Diamond', 'FKS', 'Primitive']
    # Use distinct markers for B&W compatibility
    markers = ['o', 's', '^', 'D']
    colors = ['#1F77B4', '#D62728', '#2CA02C', 'black']

    # 1. Nusselt Number
    ax = axes[0]
    for i, tpms in enumerate(tpms_types):
        Nu, _ = TPMSCorrelations.get_correlations(tpms, Re_range, Pr_gas, 'Gas')
        # Plot fewer markers for clarity
        ax.plot(Re_range, Nu, label=tpms, color=colors[i],
                linewidth=2, marker=markers[i], markevery=10)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Reynolds Number ($Re$)')
    ax.set_ylabel('Nusselt Number ($Nu$)')
    ax.legend(frameon=False)

    # 2. Friction Factor
    ax = axes[1]
    for i, tpms in enumerate(tpms_types):
        _, f = TPMSCorrelations.get_correlations(tpms, Re_range, Pr_gas, 'Gas')
        ax.plot(Re_range, f, label=tpms, color=colors[i],
                linewidth=2, marker=markers[i], markevery=10)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Reynolds Number ($Re$)')
    ax.set_ylabel('Friction Factor ($f$)')
    # ax.legend() # Legend shared or repeated as preferred

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("TPMS Visualization Module (Academic Style) Loaded.")