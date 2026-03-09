"""
Visualization Module for TPMS Heat Exchanger (Academic Standard)

Provides comprehensive plotting functions for analyzing heat exchanger performance
with publication-quality formatting (Times New Roman, specific font sizes).
Updated for compatibility with the new dictionary-based TPMSHeatExchanger class.
"""

import os
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

        # --- 3. Performance Evaluation Metrics (from self.perf, if computed) ---
        perf = getattr(he, 'perf', {})
        if perf:
            data['j_Hot']      = pad(perf['j_h'])
            data['j_Cold']     = pad(perf['j_c'])
            data['PEC_Hot']    = pad(perf['PEC_h'])
            data['PEC_Cold']   = pad(perf['PEC_c'])
            data['Ex_dest_W']  = pad(perf['Ex_dest'])
            data['S_gen_W_K']  = pad(perf['S_gen'])
        else:
            nan_col = np.full(N + 1, np.nan)
            for col in ('j_Hot', 'j_Cold', 'PEC_Hot', 'PEC_Cold', 'Ex_dest_W', 'S_gen_W_K'):
                data[col] = nan_col.copy()

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save
        try:
            df.to_csv(filename, index=False)
            print(f"[OK] Results exported successfully to: {filename}")
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
        print(f"[OK] Performance plot saved to {save_path}")
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
                f"$\\Delta P_h$: {dP_hot:.1f} kPa\n" +
                f"Effectiveness: {Q_total/self.he.Q_max_capacity*100:.1f}%"
        )

        ax.text(0.05, 0.5, txt, transform=ax.transAxes,
                verticalalignment='center', linespacing=1.8,
                fontsize=14, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    def plot_resistance_pie(self, save_path=None):
        """Pie chart of element-averaged thermal resistance breakdown.

        For packed-bed channels, the hot/cold resistance is split into
        near-wall film and bed-conduction components.  For bare channels
        it appears as a single convection slice.

        Parameters
        ----------
        save_path : str or None
            File path to save the figure.  If None the figure is not saved.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        he = self.he

        # --- Build label/value lists dynamically ---
        labels = []
        values = []
        colors = []

        hot_mode  = he.config['channels']['hot'].get('mode', 'bare')
        cold_mode = he.config['channels']['cold'].get('mode', 'bare')
        hot_struct  = he.config['channels']['hot'].get('structure', '')
        cold_struct = he.config['channels']['cold'].get('structure', '')

        # Hot-side slices
        if hot_mode == 'packed' and np.any(he.R_hot_bed_cond > 0):
            labels += [f'Hot near-wall film\n({hot_struct})',
                       f'Hot bed conduction\n({hot_struct})']
            values += [float(np.mean(he.R_hot_wall_film)),
                       float(np.mean(he.R_hot_bed_cond))]
            colors += ['#d62728', '#ff7f0e']
        else:
            labels += [f'Hot convection\n({hot_struct})']
            values += [float(np.mean(he.R_hot))]
            colors += ['#d62728']

        # Wall slice
        labels += ['Wall conduction']
        values += [float(np.mean(he.R_wall))]
        colors += ['#7f7f7f']

        # Cold-side slices
        if cold_mode == 'packed' and np.any(he.R_cold_bed_cond > 0):
            labels += [f'Cold near-wall film\n({cold_struct})',
                       f'Cold bed conduction\n({cold_struct})']
            values += [float(np.mean(he.R_cold_wall_film)),
                       float(np.mean(he.R_cold_bed_cond))]
            colors += ['#1f77b4', '#17becf']
        else:
            labels += [f'Cold convection\n({cold_struct})']
            values += [float(np.mean(he.R_cold))]
            colors += ['#1f77b4']

        values = np.array(values, dtype=float)
        total  = values.sum()

        # Explode the dominant slice slightly
        explode = np.zeros(len(values))
        if total > 0:
            explode[np.argmax(values)] = 0.05

        fig, ax = plt.subplots(figsize=(7, 6))
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            explode=explode,
            startangle=90,
            pctdistance=0.78,
            wedgeprops=dict(linewidth=0.8, edgecolor='white'),
        )
        for t in autotexts:
            t.set_fontsize(10)

        ax.set_title('Thermal Resistance Breakdown\n(element-averaged)', fontsize=13)

        # Reference info box
        mean_UA = float(np.mean(1.0 / (he.R_hot + he.R_wall + he.R_cold)))
        info = (f"Mean UA = {mean_UA:.3g} W/K\n"
                f"Total R = {total:.3g} K/W")
        ax.text(1.25, -0.05, info, transform=ax.transAxes,
                fontsize=9, va='center',
                bbox=dict(facecolor='white', edgecolor='gray',
                          boxstyle='round,pad=0.4', alpha=0.9))

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=200)
            print(f"[OK] Resistance pie chart saved to {save_path}")
        plt.close(fig)
        return fig

    def plot_performance_evaluation(self, save_path='results/performance_evaluation.png'):
        """4-panel performance evaluation figure:
        (a) Exergy destruction profile
        (b) Colburn j-factor profiles (hot & cold)
        (c) PEC = j/f^(1/3) profiles (hot & cold)
        (d) Summary table of key indicators
        """
        perf = getattr(self.he, 'perf', {})
        if not perf:
            print("[WARN] Performance metrics not computed — skipping performance evaluation plot.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("Performance Evaluation — Exergy & PEC Analysis", fontsize=13, fontweight='bold')
        x_elem = np.linspace(0, 1, self.he.N)

        col_hot  = self.colors.get('hot',  '#d62728')
        col_cold = self.colors.get('cold', '#1f77b4')
        col_dest = self.colors.get('wall', '#7f7f7f')

        hot_struct  = self.he.streams['hot']['tpms']
        cold_struct = self.he.streams['cold']['tpms']

        # --- (a) Exergy destruction profile ---
        ax = axes[0, 0]
        ax.plot(x_elem, perf['Ex_hot_lost']  * 1e3, color=col_hot,  lw=1.5,
                label=f"Hot supplied ({hot_struct})")
        ax.plot(x_elem, perf['Ex_cold_gain'] * 1e3, color=col_cold, lw=1.5,
                label=f"Cold gained ({cold_struct})")
        ax.fill_between(x_elem, np.maximum(perf['Ex_dest'], 0) * 1e3,
                         color=col_dest, alpha=0.35, label="Destruction")
        ax.set_xlabel(r'Normalised position $\xi$')
        ax.set_ylabel('Exergy flux [mW]')
        ax.set_title('(a) Elemental Exergy Balance')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # --- (b) Colburn j-factor ---
        ax = axes[0, 1]
        ax.plot(x_elem, perf['j_h'], color=col_hot,  lw=1.5,
                label=f"Hot j  (mean={perf['j_mean_h']:.4f})")
        ax.plot(x_elem, perf['j_c'], color=col_cold, lw=1.5,
                label=f"Cold j  (mean={perf['j_mean_c']:.4f})")
        ax.axhline(perf['j_mean_h'], color=col_hot,  ls='--', lw=0.8, alpha=0.6)
        ax.axhline(perf['j_mean_c'], color=col_cold, ls='--', lw=0.8, alpha=0.6)
        ax.set_xlabel(r'Normalised position $\xi$')
        ax.set_ylabel(r'Colburn $j$-factor  $j = Nu\,Pr^{-1/3}/Re$')
        ax.set_title('(b) Heat Transfer: Colburn j-factor')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # --- (c) PEC = j / f^(1/3) ---
        ax = axes[1, 0]
        ax.plot(x_elem, perf['PEC_h'], color=col_hot,  lw=1.5,
                label=f"Hot PEC  (mean={perf['PEC_mean_h']:.4f})")
        ax.plot(x_elem, perf['PEC_c'], color=col_cold, lw=1.5,
                label=f"Cold PEC  (mean={perf['PEC_mean_c']:.4f})")
        ax.axhline(perf['PEC_mean_h'], color=col_hot,  ls='--', lw=0.8, alpha=0.6)
        ax.axhline(perf['PEC_mean_c'], color=col_cold, ls='--', lw=0.8, alpha=0.6)
        ax.set_xlabel(r'Normalised position $\xi$')
        ax.set_ylabel(r'PEC $= j\,/\,f^{1/3}$')
        ax.set_title(r'(c) Performance Evaluation Criterion  PEC $= j/f^{1/3}$')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # --- (d) Summary table ---
        ax = axes[1, 1]
        ax.axis('off')
        Q_total = float(np.sum(self.he.Q))
        effectiveness = Q_total / max(self.he.Q_max_capacity, 1e-12) * 100
        summary_lines = [
            r"$\bf{Exergy\ Analysis}$",
            f"  Dead-state T₀ = {perf['T0']:.2f} K  (Tc_in)",
            f"  Ex supplied (hot)  = {perf['Ex_hot_total']*1e3:.2f} mW",
            f"  Ex recovered (cold) = {perf['Ex_cold_total']*1e3:.2f} mW",
            f"  Ex destroyed       = {float(np.sum(np.maximum(perf['Ex_dest'],0)))*1e3:.2f} mW",
            f"  η_ex               = {perf['eta_ex']*100:.1f}%",
            f"  S_gen_total        = {perf['S_gen_total']*1e3:.3f} mW/K",
            "",
            r"$\bf{Thermo-Hydraulic\ Performance}$",
            f"  Effectiveness       = {effectiveness:.1f}%",
            f"  j_mean  hot / cold  = {perf['j_mean_h']:.4f} / {perf['j_mean_c']:.4f}",
            f"  PEC_mean hot / cold = {perf['PEC_mean_h']:.4f} / {perf['PEC_mean_c']:.4f}",
            f"  Structure  hot      = {hot_struct}",
            f"  Structure  cold     = {cold_struct}",
        ]
        ax.text(0.03, 0.97, "\n".join(summary_lines),
                transform=ax.transAxes, va='top', ha='left',
                fontsize=8.5, linespacing=1.6,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', alpha=0.8))
        ax.set_title('(d) Summary')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[OK] Performance evaluation plot saved to {save_path}")
        plt.close(fig)
        return fig


if __name__ == "__main__":
    print("TPMS Visualization Module (Academic Style) Loaded.")
