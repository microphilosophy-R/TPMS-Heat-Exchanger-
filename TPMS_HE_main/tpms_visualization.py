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

        # --- 3. Hydraulic Pressure Loss ---
        ax3 = fig.add_subplot(gs[1, 0])
        # Cumulative pressure drop from each stream's inlet [kPa]
        # Hot flows 0→N: loss increases left to right
        # Cold flows N→0: loss increases right to left
        dP_hot_profile  = (self.he.Ph[0]  - self.he.Ph)  / 1e3
        dP_cold_profile = (self.he.Pc[-1] - self.he.Pc) / 1e3
        ax3.plot(x_pos, dP_hot_profile,  color=self.colors['hot'],  label=f'Hot  (total {dP_hot_profile[-1]:.2f} kPa)')
        ax3.plot(x_pos, dP_cold_profile, color=self.colors['cold'], label=f'Cold (total {dP_cold_profile[0]:.2f} kPa)')
        ax3.set_xlabel(r'Normalized Position ($\xi = x/L$)')
        ax3.set_ylabel(r'Hydraulic Pressure Loss $\Delta P$ (kPa)')
        ax3.set_title('(c) Hydraulic Pressure Loss')
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
        ax5.set_title('(e) Friction Factor Profile')
        ax5.legend()

        # 6. Performance Summary Text (Table-like)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        self._add_summary_text(ax6)

        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[OK] Performance plot saved to {save_path}")
        # plt.show() # Optional: Comment out if running in batch mode without display

    def _add_summary_text(self, ax):
        """Add summary metrics with exergy analysis to the plot"""
        he = self.he
        # Use inlet/outlet enthalpy balance (includes conversion heat)
        mh = he.streams['hot']['m']
        Q_total = mh * (he.props_h['h'][0] - he.props_h['h'][-1])

        # --- Temperature ---
        Th_in,  Th_out = he.Th[0],  he.Th[-1]
        Tc_in,  Tc_out = he.Tc[-1], he.Tc[0]   # cold inlet = Tc[-1] (counter-flow)
        dTh = Th_in  - Th_out
        dTc = Tc_out - Tc_in
        LMTD_num = (Th_in - Tc_out) - (Th_out - Tc_in)
        LMTD_den = np.log(max((Th_in - Tc_out) / max(Th_out - Tc_in, 1e-6), 1e-6))
        LMTD = abs(LMTD_num / LMTD_den) if abs(LMTD_den) > 1e-9 else abs(LMTD_num)

        # --- Pressure ---
        Ph_in,  Ph_out = he.Ph[0]  / 1e6, he.Ph[-1]  / 1e6   # MPa
        Pc_in,  Pc_out = he.Pc[-1] / 1e6, he.Pc[0]   / 1e6
        dP_hot  = (he.Ph[0]  - he.Ph[-1])  / 1e3   # kPa
        dP_cold = (he.Pc[-1] - he.Pc[0])   / 1e3

        # --- Para-H2 ---
        x_in  = he.xh[0]
        x_out = he.xh[-1]
        try:
            x_eq_in  = self.h2_props.get_equilibrium_fraction(Th_in)
            x_eq_out = self.h2_props.get_equilibrium_fraction(Th_out)
            eff_conv = (x_out - x_in) / (x_eq_out - x_in) * 100 if abs(x_eq_out - x_in) > 1e-6 else 0.0
        except:
            x_eq_in, x_eq_out, eff_conv = 0.0, 0.0, 0.0

        # --- Mean U ---
        mean_U = float(np.mean(he.U))

        lines = [
            "--- Thermo-Hydraulic Summary ---",
            "",
            "Temperatures (K)",
            f"  Hot:   {Th_in:.2f} \u2192 {Th_out:.2f}  (\u0394T = {dTh:.2f} K)",
            f"  Cold:  {Tc_in:.2f} \u2192 {Tc_out:.2f}  (\u0394T = {dTc:.2f} K)",
            f"  LMTD:  {LMTD:.2f} K",
            "",
            "Pressures",
            f"  Hot:   {Ph_in:.3f} \u2192 {Ph_out:.3f} MPa  (\u0394P = {dP_hot:.2f} kPa)",
            f"  Cold:  {Pc_in:.3f} \u2192 {Pc_out:.3f} MPa  (\u0394P = {dP_cold:.2f} kPa)",
            "",
            "Para-H2 Fraction",
            f"  Actual:  {x_in:.4f} \u2192 {x_out:.4f}",
            f"  Equil.:  {x_eq_in:.4f} \u2192 {x_eq_out:.4f}",
            f"  Conv. eff.: {eff_conv:.1f}%",
            "",
            "Heat Transfer",
            f"  Q total:       {Q_total:.1f} W",
            f"  Effectiveness: {Q_total / max(he.Q_max_capacity, 1e-12) * 100:.1f}%",
            f"  Mean U:        {mean_U:.1f} W/m\u00b2K",
        ]

        perf = getattr(he, 'perf', {})
        if perf:
            dest_grand  = perf.get('Ex_dest_grand',    0.0)
            dest_HT_tot = perf.get('Ex_dest_HT_tot',   0.0)
            dest_dP_tot = perf.get('Ex_dest_dP_tot',   0.0)
            dest_ch_tot = perf.get('Ex_dest_chem_tot', 0.0)
            pct_HT   = 100.0 * dest_HT_tot / max(dest_grand, 1e-12)
            pct_dP   = 100.0 * dest_dP_tot / max(dest_grand, 1e-12)
            pct_chem = 100.0 * dest_ch_tot / max(dest_grand, 1e-12)
            Ex_He    = perf.get('Ex_He_consumed',   perf.get('Ex_cold_net', 0.0))
            Ex_H2    = perf.get('Ex_H2_total_gain', perf.get('Ex_hot_net',  0.0))
            Ex_chem  = perf.get('Ex_chem_net', 0.0)
            lines += [
                "",
                f"--- Exergy Analysis  (T0={perf['T0']:.0f} K) ---",
                f"  \u03b7_ex (HX, thermal): {perf['eta_ex'] * 100:.1f}%",
                f"  He cold ex consumed: {Ex_He:.4f} W",
                f"  H2 total ex gained:  {Ex_H2:.4f} W",
                f"  Ex destr (total):    {dest_grand:.4f} W",
                f"    \u21b3 Heat xfer \u0394T: {dest_HT_tot:.4f} W  ({pct_HT:.0f}%)",
                f"    \u21b3 Press. drop \u0394P:{dest_dP_tot:.4f} W  ({pct_dP:.0f}%)",
                f"    \u21b3 Ortho-para:  {dest_ch_tot:.4f} W  ({pct_chem:.0f}%)",
                f"  S_gen total:   {perf['S_gen_total'] * 1e3:.3f} mW/K",
                f"    \u21b3 \u0394T: {perf.get('S_gen_HT_tot',0.0)*1e3:.3f}"
                f"  \u0394P: {perf.get('S_gen_dP_tot',0.0)*1e3:.3f}"
                f"  chem: {perf.get('S_gen_chem_tot',0.0)*1e3:.3f} mW/K",
                "",
                "--- Transfer Performance ---",
                f"  j_mean  h/c:   {perf['j_mean_h']:.4f} / {perf['j_mean_c']:.4f}",
                f"  PEC_mean h/c:  {perf['PEC_mean_h']:.4f} / {perf['PEC_mean_c']:.4f}",
            ]

        txt = "\n".join(lines)
        ax.text(0.05, 0.97, txt, transform=ax.transAxes,
                verticalalignment='top', linespacing=1.45,
                fontsize=8.5, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

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

        # --- (a) Exergy balance — cryogenic convention ---
        # Both streams operate below ambient T0. Roles:
        #   Refrigerant (cold stream, warmer) SUPPLIES cold exergy as it warms.
        #   Product     (hot  stream, cooler) RECEIVES cold exergy as it cools further.
        # Destruction is decomposed: heat-transfer ΔT (red fill) + ortho-para rxn (orange fill).
        ax = axes[0, 0]
        ax.plot(x_elem, perf['Ex_cold_supplied'], color=col_cold, lw=1.5,
                label=f"Refrigerant supplied ({cold_struct})")
        ax.plot(x_elem, perf['Ex_hot_received'],  color=col_hot,  lw=1.5,
                label=f"Product received ({hot_struct})")
        # Stacked destruction: heat-transfer (bottom) then chemical (on top)
        dest_HT   = np.maximum(perf['Ex_dest_HT'],  0)
        dest_chem = np.maximum(perf['Ex_dest_chem'], 0)
        ax.fill_between(x_elem, dest_HT,
                        color='#d62728', alpha=0.40,
                        label=r'Dest: $\Delta T$ (heat transfer)')
        ax.fill_between(x_elem, dest_HT + dest_chem,
                        dest_HT,
                        color='#ff7f0e', alpha=0.40,
                        label=r'Dest: ortho-para rxn')
        ax.set_xlabel(r'Normalised position $\xi$')
        ax.set_ylabel('Exergy flux [W]')
        ax.set_title(f'(a) Elemental Exergy Balance  ($T_0={perf["T0"]:.0f}$ K, ambient)')
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
        # Use inlet/outlet enthalpy balance (includes conversion heat)
        _mh = self.he.streams['hot']['m']
        Q_total = float(_mh * (self.he.props_h['h'][0] - self.he.props_h['h'][-1]))
        effectiveness = Q_total / max(self.he.Q_max_capacity, 1e-12) * 100

        dest_grand      = perf.get('Ex_dest_grand',    0.0)
        dest_HT_tot     = perf.get('Ex_dest_HT_tot',   0.0)
        dest_dP_tot     = perf.get('Ex_dest_dP_tot',   0.0)
        dest_ch_tot     = perf.get('Ex_dest_chem_tot', 0.0)
        pct_HT   = 100.0 * dest_HT_tot  / max(dest_grand, 1e-12)
        pct_dP   = 100.0 * dest_dP_tot  / max(dest_grand, 1e-12)
        pct_chem = 100.0 * dest_ch_tot  / max(dest_grand, 1e-12)

        Ex_He   = perf.get('Ex_He_consumed',   perf.get('Ex_cold_net', 0.0))
        Ex_H2   = perf.get('Ex_H2_total_gain', perf.get('Ex_hot_net',  0.0))
        Ex_chem = perf.get('Ex_chem_net',      0.0)
        ex_bal  = perf.get('Ex_balance_residual', Ex_He - Ex_H2 - dest_grand)

        summary_lines = [
            "--- Exergy Analysis (Gouy-Stodola, 3 sources) ---",
            f"  Dead-state T0 = {perf['T0']:.1f} K  (ambient)",
            f"  He cold exergy consumed  = {Ex_He:.4f} W",
            f"  H2 total exergy gained   = {Ex_H2:.4f} W",
            f"  Ex destroyed (total)        = {dest_grand:.4f} W",
            f"    \u21b3 Heat transfer \u0394T     = {dest_HT_tot:.4f} W  ({pct_HT:.0f}%)",
            f"    \u21b3 Pressure drop \u0394P     = {dest_dP_tot:.4f} W  ({pct_dP:.0f}%)",
            f"    \u21b3 Ortho-para rxn        = {dest_ch_tot:.4f} W  ({pct_chem:.0f}%)",
            f"  Balance residual         = {ex_bal:.4f} W  (\u2248 0)",
            f"  \u03b7_ex (HX, thermal)       = {perf['eta_ex']*100:.1f}%  (\u2264 100%)",
            f"  S_gen total  = {perf['S_gen_total']*1e3:.3f} mW/K",
            f"    \u21b3 Heat xfer: {perf.get('S_gen_HT_tot', 0.0)*1e3:.3f}"
            f"  | \u0394P: {perf.get('S_gen_dP_tot', 0.0)*1e3:.3f}"
            f"  | chem: {perf.get('S_gen_chem_tot', 0.0)*1e3:.3f} mW/K",
            "",
            "--- Thermo-Hydraulic Performance ---",
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
