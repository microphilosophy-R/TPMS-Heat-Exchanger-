"""
Visualization Module for TPMS Heat Exchanger

Provides comprehensive plotting functions for analyzing heat exchanger performance.

Author: Based on research by Zhang et al. (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class TPMSVisualizer:
    """Visualization tools for TPMS heat exchanger analysis"""
    
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
        
        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'hot': '#d62728',    # Red
            'cold': '#1f77b4',   # Blue
            'equilibrium': '#2ca02c',  # Green
            'conversion': '#ff7f0e'    # Orange
        }
    
    def plot_comprehensive(self, save_path=None):
        """
        Create comprehensive 9-panel performance plot
        
        Parameters
        ----------
        save_path : str, optional
            Path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Normalized position
        x_pos = np.linspace(0, 1, len(self.he.Th))
        x_elem = np.linspace(0, 1, self.he.N_elements)
        
        # 1. Temperature profiles
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_temperatures(ax1, x_pos)
        
        # 2. Para-H2 concentration
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_concentration(ax2, x_pos)
        
        # 3. Degree of non-equilibrium
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_non_equilibrium(ax3, x_pos)
        
        # 4. Pressure profiles
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_pressures(ax4, x_pos)
        
        # 5. Heat transfer rate
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_heat_transfer(ax5, x_elem)
        
        # 6. Overall U
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_U_coefficient(ax6, x_elem)
        
        # 7. Nusselt numbers
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_nusselt(ax7, x_pos)
        
        # 8. Friction factors
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_friction(ax8, x_elem)
        
        # 9. Performance summary
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_summary(ax9)
        
        # Overall title
        fig.suptitle(f'TPMS Heat Exchanger Performance: {self.he.TPMS_hot} (Hot) / {self.he.TPMS_cold} (Cold)',
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def _plot_temperatures(self, ax, x_pos):
        """Plot temperature profiles"""
        ax.plot(x_pos, self.he.Th, color=self.colors['hot'], 
               linewidth=2, label='Hot Fluid')
        ax.plot(x_pos, self.he.Tc, color=self.colors['cold'], 
               linewidth=2, label='Cold Fluid')
        ax.set_xlabel('Normalized Position [-]')
        ax.set_ylabel('Temperature [K]')
        ax.set_title('Temperature Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_concentration(self, ax, x_pos):
        """Plot para-hydrogen concentration"""
        # Calculate equilibrium concentration
        x_eq = self.h2_props.get_equilibrium_fraction(self.he.Th)
        
        ax.plot(x_pos, self.he.xh, color=self.colors['hot'], 
               linewidth=2, label='Actual')
        ax.plot(x_pos, x_eq, color=self.colors['equilibrium'], 
               linestyle='--', linewidth=1.5, label='Equilibrium')
        ax.set_xlabel('Normalized Position [-]')
        ax.set_ylabel('Para-H₂ Concentration [-]')
        ax.set_title('Ortho-Para Hydrogen Conversion')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_non_equilibrium(self, ax, x_pos):
        """Plot degree of non-equilibrium"""
        x_eq = self.h2_props.get_equilibrium_fraction(self.he.Th)
        DNE = (x_eq - self.he.xh) / x_eq * 100
        
        ax.plot(x_pos, DNE, color='black', linewidth=2)
        ax.set_xlabel('Normalized Position [-]')
        ax.set_ylabel('Degree of Non-Equilibrium [%]')
        ax.set_title('Conversion Non-Equilibrium')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    def _plot_pressures(self, ax, x_pos):
        """Plot pressure profiles"""
        ax.plot(x_pos, self.he.Ph/1e6, color=self.colors['hot'], 
               linewidth=2, label='Hot Fluid')
        ax.plot(x_pos, self.he.Pc/1e6, color=self.colors['cold'], 
               linewidth=2, label='Cold Fluid')
        ax.set_xlabel('Normalized Position [-]')
        ax.set_ylabel('Pressure [MPa]')
        ax.set_title('Pressure Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_heat_transfer(self, ax, x_elem):
        """Plot element-wise heat transfer"""
        props_h = self.he._calculate_hot_properties()
        props_c = self.he._calculate_cold_properties()
        _, Q = self.he._calculate_heat_transfer(props_h, props_c)
        
        ax.bar(x_elem, Q, width=1/len(Q), color=self.colors['conversion'], alpha=0.7)
        ax.set_xlabel('Normalized Position [-]')
        ax.set_ylabel('Heat Transfer Rate [W]')
        ax.set_title('Element-wise Heat Transfer')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_U_coefficient(self, ax, x_elem):
        """Plot overall heat transfer coefficient"""
        props_h = self.he._calculate_hot_properties()
        props_c = self.he._calculate_cold_properties()
        U, _ = self.he._calculate_heat_transfer(props_h, props_c)
        
        ax.plot(x_elem, U, color=self.colors['conversion'], linewidth=2)
        ax.set_xlabel('Normalized Position [-]')
        ax.set_ylabel('U [W/(m²·K)]')
        ax.set_title('Overall Heat Transfer Coefficient')
        ax.grid(True, alpha=0.3)
    
    def _plot_nusselt(self, ax, x_pos):
        """Plot Nusselt numbers"""
        props_h = self.he._calculate_hot_properties()
        props_c = self.he._calculate_cold_properties()
        
        ax.plot(x_pos, props_h['Nu'], color=self.colors['hot'], 
               linewidth=2, label=f'Hot ({self.he.TPMS_hot})')
        ax.plot(x_pos, props_c['Nu'], color=self.colors['cold'], 
               linewidth=2, label=f'Cold ({self.he.TPMS_cold})')
        ax.set_xlabel('Normalized Position [-]')
        ax.set_ylabel('Nusselt Number [-]')
        ax.set_title('TPMS Heat Transfer Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_friction(self, ax, x_elem):
        """Plot friction factors"""
        props_h = self.he._calculate_hot_properties()
        props_c = self.he._calculate_cold_properties()
        
        ax.plot(x_elem, props_h['f'][:-1], color=self.colors['hot'], 
               linewidth=2, label=f'Hot ({self.he.TPMS_hot})')
        ax.plot(x_elem, props_c['f'][:-1], color=self.colors['cold'], 
               linewidth=2, label=f'Cold ({self.he.TPMS_cold})')
        ax.set_xlabel('Normalized Position [-]')
        ax.set_ylabel('Friction Factor [-]')
        ax.set_title('TPMS Flow Resistance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_summary(self, ax):
        """Plot performance summary"""
        ax.axis('off')
        
        # Calculate metrics
        props_h = self.he._calculate_hot_properties()
        props_c = self.he._calculate_cold_properties()
        U, Q = self.he._calculate_heat_transfer(props_h, props_c)
        
        Q_total = np.sum(Q)
        dP_hot = self.he.Ph[0] - self.he.Ph[-1]
        dP_cold = self.he.Pc[0] - self.he.Pc[-1]
        dT_hot = self.he.Th[0] - self.he.Th[-1]
        dT_cold = self.he.Tc[-1] - self.he.Tc[0]
        
        x_eq_out = self.h2_props.get_equilibrium_fraction(self.he.Th[-1])
        x_eq_in = self.h2_props.get_equilibrium_fraction(self.he.Th[0])
        conv_eff = (self.he.xh[-1] - self.he.xh[0]) / (x_eq_out - x_eq_in) * 100
        
        # Create text summary
        summary_text = [
            r'$\bf{TPMS\ Heat\ Exchanger\ Summary}$',
            '',
            r'$\bf{Structure:}$',
            f'Hot side: {self.he.TPMS_hot}',
            f'Cold side: {self.he.TPMS_cold}',
            '',
            r'$\bf{Thermal\ Performance:}$',
            f'Total heat: {Q_total:.2f} W',
            f'Hot ΔT: {dT_hot:.2f} K',
            f'Cold ΔT: {dT_cold:.2f} K',
            f'Avg U: {np.mean(U):.2f} W/(m²·K)',
            '',
            r'$\bf{Flow\ Performance:}$',
            f'Hot ΔP: {dP_hot/1e3:.2f} kPa',
            f'Cold ΔP: {dP_cold/1e3:.2f} kPa',
            '',
            r'$\bf{Conversion:}$',
            f'Efficiency: {conv_eff:.2f}%',
            f'$x_{{in}}$: {self.he.xh[0]:.4f}',
            f'$x_{{out}}$: {self.he.xh[-1]:.4f}'
        ]
        
        ax.text(0.1, 0.95, '\n'.join(summary_text), 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', family='monospace')
    
    def plot_performance_metrics(self, save_path=None):
        """
        Create additional performance analysis plots
        
        Parameters
        ----------
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('TPMS Performance Metrics Analysis', fontsize=16, fontweight='bold')
        
        props_h = self.he._calculate_hot_properties()
        props_c = self.he._calculate_cold_properties()
        
        # Temperature-Nusselt relationship
        ax = axes[0, 0]
        ax2 = ax.twinx()
        ln1 = ax.plot(self.he.Th, props_h['Nu'], 'ro-', label='Nu_hot')
        ln2 = ax2.plot(self.he.Th, props_c['Nu'], 'bo-', label='Nu_cold')
        ax.set_xlabel('Hot Fluid Temperature [K]')
        ax.set_ylabel('Nu_hot', color='r')
        ax2.set_ylabel('Nu_cold', color='b')
        ax.set_title('Nusselt Number vs Temperature')
        ax.grid(True, alpha=0.3)
        
        # Temperature-Friction relationship
        ax = axes[0, 1]
        ax2 = ax.twinx()
        ln1 = ax.plot(self.he.Th[:-1], props_h['f'][:-1], 'ro-', label='f_hot')
        ln2 = ax2.plot(self.he.Th[:-1], props_c['f'][:-1], 'bo-', label='f_cold')
        ax.set_xlabel('Hot Fluid Temperature [K]')
        ax.set_ylabel('f_hot', color='r')
        ax2.set_ylabel('f_cold', color='b')
        ax.set_title('Friction Factor vs Temperature')
        ax.grid(True, alpha=0.3)
        
        # Conversion rate distribution
        ax = axes[0, 2]
        dx = np.diff(self.he.xh)
        dx_pos = np.diff(np.linspace(0, 1, len(self.he.xh)))
        conv_rate = dx / dx_pos
        x_pos = np.linspace(0, 1, len(conv_rate))
        ax.plot(x_pos, conv_rate, 'k-', linewidth=2)
        ax.set_xlabel('Normalized Position [-]')
        ax.set_ylabel('dx_para/dx [-]')
        ax.set_title('Conversion Rate Distribution')
        ax.grid(True, alpha=0.3)
        
        # PEC distribution
        ax = axes[1, 0]
        Nu0_hot = props_h['Nu'][0]
        Nu0_cold = props_c['Nu'][0]
        f0_hot = props_h['f'][0]
        f0_cold = props_c['f'][0]
        
        PEC_hot = (props_h['Nu'][:-1]/Nu0_hot) / (props_h['f'][:-1]/f0_hot)**(1/3)
        PEC_cold = (props_c['Nu'][:-1]/Nu0_cold) / (props_c['f'][:-1]/f0_cold)**(1/3)
        
        x_elem = np.linspace(0, 1, len(PEC_hot))
        ax.plot(x_elem, PEC_hot, 'r-', linewidth=2, label='Hot')
        ax.plot(x_elem, PEC_cold, 'b-', linewidth=2, label='Cold')
        ax.axhline(y=1, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel('Normalized Position [-]')
        ax.set_ylabel('PEC [-]')
        ax.set_title('Performance Evaluation Criterion')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Thermal effectiveness
        ax = axes[1, 1]
        U, Q = self.he._calculate_heat_transfer(props_h, props_c)
        Q_total = np.sum(Q)
        mh = self.he.config['operating']['mh']
        mc = self.he.config['operating']['mc']
        dT_hot = self.he.Th[0] - self.he.Th[-1]
        dT_cold = self.he.Tc[-1] - self.he.Tc[0]
        C_hot = Q_total / dT_hot
        C_cold = Q_total / dT_cold
        C_min = min(C_hot, C_cold)
        effectiveness = Q_total / (C_min * (self.he.Th[0] - self.he.Tc[0]))
        
        bars = ax.bar(['Effectiveness', 'Loss'], [effectiveness, 1-effectiveness],
                     color=[self.colors['conversion'], 'gray'], alpha=0.7)
        ax.set_ylabel('Fraction [-]')
        ax.set_title(f'Thermal Effectiveness: {effectiveness*100:.2f}%')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Reynolds number distribution
        ax = axes[1, 2]
        x_pos = np.linspace(0, 1, len(props_h['Re']))
        ax.plot(x_pos, props_h['Re'], 'r-', linewidth=2, label='Hot')
        ax.plot(x_pos, props_c['Re'], 'b-', linewidth=2, label='Cold')
        ax.set_xlabel('Normalized Position [-]')
        ax.set_ylabel('Reynolds Number [-]')
        ax.set_title('Reynolds Number Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()


def compare_tpms_structures():
    """Create comparison plots for different TPMS structures"""
    from tpms_correlations import TPMSCorrelations
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Comprehensive TPMS Structure Comparison', fontsize=16, fontweight='bold')
    
    Re_range = np.logspace(1, 4, 100)
    Pr_gas = 0.7
    Pr_water = 6.0
    
    tpms_types = ['Gyroid', 'Diamond', 'Primitive', 'FKS']
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    
    # Gas correlations: Nu-Re
    ax = axes[0, 0]
    for i, tpms in enumerate(tpms_types):
        Nu, _ = TPMSCorrelations.get_correlations(tpms, Re_range, Pr_gas, 'Gas')
        valid = ~np.isnan(Nu)
        ax.loglog(Re_range[valid], Nu[valid], linewidth=2.5, 
                 color=colors[i], label=tpms)
    ax.set_xlabel('Reynolds Number, Re [-]')
    ax.set_ylabel('Nusselt Number, Nu [-]')
    ax.set_title('Gas (H₂/He, Pr = 0.7)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gas correlations: f-Re
    ax = axes[0, 1]
    for i, tpms in enumerate(tpms_types):
        _, f = TPMSCorrelations.get_correlations(tpms, Re_range, Pr_gas, 'Gas')
        valid = ~np.isnan(f) & (f > 0)
        ax.loglog(Re_range[valid], f[valid], linewidth=2.5, 
                 color=colors[i], label=tpms)
    ax.set_xlabel('Reynolds Number, Re [-]')
    ax.set_ylabel('Friction Factor, f [-]')
    ax.set_title('Gas (H₂/He, Pr = 0.7)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gas correlations: PEC
    ax = axes[0, 2]
    for i, tpms in enumerate(tpms_types):
        Nu, f = TPMSCorrelations.get_correlations(tpms, Re_range, Pr_gas, 'Gas')
        valid = ~np.isnan(Nu) & ~np.isnan(f) & (f > 0)
        if np.any(valid):
            idx = np.where(valid)[0][0]
            PEC = (Nu[valid]/Nu[valid][0]) / (f[valid]/f[valid][0])**(1/3)
            ax.semilogx(Re_range[valid], PEC, linewidth=2.5, 
                       color=colors[i], label=tpms)
    ax.set_xlabel('Reynolds Number, Re [-]')
    ax.set_ylabel('PEC [-]')
    ax.set_title('Gas - Thermohydraulic Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Water correlations: Nu-Re
    ax = axes[1, 0]
    for i, tpms in enumerate(tpms_types):
        Nu, _ = TPMSCorrelations.get_correlations(tpms, Re_range, Pr_water, 'Water')
        valid = ~np.isnan(Nu)
        if np.any(valid):
            ax.loglog(Re_range[valid], Nu[valid], linewidth=2.5, 
                     color=colors[i], label=tpms)
    ax.set_xlabel('Reynolds Number, Re [-]')
    ax.set_ylabel('Nusselt Number, Nu [-]')
    ax.set_title('Water (Pr = 6.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Water correlations: f-Re
    ax = axes[1, 1]
    for i, tpms in enumerate(tpms_types):
        _, f = TPMSCorrelations.get_correlations(tpms, Re_range, Pr_water, 'Water')
        valid = ~np.isnan(f) & (f > 0)
        if np.any(valid):
            ax.loglog(Re_range[valid], f[valid], linewidth=2.5, 
                     color=colors[i], label=tpms)
    ax.set_xlabel('Reynolds Number, Re [-]')
    ax.set_ylabel('Friction Factor, f [-]')
    ax.set_title('Water (Pr = 6.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Water correlations: PEC
    ax = axes[1, 2]
    for i, tpms in enumerate(tpms_types):
        Nu, f = TPMSCorrelations.get_correlations(tpms, Re_range, Pr_water, 'Water')
        valid = ~np.isnan(Nu) & ~np.isnan(f) & (f > 0)
        if np.any(valid):
            idx = np.where(valid)[0][0]
            PEC = (Nu[valid]/Nu[valid][0]) / (f[valid]/f[valid][0])**(1/3)
            ax.semilogx(Re_range[valid], PEC, linewidth=2.5, 
                       color=colors[i], label=tpms)
    ax.set_xlabel('Reynolds Number, Re [-]')
    ax.set_ylabel('PEC [-]')
    ax.set_title('Water - Thermohydraulic Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Run this module after solving heat exchanger")
    print("Example:")
    print("  from tpms_heat_exchanger import *")
    print("  config = create_default_config()")
    print("  he = TPMSHeatExchanger(config)")
    print("  he.solve()")
    print("  vis = TPMSVisualizer(he)")
    print("  vis.plot_comprehensive()")
