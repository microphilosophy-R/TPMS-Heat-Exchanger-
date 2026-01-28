import pandas as pd
from matplotlib import pyplot as plt


class ConvergenceTracker:
    """Tracks solution convergence history and generates diagnostics."""

    def __init__(self):
        self.history = {
            'iteration': [], 'unified_error': [], 'dT': [], 'dP': [], 'dQ': [],
            'Q_total': [], 'energy_imbalance': [], 'damping': []
        }
        self.set_academic_style()
        # Color Groups (optimized for contrast)
        # self.colors = {
        #     'hot': '#D62728',  # Deep Red
        #     'cold': '#1F77B4',  # Muted Blue
        #     'equilibrium': '#2CA02C',  # Green
        #     'conversion': '#FF7F0E',  # Orange
        #     'black': '#000000',
        #     'gray': '#666666'
        # }

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

    def update(self, iteration, error_dict, solver, damping_factor):
        """Update history with current iteration state."""
        self.history['iteration'].append(iteration)
        self.history['unified_error'].append(error_dict['total'])
        self.history['dT'].append(error_dict['dT'])
        self.history['dP'].append(error_dict['dP'])
        self.history['dQ'].append(error_dict['dQ'])
        self.history['damping'].append(damping_factor)

        # Calculate Energy Imbalance
        try:
            # Access properties from the new dict structure
            h_hot_in = solver.props_h['h'][0]
            h_hot_out = solver.props_h['h'][-1]
            h_cold_out = solver.props_c['h'][0]  # Cold flows N->0, so 0 is outlet
            h_cold_in = solver.props_c['h'][-1]

            Q_hot = solver.streams['hot']['m'] * (h_hot_in - h_hot_out)
            Q_cold = solver.streams['cold']['m'] * (h_cold_out - h_cold_in)

            imbalance = abs(Q_hot - Q_cold) / (max(abs(Q_hot), 1e-5)) * 100
            self.history['energy_imbalance'].append(imbalance)
            self.history['Q_total'].append(Q_hot)
        except Exception:
            self.history['energy_imbalance'].append(0.0)
            self.history['Q_total'].append(0.0)

    def plot(self, save_path='convergence_history.png'):
        """Generates a summary plot of the convergence."""
        if not self.history['iteration']:
            print("No iteration history to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Solver Convergence Diagnostics', fontsize=14, fontweight='bold')

        iters = self.history['iteration']

        # 1. Residuals
        ax = axes[0, 0]
        ax.semilogy(iters, self.history['unified_error'], 'k-', lw=2, label='Unified Error')
        ax.semilogy(iters, self.history['dT'], 'r--', label='dT (Temp)')
        ax.semilogy(iters, self.history['dP'], 'b--', label='dP (Press)')
        ax.semilogy(iters, self.history['dQ'], 'g--', label='dQ (Heat)')
        ax.set_title('Residual Decay')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative Error')
        ax.legend()
        ax.grid(True, which="both", ls="-")

        # 2. Energy Imbalance
        ax = axes[0, 1]
        ax.plot(iters, self.history['energy_imbalance'], 'r-')
        ax.set_title('Energy Imbalance [%]')
        ax.set_xlabel('Iteration')
        ax.grid(True)

        # 3. Total Heat Transfer
        ax = axes[1, 0]
        ax.plot(iters, self.history['Q_total'], 'b-')
        ax.set_title('Total Heat Load [W]')
        ax.set_xlabel('Iteration')
        ax.grid(True)

        # 4. Damping Factor
        ax = axes[1, 1]
        ax.plot(iters, self.history['damping'], 'g-')
        ax.set_title('Adaptive Relaxation Factor')
        ax.set_xlabel('Iteration')
        ax.set_ylim(0, 1.0)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✓ Convergence plot saved to {save_path}")

    def export_csv(self, filepath='convergence_data.csv'):
        df = pd.DataFrame(self.history)
        df.to_csv(filepath, index=False)
        print(f"✓ Convergence data exported to {filepath}")
