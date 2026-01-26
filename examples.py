"""
Example Script: TPMS Heat Exchanger for Hydrogen Liquefaction

This script demonstrates how to use the TPMS heat exchanger package
for ortho-para hydrogen conversion calculations.

Author: Based on research by Zhang et al. (2025)
"""

import numpy as np
import matplotlib.pyplot as plt

# Import our modules
from hydrogen_properties import HydrogenProperties, test_hydrogen_properties
from tpms_correlations import TPMSCorrelations, test_tpms_correlations
from tpms_heat_exchanger import TPMSHeatExchanger, create_default_config
from tpms_visualization import TPMSVisualizer, compare_tpms_structures


def example_1_test_properties():
    """Example 1: Test hydrogen property calculations"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Testing Hydrogen Property Calculations")
    print("="*70 + "\n")
    
    test_hydrogen_properties()


def example_2_test_correlations():
    """Example 2: Test TPMS correlations"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Testing TPMS Correlations")
    print("="*70 + "\n")
    
    test_tpms_correlations()


def example_3_basic_heat_exchanger():
    """Example 3: Run basic heat exchanger calculation"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Basic TPMS Heat Exchanger")
    print("="*70 + "\n")
    
    # Create default configuration
    config = create_default_config()
    
    # Customize if needed
    config['tpms']['type_hot'] = 'Diamond'
    config['tpms']['type_cold'] = 'Gyroid'
    
    # Create and solve3
    he = TPMSHeatExchanger(config)
    converged = he.solve(max_iter=500, tolerance=1e-3)
    
    if converged:
        # Visualize results
        vis = TPMSVisualizer(he)
        vis.plot_comprehensive(save_path='tpms_comprehensive.png')
        vis.plot_performance_metrics(save_path='tpms_metrics.png')
    
    return he


def example_4_parametric_study():
    """Example 4: Parametric study of mass flow rate"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Parametric Study - Mass Flow Rate Effect")
    print("="*70 + "\n")
    
    # Mass flow rate range
    mh_range = np.linspace(0.5e-1, 2e-1, 5)  # kg/s
    
    results = {
        'mh': [],
        'Th_out': [],
        'xh_out': [],
        'dP_hot': [],
        'Q_total': [],
        'conv_eff': []
    }
    
    for mh in mh_range:
        print(f"\nSolving for mh = {mh*1e3:.2f} g/s...")
        
        config = create_default_config()
        config['operating']['mh'] = mh
        config['operating']['mc'] = 2 * mh  # Maintain ratio
        config['solver']['n_elements'] = 15  # Reduce for speed
        
        he = TPMSHeatExchanger(config)
        converged = he.solve(max_iter=300, tolerance=5e-3)
        
        if converged:
            h2_props = HydrogenProperties()
            x_eq_out = h2_props.get_equilibrium_fraction(he.Th[-1])
            x_eq_in = h2_props.get_equilibrium_fraction(he.Th[0])
            conv_eff = (he.xh[-1] - he.xh[0]) / (x_eq_out - x_eq_in) * 100
            
            props_h = he._calculate_hot_properties()
            props_c = he._calculate_cold_properties()
            _, Q = he._calculate_heat_transfer(props_h, props_c)
            
            results['mh'].append(mh * 1e3)  # Convert to g/s
            results['Th_out'].append(he.Th[-1])
            results['xh_out'].append(he.xh[-1])
            results['dP_hot'].append((he.Ph[0] - he.Ph[-1]) / 1e3)  # kPa
            results['Q_total'].append(np.sum(Q))
            results['conv_eff'].append(conv_eff)
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Parametric Study: Mass Flow Rate Effect', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(results['mh'], results['Th_out'], 'ro-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Mass Flow Rate [g/s]')
    axes[0, 0].set_ylabel('Outlet Temperature [K]')
    axes[0, 0].set_title('Hot Outlet Temperature')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(results['mh'], results['xh_out'], 'bo-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Mass Flow Rate [g/s]')
    axes[0, 1].set_ylabel('Para-H₂ Fraction [-]')
    axes[0, 1].set_title('Outlet Para-Hydrogen')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(results['mh'], results['dP_hot'], 'go-', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('Mass Flow Rate [g/s]')
    axes[0, 2].set_ylabel('Pressure Drop [kPa]')
    axes[0, 2].set_title('Hot Side Pressure Drop')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(results['mh'], results['Q_total'], 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Mass Flow Rate [g/s]')
    axes[1, 0].set_ylabel('Heat Load [W]')
    axes[1, 0].set_title('Total Heat Transfer')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(results['mh'], results['conv_eff'], 'co-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Mass Flow Rate [g/s]')
    axes[1, 1].set_ylabel('Conversion Efficiency [%]')
    axes[1, 1].set_title('Catalytic Efficiency')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=90, color='r', linestyle='--', label='Target: 90%')
    axes[1, 1].legend()
    
    # Summary table
    axes[1, 2].axis('off')
    table_data = []
    for i in range(len(results['mh'])):
        table_data.append([
            f"{results['mh'][i]:.2f}",
            f"{results['Th_out'][i]:.1f}",
            f"{results['xh_out'][i]:.4f}",
            f"{results['conv_eff'][i]:.1f}"
        ])
    
    table = axes[1, 2].table(
        cellText=table_data,
        colLabels=['ṁ [g/s]', 'T_out [K]', 'x_out [-]', 'η [%]'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1, 2].set_title('Results Summary')
    
    plt.tight_layout()
    plt.savefig('parametric_study.png', dpi=300, bbox_inches='tight')
    print(f"\nParametric study plot saved to parametric_study.png")
    plt.show()


def example_5_compare_structures():
    """Example 5: Compare different TPMS structures"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Comparing TPMS Structures")
    print("="*70 + "\n")
    
    compare_tpms_structures()


def example_6_design_optimization():
    """Example 6: Simple design optimization"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Design Optimization")
    print("="*70 + "\n")
    
    # Test different TPMS combinations
    combinations = [
        ('Diamond', 'Gyroid'),
        ('Gyroid', 'Diamond'),
        ('Diamond', 'Diamond'),
        ('FKS', 'Gyroid'),
    ]
    
    print("Testing different TPMS combinations:")
    print("-" * 70)
    print(f"{'Hot TPMS':<12} {'Cold TPMS':<12} {'Conv Eff [%]':>12} {'dP_hot [kPa]':>12} {'Q [W]':>10}")
    print("-" * 70)
    
    best_config = None
    best_score = -np.inf
    
    for hot_tpms, cold_tpms in combinations:
        config = create_default_config()
        config['tpms']['type_hot'] = hot_tpms
        config['tpms']['type_cold'] = cold_tpms
        config['solver']['n_elements'] = 15
        
        he = TPMSHeatExchanger(config)
        converged = he.solve(max_iter=200, tolerance=1e-2)
        
        if converged:
            h2_props = HydrogenProperties()
            x_eq_out = h2_props.get_equilibrium_fraction(he.Th[-1])
            x_eq_in = h2_props.get_equilibrium_fraction(he.Th[0])
            conv_eff = (he.xh[-1] - he.xh[0]) / (x_eq_out - x_eq_in) * 100
            
            dP_hot = (he.Ph[0] - he.Ph[-1]) / 1e3
            
            props_h = he._calculate_hot_properties()
            props_c = he._calculate_cold_properties()
            _, Q = he._calculate_heat_transfer(props_h, props_c)
            Q_total = np.sum(Q)
            
            # Score: maximize conversion efficiency, minimize pressure drop
            score = conv_eff - 0.1 * dP_hot
            
            print(f"{hot_tpms:<12} {cold_tpms:<12} {conv_eff:>12.2f} {dP_hot:>12.2f} {Q_total:>10.2f}")
            
            if score > best_score:
                best_score = score
                best_config = (hot_tpms, cold_tpms, conv_eff, dP_hot, Q_total)
    
    print("-" * 70)
    print(f"\nBest configuration: {best_config[0]} (Hot) / {best_config[1]} (Cold)")
    print(f"  Conversion efficiency: {best_config[2]:.2f}%")
    print(f"  Pressure drop: {best_config[3]:.2f} kPa")
    print(f"  Heat load: {best_config[4]:.2f} W")
    print()


def main():
    """Main function to run all examples"""
    print("\n" + "="*70)
    print("  TPMS HEAT EXCHANGER EXAMPLES")
    print("  Ortho-Para Hydrogen Conversion for Hydrogen Liquefaction")
    print("="*70)
    
    examples = {
        '1': ('Test Hydrogen Properties', example_1_test_properties),
        '2': ('Test TPMS Correlations', example_2_test_correlations),
        '3': ('Basic Heat Exchanger', example_3_basic_heat_exchanger),
        '4': ('Parametric Study', example_4_parametric_study),
        '5': ('Compare TPMS Structures', example_5_compare_structures),
        '6': ('Design Optimization', example_6_design_optimization),
    }
    
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  0. Run all examples")
    print("  q. Quit")
    
    while True:
        choice = input("\nSelect example (0-6, q to quit): ").strip()
        
        if choice == 'q':
            print("\nExiting...")
            break
        elif choice == '0':
            for key in sorted(examples.keys()):
                examples[key][1]()
            break
        elif choice in examples:
            examples[choice][1]()
        else:
            print("Invalid choice. Please try again.")
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    # Run specific example or show menu
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == '1':
            example_1_test_properties()
        elif example_num == '2':
            example_2_test_correlations()
        elif example_num == '3':
            example_3_basic_heat_exchanger()
        elif example_num == '4':
            example_4_parametric_study()
        elif example_num == '5':
            example_5_compare_structures()
        elif example_num == '6':
            example_6_design_optimization()
        else:
            print(f"Unknown example: {example_num}")
            print("Usage: python examples.py [1-6]")
    else:
        main()
