"""
空TPMS vs 填充TPMS 传热性能对比分析
Bare TPMS vs Packed-Bed TPMS Heat Transfer Comparison

本脚本对比两种传热模型:
1. 空通道TPMS (现有 tpms_correlations.py)
2. 填充催化剂的TPMS (packed_bed_model.py)

输出:
- h_eff 随 Re 的变化对比 (含区间带)
- 总传热系数 U 对比
- 压降对比
- 敏感性分析 (催化剂参数对h_eff的影响)
- CSV数据导出

用于第四章: 论证填充床对TPMS换热器性能的定量影响。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

from tpms_correlations import TPMSCorrelations
from packed_bed_model import PackedBedTPMSModel


# ====================================================================
# 全局绘图设置
# ====================================================================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def setup_output_dir(dirname='results_packed_vs_bare'):
    """创建输出目录。"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


# ====================================================================
# 1. 主要对比: h_eff vs Re
# ====================================================================

def compare_htc_vs_re(tpms_type='Diamond', output_dir='results'):
    """
    对比空TPMS和填充TPMS的传热系数随Re变化。

    生成图表: h_eff vs Re_channel, 包含区间带。
    """
    # --- 参数 ---
    Pr = 0.80
    k_f = 0.10  # W/m·K (低温氢气)

    # TPMS几何
    porosity = 0.65
    cell_size = 5e-3  # m
    D_h = 4.0 * porosity * cell_size / (2.0 * np.pi)

    # 填充床参数
    catalyst_config = {
        'particle_diameter': 1.0e-3,
        'bed_porosity': 0.40,
        'k_solid': 10.0,
        'shape_factor': 1.0,
    }
    tpms_geometry = {
        'D_h': D_h,
        'wall_thickness': 0.5e-3,
        'k_wall': 237.0,
    }

    packed_model = PackedBedTPMSModel(catalyst_config, tpms_geometry)

    # Re范围
    Re_range = np.logspace(2, 4, 50)  # 100 ~ 10000

    # --- 计算 ---
    h_bare = np.zeros_like(Re_range)
    h_packed_lower = np.zeros_like(Re_range)
    h_packed_nominal = np.zeros_like(Re_range)
    h_packed_upper = np.zeros_like(Re_range)
    f_bare = np.zeros_like(Re_range)
    f_packed = np.zeros_like(Re_range)

    for i, Re in enumerate(Re_range):
        # 空TPMS
        Nu, f_val = TPMSCorrelations.get_correlations(tpms_type, Re, Pr, 'Gas')
        h_bare[i] = Nu * k_f / D_h
        f_bare[i] = f_val

        # 填充TPMS (三档)
        for mode, arr in [('lower', h_packed_lower),
                          ('nominal', h_packed_nominal),
                          ('upper', h_packed_upper)]:
            h_eff, f_eq, _ = packed_model.get_htc_and_friction(
                Re, Pr, k_f, tpms_type, mode
            )
            arr[i] = h_eff

        # 填充TPMS 摩擦因子 (nominal)
        _, f_packed[i], _ = packed_model.get_htc_and_friction(
            Re, Pr, k_f, tpms_type, 'nominal'
        )

    # --- 绘图 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) 传热系数
    ax = axes[0]
    ax.loglog(Re_range, h_bare, 'b-', lw=2, label=f'Bare {tpms_type} (Nu correlation)')
    ax.loglog(Re_range, h_packed_nominal, 'r-', lw=2,
              label=f'Packed {tpms_type} (nominal)')
    ax.fill_between(Re_range, h_packed_lower, h_packed_upper,
                     alpha=0.25, color='red', label='Packed (lower–upper)')
    ax.set_xlabel('Re (channel)')
    ax.set_ylabel('h$_{eff}$ [W/m²·K]')
    ax.set_title('(a) Heat Transfer Coefficient')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # (b) 摩擦因子
    ax = axes[1]
    ax.loglog(Re_range, f_bare, 'b-', lw=2, label=f'Bare {tpms_type}')
    ax.loglog(Re_range, f_packed, 'r-', lw=2, label=f'Packed {tpms_type} (Ergun)')
    ax.set_xlabel('Re (channel)')
    ax.set_ylabel('f (Fanning) [-]')
    ax.set_title('(b) Friction Factor')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'htc_friction_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Saved: {save_path}")

    # --- CSV导出 ---
    df = pd.DataFrame({
        'Re_channel': Re_range,
        'h_bare': h_bare,
        'h_packed_lower': h_packed_lower,
        'h_packed_nominal': h_packed_nominal,
        'h_packed_upper': h_packed_upper,
        'f_bare': f_bare,
        'f_packed': f_packed,
    })
    csv_path = os.path.join(output_dir, 'htc_friction_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.6e')
    print(f"[OK] Saved: {csv_path}")

    return df


# ====================================================================
# 2. 总传热系数 U 对比
# ====================================================================

def compare_overall_U(tpms_type='Diamond', output_dir='results'):
    """
    对比空TPMS和填充TPMS的总传热系数U。

    U的计算:
    1/U = 1/h_hot + t_wall/k_wall + 1/h_cold

    假设:
    - 热侧(氢气): 填充催化剂 或 空TPMS
    - 冷侧(氦气): 空TPMS (两种情况相同)
    """
    Pr_hot, k_f_hot = 0.80, 0.10   # 氢气
    Pr_cold, k_f_cold = 0.67, 0.08  # 氦气

    porosity_hot = 0.65
    porosity_cold = 0.70
    cell_size = 5e-3
    D_h_hot = 4.0 * porosity_hot * cell_size / (2.0 * np.pi)
    D_h_cold = 4.0 * porosity_cold * cell_size / (2.0 * np.pi)

    t_wall = 0.5e-3
    k_wall = 237.0

    # 填充床模型
    packed_model = PackedBedTPMSModel(
        catalyst_config={
            'particle_diameter': 1.0e-3,
            'bed_porosity': 0.40,
            'k_solid': 10.0,
        },
        tpms_geometry={
            'D_h': D_h_hot,
            'wall_thickness': t_wall,
            'k_wall': k_wall,
        },
    )

    Re_range = np.logspace(2, 3.7, 40)  # 100 ~ 5000

    U_bare = np.zeros_like(Re_range)
    U_packed_lower = np.zeros_like(Re_range)
    U_packed_nominal = np.zeros_like(Re_range)
    U_packed_upper = np.zeros_like(Re_range)

    for i, Re in enumerate(Re_range):
        # 冷侧始终用空TPMS
        Nu_cold, _ = TPMSCorrelations.get_correlations('Gyroid', Re * 0.8, Pr_cold, 'Gas')
        h_cold = Nu_cold * k_f_cold / D_h_cold

        # 热侧: 空TPMS
        Nu_hot_bare, _ = TPMSCorrelations.get_correlations(tpms_type, Re, Pr_hot, 'Gas')
        h_hot_bare = Nu_hot_bare * k_f_hot / D_h_hot

        R_wall = t_wall / k_wall
        U_bare[i] = 1.0 / (1.0 / h_hot_bare + R_wall + 1.0 / h_cold)

        # 热侧: 填充TPMS (三档)
        for mode, arr in [('lower', U_packed_lower),
                          ('nominal', U_packed_nominal),
                          ('upper', U_packed_upper)]:
            h_hot_packed, _, _ = packed_model.get_htc_and_friction(
                Re, Pr_hot, k_f_hot, tpms_type, mode
            )
            arr[i] = 1.0 / (1.0 / h_hot_packed + R_wall + 1.0 / h_cold)

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(Re_range, U_bare, 'b-', lw=2, label=f'Bare {tpms_type} (both sides)')
    ax.loglog(Re_range, U_packed_nominal, 'r-', lw=2,
              label=f'Packed hot / Bare cold (nominal)')
    ax.fill_between(Re_range, U_packed_lower, U_packed_upper,
                     alpha=0.25, color='red', label='Packed (lower–upper)')
    ax.set_xlabel('Re$_{hot}$ (channel)')
    ax.set_ylabel('U [W/m²·K]')
    ax.set_title('Overall Heat Transfer Coefficient Comparison')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    save_path = os.path.join(output_dir, 'overall_U_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Saved: {save_path}")

    # CSV
    df = pd.DataFrame({
        'Re_hot': Re_range,
        'U_bare': U_bare,
        'U_packed_lower': U_packed_lower,
        'U_packed_nominal': U_packed_nominal,
        'U_packed_upper': U_packed_upper,
    })
    csv_path = os.path.join(output_dir, 'overall_U_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.6e')
    print(f"[OK] Saved: {csv_path}")

    return df


# ====================================================================
# 3. 催化剂参数敏感性分析
# ====================================================================

def sensitivity_analysis(output_dir='results'):
    """
    分析催化剂关键参数 (d_p, ε_bed, k_s) 对 h_eff 的影响。

    固定 Re_p = 50, Pr = 0.8, k_f = 0.1 W/m·K
    """
    Re_p_ref = 50
    Pr = 0.80
    k_f = 0.10

    D_h = 3.3e-3
    base_tpms = {'D_h': D_h, 'wall_thickness': 0.5e-3, 'k_wall': 237.0}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) 粒径 d_p 敏感性
    d_p_range = np.linspace(0.3e-3, 3.0e-3, 30)
    h_dp = {'lower': [], 'nominal': [], 'upper': []}
    for dp in d_p_range:
        model = PackedBedTPMSModel(
            {'particle_diameter': dp, 'bed_porosity': 0.40, 'k_solid': 10.0},
            base_tpms
        )
        for mode in h_dp:
            h, _ = model.overall_htc_packed_side(Re_p_ref, Pr, k_f, mode)
            h_dp[mode].append(h)

    ax = axes[0]
    ax.plot(d_p_range * 1e3, h_dp['nominal'], 'r-', lw=2)
    ax.fill_between(d_p_range * 1e3, h_dp['lower'], h_dp['upper'],
                     alpha=0.25, color='red')
    ax.set_xlabel('d$_p$ [mm]')
    ax.set_ylabel('h$_{eff}$ [W/m²·K]')
    ax.set_title('(a) Particle Diameter')
    ax.grid(True, alpha=0.3)

    # (b) 床层孔隙率 ε_bed 敏感性
    eps_range = np.linspace(0.30, 0.50, 30)
    h_eps = {'lower': [], 'nominal': [], 'upper': []}
    for eps in eps_range:
        model = PackedBedTPMSModel(
            {'particle_diameter': 1e-3, 'bed_porosity': eps, 'k_solid': 10.0},
            base_tpms
        )
        for mode in h_eps:
            h, _ = model.overall_htc_packed_side(Re_p_ref, Pr, k_f, mode)
            h_eps[mode].append(h)

    ax = axes[1]
    ax.plot(eps_range, h_eps['nominal'], 'r-', lw=2)
    ax.fill_between(eps_range, h_eps['lower'], h_eps['upper'],
                     alpha=0.25, color='red')
    ax.set_xlabel('ε$_{bed}$ [-]')
    ax.set_ylabel('h$_{eff}$ [W/m²·K]')
    ax.set_title('(b) Bed Porosity')
    ax.grid(True, alpha=0.3)

    # (c) 固体导热 k_s 敏感性
    ks_range = np.logspace(-1, 2, 30)  # 0.1 ~ 100 W/m·K
    h_ks = {'lower': [], 'nominal': [], 'upper': []}
    for ks in ks_range:
        model = PackedBedTPMSModel(
            {'particle_diameter': 1e-3, 'bed_porosity': 0.40, 'k_solid': ks},
            base_tpms
        )
        for mode in h_ks:
            h, _ = model.overall_htc_packed_side(Re_p_ref, Pr, k_f, mode)
            h_ks[mode].append(h)

    ax = axes[2]
    ax.semilogx(ks_range, h_ks['nominal'], 'r-', lw=2)
    ax.fill_between(ks_range, h_ks['lower'], h_ks['upper'],
                     alpha=0.25, color='red')
    ax.set_xlabel('k$_s$ [W/m·K]')
    ax.set_ylabel('h$_{eff}$ [W/m²·K]')
    ax.set_title('(c) Catalyst Conductivity')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'sensitivity_analysis.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Saved: {save_path}")


# ====================================================================
# 4. 热阻分解分析
# ====================================================================

def resistance_breakdown(tpms_type='Diamond', output_dir='results'):
    """
    展示填充床各热阻分量占比随Re的变化。

    帮助理解哪个热阻是瓶颈。
    """
    Pr = 0.80
    k_f = 0.10

    packed_model = PackedBedTPMSModel(
        catalyst_config={
            'particle_diameter': 1.0e-3,
            'bed_porosity': 0.40,
            'k_solid': 10.0,
        },
        tpms_geometry={
            'D_h': 3.3e-3,
            'wall_thickness': 0.5e-3,
            'k_wall': 237.0,
        },
    )

    Re_p_range = np.logspace(0.5, 2.5, 40)  # 3 ~ 300

    R_wall_film = np.zeros_like(Re_p_range)
    R_bed_cond = np.zeros_like(Re_p_range)

    for i, Re_p in enumerate(Re_p_range):
        _, details = packed_model.overall_htc_packed_side(Re_p, Pr, k_f, 'nominal')
        R_wall_film[i] = details['R_wall_film']
        R_bed_cond[i] = details['R_bed_conduction']

    R_total = R_wall_film + R_bed_cond
    frac_wall = R_wall_film / R_total * 100
    frac_bed = R_bed_cond / R_total * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) 热阻值
    ax1.loglog(Re_p_range, R_wall_film, 'b-', lw=2, label='R$_{wall-film}$ (1/h$_w$)')
    ax1.loglog(Re_p_range, R_bed_cond, 'r-', lw=2,
               label='R$_{bed}$ (D$_h$/Ck$_{r,eff}$)')
    ax1.loglog(Re_p_range, R_total, 'k--', lw=1.5, label='R$_{total}$')
    ax1.set_xlabel('Re$_p$ (particle)')
    ax1.set_ylabel('Thermal Resistance [m²·K/W]')
    ax1.set_title('(a) Resistance Components')
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', alpha=0.3)

    # (b) 占比
    ax2.semilogx(Re_p_range, frac_wall, 'b-', lw=2, label='Wall film')
    ax2.semilogx(Re_p_range, frac_bed, 'r-', lw=2, label='Bed conduction')
    ax2.set_xlabel('Re$_p$ (particle)')
    ax2.set_ylabel('Fraction of Total Resistance [%]')
    ax2.set_title('(b) Resistance Breakdown')
    ax2.set_ylim([0, 100])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'resistance_breakdown.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Saved: {save_path}")


# ====================================================================
# 5. 多种TPMS类型对比
# ====================================================================

def compare_tpms_types(output_dir='results'):
    """
    对比不同TPMS类型在填充工况下的传热与压降差异。
    """
    tpms_types = ['Gyroid', 'Diamond', 'Primitive', 'FKS']
    Re_channel = 1500
    Pr = 0.80
    k_f = 0.10

    porosity = 0.65
    cell_size = 5e-3
    D_h = 4.0 * porosity * cell_size / (2.0 * np.pi)

    packed_model = PackedBedTPMSModel(
        catalyst_config={
            'particle_diameter': 1.0e-3,
            'bed_porosity': 0.40,
            'k_solid': 10.0,
        },
        tpms_geometry={'D_h': D_h, 'wall_thickness': 0.5e-3, 'k_wall': 237.0},
    )

    print("\n" + "=" * 80)
    print(f"TPMS Type Comparison | Re_channel = {Re_channel}")
    print("=" * 80)
    print(f"{'Type':<12} | {'h_bare':>8} {'h_packed':>8} {'ratio':>6} | "
          f"{'f_bare':>8} {'f_packed':>10} {'f_ratio':>8} | {'ψ_TPMS':>6}")
    print("-" * 80)

    rows = []
    for tpms in tpms_types:
        # 空TPMS
        Nu_b, f_b = TPMSCorrelations.get_correlations(tpms, Re_channel, Pr, 'Gas')
        h_b = Nu_b * k_f / D_h

        # 填充TPMS
        h_p, f_p, det = packed_model.get_htc_and_friction(
            Re_channel, Pr, k_f, tpms, 'nominal'
        )

        ratio_h = h_p / h_b
        ratio_f = f_p / f_b
        psi = packed_model.tpms_pressure_correction(tpms)

        print(f"{tpms:<12} | {h_b:8.1f} {h_p:8.1f} {ratio_h:6.3f} | "
              f"{f_b:8.4f} {f_p:10.1f} {ratio_f:8.0f} | {psi:6.2f}")

        rows.append({
            'tpms_type': tpms,
            'h_bare': h_b, 'h_packed_nominal': h_p, 'h_ratio': ratio_h,
            'f_bare': f_b, 'f_packed': f_p, 'f_ratio': ratio_f,
            'psi_tpms': psi,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'tpms_type_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.6e')
    print(f"\n[OK] Saved: {csv_path}")
    return df


# ====================================================================
# Main
# ====================================================================

def main():
    output_dir = setup_output_dir('results_packed_vs_bare')

    print("=" * 70)
    print("Packed Bed + TPMS: Comprehensive Comparison Analysis")
    print("=" * 70)

    # 1. h_eff & f 对比
    print("\n--- 1. HTC & Friction Factor vs Re ---")
    compare_htc_vs_re(tpms_type='Diamond', output_dir=output_dir)

    # 2. 总传热系数 U
    print("\n--- 2. Overall U Comparison ---")
    compare_overall_U(tpms_type='Diamond', output_dir=output_dir)

    # 3. 敏感性分析
    print("\n--- 3. Sensitivity Analysis ---")
    sensitivity_analysis(output_dir=output_dir)

    # 4. 热阻分解
    print("\n--- 4. Resistance Breakdown ---")
    resistance_breakdown(output_dir=output_dir)

    # 5. TPMS类型对比
    print("\n--- 5. TPMS Type Comparison ---")
    compare_tpms_types(output_dir=output_dir)

    print("\n" + "=" * 70)
    print(f"All results saved to: {os.path.abspath(output_dir)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
