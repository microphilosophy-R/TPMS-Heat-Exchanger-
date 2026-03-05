"""
多孔介质 + TPMS 耦合传热模型
Packed Bed + TPMS Combined Heat Transfer Model

当TPMS流道内填充催化剂颗粒时，壁面到流体的传热路径变为：
  流体主体 → 填充床弥散/对流 → 有效导热 → 近壁区 → TPMS壁面

本模块包含：
1. 有效导热系数: Zehner-Bauer-Schlünder (静态) + 弥散项 (Wen-Fan)
2. 壁面传热系数: Martin-Nilles
3. 综合壁面传热系数: 双区域模型 + TPMS翅片效应
4. 压降: 修正Ergun方程 + TPMS校正因子
5. 区间估计: lower / nominal / upper

接口兼容现有 TPMSHeatExchanger 求解器。

References
----------
- Zehner & Schlünder (1970), Chemie Ingenieur Technik, 42(14), 933-941.
- Martin & Nilles (1993), Chem. Eng. Process., 32(2), 77-83.
- Ergun (1952), Chem. Eng. Progress, 48, 89-94.
- Dixon & Cresswell (1979), AIChE Journal, 25(4), 663-676.
"""

import numpy as np
import warnings

SUPPORTED_PACKED_MODES = ('lower', 'nominal', 'upper')


class PackedBedTPMSModel:
    """
    TPMS流道内填充床的耦合传热与压降模型。

    热阻分解:
        1/h_eff = 1/h_w + D_h / (C_shape * k_r,eff)

    其中 h_w 为壁面传热系数, k_r,eff 为径向有效导热系数,
    D_h 为TPMS水力直径, C_shape 为几何因子。

    Parameters
    ----------
    catalyst_config : dict
        particle_diameter : float  催化剂粒径 [m]
        bed_porosity : float       床层孔隙率 [-]
        k_solid : float            催化剂固体导热系数 [W/m·K]
        shape_factor : float       颗粒球形度 [-], 默认1.0
    tpms_geometry : dict
        D_h : float                TPMS通道水力直径 [m]
        wall_thickness : float     TPMS壁厚 [m]
        k_wall : float             TPMS壁面材料导热系数 [W/m·K]
    """

    def __init__(self, catalyst_config, tpms_geometry):
        # 催化剂填充床参数
        self.d_p = catalyst_config['particle_diameter']
        self.eps_bed = catalyst_config['bed_porosity']
        self.k_s = catalyst_config['k_solid']
        self.sphericity = catalyst_config.get('shape_factor', 1.0)

        # TPMS几何参数
        self.D_h = tpms_geometry['D_h']
        self.t_wall = tpms_geometry['wall_thickness']
        self.k_wall = tpms_geometry['k_wall']

        if self.d_p <= 0:
            raise ValueError("particle_diameter must be > 0")
        if self.k_s <= 0:
            raise ValueError("k_solid must be > 0")
        if self.sphericity <= 0:
            raise ValueError("shape_factor must be > 0")
        if not (0.05 <= self.eps_bed <= 0.95):
            raise ValueError("bed_porosity must be within [0.05, 0.95]")

        # 派生参数
        self.N_ratio = self.D_h / self.d_p  # 管径/粒径比

        if self.N_ratio < 2:
            warnings.warn(
                f"D_h/d_p = {self.N_ratio:.1f} < 2: 填充床在此尺度下"
                f"可能不满足连续介质假设, 结果仅供参考"
            )

    # ================================================================
    # 1. 有效导热系数
    # ================================================================

    def effective_conductivity_stagnant(self, k_f):
        """
        Zehner-Bauer-Schlünder 静态有效导热系数。

        无流动时填充床的等效导热系数，考虑固-液两相导热。

        Parameters
        ----------
        k_f : float
            流体导热系数 [W/m·K]

        Returns
        -------
        k_eff_0 : float
            静态有效导热系数 [W/m·K]
        """
        eps = self.eps_bed
        kappa = self.k_s / k_f  # 固液导热系数比
        k_s_val = self.k_s

        # --- Maxwell 有效介质模型 (对所有正kappa稳定) ---
        num = k_s_val + 2.0 * k_f + 2.0 * (1.0 - eps) * (k_s_val - k_f)
        den = k_s_val + 2.0 * k_f - (1.0 - eps) * (k_s_val - k_f)
        k_maxwell = k_f * num / den

        # --- ZBS 模型 (高kappa时更准确, 但在kappa≈B附近不稳定) ---
        B = 1.25 * ((1.0 - eps) / eps) ** (10.0 / 9.0)
        N = 1.0 - B / kappa

        if kappa < 1.01 or abs(N) < 0.05:
            # kappa接近1或N接近0的退化区: 仅用Maxwell
            return max(k_maxwell, k_f)

        if N > 0:
            ln_term = np.log(kappa / B)
            term_a = (B * (kappa - 1.0) / kappa) * ln_term / N
            term_b = (B - 1.0) / N
            term_c = (B + 1.0) / 2.0
            k_cell = k_f * (2.0 / N) * (term_a - term_b + term_c)
            k_zbs = (1.0 - np.sqrt(1.0 - eps)) * k_f + np.sqrt(1.0 - eps) * k_cell
        else:
            # N < 0: ZBS不适用
            k_zbs = k_f

        # 取两种模型中的较大值, 保证单调性和稳定性
        # Maxwell在低kappa更准确, ZBS在高kappa更准确
        k_eff_0 = max(k_maxwell, k_zbs)

        # 物理下界: k_eff 不可能低于纯流体导热
        return max(k_eff_0, k_f)

    def effective_conductivity_dispersion(self, k_f, Re_p, Pr):
        """
        弥散项对径向有效导热的贡献。

        k_disp = C_disp * Pe_p * k_f
        其中 Pe_p = Re_p * Pr (颗粒Peclet数)

        径向弥散系数 C_disp ≈ 0.1 (Wen-Fan)

        Parameters
        ----------
        k_f : float
            流体导热系数 [W/m·K]
        Re_p : float
            颗粒Reynolds数 = ρ*u_s*d_p/μ
        Pr : float
            Prandtl数

        Returns
        -------
        k_disp : float
            弥散导热系数 [W/m·K]
        """
        Pe_p = Re_p * Pr
        C_disp = 0.1  # 径向弥散 (Wen-Fan)
        return C_disp * Pe_p * k_f

    def effective_conductivity_total(self, k_f, Re_p, Pr):
        """总径向有效导热系数 = 静态 + 弥散。"""
        k_0 = self.effective_conductivity_stagnant(k_f)
        k_d = self.effective_conductivity_dispersion(k_f, Re_p, Pr)
        return k_0 + k_d

    # ================================================================
    # 2. 壁面传热系数
    # ================================================================

    def wall_htc_packed_bed(self, Re_p, Pr, k_f):
        """
        填充床壁面传热系数 (Martin-Nilles关联式)。

        Nu_w = (1.3 + 5/(D_h/d_p)) * (k_r,eff/k_f) + 0.19 * Re_p^0.75 * Pr^(1/3)

        该关联式在低D_h/d_p比(壁面效应主导)时仍有效。

        Parameters
        ----------
        Re_p : float
            颗粒Reynolds数
        Pr : float
            Prandtl数
        k_f : float
            流体导热系数 [W/m·K]

        Returns
        -------
        h_w : float
            壁面传热系数 [W/m²·K]
        Nu_w : float
            壁面Nusselt数 (基于d_p)
        """
        k_r_eff = self.effective_conductivity_total(k_f, Re_p, Pr)
        ratio_k = k_r_eff / k_f
        ratio_D = max(self.N_ratio, 1.5)  # 避免除以过小值

        # Martin-Nilles
        Nu_w = (1.3 + 5.0 / ratio_D) * ratio_k + 0.19 * Re_p**0.75 * Pr**(1.0 / 3.0)

        h_w = Nu_w * k_f / self.d_p
        return h_w, Nu_w

    # ================================================================
    # 3. TPMS 翅片增强
    # ================================================================

    def tpms_fin_efficiency(self, h_local):
        """
        TPMS壁面作为翅片深入填充床的增强效率。

        将TPMS壁面建模为长度 L_fin ≈ D_h/4 的翅片,
        在局部对流系数 h_local 下计算翅片效率。

        Parameters
        ----------
        h_local : float
            局部对流换热系数 [W/m²·K]

        Returns
        -------
        eta : float
            翅片效率 [-], 范围 (0, 1]
        """
        L_fin = self.D_h / 4.0

        if h_local <= 0 or self.k_wall <= 0 or self.t_wall <= 0:
            return 1.0

        m = np.sqrt(2.0 * h_local / (self.k_wall * self.t_wall))
        mL = m * L_fin

        if mL < 0.01:
            return 1.0
        elif mL > 20.0:
            return 1.0 / mL
        else:
            return np.tanh(mL) / mL

    # ================================================================
    # 4. 综合壁面传热系数 (含区间估计)
    # ================================================================

    def overall_htc_packed_side(self, Re_p, Pr, k_f, mode='nominal'):
        """
        填充床侧的综合有效传热系数。

        双区域模型: 1/h_eff = 1/h_w + D_h / (C_shape * k_r,eff)

        三种模式提供不确定性区间:
        - 'lower':  保守估计 (标准圆管, 无TPMS增强)
        - 'nominal': 最佳估计 (TPMS中等增强)
        - 'upper':  乐观估计 (完全TPMS增强)

        Parameters
        ----------
        Re_p : float
            颗粒Reynolds数
        Pr : float
            Prandtl数
        k_f : float
            流体导热系数 [W/m·K]
        mode : str
            'lower', 'nominal', 'upper'

        Returns
        -------
        h_eff : float
            有效传热系数 [W/m²·K]
        details : dict
            热阻分解细节
        """
        # 壁面传热系数
        mode = str(mode).strip().lower()
        if mode not in SUPPORTED_PACKED_MODES:
            raise ValueError(
                f"Invalid packed mode '{mode}'. Use one of {SUPPORTED_PACKED_MODES}."
            )

        h_w, Nu_w = self.wall_htc_packed_bed(Re_p, Pr, k_f)

        # 有效导热系数
        k_r_eff = self.effective_conductivity_total(k_f, Re_p, Pr)

        # 模式相关参数
        # TPMS增强体现在三个方面:
        # 1. C_shape: 几何因子 (TPMS缩短导热路径 → 更小的C)
        # 2. k_enhance: 弥散增强 (TPMS流动重分配 → 更强的弥散)
        # 3. area_enhance: 翅片面积增强 (TPMS壁面深入填充床 → 额外传热面积)
        #    公式: h_eff = (1 + α*(η-1) 保守处理) / R_total
        #    其中 α = A_fin/A_base 为翅片面积比

        eta_fin = self.tpms_fin_efficiency(h_w)

        if mode == 'lower':
            C_shape = 8.0      # 圆管几何 (最长导热路径)
            k_enhance = 1.0    # 无额外弥散
            area_factor = 1.0  # 无翅片面积增益
        elif mode == 'nominal':
            C_shape = 6.0      # TPMS中间值
            k_enhance = 1.2    # 适度弥散增强
            # 翅片增益: TPMS壁面提供约20%额外有效面积
            area_factor = 1.0 + 0.2 * eta_fin
        else:  # upper
            C_shape = 4.0      # 短导热路径
            k_enhance = 1.5    # TPMS混合强化
            # 翅片增益: TPMS壁面提供约40%额外有效面积
            area_factor = 1.0 + 0.4 * eta_fin

        k_r_adj = k_r_eff * k_enhance

        # 热阻
        R_wall_film = 1.0 / h_w
        R_bed_cond = self.D_h / (C_shape * k_r_adj)
        R_total = R_wall_film + R_bed_cond

        # 有效传热系数 = 面积增强 / 总热阻
        h_eff = area_factor / R_total

        details = {
            'h_w': h_w,
            'Nu_w': Nu_w,
            'k_eff_stagnant': self.effective_conductivity_stagnant(k_f),
            'k_r_eff': k_r_eff,
            'k_r_adjusted': k_r_adj,
            'R_wall_film': R_wall_film,
            'R_bed_conduction': R_bed_cond,
            'R_total': R_total,
            'eta_fin': eta_fin,
            'area_factor': area_factor,
            'C_shape': C_shape,
            'h_eff': h_eff,
            'mode': mode,
            'D_h_over_d_p': self.N_ratio,
        }
        return h_eff, details

    # ================================================================
    # 5. 压降模型
    # ================================================================

    def friction_factor_ergun(self, Re_p):
        """
        Ergun方程等效摩擦因子。

        转换为与现有求解器兼容的Fanning摩擦因子格式:
        dP = f_equiv * (L/D_h) * (ρ*u²/2)

        其中 u 为TPMS通道内的表观速度 (= m_dot / (ρ * Ac_TPMS))

        Parameters
        ----------
        Re_p : float
            颗粒Reynolds数 = ρ*u_s*d_p/μ

        Returns
        -------
        f_equiv : float
            等效Fanning摩擦因子 [-]
        """
        eps = self.eps_bed
        f_equiv = (self.D_h / self.d_p) * (1.0 - eps) / eps**3 * (
            300.0 * (1.0 - eps) / max(Re_p, 0.1) + 3.5
        )
        return f_equiv

    def pressure_drop_ergun(self, rho, mu, u_s, L):
        """
        Ergun方程直接计算压降。

        ΔP/L = 150*μ*u_s*(1-ε)² / (ε³*d_p²) + 1.75*ρ*u_s²*(1-ε) / (ε³*d_p)

        Parameters
        ----------
        rho : float  流体密度 [kg/m³]
        mu : float   动力粘度 [Pa·s]
        u_s : float  表观速度 [m/s]
        L : float    床层长度 [m]

        Returns
        -------
        dP : float       总压降 [Pa]
        breakdown : dict  粘性/惯性项分解
        """
        eps = self.eps_bed
        d_p_eff = self.d_p * self.sphericity

        term_viscous = 150.0 * mu * u_s * (1 - eps)**2 / (eps**3 * d_p_eff**2)
        term_inertial = 1.75 * rho * u_s**2 * (1 - eps) / (eps**3 * d_p_eff)

        dP = (term_viscous + term_inertial) * L

        return dP, {
            'dP_viscous': term_viscous * L,
            'dP_inertial': term_inertial * L,
            'dP_per_meter': term_viscous + term_inertial,
        }

    @staticmethod
    def tpms_pressure_correction(tpms_type):
        """
        TPMS对填充床压降的校正因子。

        不同TPMS骨架对流道的宏观扭曲程度不同(第三章结论),
        导致相同填充床在不同TPMS中的压降存在差异。
        此校正因子乘以Ergun基础压降。

        数值来源: 基于第三章实验趋势的估计值。

        Parameters
        ----------
        tpms_type : str
            TPMS类型

        Returns
        -------
        psi : float
            校正因子 [-], >= 1.0
        """
        corrections = {
            'Gyroid': 1.15,
            'Diamond': 1.20,
            'Primitive': 1.10,
            'Neovius': 1.30,
            'FRD': 1.18,
            'FKS': 1.12,
        }
        return corrections.get(tpms_type, 1.15)

    # ================================================================
    # 6. 统一接口 (兼容现有求解器)
    # ================================================================

    def get_htc_and_friction(self, Re_channel, Pr, k_f, tpms_type='Diamond',
                             mode='nominal'):
        """
        统一接口: 返回有效传热系数和等效摩擦因子。

        将通道Re自动转换为颗粒Re, 并应用TPMS压降校正。

        Parameters
        ----------
        Re_channel : float
            基于TPMS通道水力直径的Reynolds数
        Pr : float
            Prandtl数
        k_f : float
            流体导热系数 [W/m·K]
        tpms_type : str
            TPMS类型
        mode : str
            估计模式: 'lower', 'nominal', 'upper'

        Returns
        -------
        h_eff : float
            有效传热系数 [W/m²·K]
        f_equiv : float
            等效Fanning摩擦因子 [-]
        details : dict
            计算细节
        """
        # 通道Re → 颗粒Re
        mode = str(mode).strip().lower()
        if mode not in SUPPORTED_PACKED_MODES:
            raise ValueError(
                f"Invalid packed mode '{mode}'. Use one of {SUPPORTED_PACKED_MODES}."
            )

        Re_p = Re_channel * (self.d_p / self.D_h)

        # 传热
        h_eff, details = self.overall_htc_packed_side(Re_p, Pr, k_f, mode)

        # 压降
        f_base = self.friction_factor_ergun(Re_p)
        psi = self.tpms_pressure_correction(tpms_type)
        f_equiv = f_base * psi

        details['Re_p'] = Re_p
        details['f_ergun_base'] = f_base
        details['psi_tpms'] = psi
        details['f_equiv'] = f_equiv

        return h_eff, f_equiv, details

    # ================================================================
    # 7. 区间估计
    # ================================================================

    def interval_estimate(self, Re_p, Pr, k_f):
        """
        返回 lower / nominal / upper 三档传热系数估计。

        用于不确定性传播分析和论文中的置信区间。

        Parameters
        ----------
        Re_p : float
            颗粒Reynolds数
        Pr : float
            Prandtl数
        k_f : float
            流体导热系数 [W/m·K]

        Returns
        -------
        results : dict
            keys = 'lower', 'nominal', 'upper'
            每项包含 h_eff 和 details
        """
        results = {}
        for mode in ['lower', 'nominal', 'upper']:
            h_eff, details = self.overall_htc_packed_side(Re_p, Pr, k_f, mode)
            results[mode] = {'h_eff': h_eff, 'details': details}
        return results


# ====================================================================
# 辅助函数: 从现有config生成PackedBedTPMSModel
# ====================================================================

def create_packed_bed_model(config, stream_key='hot'):
    """
    从现有换热器config字典创建PackedBedTPMSModel实例。

    在config中需添加 'catalyst' 部分:
        config['catalyst'] = {
            'particle_diameter': 1e-3,    # [m]
            'bed_porosity': 0.40,         # [-]
            'k_solid': 10.0,              # [W/m·K]
            'shape_factor': 1.0,          # [-]
        }

    Parameters
    ----------
    config : dict
        换热器配置字典

    Returns
    -------
    model : PackedBedTPMSModel
    """
    if stream_key not in ('hot', 'cold'):
        raise ValueError("stream_key must be 'hot' or 'cold'")

    cat = config.get('catalyst', {})
    channel_cfg = config.get('channels', {}).get(stream_key, {})
    packed_cfg = channel_cfg.get('packed', {})
    catalyst_config = {
        'particle_diameter': packed_cfg.get('particle_diameter', cat.get('particle_diameter', 1e-3)),
        'bed_porosity': packed_cfg.get('bed_porosity', cat.get('bed_porosity', 0.40)),
        'k_solid': packed_cfg.get('k_solid', cat.get('k_solid', 10.0)),
        'shape_factor': packed_cfg.get('shape_factor', cat.get('shape_factor', 1.0)),
    }

    geo = config['geometry']
    porosity_key = f'porosity_{stream_key}'
    porosity_default = 0.65 if stream_key == 'hot' else 0.70
    porosity = config['geometry'].get(porosity_key, porosity_default)
    cell_size = geo['unit_cell_size']
    D_h = 4.0 * porosity * cell_size / (2.0 * np.pi)

    tpms_geometry = {
        'D_h': D_h,
        'wall_thickness': geo['wall_thickness'],
        'k_wall': config['material']['k_wall'],
    }

    return PackedBedTPMSModel(catalyst_config, tpms_geometry)


# ====================================================================
# 自检与验证
# ====================================================================

def test_packed_bed_model():
    """模型自检: 打印典型工况下的计算结果。"""
    print("=" * 70)
    print("Packed Bed + TPMS Combined Model - Self Test")
    print("=" * 70)

    # 典型低温氢液化工况参数
    catalyst_config = {
        'particle_diameter': 1.0e-3,   # 1 mm
        'bed_porosity': 0.40,
        'k_solid': 10.0,               # Fe2O3/Al2O3 类催化剂
        'shape_factor': 1.0,
    }
    tpms_geometry = {
        'D_h': 3.3e-3,                 # 典型 TPMS D_h (cell=5mm, ε=0.65)
        'wall_thickness': 0.5e-3,
        'k_wall': 237.0,               # 铝
    }

    model = PackedBedTPMSModel(catalyst_config, tpms_geometry)
    print(f"\nD_h/d_p = {model.N_ratio:.1f}")

    # 低温氢气典型物性 (T≈50K, P≈2MPa)
    k_f = 0.10     # W/m·K
    Pr = 0.80
    mu = 3e-6      # Pa·s
    rho = 3.0      # kg/m³

    print(f"\nFluid: k_f={k_f} W/m-K, Pr={Pr}, mu={mu:.1e} Pa-s, rho={rho} kg/m3")

    # --- 有效导热 ---
    k_0 = model.effective_conductivity_stagnant(k_f)
    print(f"\nStagnant k_eff,0 = {k_0:.4f} W/m-K  (k_eff,0/k_f = {k_0/k_f:.2f})")

    # --- Re扫描 ---
    print(f"\n{'Re_p':>6} | {'h_lower':>10} {'h_nominal':>10} {'h_upper':>10} | {'f_equiv':>10}")
    print("-" * 65)

    for Re_p in [5, 10, 20, 50, 100, 200, 500]:
        results = model.interval_estimate(Re_p, Pr, k_f)
        f_eq = model.friction_factor_ergun(Re_p)
        print(
            f"{Re_p:6d} | "
            f"{results['lower']['h_eff']:10.1f} "
            f"{results['nominal']['h_eff']:10.1f} "
            f"{results['upper']['h_eff']:10.1f} | "
            f"{f_eq:10.1f}"
        )

    # --- 压降对比 ---
    u_s = 0.5  # m/s
    dP, breakdown = model.pressure_drop_ergun(rho, mu, u_s, L=1.0)
    print(f"\n压降 (u_s={u_s} m/s, L=1m):")
    print(f"  粘性项: {breakdown['dP_viscous']:.0f} Pa")
    print(f"  惯性项: {breakdown['dP_inertial']:.0f} Pa")
    print(f"  总计:   {dP:.0f} Pa  ({dP/1e3:.2f} kPa)")

    # --- 统一接口测试 ---
    print(f"\n--- 统一接口 (Re_channel=1000) ---")
    for mode in ['lower', 'nominal', 'upper']:
        h_eff, f_eff, det = model.get_htc_and_friction(
            Re_channel=1000, Pr=Pr, k_f=k_f,
            tpms_type='Diamond', mode=mode
        )
        print(f"  {mode:8s}: h_eff={h_eff:8.1f} W/m2K, f_equiv={f_eff:8.1f}, "
              f"Re_p={det['Re_p']:.1f}, eta_fin={det['eta_fin']:.3f}")

    print("\n" + "=" * 70)
    print("Self-test completed.")
    print("=" * 70)


if __name__ == "__main__":
    test_packed_bed_model()
