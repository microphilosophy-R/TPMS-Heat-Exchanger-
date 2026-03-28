"""
澶氬瓟浠嬭川 + TPMS 鑰﹀悎浼犵儹妯″瀷
Packed Bed + TPMS Combined Heat Transfer Model

褰揟PMS娴侀亾鍐呭～鍏呭偓鍖栧墏棰楃矑鏃讹紝澹侀潰鍒版祦浣撶殑浼犵儹璺緞鍙樹负锛?
  娴佷綋涓讳綋 鈫?濉厖搴婂讥鏁?瀵规祦 鈫?鏈夋晥瀵肩儹 鈫?杩戝鍖?鈫?TPMS澹侀潰

鏈ā鍧楀寘鍚細
1. 鏈夋晥瀵肩儹绯绘暟 (妯″瀷A): Zehner-Bauer-Schl眉nder (闈欐€? + 寮ユ暎椤?(Wen-Fan)
2. 澹侀潰浼犵儹绯绘暟 (妯″瀷A): Martin-Nilles
3. 鏈夋晥瀵肩儹绯绘暟 (妯″瀷B): Dixon骞傚緥闈欐€佸叧鑱?+ Re渚濊禆Peclet寮ユ暎椤?
4. 澹侀潰浼犵儹绯绘暟 (妯″瀷B): Dixon Nu_w鍏宠仈寮?
5. 缁煎悎澹侀潰浼犵儹绯绘暟: 鍙屽尯鍩熸ā鍨?+ TPMS缈呯墖鏁堝簲 (涓ょ妯″瀷鍏辩敤)
6. 鍘嬮檷: 淇Ergun鏂圭▼ + TPMS鏍℃鍥犲瓙
7. 鍖洪棿浼拌: lower / nominal / upper

htc_model 鍙傛暟閫夋嫨浼犵儹瀛愭ā鍨?
  'martin_nilles' (榛樿): ZBS/Maxwell 闈欐€佸鐑?+ Martin-Nilles 澹侀潰浼犵儹
  'dixon':                Dixon 骞傚緥闈欐€佸鐑?+ Dixon Nu_w 澹侀潰浼犵儹 + Dixon 鐑樆鍏紡

鎺ュ彛鍏煎鐜版湁 TPMSHeatExchanger 姹傝В鍣ㄣ€?

References
----------
- Zehner & Schl眉nder (1970), Chemie Ingenieur Technik, 42(14), 933-941.
- Martin & Nilles (1993), Chem. Eng. Process., 32(2), 77-83.
- Ergun (1952), Chem. Eng. Progress, 48, 89-94.
- Dixon & Cresswell (1979), AIChE Journal, 25(4), 663-676.
- Dixon (1988), Int. J. Heat Mass Transfer, 31(2), 337-344.
"""

import numpy as np
import warnings
from models.packed_closures import (
    SUPPORTED_PACKED_MODES,
    SUPPORTED_HYDRAULIC_MODELS,
    SUPPORTED_PHI_SOURCES,
    SUPPORTED_HT_NOMINAL_RULES,
    SUPPORTED_WALL_ENHANCEMENT_MODELS as SUPPORTED_HT_ENHANCEMENT_MODELS,
    normalize_hydraulic_model,
    normalize_packed_heat_transfer_model,
    normalize_wall_enhancement_model,
    get_hydraulic_closure,
    get_packed_heat_transfer_closure,
    get_wall_enhancement_policy,
)

# Chapter 3 fit summary: f = A * Re^b
_CH3_F_RE_COEFFS = {
    'Diamond': (3263.281605, -0.905257),
    'Gyroid': (2005.814133, -0.873869),
    'Plate': (522.376132, -0.924664),
    'Wavy': (1626.060415, -0.866665),
}


class PackedBedTPMSModel:
    """
    TPMS娴侀亾鍐呭～鍏呭簥鐨勮€﹀悎浼犵儹涓庡帇闄嶆ā鍨嬨€?

    鐑樆鍒嗚В:
        1/h_eff = 1/h_w + D_h / (C_shape * k_r,eff)

    鍏朵腑 h_w 涓哄闈紶鐑郴鏁? k_r,eff 涓哄緞鍚戞湁鏁堝鐑郴鏁?
    D_h 涓篢PMS姘村姏鐩村緞, C_shape 涓哄嚑浣曞洜瀛愩€?

    Parameters
    ----------
    catalyst_config : dict
        particle_diameter : float  鍌寲鍓傜矑寰?[m]
        bed_porosity : float       搴婂眰瀛旈殭鐜?[-]
        k_solid : float            鍌寲鍓傚浐浣撳鐑郴鏁?[W/m路K]
        shape_factor : float       棰楃矑鐞冨舰搴?[-], 榛樿1.0
    tpms_geometry : dict
        D_h : float                TPMS閫氶亾姘村姏鐩村緞 [m]
        wall_thickness : float     TPMS澹佸帤 [m]
        k_wall : float             TPMS澹侀潰鏉愭枡瀵肩儹绯绘暟 [W/m路K]
    """

    def __init__(self, catalyst_config, tpms_geometry):
        # 鍌寲鍓傚～鍏呭簥鍙傛暟
        self.d_p = catalyst_config['particle_diameter']
        self.eps_bed = catalyst_config['bed_porosity']

        # Support both scalar and callable k_solid
        k_solid_input = catalyst_config.get('k_solid', 10.0)
        k_solid_material = catalyst_config.get('k_solid_material', None)

        if k_solid_material:
            try:
                from models.solid_props import get_k_solid
                self.k_s = lambda T: get_k_solid(k_solid_material, T)
                self._k_s_is_callable = True
            except Exception as e:
                warnings.warn(f"Could not load material '{k_solid_material}': {e}. Using constant k_solid")
                self.k_s = float(k_solid_input)
                self._k_s_is_callable = False
        else:
            self.k_s = float(k_solid_input)
            self._k_s_is_callable = False

        self.sphericity = catalyst_config.get('shape_factor', 1.0)

        # TPMS/PlateFin鍑犱綍鍙傛暟
        self.D_h = tpms_geometry['D_h']
        self.t_wall = tpms_geometry['wall_thickness']
        self.k_wall = tpms_geometry['k_wall']
        # PlateFin-specific: fin height and Af/Ah ratio for plate-fin efficiency (Eqs. 10鈥?2)
        self.fin_height_for_eff = tpms_geometry.get('fin_height', None)
        self.Af_Ah_ratio        = tpms_geometry.get('Af_Ah_ratio', None)
        # Fin-to-base-plate area ratio: Afin/Abase where Abase = width 脳 length
        # TPMS: alpha * H (SAD 脳 channel height); PlateFin: (2*Hf - tf) / sf
        self.Afin_Abase = tpms_geometry.get('Afin_Abase_ratio', 0.0)
        # Model controls for hydraulic/thermal enhancement
        self.hydraulic_model = normalize_hydraulic_model(
            catalyst_config.get('hydraulic_model', 'ergun_psi_tpms')
        )
        self.phi_source = str(catalyst_config.get('phi_source', 'ch3_f_re_fit')).strip().lower()
        self.ht_enhancement_model = normalize_wall_enhancement_model(
            catalyst_config.get('ht_enhancement_model', 'off')
        )
        self.ht_nominal_rule = str(catalyst_config.get('ht_nominal_rule', 'geometric')).strip().lower()

        if self.d_p <= 0:
            raise ValueError("particle_diameter must be > 0")
        if not self._k_s_is_callable and self.k_s <= 0:
            raise ValueError("k_solid must be > 0")
        if self.sphericity <= 0:
            raise ValueError("shape_factor must be > 0")
        if not (0.05 <= self.eps_bed <= 0.95):
            raise ValueError("bed_porosity must be within [0.05, 0.95]")
        if self.phi_source not in SUPPORTED_PHI_SOURCES:
            raise ValueError(f"phi_source must be one of {SUPPORTED_PHI_SOURCES}")
        if self.ht_nominal_rule not in SUPPORTED_HT_NOMINAL_RULES:
            raise ValueError(f"ht_nominal_rule must be one of {SUPPORTED_HT_NOMINAL_RULES}")

        self.hydraulic_closure = get_hydraulic_closure(self.hydraulic_model)
        self.wall_enhancement_policy = get_wall_enhancement_policy(
            self.ht_enhancement_model
        )

        # 娲剧敓鍙傛暟
        self.N_ratio = self.D_h / self.d_p  # 绠″緞/绮掑緞姣?

        if self.N_ratio < 2:
            warnings.warn(
                f"D_h/d_p = {self.N_ratio:.1f} < 2: continuum assumption may be weak; "
                "use results with caution."
            )

    def _eval_k_s(self, T=None):
        """Evaluate k_solid at temperature T (or use constant)"""
        if self._k_s_is_callable:
            return self.k_s(T) if T is not None else self.k_s(50.0)
        return self.k_s

    # ================================================================
    # 1. 鏈夋晥瀵肩儹绯绘暟
    # ================================================================

    def effective_conductivity_stagnant(self, k_f, T=None):
        """
        Zehner-Bauer-Schl眉nder 闈欐€佹湁鏁堝鐑郴鏁般€?

        鏃犳祦鍔ㄦ椂濉厖搴婄殑绛夋晥瀵肩儹绯绘暟锛岃€冭檻鍥?娑蹭袱鐩稿鐑€?

        Parameters
        ----------
        k_f : float
            娴佷綋瀵肩儹绯绘暟 [W/m路K]
        T : float, optional
            娓╁害 [K], 鐢ㄤ簬娓╁害渚濊禆鐨刱_solid

        Returns
        -------
        k_eff_0 : float
            闈欐€佹湁鏁堝鐑郴鏁?[W/m路K]
        """
        eps = self.eps_bed
        k_s_val = self._eval_k_s(T)
        kappa = k_s_val / k_f  # 鍥烘恫瀵肩儹绯绘暟姣?

        # --- Maxwell 鏈夋晥浠嬭川妯″瀷 (瀵规墍鏈夋kappa绋冲畾) ---
        num = k_s_val + 2.0 * k_f + 2.0 * (1.0 - eps) * (k_s_val - k_f)
        den = k_s_val + 2.0 * k_f - (1.0 - eps) * (k_s_val - k_f)
        k_maxwell = k_f * num / den

        # --- ZBS 妯″瀷 (楂榢appa鏃舵洿鍑嗙‘, 浣嗗湪kappa鈮圔闄勮繎涓嶇ǔ瀹? ---
        B = 1.25 * ((1.0 - eps) / eps) ** (10.0 / 9.0)
        N = 1.0 - B / kappa

        if kappa < 1.01 or abs(N) < 0.05:
            # kappa鎺ヨ繎1鎴朜鎺ヨ繎0鐨勯€€鍖栧尯: 浠呯敤Maxwell
            return max(k_maxwell, k_f)

        if N > 0:
            ln_term = np.log(kappa / B)
            term_a = (B * (kappa - 1.0) / kappa) * ln_term / N
            term_b = (B - 1.0) / N
            term_c = (B + 1.0) / 2.0
            k_cell = k_f * (2.0 / N) * (term_a - term_b + term_c)
            k_zbs = (1.0 - np.sqrt(1.0 - eps)) * k_f + np.sqrt(1.0 - eps) * k_cell
        else:
            # N < 0: ZBS涓嶉€傜敤
            k_zbs = k_f

        # 鍙栦袱绉嶆ā鍨嬩腑鐨勮緝澶у€? 淇濊瘉鍗曡皟鎬у拰绋冲畾鎬?
        # Maxwell鍦ㄤ綆kappa鏇村噯纭? ZBS鍦ㄩ珮kappa鏇村噯纭?
        k_eff_0 = max(k_maxwell, k_zbs)

        # 鐗╃悊涓嬬晫: k_eff 涓嶅彲鑳戒綆浜庣函娴佷綋瀵肩儹
        return max(k_eff_0, k_f)

    def effective_conductivity_dispersion(self, k_f, Re_p, Pr):
        """
        寮ユ暎椤瑰寰勫悜鏈夋晥瀵肩儹鐨勮础鐚€?

        k_disp = C_disp * Pe_p * k_f
        鍏朵腑 Pe_p = Re_p * Pr (棰楃矑Peclet鏁?

        寰勫悜寮ユ暎绯绘暟 C_disp 鈮?0.1 (Wen-Fan)

        Parameters
        ----------
        k_f : float
            娴佷綋瀵肩儹绯绘暟 [W/m路K]
        Re_p : float
            棰楃矑Reynolds鏁?= 蟻*u_s*d_p/渭
        Pr : float
            Prandtl鏁?

        Returns
        -------
        k_disp : float
            寮ユ暎瀵肩儹绯绘暟 [W/m路K]
        """
        Pe_p = Re_p * Pr
        C_disp = 0.1  # 寰勫悜寮ユ暎 (Wen-Fan)
        return C_disp * Pe_p * k_f

    def effective_conductivity_total(self, k_f, Re_p, Pr, T=None):
        """Total radial effective conductivity = stagnant + dispersion."""
        k_0 = self.effective_conductivity_stagnant(k_f, T)
        k_d = self.effective_conductivity_dispersion(k_f, Re_p, Pr)
        return k_0 + k_d

    # ================================================================
    # 1b. Dixon 寰勫悜鏈夋晥瀵肩儹绯绘暟妯″瀷 (Eq 0.3, Eq 0.6)
    # ================================================================

    def _radial_peclet_dixon(self, Re_p):
        """
        寰勫悜浼犵儹Peclet鏁?(Eq 0.6):
            Pe_r = 1 / (0.11 + 20.64/Re)

        璇ュ叕寮忓湪 Re鈫? 鏃惰秼浜?Re/20.64 閬垮厤濂囩偣锛?
        鍦ㄩ珮Re鏃惰秼浜庡父鏁?1/0.11 鈮?9.1銆?
        """
        Re_safe = max(Re_p, 1e-6)
        return 1.0 / (0.11 + 20.64 / Re_safe)

    def effective_conductivity_dixon_stagnant(self, k_f, T=None):
        """
        Dixon骞傚緥闈欐€佹湁鏁堝緞鍚戝鐑郴鏁?(Eq 0.3 闈欐€侀」):
            k_r^0 = 位_h * (位_s/位_h)^(0.28 - 0.757路log10(蔚) - 0.057路log10(位_s/位_h))

        Parameters
        ----------
        k_f : float
            娴佷綋瀵肩儹绯绘暟 位_h [W/m路K]
        T : float, optional
            娓╁害 [K], 鐢ㄤ簬娓╁害渚濊禆鐨刱_solid

        Returns
        -------
        k_r0 : float
            闈欐€佹湁鏁堝緞鍚戝鐑郴鏁?[W/m路K]
        """
        k_s_val = self._eval_k_s(T)
        kappa = k_s_val / k_f
        exponent = (
            0.28
            - 0.757 * np.log10(self.eps_bed)
            - 0.057 * np.log10(kappa)
        )
        return k_f * (kappa ** exponent)

    def effective_conductivity_dixon(self, k_f, Re_p, Pr, T=None):
        """
        Dixon鎬绘湁鏁堝緞鍚戝鐑郴鏁? 闈欐€侀」 + Pe_r寮ユ暎椤?(Eq 0.3):
            k_r = k_r^0 + (位_h / Pe_r) 路 Re 路 Pr

        Parameters
        ----------
        k_f : float   娴佷綋瀵肩儹绯绘暟 [W/m路K]
        Re_p : float  棰楃矑Reynolds鏁?
        Pr : float    Prandtl鏁?
        T : float, optional  娓╁害 [K]

        Returns
        -------
        k_r : float   鎬绘湁鏁堝緞鍚戝鐑郴鏁?[W/m路K]
        """
        k_r0 = self.effective_conductivity_dixon_stagnant(k_f, T)
        Pe_r = self._radial_peclet_dixon(Re_p)
        k_disp = k_f * Re_p * Pr / Pe_r
        return k_r0 + k_disp

    def wall_htc_dixon(self, Re_p, Pr, k_f, Nu_w0=20.0):
        """
        Dixon澹侀潰浼犵儹鍏宠仈寮?(Eqs 0.4, 0.5):
            Nu_w = Nu_{w,0} + 1 / (1/(0.3路Pr^(1/3)路Re^0.75) + 1/(0.054路Re路Pr))
            h_w  = Nu_w 路 位_h / d_p

        涓ら」鍙栬皟鍜屽钩鍧?(骞惰仈鐑樆褰㈠紡), 鍒嗗埆浠ｈ〃瀵规祦鑶滀紶鐑笌婀嶆祦寮ユ暎浼犵儹鐨勬瀬闄愩€?

        Parameters
        ----------
        Re_p : float    棰楃矑Reynolds鏁?
        Pr : float      Prandtl鏁?
        k_f : float     娴佷綋瀵肩儹绯绘暟 位_h [W/m路K]
        Nu_w0 : float   鏃犳祦閲忓闈usselt鏁? 鐞冨舰棰楃矑鍙栦腑浣嶆暟20 [111]

        Returns
        -------
        h_w : float   澹侀潰浼犵儹绯绘暟 [W/m虏路K]
        Nu_w : float  澹侀潰Nusselt鏁?(鍩轰簬d_p)
        """
        term1 = 0.3 * Pr ** (1.0 / 3.0) * Re_p ** 0.75
        term2 = 0.054 * Re_p * Pr
        Nu_w = Nu_w0 + 1.0 / (1.0 / max(term1, 1e-30) + 1.0 / max(term2, 1e-30))
        h_w = Nu_w * k_f / self.d_p
        return h_w, Nu_w

    def wall_htc_wang_experiment(self, Re_p, Pr, k_f):
        """
        Wang experiment correlation for packed bed in plate-fin channel.

        Nu_ce = 0.028535 * Re^1.0651 * Pr^5.3106

        Fitted to 90 experimental data points:
        - Re range: [67.6, 1331.8]
        - Pr range: [0.693, 0.735]
        - Nu range: [0.43, 11.42]

        Parameters
        ----------
        Re_p : float    棰楃矑Reynolds鏁?
        Pr : float      Prandtl鏁?
        k_f : float     娴佷綋瀵肩儹绯绘暟 [W/m路K]

        Returns
        -------
        h_w : float   澹侀潰浼犵儹绯绘暟 [W/m虏路K]
        Nu_w : float  澹侀潰Nusselt鏁?(鍩轰簬d_p)
        """
        Nu_w = 0.028535 * Re_p**1.0651 * Pr**5.3106
        h_w = Nu_w * k_f / self.d_p
        return h_w, Nu_w

    # ================================================================
    # 2. 澹侀潰浼犵儹绯绘暟 (Martin-Nilles)
    # ================================================================

    def wall_htc_packed_bed(self, Re_p, Pr, k_f):
        """
        濉厖搴婂闈紶鐑郴鏁?(Martin-Nilles鍏宠仈寮?銆?

        Nu_w = (1.3 + 5/(D_h/d_p)) * (k_r,eff/k_f) + 0.19 * Re_p^0.75 * Pr^(1/3)

        璇ュ叧鑱斿紡鍦ㄤ綆D_h/d_p姣?澹侀潰鏁堝簲涓诲)鏃朵粛鏈夋晥銆?

        Parameters
        ----------
        Re_p : float
            棰楃矑Reynolds鏁?
        Pr : float
            Prandtl鏁?
        k_f : float
            娴佷綋瀵肩儹绯绘暟 [W/m路K]

        Returns
        -------
        h_w : float
            澹侀潰浼犵儹绯绘暟 [W/m虏路K]
        Nu_w : float
            澹侀潰Nusselt鏁?(鍩轰簬d_p)
        """
        k_r_eff = self.effective_conductivity_total(k_f, Re_p, Pr)
        ratio_k = k_r_eff / k_f
        ratio_D = max(self.N_ratio, 1.5)  # 閬垮厤闄や互杩囧皬鍊?

        # Martin-Nilles
        Nu_w = (1.3 + 5.0 / ratio_D) * ratio_k + 0.19 * Re_p**0.75 * Pr**(1.0 / 3.0)

        h_w = Nu_w * k_f / self.d_p
        return h_w, Nu_w

    # ================================================================
    # 3. TPMS 缈呯墖澧炲己
    # ================================================================

    def tpms_fin_efficiency(self, h_local):
        """
        Fin efficiency for TPMS or PlateFin channels.

        For TPMS: models the TPMS wall as a fin of length L_fin 鈮?D_h/4.
        For PlateFin: uses the perforated-fin formulas (Wang et al. 2024, Eqs. 10鈥?2):
            畏_f = tanh(m路Hf)/(m路Hf),  m = sqrt(2h/(k_wall路tf))
            畏_h = 1 鈭?(Af/Ah)路(1 鈭?畏_f)

        Parameters
        ----------
        h_local : float
            Local convective HTC [W/m虏路K]

        Returns
        -------
        eta : float
            Fin efficiency [-], range (0, 1]
        """
        if h_local <= 0 or self.k_wall <= 0 or self.t_wall <= 0:
            return 1.0

        if self.fin_height_for_eff is not None:
            # PlateFin: Eqs. 11鈥?2 for 畏_f, then Eq. 10 for 畏_h
            Hf = self.fin_height_for_eff
            m  = np.sqrt(2.0 * h_local / (self.k_wall * self.t_wall))
            mL = m * Hf
            if mL < 0.01:
                eta_f = 1.0
            elif mL > 20.0:
                eta_f = 1.0 / mL
            else:
                eta_f = np.tanh(mL) / mL
            Af_Ah = self.Af_Ah_ratio if self.Af_Ah_ratio is not None else 0.5
            return 1.0 - Af_Ah * (1.0 - eta_f)
        else:
            # TPMS: approximate fin half-length as D_h/4
            L_fin = self.D_h / 4.0
            m  = np.sqrt(2.0 * h_local / (self.k_wall * self.t_wall))
            mL = m * L_fin
            if mL < 0.01:
                return 1.0
            elif mL > 20.0:
                return 1.0 / mL
            else:
                return np.tanh(mL) / mL

    @staticmethod
    def _normalize_phi_structure_name(tpms_type):
        name = str(tpms_type or "").strip()
        if name in ("Plate", "PlateFin", "SmoothPlateFin"):
            return "Plate"
        return name

    def hydraulic_enhancement_phi(self, tpms_type, Re_channel):
        """
        Hydraulic enhancement factor from Chapter 3 fit: f = A*Re^b.
        """
        if self.phi_source != 'ch3_f_re_fit':
            return 1.0

        tpms_key = self._normalize_phi_structure_name(tpms_type)
        if tpms_key == "Plate":
            return 1.0

        coeff_tpms = _CH3_F_RE_COEFFS.get(tpms_key)
        coeff_plate = _CH3_F_RE_COEFFS.get("Plate")
        if coeff_tpms is None or coeff_plate is None:
            return 1.0

        re_safe = max(float(Re_channel), 1e-6)
        a_t, b_t = coeff_tpms
        a_p, b_p = coeff_plate
        phi = (a_t / a_p) * (re_safe ** (b_t - b_p))
        return float(max(phi, 1.0))

    def _ht_enhancement_bounds(self, phi_value):
        phi = max(float(phi_value), 1.0)
        lower = 1.0
        upper = phi
        if self.ht_nominal_rule == 'similarity':
            nominal = phi
        elif self.ht_nominal_rule == 'arithmetic':
            nominal = 0.5 * (lower + upper)
        else:
            nominal = np.sqrt(phi)
        return lower, nominal, upper

    def _ht_enhancement_factor(self, mode, phi_value):
        """
        Returns (lower, nominal, upper, used_factor) for traceability.
        """
        mode = str(mode).strip().lower()
        return self.wall_enhancement_policy.select_factor(self, mode, phi_value)

    # ================================================================
    # 4. 缁煎悎澹侀潰浼犵儹绯绘暟 (鍚尯闂翠及璁?
    # ================================================================

    def _single_fin_efficiency(self, h_ref):
        if self.fin_height_for_eff is not None:
            m_value = np.sqrt(2.0 * h_ref / max(self.k_wall * self.t_wall, 1e-30))
            m_l = m_value * self.fin_height_for_eff
            if m_l < 0.01:
                return 1.0
            if m_l > 20.0:
                return 1.0 / m_l
            return np.tanh(m_l) / m_l
        return self.tpms_fin_efficiency(h_ref)

    def _shape_factor_from_uncertainty(self, uncertainty_mode):
        if uncertainty_mode == 'lower':
            return 8.0
        if uncertainty_mode == 'upper':
            return 4.0
        return 6.0

    def _apply_area_enhancement(self, h_pure, h_ref_base):
        # Area gain is treated as a static geometry effect: it uses the
        # unenhanced wall-side reference HTC and is no longer switched by
        # wall_from_phi or packed-bed uncertainty mode.
        eta_fin = self._single_fin_efficiency(h_ref_base)
        area_factor = 1.0 + self.Afin_Abase * eta_fin
        h_eff = h_pure * area_factor
        return h_eff, eta_fin, area_factor

    def _overall_htc_dixon_impl(self, Re_p, Pr, k_f, uncertainty_mode='nominal',
                                Nu_w0=20.0, t=None, phi_value=1.0):
        enh_lower, enh_nominal, enh_upper, enh_used = self._ht_enhancement_factor(
            uncertainty_mode, phi_value
        )

        h_w_raw, Nu_w_raw = self.wall_htc_dixon(Re_p, Pr, k_f, Nu_w0)
        h_w = h_w_raw * enh_used
        Nu_w = Nu_w_raw * enh_used
        k_r = self.effective_conductivity_dixon(k_f, Re_p, Pr, t)

        d_i = self.D_h
        Bi = h_w * d_i / (2.0 * k_r)
        R_w = 1.0 / h_w
        R_bed = (d_i / (6.0 * k_r)) * (Bi + 3.0) / (Bi + 4.0)
        h_pure = 1.0 / (R_w + R_bed)
        h_i, eta_fin, area_factor = self._apply_area_enhancement(h_pure, h_w_raw)

        details = {
            'htc_model': 'dixon',
            'wall_htc_source': 'dixon',
            'bed_conduction_source': 'dixon',
            'wall_enhancement_scope': 'wall_htc_only',
            'h_w_raw': h_w_raw,
            'h_w': h_w,
            'Nu_w_raw': Nu_w_raw,
            'Nu_w': Nu_w,
            'k_r_stagnant': self.effective_conductivity_dixon_stagnant(k_f, t),
            'k_r': k_r,
            'Bi': Bi,
            'R_wall_film': R_w,
            'R_bed_conduction': R_bed,
            'R_total': R_w + R_bed,
            'h_pure': h_pure,
            'Afin_Abase': self.Afin_Abase,
            'eta_fin': eta_fin,
            'area_factor': area_factor,
            'h_eff': h_i,
            'mode': uncertainty_mode,
            'D_h_over_d_p': self.N_ratio,
            'phi': float(max(phi_value, 1.0)),
            'enh_lower': enh_lower,
            'enh_nominal': enh_nominal,
            'enh_upper': enh_upper,
            'enh_used': enh_used,
            'ht_enhancement_model': self.ht_enhancement_model,
        }
        return h_i, details

    def overall_htc_dixon(self, Re_p, Pr, k_f, mode='nominal', Nu_w0=20.0, T=None,
                          phi_value=1.0):
        uncertainty_mode = str(mode).strip().lower()
        if uncertainty_mode not in SUPPORTED_PACKED_MODES:
            raise ValueError(
                f"Invalid packed mode '{uncertainty_mode}'. Use one of {SUPPORTED_PACKED_MODES}."
            )
        return self._overall_htc_dixon_impl(
            Re_p,
            Pr,
            k_f,
            uncertainty_mode=uncertainty_mode,
            Nu_w0=Nu_w0,
            t=T,
            phi_value=phi_value,
        )

    def _overall_htc_martin_like_impl(self, Re_p, Pr, k_f, uncertainty_mode='nominal',
                                      t=None, phi_value=1.0,
                                      wall_htc_source='martin_nilles'):
        enh_lower, enh_nominal, enh_upper, enh_used = self._ht_enhancement_factor(
            uncertainty_mode, phi_value
        )
        if wall_htc_source == 'wang_wall_htc':
            h_w_raw, Nu_w_raw = self.wall_htc_wang_experiment(Re_p, Pr, k_f)
        else:
            h_w_raw, Nu_w_raw = self.wall_htc_packed_bed(Re_p, Pr, k_f)

        k_r_eff = self.effective_conductivity_total(k_f, Re_p, Pr, t)
        h_w = h_w_raw * enh_used
        Nu_w = Nu_w_raw * enh_used
        C_shape = self._shape_factor_from_uncertainty(uncertainty_mode)
        k_r_adj = k_r_eff
        R_wall_film = 1.0 / h_w
        R_bed_cond = self.D_h / (C_shape * k_r_adj)
        R_total = R_wall_film + R_bed_cond
        h_pure = 1.0 / R_total
        h_eff, eta_fin, area_factor = self._apply_area_enhancement(h_pure, h_w_raw)

        details = {
            'htc_model': wall_htc_source,
            'wall_htc_source': wall_htc_source,
            'bed_conduction_source': 'martin_nilles',
            'wall_enhancement_scope': 'wall_htc_only',
            'h_w_raw': h_w_raw,
            'h_w': h_w,
            'Nu_w_raw': Nu_w_raw,
            'Nu_w': Nu_w,
            'k_eff_stagnant': self.effective_conductivity_stagnant(k_f, t),
            'k_r_eff': k_r_eff,
            'k_r_adjusted': k_r_adj,
            'R_wall_film': R_wall_film,
            'R_bed_conduction': R_bed_cond,
            'R_total': R_total,
            'h_pure': h_pure,
            'Afin_Abase': self.Afin_Abase,
            'eta_fin': eta_fin,
            'area_factor': area_factor,
            'C_shape': C_shape,
            'h_eff': h_eff,
            'mode': uncertainty_mode,
            'D_h_over_d_p': self.N_ratio,
            'phi': float(max(phi_value, 1.0)),
            'enh_lower': enh_lower,
            'enh_nominal': enh_nominal,
            'enh_upper': enh_upper,
            'enh_used': enh_used,
            'ht_enhancement_model': self.ht_enhancement_model,
        }
        return h_eff, details

    def overall_htc_packed_side(self, Re_p, Pr, k_f, mode='nominal',
                                htc_model='martin_nilles', T=None,
                                phi_value=1.0):
        uncertainty_mode = str(mode).strip().lower()
        if uncertainty_mode not in SUPPORTED_PACKED_MODES:
            raise ValueError(
                f"Invalid packed mode '{uncertainty_mode}'. Use one of {SUPPORTED_PACKED_MODES}."
            )

        closure = get_packed_heat_transfer_closure(htc_model)
        return closure.overall_htc(
            self,
            re_p=Re_p,
            pr=Pr,
            k_f=k_f,
            uncertainty_mode=uncertainty_mode,
            t=T,
            phi_value=phi_value,
        )

    # ================================================================
    # 5. 鍘嬮檷妯″瀷
    # ================================================================

    def friction_factor_ergun(self, Re_p):
        """
        Ergun鏂圭▼绛夋晥鎽╂摝鍥犲瓙銆?

        杞崲涓轰笌鐜版湁姹傝В鍣ㄥ吋瀹圭殑Fanning鎽╂摝鍥犲瓙鏍煎紡:
        dP = f_equiv * (L/D_h) * (蟻*u虏/2)

        鍏朵腑 u 涓篢PMS閫氶亾鍐呯殑琛ㄨ閫熷害 (= m_dot / (蟻 * Ac_TPMS))

        Parameters
        ----------
        Re_p : float
            棰楃矑Reynolds鏁?= 蟻*u_s*d_p/渭

        Returns
        -------
        f_equiv : float
            绛夋晥Fanning鎽╂摝鍥犲瓙 [-]
        """
        eps = self.eps_bed
        f_equiv = (self.D_h / self.d_p) * (1.0 - eps) / eps**3 * (
            300.0 * (1.0 - eps) / max(Re_p, 0.1) + 3.5
        )
        return f_equiv

    def pressure_drop_ergun(self, rho, mu, u_s, L):
        """
        Ergun鏂圭▼鐩存帴璁＄畻鍘嬮檷銆?

        螖P/L = 150*渭*u_s*(1-蔚)虏 / (蔚鲁*d_p虏) + 1.75*蟻*u_s虏*(1-蔚) / (蔚鲁*d_p)

        Parameters
        ----------
        rho : float  娴佷綋瀵嗗害 [kg/m鲁]
        mu : float   鍔ㄥ姏绮樺害 [Pa路s]
        u_s : float  琛ㄨ閫熷害 [m/s]
        L : float    搴婂眰闀垮害 [m]

        Returns
        -------
        dP : float       鎬诲帇闄?[Pa]
        breakdown : dict  绮樻€?鎯€ч」鍒嗚В
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
        TPMS瀵瑰～鍏呭簥鍘嬮檷鐨勬牎姝ｅ洜瀛愩€?

        涓嶅悓TPMS楠ㄦ灦瀵规祦閬撶殑瀹忚鎵洸绋嬪害涓嶅悓(绗笁绔犵粨璁?,
        瀵艰嚧鐩稿悓濉厖搴婂湪涓嶅悓TPMS涓殑鍘嬮檷瀛樺湪宸紓銆?
        姝ゆ牎姝ｅ洜瀛愪箻浠rgun鍩虹鍘嬮檷銆?

        鏁板€兼潵婧? 鍩轰簬绗笁绔犲疄楠岃秼鍔跨殑浼拌鍊笺€?

        Parameters
        ----------
        tpms_type : str
            TPMS绫诲瀷

        Returns
        -------
        psi : float
            鏍℃鍥犲瓙 [-], >= 1.0
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
    # 6. 缁熶竴鎺ュ彛 (鍏煎鐜版湁姹傝В鍣?
    # ================================================================

    def get_htc_and_friction(self, Re_channel, Pr, k_f, tpms_type='Diamond',
                             mode='nominal', htc_model='martin_nilles', T=None):
        """
        缁熶竴鎺ュ彛: 杩斿洖鏈夋晥浼犵儹绯绘暟鍜岀瓑鏁堟懇鎿﹀洜瀛愩€?

        灏嗛€氶亾Re鑷姩杞崲涓洪绮扲e, 骞跺簲鐢═PMS鍘嬮檷鏍℃銆?

        Parameters
        ----------
        Re_channel : float
            鍩轰簬TPMS閫氶亾姘村姏鐩村緞鐨凴eynolds鏁?
        Pr : float
            Prandtl鏁?
        k_f : float
            娴佷綋瀵肩儹绯绘暟 [W/m路K]
        tpms_type : str
            TPMS绫诲瀷
        mode : str
            浼拌妯″紡: 'lower', 'nominal', 'upper'
        htc_model : str
            浼犵儹瀛愭ā鍨? 'martin_nilles' (榛樿) 鎴?'dixon'
        T : float, optional
            娓╁害 [K]

        Returns
        -------
        h_eff : float
            鏈夋晥浼犵儹绯绘暟 [W/m虏路K]
        f_equiv : float
            绛夋晥Fanning鎽╂摝鍥犲瓙 [-]
        details : dict
            璁＄畻缁嗚妭
        """
        # 閫氶亾Re 鈫?棰楃矑Re
        mode = str(mode).strip().lower()
        if mode not in SUPPORTED_PACKED_MODES:
            raise ValueError(
                f"Invalid packed mode '{mode}'. Use one of {SUPPORTED_PACKED_MODES}."
            )

        Re_p = Re_channel * (self.d_p / self.D_h)
        phi = self.hydraulic_enhancement_phi(tpms_type, Re_channel)
        htc_model_canonical = normalize_packed_heat_transfer_model(htc_model)

        # 浼犵儹
        h_eff, details = self.overall_htc_packed_side(Re_p, Pr, k_f, mode,
                                                      htc_model=htc_model_canonical, T=T,
                                                      phi_value=phi)

        # 鍘嬮檷
        f_equiv, hydraulic_details = self.hydraulic_closure.compute_friction(
            self,
            re_p=Re_p,
            re_channel=Re_channel,
            tpms_type=tpms_type,
            phi_value=phi,
        )

        details.update(hydraulic_details)
        details['Re_p'] = Re_p
        details['htc_model'] = htc_model_canonical

        return h_eff, f_equiv, details

    # ================================================================
    # 7. 鍖洪棿浼拌
    # ================================================================

    def interval_estimate(self, Re_p, Pr, k_f, htc_model='martin_nilles', T=None):
        """
        杩斿洖 lower / nominal / upper 涓夋。浼犵儹绯绘暟浼拌銆?

        鐢ㄤ簬涓嶇‘瀹氭€т紶鎾垎鏋愬拰璁烘枃涓殑缃俊鍖洪棿銆?

        Parameters
        ----------
        Re_p : float
            棰楃矑Reynolds鏁?
        Pr : float
            Prandtl鏁?
        k_f : float
            娴佷綋瀵肩儹绯绘暟 [W/m路K]
        htc_model : str
            浼犵儹瀛愭ā鍨? 'martin_nilles' (榛樿) 鎴?'dixon'
        T : float, optional
            娓╁害 [K]

        Returns
        -------
        results : dict
            keys = 'lower', 'nominal', 'upper'
            姣忛」鍖呭惈 h_eff 鍜?details
        """
        results = {}
        for mode in ['lower', 'nominal', 'upper']:
            h_eff, details = self.overall_htc_packed_side(Re_p, Pr, k_f, mode,
                                                         htc_model=htc_model, T=T)
            results[mode] = {'h_eff': h_eff, 'details': details}
        return results


# ====================================================================
# 杈呭姪鍑芥暟: 浠庣幇鏈塩onfig鐢熸垚PackedBedTPMSModel
# ====================================================================

def create_packed_bed_model(config, stream_key='hot',
                            cell_size_override=None, t_wall_override=None):
    """
    浠庣幇鏈夋崲鐑櫒config瀛楀吀鍒涘缓PackedBedTPMSModel瀹炰緥銆?

    鍦╟onfig涓渶娣诲姞 'catalyst' 閮ㄥ垎:
        config['catalyst'] = {
            'particle_diameter': 1e-3,    # [m]
            'bed_porosity': 0.40,         # [-]
            'k_solid': 10.0,              # [W/m路K]
            'shape_factor': 1.0,          # [-]
        }

    Parameters
    ----------
    config : dict
        鎹㈢儹鍣ㄩ厤缃瓧鍏?

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
        'k_solid_material': packed_cfg.get('k_solid_material', cat.get('k_solid_material', None)),
        'shape_factor': packed_cfg.get('shape_factor', cat.get('shape_factor', 1.0)),
        'hydraulic_model': packed_cfg.get('hydraulic_model', cat.get('hydraulic_model', 'ergun_psi_tpms')),
        'phi_source': packed_cfg.get('phi_source', cat.get('phi_source', 'ch3_f_re_fit')),
        'ht_enhancement_model': packed_cfg.get(
            'ht_enhancement_model', cat.get('ht_enhancement_model', 'off')
        ),
        'ht_nominal_rule': packed_cfg.get('ht_nominal_rule', cat.get('ht_nominal_rule', 'geometric')),
    }

    geo = config['geometry']
    porosity_default = 0.65 if stream_key == 'hot' else 0.70
    # Prefer per-channel geometry porosity; fall back to legacy global keys
    ch_geo_for_por = config.get('channels', {}).get(stream_key, {}).get('geometry', {}) or {}
    porosity = (ch_geo_for_por.get('porosity')
                or geo.get(f'porosity_{stream_key}', porosity_default))

    structure  = config.get('channels', {}).get(stream_key, {}).get('structure', '')
    ch_geo_raw = config.get('channels', {}).get(stream_key, {}).get('geometry', {}) or {}

    if structure == 'PlateFin':
        # PlateFin: use fin-geometry hydraulic diameter (Wang et al. 2024, Eq. 2)
        Hf = ch_geo_raw.get('fin_height')    or geo.get('fin_height',    9.5e-3)
        sf = ch_geo_raw.get('fin_spacing')   or geo.get('fin_spacing',   3.2e-3)
        tf = ch_geo_raw.get('fin_thickness') or geo.get('fin_thickness', 0.6e-3)
        D_h       = 2.0 * (Hf - tf) * (sf - tf) / max(Hf + sf - 2.0 * tf, 1e-12)
        t_wall_eff = tf   # fin thickness acts as the fin wall thickness
        # Symmetric fin model (cold-hot-cold-hot stacking):
        # Each hot fin layer has a cold dividing plate on BOTH sides.
        # By symmetry the effective fin half-height = Hf/2 (adiabatic midplane).
        fin_height_eff = Hf / 2.0
        # Af/Ah ratio (no perforations by default)
        n_d = ch_geo_raw.get('perf_density', geo.get('perf_density', 0.0))
        r_p = ch_geo_raw.get('perf_radius',  geo.get('perf_radius',  0.0))
        if n_d > 0 and r_p > 0:
            Af_base = (2 * Hf - tf) + (sf - tf)
            Ah_base = (2 * Hf - tf) + 2 * (sf - tf)
            perf_af = 3 * n_d * np.pi * r_p**2 - 2 * n_d * np.pi * (2 * r_p) * tf
            perf_ah = 2 * n_d * np.pi * r_p**2 - 2 * n_d * np.pi * (2 * r_p) * tf
            Af_Ah_ratio = max(Af_base - perf_af, 1e-12) / max(Ah_base - perf_ah, 1e-12)
        else:
            Af_val = (2 * Hf - tf) + (sf - tf)
            Ah_val = (2 * Hf - tf) + 2 * (sf - tf)
            Af_Ah_ratio = Af_val / max(Ah_val, 1e-12)
        # Afin/Abase: each dividing plate owns half the fin area (symmetric arrangement)
        # Full perimeter per pitch = (2*Hf - tf); split equally between top and bottom plate
        Afin_Abase_ratio = (2.0 * Hf - tf) / (2.0 * max(sf, 1e-12))
    else:
        # TPMS: existing formula
        _cell_default = ch_geo_raw.get('unit_cell_size') or geo.get('unit_cell_size', 5e-3)
        _wall_default = ch_geo_raw.get('wall_thickness') or geo.get('wall_thickness', 5e-4)
        cell_size  = cell_size_override if cell_size_override is not None else _cell_default
        D_h        = 4.0 * porosity * cell_size / (2.0 * np.pi)
        t_wall_eff = t_wall_override if t_wall_override is not None else _wall_default
        fin_height_eff = None
        Af_Ah_ratio    = None
        # Afin/Abase: SAD 脳 channel height (TPMS ligaments act as fins on the base plate)
        alpha = config.get('channels', {}).get(stream_key, {}).get('surface_area_density', 0.0)
        H_ch  = float(ch_geo_raw.get('height') or geo.get('height', 0.25))
        Afin_Abase_ratio = alpha * H_ch   # [1/m] 脳 [m] = dimensionless

    tpms_geometry = {
        'D_h':             D_h,
        'wall_thickness':  t_wall_eff,
        'k_wall':          config['material']['k_wall'],
        'fin_height':      fin_height_eff,   # None -> TPMS logic; Hf/2 -> PlateFin symmetric
        'Af_Ah_ratio':     Af_Ah_ratio,
        'Afin_Abase_ratio': Afin_Abase_ratio,
    }

    return PackedBedTPMSModel(catalyst_config, tpms_geometry)


# ====================================================================
# 鑷涓庨獙璇?
# ====================================================================

def test_packed_bed_model():
    """Model self-check: print representative calculation results."""
    print("=" * 70)
    print("Packed Bed + TPMS Combined Model - Self Test")
    print("=" * 70)

    # 鍏稿瀷浣庢俯姘㈡恫鍖栧伐鍐靛弬鏁?
    catalyst_config = {
        'particle_diameter': 1.0e-3,   # 1 mm
        'bed_porosity': 0.40,
        'k_solid': 10.0,               # Fe2O3/Al2O3 绫诲偓鍖栧墏
        'shape_factor': 1.0,
    }
    tpms_geometry = {
        'D_h': 3.3e-3,                 # 鍏稿瀷 TPMS D_h (cell=5mm, 蔚=0.65)
        'wall_thickness': 0.5e-3,
        'k_wall': 237.0,               # 閾?
    }

    model = PackedBedTPMSModel(catalyst_config, tpms_geometry)
    print(f"\nD_h/d_p = {model.N_ratio:.1f}")

    # 浣庢俯姘㈡皵鍏稿瀷鐗╂€?(T鈮?0K, P鈮?MPa)
    k_f = 0.10     # W/m路K
    Pr = 0.80
    mu = 3e-6      # Pa路s
    rho = 3.0      # kg/m鲁

    print(f"\nFluid: k_f={k_f} W/m-K, Pr={Pr}, mu={mu:.1e} Pa-s, rho={rho} kg/m3")

    # --- 鏈夋晥瀵肩儹 ---
    k_0 = model.effective_conductivity_stagnant(k_f)
    print(f"\nStagnant k_eff,0 = {k_0:.4f} W/m-K  (k_eff,0/k_f = {k_0/k_f:.2f})")

    # --- Re鎵弿 ---
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

    # --- 鍘嬮檷瀵规瘮 ---
    u_s = 0.5  # m/s
    dP, breakdown = model.pressure_drop_ergun(rho, mu, u_s, L=1.0)
    print(f"\n鍘嬮檷 (u_s={u_s} m/s, L=1m):")
    print(f"  绮樻€ч」: {breakdown['dP_viscous']:.0f} Pa")
    print(f"  鎯€ч」: {breakdown['dP_inertial']:.0f} Pa")
    print(f"  鎬昏:   {dP:.0f} Pa  ({dP/1e3:.2f} kPa)")

    # --- 缁熶竴鎺ュ彛娴嬭瘯 (Martin-Nilles) ---
    print(f"\n--- 缁熶竴鎺ュ彛 Martin-Nilles (Re_channel=1000) ---")
    for mode in ['lower', 'nominal', 'upper']:
        h_eff, f_eff, det = model.get_htc_and_friction(
            Re_channel=1000, Pr=Pr, k_f=k_f,
            tpms_type='Diamond', mode=mode, htc_model='martin_nilles'
        )
        print(f"  {mode:8s}: h_eff={h_eff:8.1f} W/m2K, f_equiv={f_eff:8.1f}, "
              f"Re_p={det['Re_p']:.1f}, eta_fin={det['eta_fin']:.3f}")

    # --- Dixon 妯″瀷: 闈欐€佸鐑郴鏁伴獙璇?---
    print(f"\n--- Dixon k_r^0 vs ZBS k_eff,0 ---")
    k_r0_dixon = model.effective_conductivity_dixon_stagnant(k_f)
    k_r0_zbs = model.effective_conductivity_stagnant(k_f)
    print(f"  Dixon k_r^0 = {k_r0_dixon:.4f} W/m-K  (k_r^0/k_f = {k_r0_dixon/k_f:.2f})")
    print(f"  ZBS   k_0   = {k_r0_zbs:.4f} W/m-K  (k_0/k_f   = {k_r0_zbs/k_f:.2f})")

    # --- Dixon vs Martin-Nilles Re鎵弿瀵规瘮 ---
    print(f"\n{'Re_p':>6} | {'MN-lower':>10} {'MN-nom':>10} {'MN-upper':>10}"
          f" | {'DX-lower':>10} {'DX-nom':>10} {'DX-upper':>10}")
    print("-" * 80)
    for Re_p in [5, 10, 20, 50, 100, 200, 500]:
        res_mn = model.interval_estimate(Re_p, Pr, k_f, htc_model='martin_nilles')
        res_dx = model.interval_estimate(Re_p, Pr, k_f, htc_model='dixon')
        print(
            f"{Re_p:6d} | "
            f"{res_mn['lower']['h_eff']:10.1f} "
            f"{res_mn['nominal']['h_eff']:10.1f} "
            f"{res_mn['upper']['h_eff']:10.1f} | "
            f"{res_dx['lower']['h_eff']:10.1f} "
            f"{res_dx['nominal']['h_eff']:10.1f} "
            f"{res_dx['upper']['h_eff']:10.1f}"
        )

    # --- Dixon 鐑樆鍒嗚В缁嗚妭 (Re_p=100, nominal) ---
    print(f"\n--- Dixon 鐑樆鍒嗚В (Re_p=100, nominal) ---")
    _, det_dx = model.overall_htc_dixon(100, Pr, k_f, mode='nominal')
    print(f"  Nu_w         = {det_dx['Nu_w']:.2f}")
    print(f"  h_w          = {det_dx['h_w']:.1f} W/m2K")
    print(f"  k_r (stagnant) = {det_dx['k_r_stagnant']:.4f} W/m-K")
    print(f"  k_r (total)    = {det_dx['k_r']:.4f} W/m-K")
    print(f"  Bi           = {det_dx['Bi']:.3f}")
    print(f"  R_wall_film  = {det_dx['R_wall_film']*1e4:.3f} e-4 m2K/W")
    print(f"  R_bed_cond   = {det_dx['R_bed_conduction']*1e4:.3f} e-4 m2K/W")
    print(f"  h_i (Dixon)  = {det_dx['h_eff']:.1f} W/m2K")

    print("\n" + "=" * 70)
    print("Self-test completed.")
    print("=" * 70)


if __name__ == "__main__":
    test_packed_bed_model()

