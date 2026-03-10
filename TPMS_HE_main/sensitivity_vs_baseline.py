"""
Sensitivity Analysis: TPMS Structures vs. Smooth Plate-Fin Baseline
====================================================================

Compares all 6 TPMS types (Gyroid, Diamond, Primitive, Neovius, FRD, FKS)
against the SmoothPlateFin baseline using three parametric sweeps:

  Study 1 — PEC, j, f  vs. Reynolds number (Re = 100 → 10,000)
  Study 2 — PEC vs. porosity ε  at Re = 500, 2000, 5000
  Study 3 — PEC vs. unit cell size a  at fixed mass flux G
  Study 4 — Colburn j–friction f parametric map

Metrics:
  j   = Nu / (Re · Pr^(1/3))          Colburn j-factor
  f   = Fanning friction factor         from correlations
  PEC = (j/j₀) / (f/f₀)^(1/3)        Performance Evaluation Criterion
        (subscript 0 = SmoothPlateFin at same Re)

All analyses are correlation-level only (no full system solver).
Run from TPMS_HE_main/:  python sensitivity_vs_baseline.py
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# Suppress range-extrapolation warnings from the correlations module
warnings.filterwarnings("ignore")

from tpms_correlations import TPMSCorrelations

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
TPMS_ALL = ['Gyroid', 'Diamond', 'Primitive', 'Neovius', 'FRD', 'FKS']
BASELINE = 'SmoothPlateFin'
ALL_TYPES = TPMS_ALL + [BASELINE]

PR = 0.70          # Prandtl number  (H2/He cryogenic Gas)
K_F = 0.10         # W/(m·K)  fluid thermal conductivity
MU = 1.0e-5        # Pa·s     dynamic viscosity (H2 cryogenic)
EPS_REF = 0.65     # reference porosity
A_REF = 5.0e-3     # m, reference unit cell size
DH_REF = 4 * EPS_REF * A_REF / (2 * np.pi)   # ≈ 2.07 mm

# Consistent color + linestyle scheme across all figures
COLORS = {
    'Gyroid':        '#1f77b4',
    'Diamond':       '#ff7f0e',
    'Primitive':     '#2ca02c',
    'Neovius':       '#d62728',
    'FRD':           '#9467bd',
    'FKS':           '#8c564b',
    'SmoothPlateFin':'black',
}
LWIDTH = {t: 1.8 for t in TPMS_ALL}
LWIDTH['SmoothPlateFin'] = 2.5
LSTYLE = {t: '-' for t in TPMS_ALL}
LSTYLE['SmoothPlateFin'] = '--'

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':      'Times New Roman',
    'font.size':        12,
    'axes.labelsize':   14,
    'axes.titlesize':   13,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'savefig.dpi':      300,
})


# ---------------------------------------------------------------------------
# Helper: compute j and f arrays for all TPMS types at given Re array
# ---------------------------------------------------------------------------
def _compute_jf(Re_arr):
    """
    Returns dicts {tpms_type: array} for Nu, f, j evaluated at Re_arr.

    Re_arr : 1-D numpy array of Reynolds numbers
    """
    Re_arr = np.atleast_1d(np.asarray(Re_arr, dtype=float))
    Nu_d, f_d, j_d = {}, {}, {}
    for tpms in ALL_TYPES:
        Nu, f = TPMSCorrelations.get_correlations(tpms, Re_arr, PR, 'Gas')
        Nu = np.atleast_1d(np.nan_to_num(np.asarray(Nu, dtype=float), nan=np.nan))
        f  = np.atleast_1d(np.nan_to_num(np.asarray(f,  dtype=float), nan=np.nan))
        j  = Nu / (Re_arr * PR ** (1.0 / 3.0))
        Nu_d[tpms] = Nu
        f_d[tpms]  = f
        j_d[tpms]  = j
    return Nu_d, f_d, j_d


def _pec(j_tpms, f_tpms, j0, f0):
    """Vectorised PEC = (j/j₀) / (f/f₀)^(1/3)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        pec = (j_tpms / j0) / ((f_tpms / f0) ** (1.0 / 3.0))
    return np.where(np.isfinite(pec), pec, np.nan)


# ===========================================================================
# Study 1 — PEC, j, f vs. Reynolds Number
# ===========================================================================
def pec_vs_re(output_dir='results_sensitivity'):
    """
    Study 1: j-factor, friction factor, and PEC vs. Reynolds number.

    Re sweep: 100 → 10,000 (60 log-spaced points)
    Fixed:    ε = 0.65, a = 5 mm, Pr = 0.70, fluid = Gas
    Outputs:  pec_vs_re.png, pec_vs_re.csv
    """
    Re_arr = np.logspace(2, 4, 60)   # 100 → 10,000

    Nu_d, f_d, j_d = _compute_jf(Re_arr)
    j0 = j_d[BASELINE]
    f0 = f_d[BASELINE]

    # ---- figure -------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax_j, ax_f, ax_pec = axes

    for tpms in TPMS_ALL + [BASELINE]:
        mask = np.isfinite(j_d[tpms]) & np.isfinite(f_d[tpms])
        kw = dict(color=COLORS[tpms], lw=LWIDTH[tpms], ls=LSTYLE[tpms])

        ax_j.loglog(Re_arr[mask], j_d[tpms][mask], label=tpms, **kw)
        ax_f.loglog(Re_arr[mask], f_d[tpms][mask], label=tpms, **kw)

    for tpms in TPMS_ALL:
        pec = _pec(j_d[tpms], f_d[tpms], j0, f0)
        mask = np.isfinite(pec)
        ax_pec.semilogx(Re_arr[mask], pec[mask],
                        label=tpms,
                        color=COLORS[tpms], lw=LWIDTH[tpms])

    ax_pec.axhline(1.0, color='black', lw=1.8, ls='--', label='Baseline (PEC = 1)')

    # formatting
    ax_j.set_xlabel('Re [–]')
    ax_j.set_ylabel('j (Colburn factor) [–]')
    ax_j.set_title('(a) Colburn j-factor')
    ax_j.legend(loc='upper right', framealpha=0.85)
    ax_j.grid(True, which='both', alpha=0.3)

    ax_f.set_xlabel('Re [–]')
    ax_f.set_ylabel('f (Fanning) [–]')
    ax_f.set_title('(b) Fanning Friction Factor')
    ax_f.legend(loc='upper right', framealpha=0.85)
    ax_f.grid(True, which='both', alpha=0.3)

    ax_pec.set_xlabel('Re [–]')
    ax_pec.set_ylabel('PEC [–]')
    ax_pec.set_title('(c) Performance Evaluation Criterion')
    ax_pec.legend(loc='lower right', framealpha=0.85)
    ax_pec.grid(True, which='both', alpha=0.3)
    ax_pec.set_ylim(bottom=0)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'pec_vs_re.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {save_path}')

    # ---- CSV ----------------------------------------------------------
    rows = {'Re': Re_arr}
    for tpms in ALL_TYPES:
        rows[f'j_{tpms}']  = j_d[tpms]
        rows[f'f_{tpms}']  = f_d[tpms]
        rows[f'Nu_{tpms}'] = Nu_d[tpms]
    for tpms in TPMS_ALL:
        rows[f'PEC_{tpms}'] = _pec(j_d[tpms], f_d[tpms], j0, f0)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'pec_vs_re.csv')
    df.to_csv(csv_path, index=False, float_format='%.6e')
    print(f'[OK] Saved: {csv_path}')
    return df


# ===========================================================================
# Study 2 — PEC vs. Porosity ε
# ===========================================================================
def pec_vs_porosity(output_dir='results_sensitivity'):
    """
    Study 2: PEC vs. porosity ε at three representative Reynolds numbers.

    ε sweep:   0.50 → 0.80 (30 points)
    Fixed Re:  500, 2000, 5000
    Fixed:     a = 5 mm, Pr = 0.70, fluid = Gas

    Note: Nu/f correlations depend on Re, not on ε directly.
    PEC is therefore constant w.r.t. ε at fixed Re.  However, the Dh
    (= 4εa/2π) changes, so the *dimensional* h = Nu·k/Dh and the
    dimensional ΔP change — which matters for equal-pumping-power
    comparisons.  This study shows the PEC stability with ε while
    documenting which structure wins at each flow regime.

    Outputs: pec_vs_porosity.png, pec_vs_porosity.csv
    """
    eps_range = np.linspace(0.50, 0.80, 30)
    RE_FIXED = [500, 2000, 5000]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    csv_rows = {'eps': eps_range}

    for ax, re_val in zip(axes, RE_FIXED):
        # At fixed Re and Pr the correlations give fixed Nu, f → PEC is
        # independent of ε.  We still loop over eps to compute Dh-dependent
        # metrics and to future-proof against correlations that include Dh.
        Nu_d, f_d, j_d = _compute_jf(np.array([re_val]))
        j0 = float(j_d[BASELINE][0])
        f0 = float(f_d[BASELINE][0])

        for tpms in TPMS_ALL:
            pec_arr = np.full(len(eps_range), np.nan)
            for k, eps in enumerate(eps_range):
                pec_arr[k] = float(_pec(
                    float(j_d[tpms][0]), float(f_d[tpms][0]), j0, f0
                ))
            ax.plot(eps_range, pec_arr,
                    color=COLORS[tpms], lw=LWIDTH[tpms], label=tpms)
            csv_rows[f'PEC_{tpms}_Re{re_val}'] = pec_arr

        ax.axhline(1.0, color='black', lw=1.8, ls='--', label='Baseline')
        ax.set_xlabel('ε (porosity) [–]')
        ax.set_ylabel('PEC [–]')
        ax.set_title(f'Re = {re_val}')
        ax.legend(loc='center right', framealpha=0.85, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('PEC vs. Porosity at Fixed Re', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'pec_vs_porosity.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {save_path}')

    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(output_dir, 'pec_vs_porosity.csv')
    df.to_csv(csv_path, index=False, float_format='%.6e')
    print(f'[OK] Saved: {csv_path}')
    return df


# ===========================================================================
# Study 3 — PEC vs. Unit Cell Size a  (fixed mass flux)
# ===========================================================================
def pec_vs_cell_size(output_dir='results_sensitivity'):
    """
    Study 3: PEC vs. unit cell size a at a fixed mass flux G.

    a sweep:   2 → 10 mm (30 points)
    Fixed:     ε = 0.65, G = 1.0 kg/(m²·s), μ = 1e-5 Pa·s, Pr = 0.70

    Physical interpretation:
        Dh(a) = 4·ε·a / (2π)          [hydraulic diameter scales with a]
        Re(a) = G · Dh(a) / μ          [Re increases linearly with a]
    Larger cells → larger Dh → higher Re → different flow regime.

    Outputs: pec_vs_cell_size.png, pec_vs_cell_size.csv
    """
    a_range = np.linspace(2e-3, 10e-3, 30)          # 2 → 10 mm
    G = 1.0                                           # kg/(m²·s) mass flux
    Dh_arr = 4.0 * EPS_REF * a_range / (2.0 * np.pi)
    Re_arr = G * Dh_arr / MU

    print(f'  Cell-size sweep: a = {a_range[0]*1e3:.1f}–{a_range[-1]*1e3:.1f} mm  '
          f'→  Re = {Re_arr[0]:.0f}–{Re_arr[-1]:.0f}')

    # Compute j,f at each Re value
    Nu_d_all = {t: np.zeros(len(a_range)) for t in ALL_TYPES}
    f_d_all  = {t: np.zeros(len(a_range)) for t in ALL_TYPES}
    j_d_all  = {t: np.zeros(len(a_range)) for t in ALL_TYPES}

    for i, Re in enumerate(Re_arr):
        Nu_d, f_d, j_d = _compute_jf(np.array([Re]))
        for tpms in ALL_TYPES:
            Nu_d_all[tpms][i] = float(Nu_d[tpms][0])
            f_d_all[tpms][i]  = float(f_d[tpms][0])
            j_d_all[tpms][i]  = float(j_d[tpms][0])

    j0 = j_d_all[BASELINE]
    f0 = f_d_all[BASELINE]

    # ---- figure -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for tpms in TPMS_ALL:
        pec = _pec(j_d_all[tpms], f_d_all[tpms], j0, f0)
        mask = np.isfinite(pec)
        ax.plot(a_range[mask] * 1e3, pec[mask],
                color=COLORS[tpms], lw=LWIDTH[tpms], label=tpms)

    ax.axhline(1.0, color='black', lw=1.8, ls='--', label='Baseline (PEC = 1)')

    ax.set_xlabel('Unit cell size  a  [mm]')
    ax.set_ylabel('PEC [–]')
    ax.set_title('PEC vs. Unit Cell Size  (G = 1.0 kg m⁻² s⁻¹,  ε = 0.65)')
    ax.legend(loc='upper left', framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Secondary x-axis: Re
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # Place Re tick marks at the same positions as primary ticks
    primary_ticks = ax.get_xticks()
    primary_ticks = primary_ticks[(primary_ticks >= a_range[0] * 1e3) &
                                   (primary_ticks <= a_range[-1] * 1e3)]
    re_at_ticks = G * (4.0 * EPS_REF * primary_ticks * 1e-3 / (2.0 * np.pi)) / MU
    ax2.set_xticks(primary_ticks)
    ax2.set_xticklabels([f'{r:.0f}' for r in re_at_ticks])
    ax2.set_xlabel('Re [–]', labelpad=8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'pec_vs_cell_size.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {save_path}')

    # ---- CSV ----------------------------------------------------------
    csv_rows = {
        'a_mm':  a_range * 1e3,
        'Dh_mm': Dh_arr * 1e3,
        'Re':    Re_arr,
    }
    for tpms in TPMS_ALL:
        csv_rows[f'PEC_{tpms}'] = _pec(j_d_all[tpms], f_d_all[tpms], j0, f0)
        csv_rows[f'j_{tpms}']   = j_d_all[tpms]
        csv_rows[f'f_{tpms}']   = f_d_all[tpms]
    csv_rows[f'j_{BASELINE}'] = j_d_all[BASELINE]
    csv_rows[f'f_{BASELINE}'] = f_d_all[BASELINE]

    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(output_dir, 'pec_vs_cell_size.csv')
    df.to_csv(csv_path, index=False, float_format='%.6e')
    print(f'[OK] Saved: {csv_path}')
    return df


# ===========================================================================
# Study 4 — Colburn j – Friction f Parametric Map
# ===========================================================================
def colburn_friction_map(output_dir='results_sensitivity'):
    """
    Study 4: Parametric map in (f, j) space with Re as the parameter.

    Each TPMS traces a curve; structures above-and-left of the baseline
    offer better thermal performance for the same friction penalty.

    Re sweep:  100 → 10,000 (80 log-spaced points)
    Outputs:   colburn_friction_map.png
    """
    Re_arr = np.logspace(2, 4, 80)
    Nu_d, f_d, j_d = _compute_jf(Re_arr)

    fig, ax = plt.subplots(figsize=(8, 7))

    for tpms in ALL_TYPES:
        mask = np.isfinite(j_d[tpms]) & np.isfinite(f_d[tpms])
        f_plt = f_d[tpms][mask]
        j_plt = j_d[tpms][mask]

        ax.loglog(f_plt, j_plt,
                  color=COLORS[tpms], lw=LWIDTH[tpms], ls=LSTYLE[tpms],
                  label=tpms)

        # Annotate at the midpoint of the curve
        if len(f_plt) > 4:
            mid = len(f_plt) // 2
            ax.annotate(
                tpms,
                xy=(f_plt[mid], j_plt[mid]),
                xytext=(6, 4), textcoords='offset points',
                fontsize=8, color=COLORS[tpms],
                arrowprops=dict(arrowstyle='-', color=COLORS[tpms], lw=0.6),
            )

        # Arrow showing Re direction (low Re → high Re = right direction)
        if len(f_plt) > 10:
            i1, i2 = len(f_plt) // 3, len(f_plt) // 3 + 2
            ax.annotate('',
                xy=(f_plt[i2], j_plt[i2]),
                xytext=(f_plt[i1], j_plt[i1]),
                arrowprops=dict(arrowstyle='->', color=COLORS[tpms], lw=1.0)
            )

    ax.set_xlabel('f  (Fanning friction factor) [–]')
    ax.set_ylabel('j  (Colburn j-factor) [–]')
    ax.set_title('Colburn–Friction Performance Map\n'
                 '(arrows indicate increasing Re direction)')
    ax.legend(loc='upper left', framealpha=0.85)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'colburn_friction_map.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {save_path}')


# ===========================================================================
# Summary Table — PEC at reference geometry, three Re values
# ===========================================================================
def summary_table():
    """
    Print a formatted table: PEC at Re = 500, 2000, 5000 for all TPMS types.
    Reference geometry: ε = 0.65, a = 5 mm (DH_REF).
    """
    RE_VALS = [500, 2000, 5000]

    print()
    print('=' * 72)
    print('Summary: PEC vs. SmoothPlateFin Baseline  (ε=0.65, a=5mm, Gas Pr=0.70)')
    print('=' * 72)
    hdr = f"{'TPMS Type':<15}" + ''.join(f"{'PEC@Re='+str(r):>15}" for r in RE_VALS)
    print(hdr)
    print('-' * 72)

    for tpms in TPMS_ALL + [BASELINE]:
        row = f'{tpms:<15}'
        for re_val in RE_VALS:
            Nu_d, f_d, j_d = _compute_jf(np.array([re_val]))
            j0 = float(j_d[BASELINE][0])
            f0 = float(f_d[BASELINE][0])
            if tpms == BASELINE:
                pec_val = 1.0
            else:
                pec_val = float(_pec(float(j_d[tpms][0]), float(f_d[tpms][0]), j0, f0))
            if np.isfinite(pec_val):
                row += f'{"%.3f" % pec_val:>15}'
            else:
                row += f'{"N/A":>15}'
        print(row)

    print('=' * 72)
    print()


# ===========================================================================
# Main
# ===========================================================================
def main():
    out = 'results_sensitivity'
    os.makedirs(out, exist_ok=True)

    print('=' * 65)
    print('TPMS Sensitivity Analysis vs. Smooth Plate-Fin Baseline')
    print('=' * 65)

    print('\n--- Study 1: PEC, j, f  vs. Reynolds Number ---')
    pec_vs_re(out)

    print('\n--- Study 2: PEC vs. Porosity ---')
    pec_vs_porosity(out)

    print('\n--- Study 3: PEC vs. Unit Cell Size ---')
    pec_vs_cell_size(out)

    print('\n--- Study 4: Colburn–Friction Parametric Map ---')
    colburn_friction_map(out)

    print('\n--- Summary Table ---')
    summary_table()

    print(f'\nAll outputs saved to: {os.path.abspath(out)}/')
    print('=' * 65)


if __name__ == '__main__':
    main()
