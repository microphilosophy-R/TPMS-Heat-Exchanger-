## Plan to Improve Wang Experiment Accuracy (<2K)

### Current Status
- Average error: Hot=6.22K, Cold=4.60K
- Dixon model: ✓ Working correctly
- Issue: Plate-fin correlation mismatch

### Root Cause Analysis

**MATLAB uses geometry-dependent correlation (line 160):**
```matlab
Nu = (0.233*Re^(-0.48)*(s/h)^0.192*(t/h)^(-0.14)) * Re * Pr^(1/3)
```

**Python uses j-factor correlation:**
```python
ln(j) = -0.0264*ln(Re)^3 + 0.556*ln(Re)^2 - 4.092*ln(Re) + 6.217
Nu = j * Re * Pr^(1/3)
```

### Implementation Plan

#### Option 1: Add Geometry-Dependent Correlation (Recommended)
**File:** `correlations/thermohydraulic_correlations.py`

**Steps:**
1. Modify `_plate_fin_correlations()` to accept geometry parameters
2. Add geometry-dependent Nu calculation
3. Pass fin geometry from solver to correlation

**Changes needed:**
```python
def _plate_fin_correlations(Re, Pr, fluid_type, geometry=None):
    if geometry and 'fin_spacing' in geometry:
        # Geometry-dependent (MATLAB-style)
        s = geometry['fin_spacing']
        h = geometry['fin_height']
        t = geometry['fin_thickness']
        Nu = (0.233 * Re**(-0.48) * (s/h)**0.192 * (t/h)**(-0.14)) * Re * Pr**(1/3)
        f = 0.029 * Re**(-0.09) * (s/h)**(-0.169) * (t/h)**0.034
    else:
        # j-factor (default)
        ln_j = -0.0264*ln(Re)**3 + 0.556*ln(Re)**2 - 4.092*ln(Re) + 6.217
        Nu = exp(ln_j) * Re * Pr**(1/3)
        f = 2.5 * exp(ln_j)
    return Nu, f
```

**Estimated impact:** Reduce error to <2K

---

#### Option 2: Tune Surface Area Density Per Case
**Pros:** Quick fix
**Cons:** Not physically accurate, case-dependent

---

#### Option 3: Fix Area Calculation Method
**Issue:** Python multiplies area by SAD, MATLAB multiplies HTC by SAD

**Changes needed:**
- Modify `calculator.py` line 560-561 to use base area for bare PlateFin
- Apply SAD as HTC multiplier instead of area multiplier

---

### Recommended Approach

**Phase 1: Implement geometry-dependent correlation**
- Modify `_plate_fin_correlations()`
- Update `get_correlations()` to pass geometry
- Test with Wang cases

**Phase 2: Verify area calculation**
- Ensure bare PlateFin uses base area
- Verify fin efficiency application

**Expected outcome:** <2K error across all 6 Wang cases
