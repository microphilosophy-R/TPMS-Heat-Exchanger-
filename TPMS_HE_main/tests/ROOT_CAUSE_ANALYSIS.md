## Final Analysis: 6K Error Root Cause

### What We Validated
✅ Dixon model: Working correctly with Biot correction
✅ j-factor correlation: Matches MATLAB exactly
✅ Geometry-dependent correlation: Implemented but gives worse results

### Root Cause of 6K Error

**MATLAB approach:**
```matlab
U = 1/(1/(etas_hf*faih*hi) + 1/(etas_cf*faic*ho) + thick_p/Kwall)
A = L_set * width_HE  // Base area only
Q = U * deltaT * A
```

**Python approach:**
```python
G_hot = h_hot * A_elem_h  // A_elem = base * SAD
G_cold = h_cold * A_elem_c
UA = 1/(1/G_hot + 1/G_wall + 1/G_cold)
Q = UA * deltaT
```

### The Issue
- MATLAB: `h * SAD * eta` applied to base area
- Python: `h * eta` applied to `area * SAD`

For packed channels, Python correctly uses base area.
For bare channels, Python uses `A_elem` (area×SAD), but fin efficiency already reduces h.

This creates: `(h*eta) * (A*SAD)` instead of `(h*SAD*eta) * A`

### Solution
Bare PlateFin channels should use **base area** like packed channels, with SAD applied as HTC multiplier.

**Required change:** Line 560-561 in calculator.py
```python
# Current
A_ref_c = self.Abase_elem_c if _cold_packed else self.A_elem_c

# Should be (for PlateFin)
_cold_platefin = (self.streams['cold']['tpms'] == 'PlateFin')
A_ref_c = self.Abase_elem_c if (_cold_packed or _cold_platefin) else self.A_elem_c
```

This would make bare PlateFin use base area, with SAD embedded in the HTC through fin efficiency.
