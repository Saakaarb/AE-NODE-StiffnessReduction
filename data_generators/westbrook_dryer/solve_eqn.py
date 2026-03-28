# wd_sweep_save.py
import os
import numpy as np
from scipy.integrate import solve_ivp

# -------------------- constants (SI) --------------------
R = 8.314462618  # J/mol/K
MW = {
    "CH4": 0.01604,
    "O2":  0.03200,
    "CO":  0.02801,
    "CO2": 0.04401,
    "INERT": 0.02897,  # air-ish remainder
}

# Westbrook–Dryer global 2-step kinetics (arrhenius in T, power-law in concentrations)
# Step 1: CH4 + 1.5 O2 -> CO + 2 H2O
A1 = 1.3e8
E1 = 2.43e4 * 4.184  # cal/mol -> J/mol
a1, b1 = 0.3, 1.3

# Step 2: CO + 0.5 O2 -> CO2
A2 = 2.2e7
E2 = 2.0e4 * 4.184
a2, b2 = 1.0, 0.5

# Simple thermochemistry (constant Cp)
Q1 = 5.2e5   # J/mol released by step 1 (CH4 -> CO + 2 H2O)
Q2 = 2.83e5  # J/mol released by step 2 (CO -> CO2)
Cp_mix_const = 1200.0  # J/(kg·K)

# -------------------- mixture & RHS --------------------
def _mixture_props(Y_CH4, Y_O2, Y_CO, Y_CO2, T, p_Pa):
    """Return mixture MW (kg/mol) and density (kg/m^3) including inert remainder."""
    Y_sum = Y_CH4 + Y_O2 + Y_CO + Y_CO2
    Y_inert = max(0.0, 1.0 - Y_sum)
    denom = (
        (Y_CH4 / MW["CH4"]) +
        (Y_O2  / MW["O2"])  +
        (Y_CO  / MW["CO"])  +
        (Y_CO2 / MW["CO2"]) +
        (Y_inert / MW["INERT"])
    )
    if denom <= 0.0:
        denom = 1e-30
    Wmix = 1.0 / denom                 # kg/mol
    rho  = p_Pa * Wmix / (R * max(T, 250.0))  # kg/m^3
    return Wmix, rho

def westbrook_dryer_rhs(t, y, p_Pa=101325.0):
    """
    y = [Y_CH4, Y_O2, Y_CO, Y_CO2, T]
    Constant-pressure homogeneous reactor. Inert remainder implicit.
    """
    Y_CH4, Y_O2, Y_CO, Y_CO2, T = y
    T_eff = max(T, 250.0)

    # Mixture properties
    _, rho = _mixture_props(Y_CH4, Y_O2, Y_CO, Y_CO2, T_eff, p_Pa)

    # Floors inside kinetics only (avoid fractional power of negatives/tiny)
    eps = 1e-20
    C_CH4 = rho * max(Y_CH4, eps) / MW["CH4"]  # mol/m^3
    C_O2  = rho * max(Y_O2,  eps) / MW["O2"]
    C_CO  = rho * max(Y_CO,  eps) / MW["CO"]

    # Reaction rates (mol/m^3/s)
    k1 = A1 * np.exp(-E1 / (R * T_eff))
    k2 = A2 * np.exp(-E2 / (R * T_eff))
    r1 = k1 * (C_CH4 ** a1) * (C_O2 ** b1)
    r2 = k2 * (C_CO  ** a2) * (C_O2 ** b2)

    # Species ODEs (mass fractions)
    dY_CH4 = (MW["CH4"] / rho) * (-1.0 * r1)
    dY_O2  = (MW["O2"]  / rho) * (-1.5 * r1 - 0.5 * r2)
    dY_CO  = (MW["CO"]  / rho) * (+1.0 * r1 - 1.0 * r2)
    dY_CO2 = (MW["CO2"] / rho) * (+1.0 * r2)

    # Enforce mass conservation - prevent negative rates when species is depleted
    # More robust bounds checking to prevent negative mass fractions
    if Y_CH4 <= 1e-8:  # If CH4 is nearly depleted
        dY_CH4 = 0.0  # Stop consuming CH4 completely
    elif dY_CH4 < 0 and abs(dY_CH4) > Y_CH4:  # If consumption rate exceeds available
        dY_CH4 = -Y_CH4  # Limit rate to available amount
    
    if Y_O2 <= 1e-8:  # If O2 is nearly depleted
        dY_O2 = 0.0  # Stop consuming O2 completely
    elif dY_O2 < 0 and abs(dY_O2) > Y_O2:  # If consumption rate exceeds available
        dY_O2 = -Y_O2  # Limit rate to available amount
    
    if Y_CO <= 1e-8:  # If CO is nearly depleted
        dY_CO = max(dY_CO, 0.0)  # Don't consume more than exists
    elif dY_CO < 0 and abs(dY_CO) > Y_CO:  # If consumption rate exceeds available
        dY_CO = -Y_CO  # Limit rate to prevent going negative

    # Temperature ODE
    dT = (Q1 * r1 + Q2 * r2) / (rho * Cp_mix_const)

    # Final bounds enforcement - ensure derivatives don't lead to negative values
    # This prevents the solver from taking steps that would make mass fractions negative
    if Y_CH4 + dY_CH4 * 1e-6 < 0:  # If next step would make CH4 negative
        dY_CH4 = -Y_CH4 / 1e-6  # Limit rate to prevent going negative
    
    if Y_O2 + dY_O2 * 1e-6 < 0:  # If next step would make O2 negative
        dY_O2 = -Y_O2 / 1e-6  # Limit rate to prevent going negative
    
    if Y_CO + dY_CO * 1e-6 < 0:  # If next step would make CO negative
        dY_CO = -Y_CO / 1e-6  # Limit rate to prevent going negative

    return [dY_CH4, dY_O2, dY_CO, dY_CO2, dT]

# -------------------- integration utilities --------------------
def integrate_case(y0, t_end=0.001, p_Pa=101325.0):
    t_span = (0.0, t_end)
    atol = (1e-5, 1e-5, 1e-5, 1e-5, 1e-2)  # Less strict tolerances for stability
    sol = solve_ivp(
        lambda t, y: westbrook_dryer_rhs(t, y, p_Pa),
        t_span, y0, method="Radau",
        rtol=1e-3, atol=atol, 
        max_step=1e-5,      # Limit maximum step size for stability
        first_step=1e-7,    # Start with very small step
        dense_output=True    # Enable dense output for better interpolation
    )
    return sol

def simulate_grid(
    ch4_values=np.linspace(0.01, 0.10, 10),
    T0_values=np.linspace(900.0, 1500.0, 7),
    Y_O2=0.21,
    p_Pa=101325.0,
    t_end=0.001
):
    """
    Row-major sweep: for each Y_CH4 in ch4_values, iterate over all T0 in T0_values.
    Returns list of dicts including the SciPy solution for each grid point.
    """
    results = []
    for Y_CH4 in ch4_values:
        for T0 in T0_values:
            print(Y_CH4,T0)
            if Y_CH4 + Y_O2 > 1.0:
                # skip unphysical compositions
                continue
            y0 = [Y_CH4, Y_O2, 0.0, 0.0, T0]
            sol = integrate_case(y0, t_end=t_end, p_Pa=p_Pa)
            results.append({
                "Y_CH4_0": Y_CH4,
                "Y_O2_0": Y_O2,
                "T0": T0,
                "sol": sol
            })
    return results

# -------------------- main: save one txt per grid point --------------------
if __name__ == "__main__":
    # Grid per your spec
    ch4_values = np.linspace(0.01, 0.10, 10)   # Y_CH4 in [0.01, 0.10]
    T0_values  = np.linspace(700.0, 1100.0, 7) # T0 in [900, 1500] K
    Y_O2_fixed = 0.21
    p_Pa = 101325.0
    t_end = 0.05

    out_dir = "wd_time_histories"
    os.makedirs(out_dir, exist_ok=True)

    results = simulate_grid(
        ch4_values=ch4_values,
        T0_values=T0_values,
        Y_O2=Y_O2_fixed,
        p_Pa=p_Pa,
        t_end=t_end
    )

    # Save files in row-major order; each file has raw numeric columns only
    for idx, r in enumerate(results):
        sol = r["sol"]
        # Skip if solver failed catastrophically (shouldn't with guards)
        if sol.t.size == 0 or sol.y.size == 0:
            continue

        # Check if we got the full time span
        if sol.t[-1] < t_end * 0.95:  # Allow 5% tolerance
            print(f"Warning: Case {idx} stopped early at t={sol.t[-1]:.6f}s (target: {t_end}s)")
            print(f"  Y_CH4_0={r['Y_CH4_0']:.3f}, T0={r['T0']:.0f}K")
            print(f"  Final state: CH4={sol.y[0][-1]:.6f}, O2={sol.y[1][-1]:.6f}, CO={sol.y[2][-1]:.6f}, CO2={sol.y[3][-1]:.6f}, T={sol.y[4][-1]:.1f}K")

        # Columns: t, Y_CH4, Y_O2, Y_CO, Y_CO2, T  (no headers)
        data = np.column_stack([sol.t, sol.y.T])

        # filename includes row-major index and parameters (sanitized)
        ych4 = r["Y_CH4_0"]
        T0 = r["T0"]
        fname = os.path.join(
            out_dir,
            f"run_{idx:04d}_YCH4_{ych4:.3f}_T0_{int(round(T0))}.txt"
        )

        # Save with default numeric formatting (no headers, no comments)
        np.savetxt(fname, data)
