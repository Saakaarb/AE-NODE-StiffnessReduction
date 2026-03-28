# mild_four_species.py
# 
# This script generates solutions for a mild combustion system with 4 species.
# Features:
# - Generate multiple solutions with random initial conditions
# - Optional trajectory splitting: split each trajectory into k parts
# - Command line interface for configuration
# - Automatic handling of trajectory length divisibility
#
# Usage examples:
#   python solve_eqn.py                    # Generate 100 solutions (default)
#   python solve_eqn.py --N 50            # Generate 50 solutions
#   python solve_eqn.py --split --k 2     # Split each trajectory into 2 parts
#   python solve_eqn.py --split --k 4     # Split each trajectory into 4 parts
#
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import random

# ---- constants (SI) ----
R = 8.314462618  # J/mol/K
MW = {
    "F":   0.01604,   # lumped CH4-like fuel (kg/mol)
    "O2":  0.03200,
    "CO":  0.02801,
    "CO2": 0.04401,
    "INERT": 0.02897, # air-ish
}

# ---- "mild" kinetics: less stiff than WD ----
# Step 1: F + 1.5 O2 -> CO + 2 H2O   (integer orders)
A1 = 1.0e6      # 1/s (effective, with conc units absorbed)
E1 = 6.0e4      # J/mol
# Step 2: CO + 0.5 O2 -> CO2
A2 = 5.0e5      # 1/s
E2 = 4.0e4      # J/mol

# ---- thermochemistry (smaller heats => gentler T coupling) ----
Q1 = 3.0e5      # J/mol released by step 1
Q2 = 2.0e5      # J/mol released by step 2
Cp_mix_const = 1500.0  # J/(kg·K), constant Cp for robustness

def _mixture_props(YF, YO2, YCO, YCO2, T, p_Pa):
    """Mixture MW and density at constant pressure with inert remainder."""
    Y_sum = YF + YO2 + YCO + YCO2
    Y_inert = max(0.0, 1.0 - Y_sum)
    denom = (
        (YF    / MW["F"])   +
        (YO2   / MW["O2"])  +
        (YCO   / MW["CO"])  +
        (YCO2  / MW["CO2"]) +
        (Y_inert / MW["INERT"])
    )
    if denom <= 0.0:
        denom = 1e-30
    Wmix = 1.0 / denom
    rho  = p_Pa * Wmix / (R * max(T, 250.0))
    return Wmix, rho

def rhs_mild(t, y, p_Pa=101325.0):
    """
    y = [YF, YO2, YCO, YCO2, T]
    Constant-pressure, homogeneous reactor. Inert is implicit.
    """
    YF, YO2, YCO, YCO2, T = y
    T_eff = max(T, 250.0)

    # Mixture properties
    _, rho = _mixture_props(YF, YO2, YCO, YCO2, T_eff, p_Pa)

    # Concentrations for integer orders (no fractional powers needed)
    eps = 1e-20
    CF   = rho * max(YF,  eps) / MW["F"]
    CO2g = rho * max(YO2, eps) / MW["O2"]
    CCO  = rho * max(YCO, eps) / MW["CO"]

    # Arrhenius rates (mol/m^3/s); integer reaction orders 1
    k1 = A1 * np.exp(-E1 / (R * T_eff))
    k2 = A2 * np.exp(-E2 / (R * T_eff))
    r1 = k1 * CF   * (CO2g)       # ~ [F]^1 [O2]^1   (we fold 1.5 into stoich below)
    r2 = k2 * CCO  * (CO2g**0.5)  # mimic 0.5 O2 via sqrt; still mild

    # Species mass-fraction ODEs
    dYF   = (MW["F"]   / rho) * (-1.0 * r1)
    dYO2  = (MW["O2"]  / rho) * (-1.5 * r1 - 0.5 * r2)
    dYCO  = (MW["CO"]  / rho) * (+1.0 * r1 - 1.0 * r2)
    dYCO2 = (MW["CO2"] / rho) * (+1.0 * r2)

    # Temperature ODE (gentle coupling)
    dT = (Q1 * r1 + Q2 * r2) / (rho * Cp_mix_const)

    return [dYF, dYO2, dYCO, dYCO2, dT]

def integrate_mild(
    y0, t_span=(0.0, 0.005), p_Pa=101325.0,
    rtol=1e-7, atol=(1e-10, 1e-10, 1e-10, 1e-10, 1e-6), method="Radau"
):
    t_eval = np.linspace(t_span[0], t_span[1], 600)
    sol = solve_ivp(
        lambda t, y: rhs_mild(t, y, p_Pa=p_Pa),
        t_span, y0, method=method, t_eval=t_eval,
        rtol=rtol, atol=atol
    )
    # clip tiny negatives (numerical)
    if sol.y.size:
        for i in range(4):
            sol.y[i] = np.clip(sol.y[i], 0.0, 1.0)
    return sol

def save_sol(sol, filename):
    
    combined_sol = np.concatenate((np.expand_dims(sol.t,axis=0), sol.y),axis=0)
    
    np.savetxt(filename, combined_sol, delimiter=',')

def generate_random_initial_conditions(N=100, seed=42):
    """
    Generate N random initial conditions
    YF: 0.03 to 0.1
    T: 1100K to 1300K
    O2: 0.21 (fixed air composition)
    CO, CO2: 0.0 (no initial products)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    initial_conditions = []
    
    for i in range(N):
        # Random fuel mass fraction (0.03 to 0.1)
        YF = np.random.uniform(0.03, 0.1)
        
        # Fixed O2 mass fraction (air composition)
        YO2 = 0.21
        
        # No initial CO or CO2
        YCO = 0.0
        YCO2 = 0.0
        
        # Random temperature (1100K to 1300K)
        T = np.random.uniform(1100.0, 1300.0)
        
        # Ensure mass fractions don't exceed 1.0
        total_mass = YF + YO2 + YCO + YCO2
        if total_mass > 1.0:
            # Scale down proportionally
            scale_factor = 1.0 / total_mass
            YF *= scale_factor
            YO2 *= scale_factor
        
        initial_conditions.append([YF, YO2, YCO, YCO2, T])
    
    return initial_conditions

def validate_splitting_parameters(solution, k):
    """
    Validate that trajectory splitting parameters are reasonable.
    
    Parameters
    ----------
    solution : scipy.integrate.OdeSolution
        The solution object containing time and state arrays
    k : int
        Number of parts to split the trajectory into
    
    Returns
    -------
    bool
        True if parameters are valid
    """
    if k < 2:
        raise ValueError(f"Split factor k must be at least 2, got {k}")
    
    n_points = len(solution.t)
    points_per_part = n_points // k
    
    if points_per_part < 2:
        raise ValueError(f"Cannot split trajectory of length {n_points} into {k} parts (need at least 2 points per part)")
    
    return True

def split_trajectory(solution, k):
    """
    Split a single trajectory into k parts.
    
    Parameters
    ----------
    solution : scipy.integrate.OdeSolution
        The solution object containing time and state arrays
    k : int
        Number of parts to split the trajectory into
    
    Returns
    -------
    list of tuples
        List of (t_part, y_part) tuples, each containing a portion of the trajectory
    """
    # Validate parameters first
    validate_splitting_parameters(solution, k)
    
    t = solution.t
    y = solution.y
    
    n_points = len(t)
    points_per_part = n_points // k
    
    parts = []
    
    for i in range(k):
        start_idx = i * points_per_part
        if i == k - 1:  # Last part gets any remaining points
            end_idx = n_points
        else:
            end_idx = start_idx + points_per_part
            
        t_part = t[start_idx:end_idx]
        y_part = y[:, start_idx:end_idx]
        
        parts.append((t_part, y_part))
    
    return parts

def save_split_solution(t_part, y_part, filename):
    """
    Save a split trajectory part to a file.
    
    Parameters
    ----------
    t_part : array
        Time array for this part
    y_part : array
        State array for this part
    filename : str
        Output filename
    """
    combined_sol = np.concatenate((np.expand_dims(t_part, axis=0), y_part), axis=0)
    np.savetxt(filename, combined_sol, delimiter=',')

def generate_multiple_solutions(N=100, output_dir="mild_combustion_solutions", split_trajectories=False, k=2):
    """
    Generate N solutions with random initial conditions and save to files.
    Optionally split each trajectory into k parts.
    
    Parameters
    ----------
    N : int
        Number of solutions to generate
    output_dir : str
        Output directory for solution files
    split_trajectories : bool
        If True, split each trajectory into k parts
    k : int
        Number of parts to split each trajectory into (only used if split_trajectories=True)
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Generate random initial conditions
    initial_conditions = generate_random_initial_conditions(N)
    
    if split_trajectories:
        print(f"Generating {N} solutions and splitting each into {k} parts...")
        print(f"Total output trajectories: {N * k}")
    else:
        print(f"Generating {N} solutions...")
    
    successful_solutions = 0
    failed_solutions = 0
    total_trajectories = 0
    
    for i, y0 in enumerate(initial_conditions):
        try:
            print(f"Solving case {i+1}/{N}: YF={y0[0]:.3f}, T={y0[4]:.0f}K")
            
            # Integrate the system
            solution = integrate_mild(y0)
            
            # Check if integration was successful
            if solution.success:
                if split_trajectories:
                    # Split the trajectory into k parts
                    try:
                        trajectory_parts = split_trajectory(solution, k)
                        
                        for j, (t_part, y_part) in enumerate(trajectory_parts):
                            # Create a solution-like object for the part
                            part_solution = type('Solution', (), {
                                't': t_part,
                                'y': y_part,
                                'success': True
                            })()
                            
                            # Save each part with a unique identifier
                            filename = os.path.join(output_dir, f"solution_{i+1:04d}_part_{j+1:02d}.txt")
                            save_split_solution(t_part, y_part, filename)
                            total_trajectories += 1
                            print(f"  ✓ Part {j+1}/{k} - saved to {filename}")
                        
                        successful_solutions += 1
                        print(f"  ✓ Success - split into {k} parts")
                        
                    except ValueError as e:
                        print(f"  ✗ Trajectory splitting failed: {str(e)}")
                        # Fall back to saving the full trajectory
                        filename = os.path.join(output_dir, f"solution_{i+1:04d}.txt")
                        save_sol(solution, filename)
                        total_trajectories += 1
                        successful_solutions += 1
                        print(f"  ✓ Success - saved full trajectory to {filename}")
                        
                else:
                    # Save solution to file (original behavior)
                    filename = os.path.join(output_dir, f"solution_{i+1:04d}.txt")
                    save_sol(solution, filename)
                    total_trajectories += 1
                    successful_solutions += 1
                    print(f"  ✓ Success - saved to {filename}")
                    
            else:
                print(f"  ✗ Integration failed: {solution.message}")
                failed_solutions += 1
                
        except Exception as e:
            print(f"  ✗ Error in case {i+1}: {str(e)}")
            failed_solutions += 1
    
    print(f"\nGeneration complete!")
    print(f"Successful: {successful_solutions}, Failed: {failed_solutions}")
    if split_trajectories:
        print(f"Total trajectories generated: {total_trajectories} (from {successful_solutions} solutions split into {k} parts each)")
    else:
        print(f"Total trajectories generated: {total_trajectories}")
    print(f"Files saved in: {output_dir}")
    
    return successful_solutions, failed_solutions, total_trajectories

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate mild combustion solutions with optional trajectory splitting')
    parser.add_argument('--N', type=int, default=100, help='Number of solutions to generate (default: 100)')
    parser.add_argument('--output-dir', type=str, default='mild_combustion_solutions', 
                       help='Output directory for solution files (default: mild_combustion_solutions)')
    parser.add_argument('--split', action='store_true', 
                       help='Split each trajectory into k parts')
    parser.add_argument('--k', type=int, default=2, 
                       help='Number of parts to split each trajectory into (default: 2, only used with --split)')
    parser.add_argument('--no-split', action='store_true', 
                       help='Explicitly disable trajectory splitting (default behavior)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.N < 1:
        parser.error("--N must be at least 1")
    
    if args.k < 2:
        parser.error("--k must be at least 2")
    
    # Determine if splitting should be enabled
    split_trajectories = args.split
    
    # Generate multiple solutions
    print(f"Configuration:")
    print(f"  Number of solutions: {args.N}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Trajectory splitting: {'Enabled' if split_trajectories else 'Disabled'}")
    if split_trajectories:
        print(f"  Split factor: {args.k}")
        print(f"  Expected total trajectories: {args.N * args.k}")
        print(f"  File naming pattern: solution_XXXX_part_YY.txt")
        print(f"  Example: solution_0001_part_01.txt, solution_0001_part_02.txt, ...")
    print()
    
    successful, failed, total_trajectories = generate_multiple_solutions(
        N=args.N, 
        output_dir=args.output_dir, 
        split_trajectories=split_trajectories, 
        k=args.k
    )
    
    # Also run one example case for verification
    print("\nRunning example case for verification...")
    y0 = [0.05, 0.21, 0.0, 0.0, 1200.0]  # YF, O2, CO, CO2, T
    sol = integrate_mild(y0, t_span=(0.0, 0.005))
    print("Status:", sol.message)
    print("Final state:", [float(sol.y[i, -1]) for i in range(sol.y.shape[0])])
    
    # Demonstrate trajectory splitting if enabled
    if split_trajectories:
        print(f"\nDemonstrating trajectory splitting (k={args.k}):")
        print(f"Original trajectory length: {len(sol.t)} points")
        print(f"Expected points per part: {len(sol.t) // args.k}")
        
        try:
            trajectory_parts = split_trajectory(sol, args.k)
            print(f"Successfully split into {len(trajectory_parts)} parts:")
            for i, (t_part, y_part) in enumerate(trajectory_parts):
                print(f"  Part {i+1}: {len(t_part)} time points, {y_part.shape[1]} state points")
        except ValueError as e:
            print(f"Trajectory splitting demonstration failed: {str(e)}")
    
    # Plot the verification case
    if sol.success:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Species evolution
        plt.subplot(2, 3, 1)
        plt.plot(sol.t, sol.y[0], 'b-', linewidth=2, label='Fuel (YF)')
        plt.plot(sol.t, sol.y[1], 'r-', linewidth=2, label='O2')
        plt.plot(sol.t, sol.y[2], 'g-', linewidth=2, label='CO')
        plt.plot(sol.t, sol.y[3], 'm-', linewidth=2, label='CO2')
        plt.xlabel('Time (s)')
        plt.ylabel('Mass Fraction')
        plt.title('Species Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # Plot 2: Temperature evolution
        plt.subplot(2, 3, 2)
        plt.plot(sol.t, sol.y[4], 'orange', linewidth=2, label='Temperature')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature Evolution')
        plt.legend()
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Fuel consumption
        plt.subplot(2, 3, 3)
        plt.plot(sol.t, sol.y[0], 'b-', linewidth=2, label='Fuel (YF)')
        plt.xlabel('Time (s)')
        plt.ylabel('Fuel Mass Fraction')
        plt.title('Fuel Consumption')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # Plot 4: Oxygen consumption
        plt.subplot(2, 3, 4)
        plt.plot(sol.t, sol.y[1], 'r-', linewidth=2, label='O2')
        plt.xlabel('Time (s)')
        plt.ylabel('O2 Mass Fraction')
        plt.title('Oxygen Consumption')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # Plot 5: CO production and consumption
        plt.subplot(2, 3, 5)
        plt.plot(sol.t, sol.y[2], 'g-', linewidth=2, label='CO')
        plt.xlabel('Time (s)')
        plt.ylabel('CO Mass Fraction')
        plt.title('CO Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # Plot 6: CO2 production
        plt.subplot(2, 3, 6)
        plt.plot(sol.t, sol.y[3], 'm-', linewidth=2, label='CO2')
        plt.xlabel('Time (s)')
        plt.ylabel('CO2 Mass Fraction')
        plt.title('CO2 Production')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = "mild_combustion_verification.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Verification plot saved to: {plot_filename}")
        
        # Display the plot
        plt.show()
        
        # Print summary statistics
        print(f"\nVerification Case Summary:")
        print(f"Initial conditions: YF={y0[0]:.3f}, O2={y0[1]:.3f}, T={y0[4]:.0f}K")
        print(f"Integration time: {sol.t[-1]:.6f}s")
        print(f"Final fuel: {sol.y[0][-1]:.6f}")
        print(f"Final O2: {sol.y[1][-1]:.6f}")
        print(f"Final CO: {sol.y[2][-1]:.6f}")
        print(f"Final CO2: {sol.y[3][-1]:.6f}")
        print(f"Final temperature: {sol.y[4][-1]:.1f}K")
        print(f"Temperature rise: {sol.y[4][-1] - y0[4]:.1f}K")
    else:
        print("Verification case failed - cannot plot")

