from pathlib import Path
import numpy as np
import tidy3d as td

from gratingCoupler.nested_sampler import GratingCouplerBayesianOptimizer

# Define Tidy3D Materials
SiN = td.Medium(permittivity=2.0**2)    # Core medium (~SiN n=2.0)
SiO2 = td.Medium(permittivity=1.45**2)  # BOX/Oxide medium (~SiO2 n=1.45)
Air = td.Medium(permittivity=1.0**2)   # Top Cladding medium (Air n=1.0)

# Package Fixed Physical & Domain Parameters
# These map directly to the GratingConfig dataclass inside the sampler
fixed_params = {
    "lda0": 1.550,            # Central excitation wavelength (μm)
    "theta_deg": -12.0,        # Fiber angle off-vertical (degrees)
    "mfd": 10.4,               # Mode Field Diameter of fiber source (μm)
    "source_gap": 2.0,         # Gap above cladding for source plane (μm)
    
    # Layer Heights
    "core_height": 0.800,      # Silicon Nitride core thickness (μm)
    "cladding_height": 2.0,    # Cladding height buffer (μm)
    "box_height": 4.0,         # Buried Oxide (BOX) thickness (μm)
    "substrate_height": 8.0,   # Silicon handle substrate thickness (μm)
    
    # 3D Domain & Numerical Resolution
    "w_grating": 10.4,         # Transverse width for 3D runs (μm)
    "buffer": 3.0,             # Spatial buffer margin around domain (μm)
    "pol_angle": np.pi / 2.0,  # TE polarization (90 degrees)
    "fine_steps": 15,          # Grid steps per wavelength in fine mesh region
    "coarse_steps": 10         # Grid steps per wavelength in background mesh
}

# Package Material Media Dictionary
materials = {
    "core_medium": SiN,
    "box_medium": SiO2,
    "sub_medium": None,        # Terminated at BOX layer (set Si medium to include handle)
    "cladding_medium": Air     # Cladding medium (Air or SiO2)
}

# Define Tight Prior Search Envelopes Around Design Point
# Constraining the prior space guarantees high likelihood values early in the run
tight_param_bounds = {
    "period": [0.800, 0.840],       # Centered near Lambda = 0.8201 μm
    "fill_factor": [0.480, 0.550],  # Centered near Duty Cycle = 0.5136
    "etch_depth": [0.480, 0.550],   # Centered near Etch Depth = 0.5136 μm
    "num_periods": [18.0, 26.0],    # Centered near N = 22 teeth
    "source_x": [3.500, 4.200]      # Centered near Fiber X = 3.8419 μm
}

# Output directory for chains, budget ledger, and leaderboard
demo_2_out = Path("example_results/bayesian_demo_budgeted")

# Initialize the Budget-Aware Optimizer Engine
optimizer_budgeted = GratingCouplerBayesianOptimizer(
    fixed_params=fixed_params,
    materials=materials,
    output_dir=demo_2_out,
    param_bounds=tight_param_bounds,
    sigma=0.02,
    is_3d=False,
    credit_limit=0.5   # Pre-set hard credit cap at initialization (optional)
)

# Execute Budget-Capped Nested Sampling
# Intercepts executions before cloud submission if (spent + next_run_cost) > credit_limit
results_budgeted = optimizer_budgeted.execute_search_with_budget(
    credit_limit=0.5,         # Hard cap: Max FlexCredits allowed for this entire run session
    n_live_points=25,          # Number of active MultiNest live points
    evidence_tolerance=0.5     # Convergence tolerance (if budget isn't exhausted first)
)

# Check run termination status
if results_budgeted.get("status") == "PAUSED_BUDGET_EXCEEDED":
    print(f"\n[Run Paused] Spent {results_budgeted['total_credits_used']:.4f} / {results_budgeted['credit_limit']:.4f} credits.")
    print("Pass a higher credit_limit to this directory to resume MultiNest seamlessly.")
else:
    print(f"\n[Run Converged] Max Efficiency: {results_budgeted['max_efficiency']*100:.2f}%")

# Print the Top 10 configurations collected before hitting the budget cap
optimizer_budgeted.report_top_configurations()