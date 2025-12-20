import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
FIELD_SIZE = 500  # 500x500m
NUM_NODES = 200
COMM_RANGE = 100  # Max range for TOA measurement
NOISE_STD = 1.5   # Measurement noise in meters

def solve_linear_ls(anchors_pos, ranges):
    """
    Solves for unknown position (x, y) using Linearized Least Squares (Ax = b).
    """
    N = len(anchors_pos)
    if N < 3: return None
    
    x_k, y_k = anchors_pos[0]
    r_k = ranges[0]
    
    A = []
    b = []
    
    for i in range(1, N):
        x_i, y_i = anchors_pos[i]
        r_i = ranges[i]
        
        # Linearization: 2x(x_i - x_k) + 2y(y_i - y_k)
        row_A = [2 * (x_i - x_k), 2 * (y_i - y_k)]
        
        # b value with squared terms
        val_b = (x_i**2 + y_i**2) - (x_k**2 + y_k**2) - r_i**2 + r_k**2
        
        A.append(row_A)
        b.append(val_b)
        
    try:
        # Standard Linear Least Squares
        pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return pos
    except:
        return None

def calculate_theoretical_error(neighbor_pos, noise_std):
    """
    Calculates the Theoretical Error (GDOP) based on the geometry of neighbors.
    Uses TRUE positions to determine the 'Best Case' geometry limit.
    """
    N = len(neighbor_pos)
    if N < 3: return np.nan

    x_k, y_k = neighbor_pos[0]
    A = []
    
    # Build A matrix using TRUE geometry
    for i in range(1, N):
        x_i, y_i = neighbor_pos[i]
        row_A = [2 * (x_i - x_k), 2 * (y_i - y_k)]
        A.append(row_A)
    
    A = np.array(A)
    
    try:
        # Calculate Covariance geometry factor: (A^T * A)^-1
        fisher_info = A.T @ A
        inv_fisher = np.linalg.inv(fisher_info)
        
        # GDOP Factor (Geometric Dilution)
        gdop_factor = np.sqrt(np.trace(inv_fisher))
        
        # Scale by noise approx.
        # In Linear LS for TOA, noise is scaled by 2*Range roughly.
        # We use an average range approx (COMM_RANGE/2) for scaling visualization
        avg_range = COMM_RANGE / 2
        theoretical_err = (2 * avg_range * noise_std) * gdop_factor
        
        return theoretical_err
    except:
        return np.nan # Singular matrix (collinear)

def get_anchors(scenario, true_positions):
    """Selects anchor indices based on strategy."""
    indices = np.arange(NUM_NODES)
    center = np.array([FIELD_SIZE/2, FIELD_SIZE/2])
    dists_to_center = np.linalg.norm(true_positions - center, axis=1)
    
    if scenario == "Center":
        return indices[np.argsort(dists_to_center)][:10]
    elif scenario == "Scattered":
        # Center + Edges
        center_anchors = indices[np.argsort(dists_to_center)][:5]
        edge_anchors = []
        for i in range(NUM_NODES):
            x, y = true_positions[i]
            if (x < 100 or x > 400) and (y < 100 or y > 400):
                edge_anchors.append(i)
        combined = np.concatenate((center_anchors, edge_anchors[:8]))
        return np.unique(combined)

def run_simulation_logic(scenario, ax_map, ax_drift):
    print(f"--- Simulating: {scenario} ---")
    
    # 1. Setup
    np.random.seed(45) # Different seed for variety
    true_positions = np.random.rand(NUM_NODES, 2) * FIELD_SIZE
    anchor_indices = get_anchors(scenario, true_positions)
    
    est_positions = np.zeros((NUM_NODES, 2))
    is_discovered = np.zeros(NUM_NODES, dtype=bool)
    theoretical_errors = np.full(NUM_NODES, np.nan)
    
    est_positions[anchor_indices] = true_positions[anchor_indices]
    is_discovered[anchor_indices] = True
    
    # 2. Discovery Loop
    new_discovery = True
    while new_discovery:
        new_discovery = False
        unknown_indices = np.arange(NUM_NODES)[~is_discovered]
        
        for u_idx in unknown_indices:
            dist_sq = np.sum((true_positions[u_idx] - true_positions[is_discovered])**2, axis=1)
            dists = np.sqrt(dist_sq)
            
            valid_mask = dists <= COMM_RANGE
            known_indices = np.arange(NUM_NODES)[is_discovered]
            my_neighbors = known_indices[valid_mask]
            
            if len(my_neighbors) >= 3:
                # A. Simulation Step (Uses Estimated Pos -> Has Drift)
                neighbor_est_pos = est_positions[my_neighbors]
                true_ranges = dists[valid_mask]
                measured_ranges = true_ranges + np.random.normal(0, NOISE_STD, len(true_ranges))
                
                estimated_loc = solve_linear_ls(neighbor_est_pos, measured_ranges)
                
                # B. Theoretical Step (Uses True Pos -> No Drift, Geometry Only)
                neighbor_true_pos = true_positions[my_neighbors]
                theo_err = calculate_theoretical_error(neighbor_true_pos, NOISE_STD)
                
                if estimated_loc is not None:
                    est_positions[u_idx] = estimated_loc
                    is_discovered[u_idx] = True
                    theoretical_errors[u_idx] = theo_err
                    new_discovery = True

    # 3. Analysis
    eval_mask = is_discovered.copy()
    eval_mask[anchor_indices] = False
    
    sim_errors = np.linalg.norm(est_positions[eval_mask] - true_positions[eval_mask], axis=1)
    theo_errors = theoretical_errors[eval_mask]
    
    center = np.array([FIELD_SIZE/2, FIELD_SIZE/2])
    dist_from_center = np.linalg.norm(true_positions[eval_mask] - center, axis=1)
    
    # Clean up outliers for plotting (cap at 1000m for readability)
    # But keep data for analysis
    
    # --- PLOTTING ---
    # Map
    ax_map.set_title(f"{scenario}: Network Map")
    ax_map.scatter(est_positions[eval_mask, 0], est_positions[eval_mask, 1], 
                   c='blue', s=15, alpha=0.5, label='Discovered')
    ax_map.scatter(true_positions[anchor_indices, 0], true_positions[anchor_indices, 1],
                   s=250, c='yellow', marker='*', edgecolors='black', zorder=10, label='Anchors')
    ax_map.set_xlim(0, FIELD_SIZE); ax_map.set_ylim(0, FIELD_SIZE)
    ax_map.grid(True, linestyle='--', alpha=0.5)
    
    # Drift vs Theory Plot
    ax_drift.set_title(f"{scenario}: Simulation vs Theory")
    
    # Plot Simulation Error (Red)
    ax_drift.scatter(dist_from_center, sim_errors, c='red', alpha=0.4, s=15, label='Simulated (Drift)')
    
    # Plot Theoretical Error (Black Dots/Line)
    # Sort for cleaner line
    sort_idx = np.argsort(dist_from_center)
    sorted_dist = dist_from_center[sort_idx]
    sorted_theo = theo_errors[sort_idx]
    
    # Moving average for theory line to make it readable
    window = 15
    if len(sorted_theo) > window:
        smooth_theo = np.convolve(sorted_theo, np.ones(window)/window, mode='valid')
        ax_drift.plot(sorted_dist[:len(smooth_theo)], smooth_theo, 'k-', lw=2, label='Theoretical (GDOP)')
    else:
        ax_drift.scatter(dist_from_center, theo_errors, c='black', marker='x', alpha=0.5, label='Theoretical')

    ax_drift.set_xlabel("Dist from Center (m)")
    ax_drift.set_ylabel("Error (m)")
    ax_drift.set_ylim(0, 600) # Zoom in to see the comparison, ignoring massive outliers
    ax_drift.grid(True)
    ax_drift.legend()

def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    run_simulation_logic("Center", axes[0, 0], axes[1, 0])
    run_simulation_logic("Scattered", axes[0, 1], axes[1, 1])
    plt.tight_layout()
    filename = "simulation_vs_theory_analysis.png"
    plt.savefig(filename, dpi=150)
    print(f"Graph saved to {os.getcwd()}/{filename}")
    plt.show()

if __name__ == "__main__":
    main()