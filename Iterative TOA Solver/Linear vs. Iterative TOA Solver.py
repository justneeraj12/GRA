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
    Uses the squared-difference approach to remove non-linear terms.
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
        
        # Linearization: 2x(x_i - x_k) + 2y(y_i - y_k) = ...
        row_A = [2 * (x_i - x_k), 2 * (y_i - y_k)]
        val_b = (x_i**2 + y_i**2) - (x_k**2 + y_k**2) - r_i**2 + r_k**2
        
        A.append(row_A)
        b.append(val_b)
        
    try:
        pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return pos
    except:
        return None

def get_anchors(scenario, true_positions):
    """Selects anchor indices based on the strategy."""
    indices = np.arange(NUM_NODES)
    center = np.array([FIELD_SIZE/2, FIELD_SIZE/2])
    dists_to_center = np.linalg.norm(true_positions - center, axis=1)
    
    if scenario == "Center":
        # Closest 10 to center
        return indices[np.argsort(dists_to_center)][:10]
        
    elif scenario == "Scattered":
        # 5 Center + Edge Anchors to stop drift
        center_anchors = indices[np.argsort(dists_to_center)][:5]
        edge_anchors = []
        for i in range(NUM_NODES):
            x, y = true_positions[i]
            if (x < 100 or x > 400) and (y < 100 or y > 400):
                edge_anchors.append(i)
        
        combined = np.concatenate((center_anchors, edge_anchors[:8]))
        return np.unique(combined)

def run_simulation_logic(scenario, ax_map, ax_drift):
    """Runs one simulation scenario and plots to the specific axes."""
    print(f"--- Simulating: {scenario} Anchors ---")
    
    # Generate Ground Truth
    np.random.seed(42) 
    true_positions = np.random.rand(NUM_NODES, 2) * FIELD_SIZE
    
    anchor_indices = get_anchors(scenario, true_positions)
    
    # Discovery Loop
    est_positions = np.zeros((NUM_NODES, 2))
    is_discovered = np.zeros(NUM_NODES, dtype=bool)
    est_positions[anchor_indices] = true_positions[anchor_indices]
    is_discovered[anchor_indices] = True
    
    new_discovery = True
    while new_discovery:
        new_discovery = False
        unknown_indices = np.arange(NUM_NODES)[~is_discovered]
        
        for u_idx in unknown_indices:
            # Calculate distance to DISCOVERED nodes
            dist_sq = np.sum((true_positions[u_idx] - true_positions[is_discovered])**2, axis=1)
            dists = np.sqrt(dist_sq)
            
            valid_mask = dists <= COMM_RANGE
            known_indices = np.arange(NUM_NODES)[is_discovered]
            my_neighbors = known_indices[valid_mask]
            
            if len(my_neighbors) >= 3:
                # Use ESTIMATED positions of neighbors (causes drift)
                neighbor_pos = est_positions[my_neighbors]
                # True range + Noise
                true_ranges = dists[valid_mask]
                measured_ranges = true_ranges + np.random.normal(0, NOISE_STD, len(true_ranges))
                
                estimated_loc = solve_linear_ls(neighbor_pos, measured_ranges)
                
                if estimated_loc is not None:
                    est_positions[u_idx] = estimated_loc
                    is_discovered[u_idx] = True
                    new_discovery = True

    # Analysis
    # Filter out anchors for error calculation
    eval_mask = is_discovered.copy()
    eval_mask[anchor_indices] = False
    
    errors = np.linalg.norm(est_positions[eval_mask] - true_positions[eval_mask], axis=1)
    center = np.array([FIELD_SIZE/2, FIELD_SIZE/2])
    dist_from_center = np.linalg.norm(true_positions[eval_mask] - center, axis=1)

    # --- PLOTTING ---
    
    # 1. Map
    ax_map.set_title(f"{scenario}: Network Map")
    ax_map.scatter(est_positions[eval_mask, 0], est_positions[eval_mask, 1], 
                c='blue', s=15, alpha=0.6, label='Discovered')
    # YELLOW STARS
    ax_map.scatter(true_positions[anchor_indices, 0], true_positions[anchor_indices, 1],
                s=250, c='yellow', marker='*', edgecolors='black', zorder=10, label='Anchors')
    ax_map.set_xlim(0, FIELD_SIZE); ax_map.set_ylim(0, FIELD_SIZE)
    ax_map.grid(True, linestyle='--', alpha=0.5)
    if scenario == "Center": ax_map.legend(loc='upper right')

    # 2. Drift
    ax_drift.set_title(f"{scenario}: Error vs. Distance")
    ax_drift.scatter(dist_from_center, errors, c='red', alpha=0.5, s=15)
    
    # Trendline
    if len(errors) > 0:
        z = np.polyfit(dist_from_center, errors, 1)
        p = np.poly1d(z)
        ax_drift.plot(dist_from_center, p(dist_from_center), "k--", lw=2, label=f"Trend")
    
    ax_drift.set_xlabel("Dist from Center (m)")
    ax_drift.set_ylabel("Error (m)")
    ax_drift.set_ylim(0, max(errors, default=1) * 1.1)
    ax_drift.grid(True)

def main():
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Run Center Scenario (Left Column)
    run_simulation_logic("Center", axes[0, 0], axes[1, 0])
    
    # Run Scattered Scenario (Right Column)
    run_simulation_logic("Scattered", axes[0, 1], axes[1, 1])
    
    plt.tight_layout()
    
    # Save to current directory
    filename = "simulation_comparison_linear_ls.png"
    plt.savefig(filename, dpi=150)
    print(f"\nSUCCESS: Graph saved to {os.getcwd()}/{filename}")
    plt.show()

if __name__ == "__main__":
    main()