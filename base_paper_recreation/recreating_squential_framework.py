import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# --- 1. Simulation Constants (From Paper) ---
FIELD_SIZE = 500
NUM_UNKNOWN = 200
NUM_ANCHORS = 4
COVERAGE_RADIUS = 100  # D = 100m

# Simulation Parameters
np.random.seed(42) # Fixed seed for fairness
PATH_LOSS_N = 4.0  # From Paper (a=4)
REF_POWER = -30    # A (Power at 1m)

# Noise Parameters (Realistic values)
TOA_NOISE_STD = 1.0   # 1 meter error for TOA
RSSI_NOISE_STD = 3.0  # 3 dB error for RSSI (optimistic!)

# --- 2. Generate Ground Truth (Same for both) ---
true_unknown_pos = np.random.rand(NUM_UNKNOWN, 2) * FIELD_SIZE
center = FIELD_SIZE / 2
# Anchors clustered in center
anchors_pos = center + (np.random.rand(NUM_ANCHORS, 2) * 40 - 20)

# --- 3. Localization Solvers ---

def localize_toa_solver(anchors, measurements):
    """Linearized Least Squares for TOA."""
    try:
        x0, y0 = anchors[0]
        d0_sq = measurements[0]**2
        others = anchors[1:]
        others_d_sq = measurements[1:]**2
        
        A = 2 * (others - [x0, y0])
        b = (d0_sq - others_d_sq) + \
            (others[:, 0]**2 - x0**2) + \
            (others[:, 1]**2 - y0**2)
            
        P_est, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return P_est
    except:
        return np.array([np.nan, np.nan])

def localize_rssi_solver(anchors, rssi_values):
    """Non-linear Least Squares for RSSI."""
    # Convert RSSI back to estimated distance
    # Distance = 10 ^ ((A - RSSI) / (10 * n))
    est_distances = 10 ** ((REF_POWER - rssi_values) / (10 * PATH_LOSS_N))
    
    def error_func(p_guess):
        # Calculate distance from guess to anchors
        d_calc = np.linalg.norm(anchors - p_guess, axis=1)
        return np.sum((d_calc - est_distances)**2)

    # Initial guess: Average of anchors
    guess = np.mean(anchors, axis=0)
    
    # Optimization
    res = minimize(error_func, guess, method='L-BFGS-B')
    if res.success:
        return res.x
    return np.array([np.nan, np.nan])

# --- 4. The Sequential Discovery Engine ---

def run_sequential_simulation(mode='TOA'):
    # Reset State
    estimated_pos = np.full((NUM_UNKNOWN, 2), np.nan)
    is_localized = np.zeros(NUM_UNKNOWN, dtype=bool)
    
    # Initialize Known Nodes with Anchors
    known_nodes = []
    for i in range(NUM_ANCHORS):
        known_nodes.append({'pos': anchors_pos[i]})

    new_nodes_found = True
    iteration = 0
    
    while new_nodes_found:
        new_nodes_found = False
        iteration += 1
        
        # Check every unknown node
        for i in range(NUM_UNKNOWN):
            if is_localized[i]: continue
                
            visible_anchors = []
            measurements = []
            
            # Look for neighbors
            for node in known_nodes:
                true_dist = np.linalg.norm(node['pos'] - true_unknown_pos[i])
                
                # Check Connectivity
                if true_dist <= COVERAGE_RADIUS:
                    visible_anchors.append(node['pos'])
                    
                    # Generate Measurement based on Mode
                    if mode == 'TOA':
                        meas = true_dist + np.random.randn() * TOA_NOISE_STD
                        measurements.append(meas)
                    elif mode == 'RSSI':
                        # Generate True RSSI
                        true_rssi = REF_POWER - 10 * PATH_LOSS_N * np.log10(true_dist)
                        # Add Noise
                        meas = true_rssi + np.random.randn() * RSSI_NOISE_STD
                        measurements.append(meas)

            # Try to localize if enough neighbors
            if len(visible_anchors) >= 4:
                anchors_mat = np.array(visible_anchors)
                meas_vec = np.array(measurements)
                
                if mode == 'TOA':
                    pos_est = localize_toa_solver(anchors_mat, meas_vec)
                else:
                    pos_est = localize_rssi_solver(anchors_mat, meas_vec)
                
                # Check validity (sanity check bounds)
                if not np.isnan(pos_est).any() and \
                   0 <= pos_est[0] <= FIELD_SIZE and \
                   0 <= pos_est[1] <= FIELD_SIZE:
                    
                    estimated_pos[i] = pos_est
                    is_localized[i] = True
                    new_nodes_found = True
                    
                    # Add to known list for next iteration
                    known_nodes.append({'pos': pos_est})
                    
    return estimated_pos, is_localized

# --- 5. Run Both Simulations ---
print("Running TOA Simulation...")
start = time.time()
toa_ests, toa_found = run_sequential_simulation('TOA')
print(f"TOA Done: {np.sum(toa_found)} nodes found in {time.time()-start:.2f}s")

print("Running RSSI Simulation...")
start = time.time()
rssi_ests, rssi_found = run_sequential_simulation('RSSI')
print(f"RSSI Done: {np.sum(rssi_found)} nodes found in {time.time()-start:.2f}s")

# --- 6. Plotting Comparison ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Helper to plot on axes
def plot_sim(ax, ests, found_mask, title):
    # Plot connections (Error lines)
    found_idx = np.where(found_mask)[0]
    # Draw thin lines connecting true to estimated pos
    # (Doing this first so it's behind the dots)
    segments = np.zeros((len(found_idx) * 3, 2))
    segments[0::3] = true_unknown_pos[found_idx]
    segments[1::3] = ests[found_idx]
    segments[2::3] = np.nan # Break line
    ax.plot(segments[:,0], segments[:,1], 'k-', alpha=0.3, linewidth=0.5)

    # Plot True (Green Circles)
    ax.scatter(true_unknown_pos[:, 0], true_unknown_pos[:, 1], 
               edgecolors='g', facecolors='none', s=50, alpha=0.6, label='True')
    
    # Plot Estimates (Red Crosses)
    ax.scatter(ests[found_idx, 0], ests[found_idx, 1], 
               c='r', marker='x', s=40, label='Est')
    
    # Plot Anchors
    ax.scatter(anchors_pos[:, 0], anchors_pos[:, 1], 
               c='b', marker='^', s=100, label='Anchors')
    
    # Calc Stats
    if len(found_idx) > 0:
        errors = np.linalg.norm(true_unknown_pos[found_idx] - ests[found_idx], axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        ax.set_title(f"{title}\nLocalized: {len(found_idx)}/{NUM_UNKNOWN} | RMSE: {rmse:.2f}m")
    else:
        ax.set_title(f"{title}\nLocalized: 0")

    ax.set_xlim(0, FIELD_SIZE)
    ax.set_ylim(0, FIELD_SIZE)
    ax.legend()
    ax.grid(True)

plot_sim(ax1, toa_ests, toa_found, "TOA Sequential Discovery")
plot_sim(ax2, rssi_ests, rssi_found, "RSSI Sequential Discovery")

plt.tight_layout()
plt.show()