import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# --- 1. Simulation Constants ---
FIELD_SIZE = 500
NUM_UNKNOWN = 200
COVERAGE_RADIUS = 100 

# Noise Parameters
TOA_NOISE_STD = 1.0   
RSSI_NOISE_STD = 3.0  
PATH_LOSS_N = 4.0
REF_POWER = -30

np.random.seed(42) # Keep unknown nodes consistent across all tests
TRUE_UNKNOWN_POS = np.random.rand(NUM_UNKNOWN, 2) * FIELD_SIZE

# --- 2. Anchor Generation Helper ---
def get_anchors(config_type):
    """Returns the list of anchor positions based on configuration."""
    center = FIELD_SIZE / 2
    
    # 1. Always have the 4 central anchors (The Core)
    # Small cluster in the middle
    core_anchors = center + (np.random.rand(4, 2) * 40 - 20)
    
    extra_anchors = []
    
    # 2. Define "Half-Way" coordinates (approx 125m and 375m)
    low = FIELD_SIZE * 0.25  # 125
    high = FIELD_SIZE * 0.75 # 375
    mid = FIELD_SIZE * 0.5   # 250
    
    if config_type == 'Center Only':
        pass # No extra
        
    elif config_type == '+4 Scattered':
        # Add anchors at the 4 "corners" of the half-way box
        extra_anchors = [
            [low, low], [high, low],
            [low, high], [high, high]
        ]
        
    elif config_type == '+8 Scattered':
        # Add 4 corners AND 4 midpoints
        extra_anchors = [
            [low, low], [high, low], [low, high], [high, high], # Corners
            [low, mid], [high, mid], [mid, low], [mid, high]    # Midpoints
        ]
        
    all_anchors = np.vstack([core_anchors, np.array(extra_anchors).reshape(-1, 2)])
    return all_anchors

# --- 3. Localization Solvers (Same as before) ---
def localize_toa_solver(anchors, measurements):
    try:
        x0, y0 = anchors[0]
        d0_sq = measurements[0]**2
        others = anchors[1:]
        others_d_sq = measurements[1:]**2
        A = 2 * (others - [x0, y0])
        b = (d0_sq - others_d_sq) + (others[:, 0]**2 - x0**2) + (others[:, 1]**2 - y0**2)
        P_est, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return P_est
    except:
        return np.array([np.nan, np.nan])

def localize_rssi_solver(anchors, rssi_values):
    est_distances = 10 ** ((REF_POWER - rssi_values) / (10 * PATH_LOSS_N))
    def error_func(p_guess):
        d_calc = np.linalg.norm(anchors - p_guess, axis=1)
        return np.sum((d_calc - est_distances)**2)
    guess = np.mean(anchors, axis=0)
    res = minimize(error_func, guess, method='L-BFGS-B')
    return res.x if res.success else np.array([np.nan, np.nan])

# --- 4. Simulation Loop ---
def run_simulation(anchors_pos, mode='TOA'):
    estimated_pos = np.full((NUM_UNKNOWN, 2), np.nan)
    is_localized = np.zeros(NUM_UNKNOWN, dtype=bool)
    
    # Initialize known nodes with our anchors
    known_nodes = [{'pos': p} for p in anchors_pos]

    new_nodes_found = True
    while new_nodes_found:
        new_nodes_found = False
        for i in range(NUM_UNKNOWN):
            if is_localized[i]: continue
            
            visible_anchors = []
            measurements = []
            
            for node in known_nodes:
                true_dist = np.linalg.norm(node['pos'] - TRUE_UNKNOWN_POS[i])
                if true_dist <= COVERAGE_RADIUS:
                    visible_anchors.append(node['pos'])
                    if mode == 'TOA':
                        measurements.append(true_dist + np.random.randn() * TOA_NOISE_STD)
                    else: # RSSI
                        true_rssi = REF_POWER - 10 * PATH_LOSS_N * np.log10(true_dist)
                        measurements.append(true_rssi + np.random.randn() * RSSI_NOISE_STD)

            if len(visible_anchors) >= 4:
                if mode == 'TOA':
                    pos_est = localize_toa_solver(np.array(visible_anchors), np.array(measurements))
                else:
                    pos_est = localize_rssi_solver(np.array(visible_anchors), np.array(measurements))
                
                if not np.isnan(pos_est).any() and 0 <= pos_est[0] <= FIELD_SIZE and 0 <= pos_est[1] <= FIELD_SIZE:
                    estimated_pos[i] = pos_est
                    is_localized[i] = True
                    new_nodes_found = True
                    known_nodes.append({'pos': pos_est})
    
    # Calculate RMSE
    found_idx = np.where(is_localized)[0]
    if len(found_idx) > 0:
        errors = np.linalg.norm(TRUE_UNKNOWN_POS[found_idx] - estimated_pos[found_idx], axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        return rmse, estimated_pos, is_localized
    return np.nan, estimated_pos, is_localized

# --- 5. Run Experiment Scenarios ---
configs = ['Center Only', '+4 Scattered', '+8 Scattered']
results_toa = []
results_rssi = []

# Store visualization data for the +8 case
vis_data_toa = None
vis_data_rssi = None
vis_anchors = None

print("Running Impact Analysis...")
for cfg in configs:
    anchors = get_anchors(cfg)
    
    # Run TOA
    rmse_t, est_t, mask_t = run_simulation(anchors, 'TOA')
    results_toa.append(rmse_t)
    
    # Run RSSI
    rmse_r, est_r, mask_r = run_simulation(anchors, 'RSSI')
    results_rssi.append(rmse_r)
    
    print(f"Config: {cfg:<15} | TOA RMSE: {rmse_t:.2f}m | RSSI RMSE: {rmse_r:.2f}m")
    
    if cfg == '+8 Scattered':
        vis_data_toa = (est_t, mask_t)
        vis_data_rssi = (est_r, mask_r)
        vis_anchors = anchors

# --- 6. Visualization ---
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)

# Bar Chart (RMSE Comparison)
ax_bar = fig.add_subplot(gs[0, :])
x = np.arange(len(configs))
width = 0.35
rects1 = ax_bar.bar(x - width/2, results_toa, width, label='TOA', color='royalblue')
rects2 = ax_bar.bar(x + width/2, results_rssi, width, label='RSSI', color='crimson')

ax_bar.set_ylabel('Location RMSE (meters)')
ax_bar.set_title('Impact of Scattered Anchors on Localization Error')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(configs)
ax_bar.legend()
ax_bar.bar_label(rects1, padding=3, fmt='%.1f')
ax_bar.bar_label(rects2, padding=3, fmt='%.1f')

# Map Visualizations (For the +8 Scattered case)
def plot_map(ax, ests, mask, title):
    found_idx = np.where(mask)[0]
    # Draw error lines
    segments = np.zeros((len(found_idx) * 3, 2))
    segments[0::3] = TRUE_UNKNOWN_POS[found_idx]
    segments[1::3] = ests[found_idx]
    segments[2::3] = np.nan
    ax.plot(segments[:,0], segments[:,1], 'k-', alpha=0.2, linewidth=0.5)
    
    ax.scatter(TRUE_UNKNOWN_POS[:,0], TRUE_UNKNOWN_POS[:,1], ec='g', fc='none', s=30, alpha=0.5, label='True')
    ax.scatter(ests[found_idx,0], ests[found_idx,1], c='r', marker='x', s=20, label='Est')
    # Highlight Anchors
    ax.scatter(vis_anchors[:4,0], vis_anchors[:4,1], c='b', marker='^', s=80, label='Center Anchors')
    if len(vis_anchors) > 4:
        ax.scatter(vis_anchors[4:,0], vis_anchors[4:,1], c='orange', marker='s', s=80, label='Scattered Anchors')
    
    ax.set_title(title)
    ax.set_xlim(0, FIELD_SIZE); ax.set_ylim(0, FIELD_SIZE)
    ax.legend(loc='upper right', fontsize='small')

ax_map1 = fig.add_subplot(gs[1, 0])
plot_map(ax_map1, vis_data_toa[0], vis_data_toa[1], 'TOA Map (+8 Scattered Anchors)')

ax_map2 = fig.add_subplot(gs[1, 1])
plot_map(ax_map2, vis_data_rssi[0], vis_data_rssi[1], 'RSSI Map (+8 Scattered Anchors)')

plt.tight_layout()
plt.show()