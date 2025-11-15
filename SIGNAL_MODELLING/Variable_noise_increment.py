import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import minimize
import time
import warnings
from typing import Callable, List, Dict

# --- 1. Constants and Scenario Setup ---
AREA_SIZE = 50
P_TRUE = np.array([AREA_SIZE / 2, AREA_SIZE / 2]) # [25, 25]

# We will use ALL 10 anchors for every simulation
NUM_ANCHORS_MAX = 10
np.random.seed(42)  # For reproducible results
ANCHORS_MASTER = np.random.rand(NUM_ANCHORS_MAX, 2) * AREA_SIZE

# Monte Carlo Parameters
NUM_RUNS = 1000  # Runs per noise level
ANCHOR_COUNTS_TO_TEST = range(2, NUM_ANCHORS_MAX + 1) # [2, 3, 4, ..., 10]

# X-AXIS: The base noise level
BASE_NOISE_LEVELS = np.linspace(0.1, 2.0, 20)
N_STEPS = len(BASE_NOISE_LEVELS) # For animation

# --- 2. Core Localization & Simulation Functions ---

def localize_toa(distances: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Localizes using TOA with N anchors via linearized least-squares (lstsq)."""
    try:
        x0, y0 = anchors[0]
        d0_sq = distances[0]**2
        other_anchors = anchors[1:]
        other_distances_sq = distances[1:]**2
        
        A = 2 * (other_anchors - [x0, y0])
        b = (d0_sq - other_distances_sq) + \
            (other_anchors[:, 0]**2 - x0**2) + \
            (other_anchors[:, 1]**2 - y0**2)
        
        P_est, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return P_est
    except (np.linalg.LinAlgError, ValueError):
        return np.array([np.nan, np.nan])

def sim_toa_measurements_simple(
    anchors: np.ndarray, 
    noise_std: float
) -> Dict:
    """
    Simulates N noisy TOA measurements, where all anchors
    have the *same* noise_std.
    """
    d_true = np.linalg.norm(anchors - P_TRUE, axis=1)
    
    # Generate noise using the same std dev for all anchors
    noise_values = np.random.randn(len(anchors)) * noise_std
    
    d_meas = d_true + noise_values
    return {'distances': d_meas, 'anchors': anchors}

# --- 3. Monte Carlo Simulation Engine ---

def run_profiling() -> Dict[int, List[float]]:
    """
    Runs the full simulation.
    Outer loop: Anchor counts (N)
    Inner loop: Noise levels (base_noise)
    
    Returns:
        A dictionary where key = N, value = list of RMSEs
    """
    
    # { 2: [rmse_noise_1, rmse_noise_2, ...],
    #   3: [rmse_noise_1, rmse_noise_2, ...], ... }
    all_results = {}

    for N in ANCHOR_COUNTS_TO_TEST:
        print(f"  Running simulation for N = {N} anchors...")
        anchors_current = ANCHORS_MASTER[:N, :]
        rmse_for_this_N = []
        
        for base_noise in BASE_NOISE_LEVELS:
            squared_errors = []
            
            for _ in range(NUM_RUNS):
                # 1. Simulate
                measurements = sim_toa_measurements_simple(
                    anchors_current, 
                    base_noise
                )
                
                # 2. Localize
                P_est = localize_toa(**measurements)
                
                # 3. Store error
                if not np.isnan(P_est).any():
                    error = np.linalg.norm(P_est - P_TRUE)
                    squared_errors.append(error**2)
            
            if squared_errors:
                rmse = np.sqrt(np.mean(squared_errors))
                rmse_for_this_N.append(rmse)
            else:
                rmse_for_this_N.append(np.nan)
        
        all_results[N] = rmse_for_this_N
            
    return all_results

# --- 4. Plotting Function ---

def plot_static_results(all_results: Dict[int, List[float]]):
    """Plots the static RMSE comparison graph (fanning plot)."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 7))
    
    # Use a colormap to automatically get different colors
    colors = plt.cm.jet(np.linspace(0, 1, len(all_results)))
    
    for i, (N, rmse_data) in enumerate(all_results.items()):
        plt.plot(BASE_NOISE_LEVELS, rmse_data, color=colors[i], marker='o', 
                 markersize=4, label=f'N = {N} Anchors')
    
    plt.title('Localization Error vs. Noise Level', 
              fontsize=16, weight='bold')
    plt.xlabel('Anchor Noise (Standard Deviation in meters)', fontsize=12)
    plt.ylabel('Location RMSE (meters)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("anchor_count_vs_noise.png")
    print("\nSaved static plot to anchor_count_vs_noise.png")
    plt.show()

# --- 5. Animation Function ---

def create_animation(all_results: Dict[int, List[float]]):
    """Creates and saves the animated 'fanning' plot."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.jet(np.linspace(0, 1, len(all_results)))
    lines = []
    
    # Create the line objects
    for i, N in enumerate(all_results.keys()):
        line, = ax.plot([], [], color=colors[i], marker='o', 
                        markersize=4, label=f'N = {N} Anchors')
        lines.append(line)

    # Set the plot limits
    max_rmse = np.nanmax([np.nanmax(v) for v in all_results.values()])
    ax.set_xlim(0, BASE_NOISE_LEVELS[-1] * 1.05)
    ax.set_ylim(0, max_rmse * 1.05)
    
    # Set labels
    ax.set_title('Localization Error vs. Noise Level', 
                  fontsize=16, weight='bold')
    ax.set_xlabel('Anchor Noise (Standard Deviation in meters)', fontsize=12)
    ax.set_ylabel('Location RMSE (meters)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both")
    
    def update_animation(frame: int) -> tuple:
        """This function is called for each frame of the animation."""
        
        current_step = frame + 1
        x_data = BASE_NOISE_LEVELS[:current_step]
        
        for i, (N, rmse_data) in enumerate(all_results.items()):
            y_data = rmse_data[:current_step]
            lines[i].set_data(x_data, y_data)
        
        ax.set_title(f'Noise: {x_data[-1]:.2f}m | Frame {frame}/{N_STEPS}', 
                     loc='left', fontsize=10)
        
        return tuple(lines)

    print(f"\nCreating animation from {N_STEPS} frames... this may take a minute.")
    
    ani = animation.FuncAnimation(fig, update_animation, frames=N_STEPS, 
                                  interval=200, blit=True)
    
    try:
        ani.save('anchor_count_vs_noise_animation.gif', writer='imagemagick', fps=5)
        print("Animation saved as 'anchor_count_vs_noise_animation.gif'")
    except Exception as e:
        print(f"\nCould not save animation. Do you have 'imagemagick' or 'ffmpeg' installed?")
        print(f"Error: {e}")
        print("Showing animation inline instead (close window to exit).")
        plt.show()

# --- 6. Main Execution ---

def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    print("Running anchor count vs. noise simulation...")
    start_time = time.time()
    
    all_results = run_profiling()
    
    print(f"\nSimulation complete in {time.time() - start_time:.2f} seconds.")
    
    # --- Part 2: Generate Static Plot ---
    print("\n--- Part 2: Generating Static Plot ---")
    plot_static_results(all_results)
    
    # --- Part 3: Generate Animation ---
    print("\n--- Part 3: Generating Animation ---")
    create_animation(all_results)
    
    print("\n--- All tasks complete. ---")

if __name__ == "__main__":
    main()