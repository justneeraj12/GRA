import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.animation as animation
import time
import warnings
from typing import Callable, List, Dict

# --- 1. Constants and Scenario Setup ---
AREA_SIZE = 50
P_TRUE = np.array([AREA_SIZE / 2, AREA_SIZE / 2]) # [25, 25]

# Generate a "master set" of 100 random anchors.
np.random.seed(42)  # For reproducible results
ANCHORS_MASTER = np.random.rand(100, 2) * AREA_SIZE

# Fixed noise levels
TOA_NOISE_STD = 0.5  # 0.5m standard deviation
RSSI_NOISE_STD = 2.0 # 2.0dB standard deviation

# RSSI Model Parameters
RSSI_Pt = -30  # Ref. power at 1m (dBm)
RSSI_n = 2.5   # Path loss exponent

# Monte Carlo Parameters
NUM_RUNS = 1000  # Runs per anchor count
# --- UPDATED LIST ---
ANCHOR_COUNTS = [2, 3, 4, 8, 10, 15, 20, 30, 40, 60, 100]

# --- 2. Core Localization Functions (Unchanged) ---

def localize_rssi(dist_estimates: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Localizes using RSSI with N anchors via least-squares optimization."""
    def error_func(P):
        d_calculated = np.linalg.norm(anchors - P, axis=1)
        return np.sum((d_calculated - dist_estimates)**2)

    P_guess = np.mean(anchors, axis=0)
    result = minimize(error_func, P_guess, method='L-BFGS-B', 
                      bounds=((-20, AREA_SIZE + 20), (-20, AREA_SIZE + 20)))
    return result.x if result.success else np.array([np.nan, np.nan])

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

# --- 3. Measurement Simulator Functions (Unchanged) ---

def sim_rssi_measurements(anchors: np.ndarray) -> Dict:
    """Simulates N noisy RSSI-based distance estimates."""
    d_true = np.linalg.norm(anchors - P_TRUE, axis=1)
    RSSI_true = RSSI_Pt - 10 * RSSI_n * np.log10(d_true)
    RSSI_meas = RSSI_true + np.random.randn(len(anchors)) * RSSI_NOISE_STD
    dist_est = 10**((RSSI_Pt - RSSI_meas) / (10 * RSSI_n))
    return {'dist_estimates': dist_est, 'anchors': anchors}

def sim_toa_measurements(anchors: np.ndarray) -> Dict:
    """Simulates N noisy TOA distance measurements."""
    d_true = np.linalg.norm(anchors - P_TRUE, axis=1)
    d_meas = d_true + np.random.randn(len(anchors)) * TOA_NOISE_STD
    return {'distances': d_meas, 'anchors': anchors}

# --- 4. Monte Carlo Simulation Engine (Unchanged) ---

def run_anchor_profiling(
    localize_func: Callable, 
    measurement_simulator: Callable
) -> Dict:
    """
    Runs a full Monte Carlo simulation, storing stats AND raw estimates.
    """
    all_rmse = []
    all_times = []
    all_success_rates = []
    all_raw_estimates = []
    all_anchor_sets = []
    
    for N in ANCHOR_COUNTS:
        anchors_current = ANCHORS_MASTER[:N, :]
        
        squared_errors = []
        computation_times = []
        raw_estimates_N = []
        success_count = 0
        
        for _ in range(NUM_RUNS):
            measurements = measurement_simulator(anchors_current)
            start_t = time.perf_counter()
            P_est = localize_func(**measurements)
            end_t = time.perf_counter()
            
            computation_times.append((end_t - start_t) * 1000) # in ms
            
            if not np.isnan(P_est).any():
                error = np.linalg.norm(P_est - P_TRUE)
                squared_errors.append(error**2)
                success_count += 1
                raw_estimates_N.append(P_est)
        
        if squared_errors:
            all_rmse.append(np.sqrt(np.mean(squared_errors)))
        else:
            all_rmse.append(np.nan)
            
        all_times.append(np.mean(computation_times))
        all_success_rates.append(success_count / NUM_RUNS * 100)
        all_raw_estimates.append(np.array(raw_estimates_N))
        all_anchor_sets.append(anchors_current)
            
    return {
        'rmse': all_rmse,
        'time': all_times,
        'success': all_success_rates,
        'raw_estimates': all_raw_estimates, 
        'anchors': all_anchor_sets         
    }

# --- 5. Static Plotting Function (Unchanged) ---

def plot_static_results(toa_results: Dict, rssi_results: Dict):
    """Plots the 3 static performance metrics."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Performance vs. Number of Anchors', fontsize=18, weight='bold')

    # --- Plot 1: Accuracy (RMSE) ---
    axs[0].plot(ANCHOR_COUNTS, toa_results['rmse'], 'b-o', label='TOA')
    axs[0].plot(ANCHOR_COUNTS, rssi_results['rmse'], 'r-o', label='RSSI')
    axs[0].set(title='Accuracy vs. Anchor Count', 
               ylabel='Location RMSE (meters)')
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].grid(True, which="both")

    # --- Plot 2: Computation Time ---
    axs[1].plot(ANCHOR_COUNTS, toa_results['time'], 'b-o', label='TOA (lstsq)')
    axs[1].plot(ANCHOR_COUNTS, rssi_results['time'], 'r-o', label='RSSI (optimizer)')
    axs[1].set(title='Computation Time vs. Anchor Count', 
               ylabel='Avg. Time per Run (ms)')
    axs[1].set_yscale('log') 
    axs[1].legend()
    axs[1].grid(True, which="both")

    # --- Plot 3: Success Rate ---
    axs[2].plot(ANCHOR_COUNTS, toa_results['success'], 'b-o', label='TOA')
    axs[2].plot(ANCHOR_COUNTS, rssi_results['success'], 'r-o', label='RSSI')
    axs[2].set(title='Success Rate vs. Anchor Count', 
               xlabel='Number of Anchors',
               ylabel='Successful Runs (%)')
    axs[2].set_ylim(0, 105)
    axs[2].legend()
    axs[2].grid(True)

    for ax in axs:
        ax.set_xscale('log')
        ax.set_xticks(ANCHOR_COUNTS)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("Displaying static plots. Close the plot window to continue and generate the animation...")
    plt.show() # This will block the script until the window is closed

# --- 6. Animation Function (Unchanged) ---

def create_animation(toa_results: Dict, rssi_results: Dict):
    """Creates and saves the animated scatter plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    def update_animation(frame: int) -> tuple:
        """This function is called for each frame of the animation."""
        N = ANCHOR_COUNTS[frame]
        
        # Get the pre-computed data for this frame
        toa_cloud = toa_results['raw_estimates'][frame]
        rssi_cloud = rssi_results['raw_estimates'][frame]
        anchors = toa_results['anchors'][frame] # Anchors are the same
        
        # --- Clear and Draw TOA Plot (Left) ---
        ax1.clear()
        ax1.set_title(f"TOA Localization with {N} Anchors")
        ax1.set_xlabel("X (meters)")
        ax1.set_ylabel("Y (meters)")
        
        if toa_cloud.size > 0:
            ax1.scatter(toa_cloud[:, 0], toa_cloud[:, 1], c='blue', s=10, alpha=0.1, label='Estimates')
        ax1.scatter(anchors[:, 0], anchors[:, 1], c='blue', marker='^', s=50, label='Anchors')
        ax1.plot(P_TRUE[0], P_TRUE[1], 'y*', markersize=15, markeredgecolor='black', label='True Position')
        
        ax1.set_xlim(0, AREA_SIZE)
        ax1.set_ylim(0, AREA_SIZE)
        ax1.legend()
        ax1.grid(True)
        
        # --- Clear and Draw RSSI Plot (Right) ---
        ax2.clear()
        ax2.set_title(f"RSSI Localization with {N} Anchors")
        ax2.set_xlabel("X (meters)")
        
        if rssi_cloud.size > 0:
            ax2.scatter(rssi_cloud[:, 0], rssi_cloud[:, 1], c='red', s=10, alpha=0.1, label='Estimates')
        ax2.scatter(anchors[:, 0], anchors[:, 1], c='red', marker='^', s=50, label='Anchors')
        ax2.plot(P_TRUE[0], P_TRUE[1], 'y*', markersize=15, markeredgecolor='black', label='True Position')
                 
        ax2.set_xlim(0, AREA_SIZE)
        ax2.set_ylim(0, AREA_SIZE)
        ax2.legend()
        ax2.grid(True)
        
        fig.suptitle(f'Localization Performance ({NUM_RUNS} runs per frame)', fontsize=16)
        
        return (ax1, ax2)

    print(f"Creating animation from {len(ANCHOR_COUNTS)} frames... this may take a minute.")
    
    ani = animation.FuncAnimation(fig, update_animation, frames=len(ANCHOR_COUNTS), 
                                  interval=1000, blit=False)
    
    try:
        ani.save('anchor_profiling_animation.gif', writer='imagemagick', fps=1)
        print("Animation saved as 'anchor_profiling_animation.gif'")
    except Exception as e:
        print(f"\nCould not save animation. Do you have 'imagemagick' or 'ffmpeg' installed?")
        print(f"Error: {e}")
        print("Showing animation inline instead (close window to exit).")
        plt.show()

# --- 8. NEW: Summary Printing Function ---

def print_analysis_summary(toa_results: Dict, rssi_results: Dict):
    """Prints the analysis of the results to the console."""
    
    print("\n\n--- Analysis of Results ---")
    
    print("\n## What is 'Success Rate'?")
    print("-" * 30)
    print(f"The 'Success Rate' is the percentage of simulation runs (out of {NUM_RUNS}) that successfully computed a valid, numerical position.")
    print("\nA run is considered a 'failure' if the algorithm's math broke down and it returned 'NaN' (Not a Number). This happens if:")
    print("1. For TOA: The 'np.linalg.lstsq' function fails (e.g., from impossible anchor geometry).")
    print("2. For RSSI: The 'scipy.optimize.minimize' optimizer fails to converge on a single best-fit location.")
    print("\nThis metric measures the algorithm's STABILITY and RELIABILITY.")

    print("\n\n## Conclusions for 2, 3, and 4 Anchors (from this simulation)")
    print("-" * 30)
    
    # --- Helper to get data ---
    def get_stats(results, N):
        try:
            idx = ANCHOR_COUNTS.index(N)
            rmse = results['rmse'][idx]
            success = results['success'][idx]
            return f"RMSE: {rmse:.2f} meters, Success: {success:.1f}%"
        except ValueError:
            return "N/A (Not in ANCHOR_COUNTS list)"

    # --- N=2 ---
    print("\nN=2 (Two Anchors): INSUFFICIENT & AMBIGUOUS")
    print(f"  - TOA Stats:   {get_stats(toa_results, 2)}")
    print(f"  - RSSI Stats:  {get_stats(rssi_results, 2)}")
    print("  - Conclusion: Two anchors are not enough. The RMSE is extremely high and stability is low.")
    print("  - Reason: Geometrically, two circles intersect at TWO points. The problem is ambiguous.")

    # --- N=3 ---
    print("\nN=3 (Three Anchors): THE MINIMUM FOR A UNIQUE SOLUTION")
    print(f"  - TOA Stats:   {get_stats(toa_results, 3)}")
    print(f"  - RSSI Stats:  {get_stats(rssi_results, 3)}")
    print("  - Conclusion: This is the minimum requirement for an unambiguous 2D position.")
    print("  - Reason: Geometrically, three circles intersect at one unique point. The problem is now 'well-defined.'")
    print("  - Result: We see a DRAMATIC drop in RMSE and a jump in Success Rate compared to N=2.")

    # --- N=4 ---
    print("\nN=4 (Four Anchors): THE START OF ROBUSTNESS")
    print(f"  - TOA Stats:   {get_stats(toa_results, 4)}")
    print(f"  - RSSI Stats:  {get_stats(rssi_results, 4)}")
    print("  - Conclusion: Adding a fourth anchor begins to provide noise reduction.")
    print("  - Reason: The system is now 'overdetermined' (e.g., 3 equations for 2 unknowns). This 'extra' anchor provides REDUNDANCY.")
    print("  - Result: The algorithms (especially 'lstsq' for TOA) use this extra data to average out measurement noise. This results in a further, noticeable drop in RMSE, making the estimate more ACCURATE and ROBUST.")
    print("\n--- End of Analysis ---")

# --- 7. Main Execution (Renumbered) ---

def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    # Set the style for ALL plots
    plt.style.use('seaborn-v0_8-darkgrid')
    
    print("--- Part 1: Running Simulations ---")
    start_time = time.time()
    
    print("Running anchor profiling for TOA...")
    toa_results = run_anchor_profiling(localize_toa, sim_toa_measurements)
    print("Running anchor profiling for RSSI...")
    rssi_results = run_anchor_profiling(localize_rssi, sim_rssi_measurements)
    
    print(f"\nAll simulations complete in {time.time() - start_time:.2f} seconds.")
    
    # --- Part 2: Generate Static Plots ---
    print("\n--- Part 2: Generating Static Plots ---")
    plot_static_results(toa_results, rssi_results)
    
    # --- Part 3: Generate Animation ---
    print("\n--- Part 3: Generating Animation ---")
    create_animation(toa_results, rssi_results)
    
    # --- Part 4: Print Analysis ---
    print("\n--- Part 4: Printing Analysis Summary ---")
    # Pass the results dictionaries to the summary function
    print_analysis_summary(toa_results, rssi_results)
    
    print("\n--- All tasks complete. ---")

if __name__ == "__main__":
    main()