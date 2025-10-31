# SIGNAL_MODELLING/ADDED_NOISE.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import warnings
from typing import Callable, List, Dict

# --- 1. Constants and Scenario Setup ---
A1 = np.array([0, 0])
A2 = np.array([10, 0])
A3 = np.array([5, 15])
P_TRUE = np.array([7, 8])

# RSSI Model Parameters
RSSI_Pt = -30  # Ref. power at 1m (dBm)
RSSI_n = 2.5   # Path loss exponent

# Monte Carlo Parameters
NUM_RUNS = 3000  # Runs per noise level
N_STEPS = 25    # Number of noise levels

# --- 2. Core Localization Functions ---
# ... (localize_rssi, localize_toa, localize_tdoa are unchanged) ...

def localize_rssi(d1_est: float, d2_est: float, d3_est: float) -> np.ndarray:
    """
    Estimates position given three RSSI-based distance estimates.
    Uses 'L-BFGS-B' optimizer to find the best-fit (x, y) coordinate.
    """
    def error_func(P):
        d1 = np.linalg.norm(P - A1)
        d2 = np.linalg.norm(P - A2)
        d3 = np.linalg.norm(P - A3)
        return (d1 - d1_est)**2 + (d2 - d2_est)**2 + (d3 - d3_est)**2

    P_guess = (A1 + A2 + A3) / 3
    result = minimize(error_func, P_guess, method='L-BFGS-B', 
                      bounds=((-20, 20), (-20, 20)))
    
    return result.x if result.success else np.array([np.nan, np.nan])

def localize_toa(d1: float, d2: float, d3: float) -> np.ndarray:
    """
    Estimates position given three TOA-based distances (trilateration).
    Uses the linearized system of equations (A*P = b).
    """
    x1, y1 = A1
    x2, y2 = A2
    x3, y3 = A3
    
    A = np.array([
        [2 * (x2 - x1), 2 * (y2 - y1)],
        [2 * (x3 - x1), 2 * (y3 - y1)]
    ])
    b = np.array([
        (d1**2 - d2**2) + (x2**2 - x1**2) + (y2**2 - y1**2),
        (d1**2 - d3**2) + (x3**2 - x1**2) + (y3**2 - y1**2)
    ])
    
    try:
        P_est = np.linalg.solve(A, b)
        return P_est
    except np.linalg.LinAlgError:
        # Singular matrix (e.g., collinear anchors)
        return np.array([np.nan, np.nan])

def localize_tdoa(d_diff_21: float, d_diff_31: float) -> np.ndarray:
    """
    Estimates position given two TDOA measurements relative to Anchor 1.
    Uses 'L-BFGS-B' optimizer to find the hyperbolic intersection.
    """
    def error_func(P):
        d1 = np.linalg.norm(P - A1)
        d2 = np.linalg.norm(P - A2)
        d3 = np.linalg.norm(P - A3)
        return ((d2 - d1) - d_diff_21)**2 + ((d3 - d1) - d_diff_31)**2

    P_guess = (A1 + A2 + A3) / 3
    result = minimize(error_func, P_guess, method='L-BFGS-B', 
                      bounds=((-20, 20), (-20, 20)))
    
    return result.x if result.success else np.array([np.nan, np.nan])

def localize_aoa(theta1: float, theta2: float) -> np.ndarray:
    """
    Estimates position given two AOA measurements (triangulation).
    Uses the line-intersection method (A*P = b).
    """
    
    # --- FIX ---
    # Add a safety check to prevent tan() from exploding near pi/2 (90 deg)
    # We define a "danger zone" tolerance around pi/2 and -pi/2.
    angle_tolerance = 0.02  # approx 1.15 degrees
    
    # Check if theta1 or theta2 are in the danger zone
    if (np.abs(np.abs(theta1) - np.pi/2) < angle_tolerance or 
        np.abs(np.abs(theta2) - np.pi/2) < angle_tolerance):
        # This run is unstable, discard it by returning NaN
        return np.array([np.nan, np.nan])
    # --- END FIX ---

    m1 = np.tan(theta1)
    m2 = np.tan(theta2)
    
    x1, y1 = A1
    x2, y2 = A2
    
    # Check for near-parallel lines
    if np.abs(m1 - m2) < 1e-5:
        return np.array([np.nan, np.nan])

    A = np.array([[-m1, 1], [-m2, 1]])
    b = np.array([y1 - m1 * x1, y2 - m2 * x2])
    
    try:
        P_est = np.linalg.solve(A, b)
        return P_est
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan])

# --- 3. Monte Carlo Simulation Engine ---

def run_profiling(
    localize_func: Callable, 
    variances: np.ndarray, 
    measurement_simulator: Callable
) -> List[float]:
    """
    Runs a full Monte Carlo simulation for a given localization method.
    
    Args:
        localize_func: The localization function to test.
        variances: The array of noise variances to test.
        measurement_simulator: A function that generates noisy measurements.
        
    Returns:
        A list of RMSE values, one for each variance.
    """
    rmse_results = []
    
    # Calculate true measurements once
    d_true = {
        'd1': np.linalg.norm(P_TRUE - A1),
        'd2': np.linalg.norm(P_TRUE - A2),
        'd3': np.linalg.norm(P_TRUE - A3),
        'theta1': np.arctan2(P_TRUE[1] - A1[1], P_TRUE[0] - A1[0]),
        'theta2': np.arctan2(P_TRUE[1] - A2[1], P_TRUE[0] - A2[0]),
    }
    
    for var in variances:
        std_dev = np.sqrt(var)
        squared_errors = []
        
        for _ in range(NUM_RUNS):
            # 1. Simulate noisy measurements
            measurements = measurement_simulator(d_true, std_dev)
            
            # 2. Run localization
            P_est = localize_func(**measurements)
            
            # 3. Calculate and store error
            # np.isnan(P_est).any() checks if *any* element is NaN
            if not np.isnan(P_est).any():
                error = np.linalg.norm(P_est - P_TRUE)
                squared_errors.append(error**2)
        
        # Calculate RMSE, ignoring failed runs (NaNs)
        if squared_errors:
            rmse = np.sqrt(np.mean(squared_errors))
            rmse_results.append(rmse)
        else:
            rmse_results.append(np.nan) # All runs failed
            
    return rmse_results

# --- 4. Measurement Simulator Functions ---
# These functions generate the noisy measurements for each method.

def sim_rssi_measurements(d_true: Dict, std_dev: float) -> Dict:
    RSSI1_true = RSSI_Pt - 10 * RSSI_n * np.log10(d_true['d1'])
    RSSI2_true = RSSI_Pt - 10 * RSSI_n * np.log10(d_true['d2'])
    RSSI3_true = RSSI_Pt - 10 * RSSI_n * np.log10(d_true['d3'])
    
    RSSI1_meas = RSSI1_true + np.random.randn() * std_dev
    RSSI2_meas = RSSI2_true + np.random.randn() * std_dev
    RSSI3_meas = RSSI3_true + np.random.randn() * std_dev
    
    return {
        'd1_est': 10**((RSSI_Pt - RSSI1_meas) / (10 * RSSI_n)),
        'd2_est': 10**((RSSI_Pt - RSSI2_meas) / (10 * RSSI_n)),
        'd3_est': 10**((RSSI_Pt - RSSI3_meas) / (10 * RSSI_n)),
    }

def sim_toa_measurements(d_true: Dict, std_dev: float) -> Dict:
    return {
        'd1': d_true['d1'] + np.random.randn() * std_dev,
        'd2': d_true['d2'] + np.random.randn() * std_dev,
        'd3': d_true['d3'] + np.random.randn() * std_dev,
    }

def sim_tdoa_measurements(d_true: Dict, std_dev: float) -> Dict:
    d_diff_21_true = d_true['d2'] - d_true['d1']
    d_diff_31_true = d_true['d3'] - d_true['d1']
    return {
        'd_diff_21': d_diff_21_true + np.random.randn() * std_dev,
        'd_diff_31': d_diff_31_true + np.random.randn() * std_dev,
    }

def sim_aoa_measurements(d_true: Dict, std_dev: float) -> Dict:
    return {
        'theta1': d_true['theta1'] + np.random.randn() * std_dev,
        'theta2': d_true['theta2'] + np.random.randn() * std_dev,
    }

# --- 5. Plotting Function ---

def plot_results(results: Dict):
    """Plots the 2x2 grid of RMSE vs. Noise Variance."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Noise Profiling: RMSE (Location) vs. Noise Variance', fontsize=16)

    axs[0, 0].plot(results['rssi']['variances'], results['rssi']['rmse'], 'r-o')
    axs[0, 0].set(title='RSSI Localization', 
                  xlabel='Noise Variance (dB^2)', 
                  ylabel='Location RMSE (meters)')
    
    axs[0, 1].plot(results['toa']['variances'], results['toa']['rmse'], 'b-o')
    axs[0, 1].set(title='TOA Localization', 
                  xlabel='Noise Variance (meters^2)', 
                  ylabel='Location RMSE (meters)')

    axs[1, 0].plot(results['tdoa']['variances'], results['tdoa']['rmse'], 'g-o')
    axs[1, 0].set(title='TDOA Localization', 
                  xlabel='Noise Variance (meters^2)', 
                  ylabel='Location RMSE (meters)')

    axs[1, 1].plot(results['aoa']['variances'], results['aoa']['rmse'], 'm-o')
    axs[1, 1].set(title='AOA Localization (2 Anchors)', 
                  xlabel='Noise Variance (radians^2)', 
                  ylabel='Location RMSE (meters)')

    for ax in axs.flat:
        ax.grid(True)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 6. Main Execution ---

def main():
    """Main function to run all simulations and plot results."""
    # Suppress runtime warnings (e.g., from log10(negative numbers in RSSI))
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    print("Running Monte Carlo simulations...")
    start_time = time.time()

    # Define the noise levels for each method
    rssi_variances = np.linspace(0.1, 5.0, N_STEPS)
    toa_variances = np.linspace(0.01, 1.0, N_STEPS)
    tdoa_variances = np.linspace(0.01, 1.0, N_STEPS)
    aoa_variances = np.linspace(0.001, 0.05, N_STEPS)

    # Run simulations
    results = {
        'rssi': {
            'rmse': run_profiling(localize_rssi, rssi_variances, sim_rssi_measurements),
            'variances': rssi_variances
        },
        'toa': {
            'rmse': run_profiling(localize_toa, toa_variances, sim_toa_measurements),
            'variances': toa_variances
        },
        'tdoa': {
            'rmse': run_profiling(localize_tdoa, tdoa_variances, sim_tdoa_measurements),
            'variances': tdoa_variances
        },
        'aoa': {
            'rmse': run_profiling(localize_aoa, aoa_variances, sim_aoa_measurements),
            'variances': aoa_variances
        }
    }
    
    print(f"Simulations complete in {time.time() - start_time:.2f} seconds.")
    
    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main()