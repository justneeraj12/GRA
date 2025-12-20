import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1. Setup Scenario (Fig 1 in Paper) ---
AREA_SIZE = 100
NUM_SENSORS = 30
# Two sources at (60, 80) and (30, 40) as per Paper Fig 3 [cite: 266]
TRUE_SOURCES = np.array([[60, 80], [30, 40]])
SOURCE_POWERS = np.array([10, 10]) # Arbitrary power units
NOISE_VAR = 1.0 # Sigma_v^2
PATH_LOSS_ALPHA = 2.5 # Path loss exponent

np.random.seed(42)
# Random Sensor Locations
sensors = np.random.rand(NUM_SENSORS, 2) * AREA_SIZE

# --- 2. Simulate Received Power (Eq 1 in Paper) ---
def get_model_D(target_pos, sensors):
    # Calculates the distance-based decay vector (d_k) for a target
    # d_ik^(-alpha)
    diff = sensors - target_pos
    dists = np.linalg.norm(diff, axis=1)
    # Avoid div by zero
    dists[dists < 1] = 1
    return dists ** (-PATH_LOSS_ALPHA)

# Generate Synthetic Measurements (p vector)
# p = D * s + w
D_true = np.column_stack([get_model_D(src, sensors) for src in TRUE_SOURCES])
p_clean = D_true @ SOURCE_POWERS
p_measured = p_clean + np.random.randn(NUM_SENSORS) * np.sqrt(NOISE_VAR)

# --- 3. The Sequential Algorithm (Table I in Paper) ---

def calculate_projection(D_matrix):
    """Calculates Orthogonal Complement D_perp [cite: 117]"""
    if D_matrix is None or D_matrix.shape[1] == 0:
        return np.eye(NUM_SENSORS)
    
    # P = I - D(D^T D)^-1 D^T
    # We use pseudo-inverse for stability
    D_pinv = np.linalg.pinv(D_matrix)
    Projection = D_matrix @ D_pinv
    return np.eye(NUM_SENSORS) - Projection

found_sources = []
D_current = np.zeros((NUM_SENSORS, 0)) # Start with empty matrix

# We will search for up to 3 sources
for step in range(1, 4):
    print(f"--- Step {step}: Searching for Source #{step} ---")
    
    # 1. Calculate Null Space of previous sources
    P_perp = calculate_projection(D_current)
    
    # 2. GLRT Maximization: Find pos (x,y) that maximizes energy in the null space
    # L_m(p) equation [cite: 113]
    def negative_likelihood(pos):
        d_k = get_model_D(pos, sensors)
        
        # Numerator: |d_k^T * P_perp * p|^2
        # Denominator: d_k^T * P_perp * d_k
        # We maximize Fraction => Minimize Negative Fraction
        
        # Project measurement and new candidate signal
        p_proj = P_perp @ p_measured
        d_proj = P_perp @ d_k
        
        num = (d_k.T @ p_proj)**2
        den = d_k.T @ d_proj
        
        if den < 1e-9: return 0 # Avoid singularity
        return - (num / den)

    # Search for the best location (Grid search or Optimizer)
    # For speed here, we use a coarse grid then refine
    grid_x, grid_y = np.linspace(0, 100, 20), np.linspace(0, 100, 20)
    best_cost = np.inf
    best_guess = [50, 50]
    
    for x in grid_x:
        for y in grid_y:
            val = negative_likelihood([x, y])
            if val < best_cost:
                best_cost = val
                best_guess = [x, y]
                
    # Refine with optimizer
    res = minimize(negative_likelihood, best_guess, bounds=[(0, 100), (0, 100)])
    est_pos = res.x
    max_likelihood_val = -res.fun
    
    print(f"  Best Candidate Location: {est_pos.round(2)}")
    print(f"  GLRT Statistic: {max_likelihood_val:.2f}")
    
    # 3. Threshold Test (Hypothesis Testing)
    # Threshold usually depends on False Alarm Prob (P_fa) [cite: 219]
    # For this demo, we set a heuristic threshold
    THRESHOLD = 50.0 
    
    if max_likelihood_val > THRESHOLD:
        print("  >> Detection! Source Found.")
        found_sources.append(est_pos)
        
        # Update D_current with this new source's signature
        new_d_vec = get_model_D(est_pos, sensors).reshape(-1, 1)
        if D_current.shape[1] == 0:
            D_current = new_d_vec
        else:
            D_current = np.hstack([D_current, new_d_vec])
    else:
        print("  >> No significant source found. Stopping.")
        break

# --- 4. Plotting Results ---
plt.figure(figsize=(8, 8))
plt.style.use('seaborn-v0_8-whitegrid')
plt.scatter(sensors[:,0], sensors[:,1], c='k', marker='s', label='Sensors')
plt.scatter(TRUE_SOURCES[:,0], TRUE_SOURCES[:,1], c='g', s=150, marker='*', label='True Sources')

found_sources = np.array(found_sources)
if len(found_sources) > 0:
    plt.scatter(found_sources[:,0], found_sources[:,1], c='r', s=100, marker='x', label='Estimated Sources')

plt.xlim(0, 100); plt.ylim(0, 100)
plt.title(f"Sequential GLRT Localization (Found {len(found_sources)} Sources)")
plt.legend()
plt.show()