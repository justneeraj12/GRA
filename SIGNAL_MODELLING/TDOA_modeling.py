# --- 3. TDOA Localization Model (Python) ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Setup Scenario
A1 = np.array([0, 0])  # Anchor 1 (Reference)
A2 = np.array([10, 0]) # Anchor 2
A3 = np.array([5, 15]) # Anchor 3
P_true = np.array([7, 8]) # True position of the unknown node

# Noise in the *difference* measurement (in meters)
tdoa_noise_std = 0.5

# 2. Simulate Measurements
# Calculate true distances
d1_true = np.linalg.norm(P_true - A1)
d2_true = np.linalg.norm(P_true - A2)
d3_true = np.linalg.norm(P_true - A3)

# Calculate true distance *differences* relative to A1
d_diff_21_true = d2_true - d1_true
d_diff_31_true = d3_true - d1_true

# Simulate measured distance differences by adding noise
d_diff_21_meas = d_diff_21_true + np.random.randn() * tdoa_noise_std
d_diff_31_meas = d_diff_31_true + np.random.randn() * tdoa_noise_std

print(f'True Diffs:     d2-d1={d_diff_21_true:.2f}, d3-d1={d_diff_31_true:.2f}')
print(f'Measured Diffs: d2-d1={d_diff_21_meas:.2f}, d3-d1={d_diff_31_meas:.2f}')

# 3. Localize using Least-Squares Optimization
# We need to find the (x,y) that minimizes the error.
def error_func(P):
    # P is a 2-element array [x, y]
    d1 = np.linalg.norm(P - A1)
    d2 = np.linalg.norm(P - A2)
    d3 = np.linalg.norm(P - A3)
    
    # Calculate measured distance differences
    d_diff_21 = d2 - d1
    d_diff_31 = d3 - d1
    
    # Sum of Squared Errors
    return (d_diff_21 - d_diff_21_meas)**2 + (d_diff_31 - d_diff_31_meas)**2

# Initial guess for the position
P_guess = (A1 + A2 + A3) / 3

# Run the optimization
result = minimize(error_func, P_guess, method='L-BFGS-B')
P_est = result.x

# 4. Plot Results
fig, ax = plt.subplots()
ax.plot(A1[0], A1[1], 'b^', markersize=10, label='Anchors')
ax.plot(A2[0], A2[1], 'b^', markersize=10)
ax.plot(A3[0], A3[1], 'b^', markersize=10)
ax.text(A1[0] + 0.2, A1[1], ' A1 (Ref)')
ax.text(A2[0] + 0.2, A2[1], ' A2')
ax.text(A3[0] + 0.2, A3[1], ' A3')

ax.plot(P_true[0], P_true[1], 'go', markersize=10, label='True Position')
ax.plot(P_est[0], P_est[1], 'rx', markersize=12, label='Estimated Position')

# Plotting hyperbolas is complex; we'll omit for this basic demo

ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_title(f'TDOA Localization (Est. Pos: [{P_est[0]:.2f}, {P_est[1]:.2f}])')
ax.legend()
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
plt.show()