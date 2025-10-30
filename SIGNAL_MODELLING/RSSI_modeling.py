# --- 1. RSSI Localization Model (Python) ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Setup Scenario
A1 = np.array([0, 0])
A2 = np.array([10, 0])
A3 = np.array([5, 15])
P_true = np.array([7, 8])  # True position of the unknown node

# RSSI Model Parameters
Pt = -30  # Transmitted power (or ref. power) at 1 meter (in dBm)
n = 2.5   # Path loss exponent
rssi_noise_std = 2.0  # Standard deviation of RSSI noise (in dB)

# 2. Simulate Measurements
# Calculate true distances
d1_true = np.linalg.norm(P_true - A1)
d2_true = np.linalg.norm(P_true - A2)
d3_true = np.linalg.norm(P_true - A3)

# Calculate true RSSI at each anchor using the log-distance model
RSSI1_true = Pt - 10 * n * np.log10(d1_true)
RSSI2_true = Pt - 10 * n * np.log10(d2_true)
RSSI3_true = Pt - 10 * n * np.log10(d3_true)

# Simulate measured RSSI by adding noise
RSSI1_meas = RSSI1_true + np.random.randn() * rssi_noise_std
RSSI2_meas = RSSI2_true + np.random.randn() * rssi_noise_std
RSSI3_meas = RSSI3_true + np.random.randn() * rssi_noise_std

# Convert measured RSSI back into distance estimates
d1_est = 10**((Pt - RSSI1_meas) / (10 * n))
d2_est = 10**((Pt - RSSI2_meas) / (10 * n))
d3_est = 10**((Pt - RSSI3_meas) / (10 * n))

print(f'True Distances:     d1={d1_true:.2f}, d2={d2_true:.2f}, d3={d3_true:.2f}')
print(f'Estimated Distances: d1={d1_est:.2f}, d2={d2_est:.2f}, d3={d3_est:.2f}')

# 3. Localize using Least-Squares Optimization
# We need to find the (x,y) that minimizes the error.
def error_func(P):
    # P is a 2-element array [x, y]
    d1 = np.linalg.norm(P - A1)
    d2 = np.linalg.norm(P - A2)
    d3 = np.linalg.norm(P - A3)
    # Sum of Squared Errors
    return (d1 - d1_est)**2 + (d2 - d2_est)**2 + (d3 - d3_est)**2

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
for i, A in enumerate([A1, A2, A3]):
    ax.text(A[0] + 0.2, A[1], f' A{i+1}')

ax.plot(P_true[0], P_true[1], 'go', markersize=10, label='True Position')
ax.plot(P_est[0], P_est[1], 'rx', markersize=12, label='Estimated Position')

# Plot the estimated circles
circle1 = plt.Circle(A1, d1_est, color='k', linestyle='--', fill=False, label='Est. Range Circles')
circle2 = plt.Circle(A2, d2_est, color='k', linestyle='--', fill=False)
circle3 = plt.Circle(A3, d3_est, color='k', linestyle='--', fill=False)
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)

ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_title(f'RSSI Localization (Est. Pos: [{P_est[0]:.2f}, {P_est[1]:.2f}])')
ax.legend()
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
plt.show()