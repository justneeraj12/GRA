# --- 2. TOA Localization Model (Python) ---
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Scenario
A1 = np.array([0, 0])  # (x1, y1)
A2 = np.array([10, 0]) # (x2, y2)
A3 = np.array([5, 15]) # (x3, y3)
P_true = np.array([7, 8]) # True position of the unknown node

dist_noise_std = 0.5  # Standard deviation of distance noise (in meters)

# 2. Simulate Measurements
# Calculate true distances
d1_true = np.linalg.norm(P_true - A1)
d2_true = np.linalg.norm(P_true - A2)
d3_true = np.linalg.norm(P_true - A3)

# Simulate measured distances (from TOA) by adding noise
d1 = d1_true + np.random.randn() * dist_noise_std
d2 = d2_true + np.random.randn() * dist_noise_std
d3 = d3_true + np.random.randn() * dist_noise_std

# 3. Localize using Linearized System
# (x - x1)^2 + (y - y1)^2 = d1^2
# (x - x2)^2 + (y - y2)^2 = d2^2
# (x - x3)^2 + (y - y3)^2 = d3^2

# Expand and subtract (1) from (2) and (1) from (3)
# This gives a linear system A*P = b, where P = [x; y]

x1, y1 = A1
x2, y2 = A2
x3, y3 = A3

A = np.array([
    [2*(x2 - x1), 2*(y2 - y1)],
    [2*(x3 - x1), 2*(y3 - y1)]
])

b = np.array([
    (d1**2 - d2**2) + (x2**2 - x1**2) + (y2**2 - y1**2),
    (d1**2 - d3**2) + (x3**2 - x1**2) + (y3**2 - y1**2)
])

# Solve for P = [x; y]
P_est = np.linalg.solve(A, b)

# 4. Plot Results
fig, ax = plt.subplots()
ax.plot(A1[0], A1[1], 'b^', markersize=10, label='Anchors')
ax.plot(A2[0], A2[1], 'b^', markersize=10)
ax.plot(A3[0], A3[1], 'b^', markersize=10)
for i, A in enumerate([A1, A2, A3]):
    ax.text(A[0] + 0.2, A[1], f' A{i+1}')

ax.plot(P_true[0], P_true[1], 'go', markersize=10, label='True Position')
ax.plot(P_est[0], P_est[1], 'rx', markersize=12, label='Estimated Position')

# Plot the circles from the *measured* distances
circle1 = plt.Circle(A1, d1, color='k', linestyle='--', fill=False, label='Measured Range Circles')
circle2 = plt.Circle(A2, d2, color='k', linestyle='--', fill=False)
circle3 = plt.Circle(A3, d3, color='k', linestyle='--', fill=False)
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)

ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_title(f'TOA Localization (Est. Pos: [{P_est[0]:.2f}, {P_est[1]:.2f}])')
ax.legend()
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
plt.show()