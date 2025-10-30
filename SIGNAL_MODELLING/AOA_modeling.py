# --- 4. AOA Localization Model (Python) ---
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Scenario
# We only need two anchors for AOA
A1 = np.array([0, 0])  # (x1, y1)
A2 = np.array([10, 0]) # (x2, y2)
P_true = np.array([7, 8]) # True position of the unknown node

# Noise in the angle measurement (in radians)
angle_noise_std = 0.05  # ~2.8 degrees

# 2. Simulate Measurements
# Calculate true angles (bearings) from anchors to the node
# We use arctan2(Y, X)
theta1_true = np.arctan2(P_true[1] - A1[1], P_true[0] - A1[0])
theta2_true = np.arctan2(P_true[1] - A2[1], P_true[0] - A2[0])

# Simulate measured angles by adding noise
theta1_meas = theta1_true + np.random.randn() * angle_noise_std
theta2_meas = theta2_true + np.random.randn() * angle_noise_std

# 3. Localize using Line Intersection
# Line 1: y - y1 = tan(theta1) * (x - x1)
# Line 2: y - y2 = tan(theta2) * (x - x2)

# Rearrange into a linear system A*P = b, where P = [x; y]
# -tan(theta1)*x + 1*y = y1 - tan(theta1)*x1
# -tan(theta2)*x + 1*y = y2 - tan(theta2)*x2

m1 = np.tan(theta1_meas)
m2 = np.tan(theta2_meas)

x1, y1 = A1
x2, y2 = A2

A = np.array([
    [-m1, 1],
    [-m2, 1]
])

b = np.array([
    y1 - m1*x1,
    y2 - m2*x2
])

# Solve for P = [x; y]
P_est = np.linalg.solve(A, b)

# 4. Plot Results
fig, ax = plt.subplots()
ax.plot(A1[0], A1[1], 'b^', markersize=10, label='Anchors')
ax.plot(A2[0], A2[1], 'b^', markersize=10)
for i, A in enumerate([A1, A2]):
    ax.text(A[0] + 0.2, A[1], f' A{i+1}')

ax.plot(P_true[0], P_true[1], 'go', markersize=10, label='True Position')
ax.plot(P_est[0], P_est[1], 'rx', markersize=12, label='Estimated Position')

# Plot the bearing lines
t = np.linspace(-20, 20, 100)
line1_x = A1[0] + t * np.cos(theta1_meas)
line1_y = A1[1] + t * np.sin(theta1_meas)
ax.plot(line1_x, line1_y, 'k--', label='Measured Bearings')

line2_x = A2[0] + t * np.cos(theta2_meas)
line2_y = A2[1] + t * np.sin(theta2_meas)
ax.plot(line2_x, line2_y, 'k--')

ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_title(f'AOA Localization (Est. Pos: [{P_est[0]:.2f}, {P_est[1]:.2f}])')
ax.legend()
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 20)
plt.show()