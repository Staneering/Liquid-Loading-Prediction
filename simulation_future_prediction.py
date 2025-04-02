import numpy as np
import matplotlib.pyplot as plt

# Constants
sigma = 0.02  # Surface tension (N/m)
d = 0.0508  # Tubing diameter (m)
conversion_factor = 0.0283168 / (24 * 3600)  # Convert MMSCFD to m³/s

# Load dataset
data = pd.read_csv("gas_well_data.csv")

# Define decline parameters
Q_g0 = data["Q_g (MMSCFD) (t)"].mean()
D = 0.02
time_days = np.arange(0, 365 * 5, 30)

# Convert initial gas rate to m³/s
Q_g0_m3s = Q_g0 * conversion_factor

# Calculate future gas flow rate
Q_g_future = Q_g0_m3s * np.exp(-D * (time_days / 30))

# Compute gas velocity
V_g_future = (4 * Q_g_future) / (np.pi * d**2)

# Compute Turner's critical velocity
rho_g_mean = (1 / data["ρ_l/ρ_g (t)"]).mean()
V_c_future = 67.6 * (sigma / rho_g_mean) ** 0.25

# Plot results
plt.plot(time_days, V_g_future, label="Gas Velocity (V_g)")
plt.axhline(V_c_future, color="r", linestyle="--", label="Turner’s Critical Velocity (V_c)")
plt.xlabel("Time (Days)")
plt.ylabel("Velocity (m/s)")
plt.title("Future Prediction of Liquid Loading Occurrence")
plt.legend()
plt.grid(True)
plt.show()
