3.2 Code Implementation
import numpy as np
import matplotlib.pyplot as plt
 
# ==== CLOUD STORAGE GROWTH MODEL ====
# Logistic Growth Model for Corporate Cloud Storage
 
def logistic_growth(S0, r, K, days):
    storage = [S0]
    for t in range(1, days + 1):
        S = storage[-1]
        dS = r * S * (1 - S / K)   # Logistic growth
        storage.append(S + dS)
    return storage
 
def find_expansion_day(storage, K, threshold=0.80):
    for day, s in enumerate(storage):
        if s >= threshold * K:
            return day
    return None
 
# === Baseline Scenario Parameters ===
S0   = 50       # Initial storage in GB
r    = 0.002    # Daily growth rate (0.2%)
K    = 5000     # Max capacity in GB
days = 730      # Simulate 2 years
 
# Run simulation
storage = logistic_growth(S0, r, K, days)
exp_day = find_expansion_day(storage, K)
 
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(range(days+1), storage, label='Storage Used (GB)', color='steelblue', linewidth=2)
plt.axhline(y=0.80*K, color='orange', linestyle='--', label='80% Capacity Alert')
plt.axhline(y=K,       color='red',    linestyle='--', label='Max Capacity')
plt.xlabel('Days'); plt.ylabel('Storage Used (GB)')
plt.title('Cloud Storage Growth - Logistic Model')
plt.legend(); plt.grid(True)
plt.savefig('cloud_storage_growth.png', dpi=150)
plt.show()
 
print(f'Storage on Day 730: {storage[730]:.2f} GB')
if exp_day:
    print(f'Expansion Alert: Day {exp_day} (80% capacity reached)')
else:
    print('Expansion not needed within simulation period')