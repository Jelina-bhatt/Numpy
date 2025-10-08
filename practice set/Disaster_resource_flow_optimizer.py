import numpy as np

# Parameters
zones = 10               # Number of city zones
relief_centers = 3       # Number of supply bases
resources = ['Food', 'Water', 'Medicine']

# Random resource demand (per zone)
demand = np.random.randint(200, 800, size=(zones, len(resources)))

# Random resource stock at each relief center
supply = np.random.randint(1000, 2500, size=(relief_centers, len(resources)))

# Random distances between relief centers and zones (km)
distances = np.random.randint(5, 60, size=(relief_centers, zones))

print("🌐 Disaster Zones Demand (Units):\n", demand)
print("\n🏥 Relief Center Supplies (Units):\n", supply)
print("\n📍 Distance Matrix (km):\n", distances)

# Optimization: Compute cost (distance * demand/supply ratio)
cost_matrix = np.zeros((relief_centers, zones))

for i in range(relief_centers):
    for j in range(zones):
        ratio = np.sum(demand[j]) / np.sum(supply[i])
        cost_matrix[i, j] = distances[i, j] * ratio

# Select best relief center for each zone (minimum cost)
best_center = np.argmin(cost_matrix, axis=0)

# Allocate resources accordingly
allocations = np.zeros_like(demand, dtype=float)
for j in range(zones):
    center = best_center[j]
    allocations[j] = np.minimum(demand[j], supply[center])
    supply[center] -= allocations[j]

# Report results
print("\n🚁 Optimal Resource Distribution Plan:")
for j in range(zones):
    print(f"Zone {j+1} is served by Relief Center {best_center[j]+1}")
    print(f"  Allocated: Food={allocations[j,0]:.0f}, Water={allocations[j,1]:.0f}, Medicine={allocations[j,2]:.0f}\n")

# Analyze remaining supply and unmet demand
remaining_demand = np.maximum(demand - allocations, 0)
print("\n⚠️ Remaining Unmet Demand (if any):\n", remaining_demand)

remaining_supply = np.maximum(supply, 0)
print("\n🏦 Remaining Supplies at Centers:\n", remaining_supply)

# City-level summary
fulfilled = np.sum(allocations) / np.sum(demand) * 100
print(f"\n📈 Relief Efficiency: {fulfilled:.2f}% resources delivered successfully.")
if fulfilled > 90:
    print("✅ Excellent disaster response efficiency!")
elif fulfilled > 70:
    print("⚠️ Moderate efficiency, some areas need more attention.")
else:
    print("🚨 Low efficiency! Emergency response system needs improvement.")
