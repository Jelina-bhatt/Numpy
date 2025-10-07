import numpy as np

# Number of city zones (e.g., residential, commercial, industrial)
zones = 6

# Generate random data for energy generation (MW) and consumption (MW)
generation = np.random.randint(300, 800, size=zones)
consumption = np.random.randint(400, 900, size=zones)

print("🔋 Energy Generation (MW):", generation)
print("💡 Energy Consumption (MW):", consumption)

# Net energy per zone (positive = surplus, negative = shortage)
net_energy = generation - consumption
print("\n⚖️ Net Energy per Zone:", net_energy)

# Identify surplus and deficit zones
surplus_zones = np.where(net_energy > 0)[0]
deficit_zones = np.where(net_energy < 0)[0]

print("\n✅ Surplus Zones:", surplus_zones)
print("⚠️ Deficit Zones:", deficit_zones)

# Calculate total surplus and deficit
total_surplus = np.sum(net_energy[net_energy > 0])
total_deficit = np.sum(np.abs(net_energy[net_energy < 0]))

# Ratio to distribute available energy
transfer_ratio = min(1, total_surplus / total_deficit) if total_deficit != 0 else 0

# Balance energy — proportional redistribution
balanced_energy = np.copy(net_energy).astype(float)
for dz in deficit_zones:
    required = abs(net_energy[dz]) * transfer_ratio
    if total_surplus > 0:
        share = required / len(surplus_zones)
        balanced_energy[dz] += required
        balanced_energy[surplus_zones] -= share

print("\n🔁 Energy Balanced Between Zones:")
for i, val in enumerate(balanced_energy):
    status = (
        "⚡ Stable" if -50 <= val <= 50
        else ("✅ Surplus" if val > 50 else "⚠️ Shortage")
    )
    print(f"Zone {i+1}: {round(val, 2)} MW → {status}")

# Compute citywide energy status
city_total = np.sum(balanced_energy)
if abs(city_total) <= 50:
    print("\n🏙️ City Power Grid is Stable ✅")
elif city_total > 0:
    print("\n🔆 City has Surplus Energy ⚡ (Can share with nearby cities)")
else:
    print("\n⚠️ City Facing Power Shortage! Need to Import Energy")
