import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# 🧠 Cognitive Universe Simulator – Fixed & Enhanced
# -----------------------------

emotion_intensity = float(input("Enter your emotional intensity (0.1 - 5.0): "))

# Number of thoughts (particles)
n_thoughts = 100

# Initial setup
positions = np.random.rand(n_thoughts, 2) * 10
velocities = (np.random.rand(n_thoughts, 2) - 0.5) * 0.3
emotions = np.random.uniform(-1, 1, n_thoughts)

# Parameters
dt = 0.1
attraction_strength = 0.015 * emotion_intensity
repulsion_strength = 0.025 / emotion_intensity
damping = 0.96

# Plot setup
fig, ax = plt.subplots(figsize=(6,6))
sc = ax.scatter(positions[:,0], positions[:,1],
                c=emotions, cmap='coolwarm', s=80, alpha=0.85, edgecolors='white', linewidths=0.5)
ax.set_facecolor('black')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("🧠 Cognitive Universe – Thought Particle Evolution", color='white', fontsize=12)

def update(frame):
    global positions, velocities
    
    # Pairwise difference and distance
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (100,100,2)
    dist = np.linalg.norm(diff, axis=2) + 1e-6  # (100,100)
    
    # Normalize direction vectors
    direction = diff / dist[:, :, np.newaxis]
    
    # Emotional connection (outer product)
    emotion_matrix = np.outer(emotions, emotions)
    
    # Combine forces
    attraction = -attraction_strength * emotion_matrix[:, :, np.newaxis] * direction
    repulsion = repulsion_strength * direction / (dist[:, :, np.newaxis] ** 2)
    
    # Net forces (sum over all interactions)
    forces = np.sum(attraction + repulsion, axis=1)
    
    # Update physics
    velocities += forces * dt
    velocities *= damping
    positions += velocities * dt
    
    # Boundary reflection
    for i in range(2):
        mask_low = positions[:, i] < 0
        mask_high = positions[:, i] > 10
        velocities[mask_low | mask_high, i] *= -1
        positions[:, i] = np.clip(positions[:, i], 0, 10)
    
    sc.set_offsets(positions)
    return sc,

# Run animation
ani = FuncAnimation(fig, update, frames=400, interval=40, blit=True)
plt.show()
