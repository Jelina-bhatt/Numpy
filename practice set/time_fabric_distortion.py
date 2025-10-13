import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize the Time Fabric (2D temporal plane)
size = 200
time_fabric = np.random.uniform(-1, 1, (size, size))

# Constants controlling the distortion
time_gravity = 0.04   # How time curves
singularity_pull = 0.015  # Collapse rate
resonance = 0.02      # Oscillation factor
stability = 0.97      # Memory of past timelines

def distort_time(fabric):
    # Local interactions = curvature between neighboring moments
    laplacian = (
        np.roll(fabric, 1, 0)
        + np.roll(fabric, -1, 0)
        + np.roll(fabric, 1, 1)
        + np.roll(fabric, -1, 1)
        - 4 * fabric
    )

    # Quantum temporal oscillations (resonance of time)
    resonance_wave = np.sin(fabric * np.pi * 2) * resonance

    # Singularity influence (self-collapse of local time)
    singularity = np.exp(-np.abs(fabric)) * singularity_pull

    # Update time fabric
    new_fabric = (
        stability * fabric +
        time_gravity * laplacian +
        resonance_wave - singularity
    )
    return np.clip(new_fabric, -1, 1)

# Visualization
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(time_fabric, cmap="cividis", animated=True)
ax.axis("off")
fig.patch.set_facecolor("black")
title = ax.set_title("🪐 Time Fabric Distortion Simulator", color="white", fontsize=13)

# Animation logic
def update(frame):
    global time_fabric
    time_fabric = distort_time(time_fabric)
    im.set_array(time_fabric)
    if frame % 60 == 0:
        im.set_cmap(np.random.choice(["cividis", "plasma", "twilight", "inferno", "magma"]))
        title.set_text("⏳ Temporal Phase " + str(frame))
    return [im]

ani = FuncAnimation(fig, update, frames=800, interval=40, blit=True)
plt.show()
