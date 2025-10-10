import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Ask user for dream theme
theme = input("Enter a dream theme (ocean, fire, forest, galaxy, storm, desert, abstract): ").strip().lower()

# Theme-based parameters (logical mapping)
themes = {
    "ocean": {"color": "Blues", "noise": 0.03, "smoothing": 0.18, "decay": 0.96},
    "fire": {"color": "inferno", "noise": 0.07, "smoothing": 0.10, "decay": 0.92},
    "forest": {"color": "Greens", "noise": 0.04, "smoothing": 0.20, "decay": 0.95},
    "galaxy": {"color": "magma", "noise": 0.05, "smoothing": 0.12, "decay": 0.93},
    "storm": {"color": "PuBuGn", "noise": 0.06, "smoothing": 0.14, "decay": 0.90},
    "desert": {"color": "copper", "noise": 0.04, "smoothing": 0.16, "decay": 0.94},
    "abstract": {"color": "plasma", "noise": 0.05, "smoothing": 0.15, "decay": 0.93},
}

# Use chosen theme or default to abstract
params = themes.get(theme, themes["abstract"])

# Initialize dream matrix
size = 100
dream = np.random.randn(size, size)

# Extract parameters
decay_rate = params["decay"]
noise_intensity = params["noise"]
smoothing_strength = params["smoothing"]
cmap = params["color"]

# Create figure
fig, ax = plt.subplots()
im = ax.imshow(dream, cmap=cmap, animated=True)
plt.title(f"🧠 AI Dream Pattern – {theme.capitalize()} Mode")
plt.axis("off")

def evolve(frame):
    global dream

    # Random thought noise
    noise = np.random.randn(size, size) * noise_intensity

    # Apply FFT low-pass filter (smoothing neural patterns)
    f = np.fft.fft2(dream)
    freq = np.fft.fftfreq(size)[:, None] ** 2 + np.fft.fftfreq(size)[None, :] ** 2
    low_pass = np.exp(-smoothing_strength * freq * size**2)
    dream = np.fft.ifft2(f * low_pass).real

    # Evolve dream with decay and noise
    dream = dream * decay_rate + noise

    # Normalize brightness
    dream = (dream - dream.min()) / (dream.max() - dream.min())

    im.set_array(dream)
    return [im]

# Animate evolving dreams
ani = animation.FuncAnimation(fig, evolve, frames=400, interval=40, blit=True)
plt.show()
