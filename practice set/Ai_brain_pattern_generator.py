
"""
AI Brain Pattern Generator — Emergent Memory Simulation
-------------------------------------------------------
Pure NumPy project (no matplotlib, no external libs)

Concept:
- Simulates a grid of "neurons" that learn and form stable memory patterns.
- The network evolves through Hebbian-like plasticity.
- Over time, random noise leads to self-organizing stable states (artificial 'memory').

Run:
    python brain_pattern_generator.py
"""

import numpy as np

# -------------------------
# CONFIG
# -------------------------
SIZE = 32          # 32x32 neuron grid
ITER = 500         # number of evolution steps
LEARN_RATE = 0.01  # connection strengthening rate
DECAY = 0.995      # memory decay rate
THRESHOLD = 0.5    # firing threshold
SEED = 42

# -------------------------
# INITIALIZATION
# -------------------------
np.random.seed(SEED)
# Initial neuron states (random on/off)
neurons = np.random.choice([0, 1], size=(SIZE, SIZE)).astype(float)

# Synaptic weights (connections between neurons)
# Each neuron has connections to its 8 neighbors
weights = np.random.rand(SIZE, SIZE, 3, 3) * 0.1  # local connections

# -------------------------
# UTILITY FUNCTIONS
# -------------------------
def local_activity(field):
    """Compute local weighted sum (like convolution) purely with NumPy."""
    padded = np.pad(field, pad_width=1, mode='wrap')
    new_field = np.zeros_like(field)
    for i in range(3):
        for j in range(3):
            new_field += padded[i:i+SIZE, j:j+SIZE] * weights[:, :, i, j]
    return new_field

def normalize_weights():
    """Prevent runaway connection growth."""
    global weights
    norm = np.linalg.norm(weights, axis=(2,3), keepdims=True)
    weights /= (norm + 1e-9)

# -------------------------
# MAIN LOOP
# -------------------------
for t in range(ITER):
    # Compute local activation
    activation = local_activity(neurons)

    # Neuron firing rule (sigmoid-like)
    neurons = (activation > THRESHOLD).astype(float)

    # Hebbian-like learning (fire together = strengthen connection)
    delta = LEARN_RATE * (neurons[:, :, None, None] * neurons[None, None, :, :])
    # Simplify to local (3x3 neighborhood only)
    for i in range(3):
        for j in range(3):
            # local offset in neurons
            rolled = np.roll(np.roll(neurons, i-1, axis=0), j-1, axis=1)
            weights[:, :, i, j] = DECAY * weights[:, :, i, j] + LEARN_RATE * neurons * rolled

    # Normalize weights occasionally
    if t % 50 == 0:
        normalize_weights()

    # Print summary every 100 steps
    if t % 100 == 0 or t == ITER - 1:
        mean_activity = neurons.mean()
        variance = np.var(neurons)
        pattern_hash = int(neurons.sum()) % 10000
        print(f"Step {t:4d}: Activity={mean_activity:.3f}, Variance={variance:.3f}, Pattern={pattern_hash}")

# -------------------------
# OUTPUT SUMMARY
# -------------------------
print("\nFinal emergent 'memory' pattern:")
print(neurons.astype(int))

# Save final pattern as a text file
np.savetxt("brain_pattern.txt", neurons, fmt="%d")

print("\nSaved final neuron grid to brain_pattern.txt")

