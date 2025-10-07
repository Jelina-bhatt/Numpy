import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------
# Simulation parameters
# -------------------------
nx, ny = 200, 200       # Grid size
dx, dy = 0.1, 0.1       # Space step
dt = 0.001              # Time step
steps = 300             # Number of time steps
hbar = 1.0              # Reduced Planck constant
m = 1.0                 # Particle mass

# -------------------------
# Potential (Barrier in center)
# -------------------------
V = np.zeros((nx, ny))
V[nx//2-5:nx//2+5, ny//2-50:ny//2+50] = 50.0  # Barrier

# -------------------------
# Initial wavefunction (Gaussian packet)
# -------------------------
x = np.linspace(0, nx*dx, nx)
y = np.linspace(0, ny*dy, ny)
X, Y = np.meshgrid(x, y)

x0, y0 = nx*dx*0.25, ny*dy*0.5
sigma = 1.0
k0 = 5.0  # Initial momentum

psi = np.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2)) * np.exp(1j*k0*X)
psi /= np.sqrt(np.sum(np.abs(psi)**2))  # Normalize

# -------------------------
# Laplacian function
# -------------------------
def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1)
    ) / (dx*dy)

# -------------------------
# Time evolution (Crank-Nicolson simplified)
# -------------------------
def evolve(psi, V):
    lap = laplacian(psi)
    return psi - 1j*dt*( - (hbar**2)/(2*m) * lap + V*psi)/hbar

# -------------------------
# Animation setup
# -------------------------
fig, ax = plt.subplots()
prob = np.abs(psi)**2
im = ax.imshow(prob, cmap='inferno', origin='lower', extent=[0, nx*dx, 0, ny*dy])
ax.set_title("Quantum Particle Probability Density")

def update(frame):
    global psi
    psi = evolve(psi, V)
    prob = np.abs(psi)**2
    im.set_array(prob)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=steps, blit=True, interval=50)
plt.show()
