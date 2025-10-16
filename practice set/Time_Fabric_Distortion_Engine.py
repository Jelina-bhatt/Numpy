"""
Time Fabric Distortion Engine
--------------------------------
Simulates particles moving under a central mass whose gravitational potential
slows local proper time. Particles near the mass therefore "evolve" slower
relative to a distant observer.

Usage:
    python time_fabric_engine.py

Optional:
    - Edit parameters (G, M, k_time, dt, num_particles, frames) to experiment.
    - To export MP4: ensure ffmpeg is installed.
    - To export GIF without ffmpeg: uncomment the imageio section and install imageio.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse

def run_simulation(save_mp4=False, save_gif=False, outname="time_fabric_output"):
    # ----- Simulation parameters -----
    G = 1.0                # gravitational constant (scaled)
    M = 200.0              # central mass
    k_time = 0.9           # time-dilation strength (how strongly potential affects local time)
    dt = 0.03              # global timestep (observer time)
    num_particles = 60     # number of test particles
    frames = 350           # number of animation frames to generate
    lim = 8.0              # plot limits (x and y range)
    grid_res = 160         # resolution for background time-field visualization
    max_trail = 80         # how many past positions to show per particle

    # ----- Initialize particles (ring with perturbation) -----
    theta = np.linspace(0, 2*np.pi, num_particles, endpoint=False)
    radii = 3.0 + 0.6 * np.random.randn(num_particles)
    pos = np.column_stack([radii * np.cos(theta), radii * np.sin(theta)])
    speed = np.sqrt(G * M / np.maximum(radii, 0.1))
    vel = np.column_stack([-speed * np.sin(theta), speed * np.cos(theta)])
    vel += 0.06 * np.random.randn(num_particles, 2)  # small randomness

    trails = [ [] for _ in range(num_particles) ]

    # ----- Precompute time-field for background -----
    xg = np.linspace(-lim, lim, grid_res)
    yg = np.linspace(-lim, lim, grid_res)
    X, Y = np.meshgrid(xg, yg)
    R = np.sqrt(X**2 + Y**2) + 1e-9
    potential = -G * M / R
    # proper-time flow factor: tau_dot = 1 / (1 + k * |potential|)
    time_field = 1.0 / (1.0 + k_time * np.abs(potential))
    time_field = np.clip(time_field, 0.01, 1.0)  # avoid extremes

    # ----- Integrator step: Euler with local time scaling -----
    def step(p, v, dt_sub):
        r = np.linalg.norm(p, axis=1, keepdims=True) + 1e-9
        r_vec = p
        acc = -G * M * r_vec / (r**3)
        pot_p = -G * M / r.flatten()
        time_scale = 1.0 / (1.0 + k_time * np.abs(pot_p))
        time_scale = np.clip(time_scale, 0.01, 1.0).reshape(-1,1)
        v = v + acc * (dt_sub * time_scale)
        p = p + v * (dt_sub * time_scale)
        return p, v, time_scale.flatten()

    # ----- Matplotlib figure -----
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', 'box')
    ax.set_title("Time Fabric Distortion Engine — observer-time view")

    # show background time field (do not explicitly set color map)
    bg = ax.imshow(time_field, origin='lower', extent=[-lim, lim, -lim, lim], alpha=0.55, zorder=0)

    # central mass marker
    central_marker, = ax.plot(0, 0, marker='o', markersize=12)

    # particles + trails
    scat = ax.scatter(pos[:,0], pos[:,1], s=20, zorder=3)
    lines = [ax.plot([], [], linewidth=1.0, alpha=0.9, zorder=2)[0] for _ in range(num_particles)]

    # colorbar for the time-field
    cb = fig.colorbar(bg, ax=ax, shrink=0.85)
    cb.set_label("local proper time flow (1 = no dilation)")

    # Animation helpers
    def init():
        scat.set_offsets(pos)
        for line in lines:
            line.set_data([], [])
        return [scat, *lines]

    def animate(i):
        nonlocal pos, vel, trails
        substeps = 3
        for _ in range(substeps):
            pos, vel, scales = step(pos, vel, dt/substeps)
        for idx in range(num_particles):
            trails[idx].append(pos[idx].copy())
            if len(trails[idx]) > max_trail:
                trails[idx].pop(0)
        scat.set_offsets(pos)
        for idx, line in enumerate(lines):
            if trails[idx]:
                arr = np.array(trails[idx])
                line.set_data(arr[:,0], arr[:,1])
            else:
                line.set_data([], [])
        return [scat, *lines]

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=32, blit=False)

    # Display and optionally save
    plt.tight_layout()

    if save_mp4:
        # requires ffmpeg available in PATH
        print("Saving MP4 (requires ffmpeg)... this may take a while.")
        anim.save(f"{outname}.mp4", writer="ffmpeg", fps=30, dpi=160)
        print(f"Saved: {outname}.mp4")

    if save_gif:
        # lightweight GIF export using imageio (no ffmpeg required)
        try:
            import imageio
            print("Exporting GIF frames and assembling (imageio). This may take a bit.")
            frames_list = []
            for frame_num in range(frames):
                animate(frame_num)
                fig.canvas.draw()
                # grab the RGBA buffer from the figure
                w, h = fig.canvas.get_width_height()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
                frames_list.append(image)
            imageio.mimsave(f"{outname}.gif", frames_list, fps=30)
            print(f"Saved: {outname}.gif")
        except Exception as e:
            print("GIF export failed (imageio missing or other error):", e)

    try:
        plt.show()
    except Exception:
        # In some environments (headless), plt.show might fail; try saving a first frame image
        print("Plot display failed (headless environment). You can enable save_mp4 or save_gif to export.")

    return anim, fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Time Fabric Distortion Engine animation.")
    parser.add_argument("--mp4", action="store_true", help="Save animation as MP4 (requires ffmpeg).")
    parser.add_argument("--gif", action="store_true", help="Save animation as GIF (uses imageio).")
    parser.add_argument("--out", type=str, default="time_fabric_output", help="Base name for output files.")
    args = parser.parse_args()
    run_simulation(save_mp4=args.mp4, save_gif=args.gif, outname=args.out)
