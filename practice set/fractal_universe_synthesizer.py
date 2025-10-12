import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 🪐 FRACTAL UNIVERSE SYNTHESIZER – Jelina Bhatt Edition
# ==========================================

# ---- Value Noise ---------------------------------------------------
def value_noise(width, height, grid_x=8, grid_y=8, seed=None):
    """
    Generates smooth value noise.
    """
    if seed is not None:
        np.random.seed(seed)

    grid_x = max(grid_x, 2)
    grid_y = max(grid_y, 2)

    nodes = np.random.rand(grid_y, grid_x)

    x = np.linspace(0, grid_x - 1, width)
    y = np.linspace(0, grid_y - 1, height)
    xi = np.floor(x).astype(int)
    yi = np.floor(y).astype(int)
    xf = x - xi
    yf = y - yi

    xi = np.clip(xi, 0, grid_x - 2)
    yi = np.clip(yi, 0, grid_y - 2)

    def fade(t): return 3 * t ** 2 - 2 * t ** 3

    xf = fade(xf)
    yf = fade(yf)

    result = np.zeros((height, width))
    for j in range(height):
        i0, i1 = xi, xi + 1
        j0 = np.clip(yi[j], 0, grid_y - 2)
        j1 = j0 + 1

        n00 = nodes[j0, i0]
        n10 = nodes[j0, i1]
        n01 = nodes[j1, i0]
        n11 = nodes[j1, i1]

        nx0 = n00 * (1 - xf) + n10 * xf
        nx1 = n01 * (1 - xf) + n11 * xf
        result[j] = nx0 * (1 - yf[j]) + nx1 * yf[j]

    return result


# ---- Fractal Brownian Motion ---------------------------------------
def fbm(width, height, octaves=5, lacunarity=2.0, gain=0.5, seed=None):
    """
    Generates fractal Brownian motion noise by summing octaves of value noise.
    """
    total = np.zeros((height, width))
    amplitude = 1.0
    frequency = 1.0
    for o in range(octaves):
        octave_noise = value_noise(width, height,
                                   grid_x=int(4 * frequency),
                                   grid_y=int(4 * frequency),
                                   seed=None if seed is None else seed + o)
        total += amplitude * octave_noise
        amplitude *= gain
        frequency *= lacunarity
    total -= total.min()
    total /= total.max() + 1e-8
    return total


# ---- Mandelbrot Fractal --------------------------------------------
def mandelbrot(width, height, zoom=1.0, move_x=-0.5, move_y=0.0, max_iter=200):
    x = np.linspace(-2.5 / zoom + move_x, 1.5 / zoom + move_x, width)
    y = np.linspace(-1.5 / zoom + move_y, 1.5 / zoom + move_y, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    M = np.full(C.shape, True, dtype=bool)
    img = np.zeros(C.shape)

    for i in range(max_iter):
        Z[M] = Z[M] ** 2 + C[M]
        escaped = np.abs(Z) > 2
        img[M & escaped] = i
        M[M & escaped] = False
    img = img / img.max()
    return img


# ---- Starfield -----------------------------------------------------
def add_stars(width, height, density=0.0015, seed=None):
    if seed is not None:
        np.random.seed(seed)
    stars = np.zeros((height, width))
    n_stars = int(width * height * density)
    xs = np.random.randint(0, width, n_stars)
    ys = np.random.randint(0, height, n_stars)
    brightness = np.random.rand(n_stars) ** 3 * 1.5
    stars[ys, xs] = brightness
    return stars


# ---- Universe Synthesizer -----------------------------------------
def synthesize_universe(width=800, height=800,
                        mandel_params=None,
                        fbm_octaves=5,
                        fbm_lacunarity=2.0,
                        fbm_gain=0.5,
                        blend=0.5,
                        star_density=0.0015,
                        seed=42,
                        colormap_name='plasma',
                        invert_mandel=False,
                        contrast=1.0):
    if mandel_params is None:
        mandel_params = dict(zoom=1.5, move_x=-0.5, move_y=0.0, max_iter=250)

    # Generate components
    noise = fbm(width, height, octaves=fbm_octaves,
                lacunarity=fbm_lacunarity, gain=fbm_gain, seed=seed)
    mandel = mandelbrot(width, height, **mandel_params)
    if invert_mandel:
        mandel = 1 - mandel

    # Combine noise + mandelbrot
    universe = (blend * noise + (1 - blend) * mandel)
    universe -= universe.min()
    universe /= universe.max() + 1e-8

    # Add stars
    stars = add_stars(width, height, density=star_density, seed=seed)
    result = np.clip(universe + stars, 0, 1)

    # Contrast boost
    result = np.power(result, contrast)
    return result


# ---- Visualization -------------------------------------------------
def show_universe(image, cmap='plasma', title="🌌 Fractal Universe Synthesizer"):
    plt.figure(figsize=(7, 7))
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title(title, color='white', fontsize=14, pad=12)
    plt.gcf().patch.set_facecolor('black')
    plt.tight_layout()
    plt.show()


# ---- Run -----------------------------------------------------------
if __name__ == "__main__":
    print("🪐 Generating your fractal universe, Jelina... please wait 🌠")

    W, H = 800, 800
    mandel_params = dict(zoom=1.8, move_x=-0.65, move_y=0.0, max_iter=250)

    image = synthesize_universe(
        width=W, height=H,
        mandel_params=mandel_params,
        fbm_octaves=6,
        fbm_lacunarity=2.2,
        fbm_gain=0.55,
        blend=0.55,
        star_density=0.0017,
        seed=2025,
        colormap_name='plasma',
        invert_mandel=False,
        contrast=1.1
    )

    show_universe(image, cmap='plasma', title="🌌 Fractal Universe Synthesizer – Jelina's Cosmic Canvas")
