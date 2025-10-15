#!/usr/bin/env python3
"""
Global Resource Equilibrium Simulator (GRES)
NumPy-only simulation of energy, water, food flows across a grid of regions.

Save as: gres_simulator.py
Run: python gres_simulator.py
"""

import numpy as np
import argparse
import csv
import os
from textwrap import dedent

# ------------------------- DEFAULT CONFIG ----------------------------
DEFAULT_CONFIG = {
    "seed": 42,
    "grid_shape": (20, 20),  # rows, cols
    "initial_resource_scale": {"energy": 500.0, "water": 1000.0, "food": 800.0},
    "population_scale": 50.0,
    "pop_growth_rate": 0.01,
    "consumption_per_capita": {"energy": 1.2, "water": 2.5, "food": 1.0},
    "regeneration_rates": {"energy": 0.01, "water": 0.02, "food": 0.015},
    "diffusion_strength": 0.1,
    "disaster_chance": 0.08,
    "tech_progress_chance": 0.03,
    "iterations": 20,
    "logging_interval": 1,
    "output_dir": "gres_output"
}

# ------------------------- UTIL / MODEL HELPERS ------------------------
def stack_resources(eng, wat, foo):
    return np.stack([eng, wat, foo], axis=0)  # shape (3, rows, cols)

def diffuse_layer(layer, cooperation, base_strength):
    """
    Diffuse a single resource layer using 4-neighbor mean and local cooperation-scaled strength.
    """
    up = np.roll(layer, -1, axis=0)
    down = np.roll(layer, 1, axis=0)
    left = np.roll(layer, -1, axis=1)
    right = np.roll(layer, 1, axis=1)
    neighbor_mean = (up + down + left + right) / 4.0
    # local strength varies with cooperation
    strength = base_strength * (0.5 + cooperation)
    return layer + strength * (neighbor_mean - layer)

def sustainability_index(resources, population, consumption_per_capita):
    """
    resources: (3, rows, cols)
    population: (rows, cols)
    consumption_per_capita: (3,1,1)
    Returns: (global_index, cell_index_map)
    """
    per_capita = resources / (population + 1e-6)
    ratio = np.clip(per_capita / consumption_per_capita, 0.0, 5.0)
    cell_index = ratio.mean(axis=0)
    g_index = np.clip((cell_index - 0.0) / (5.0 - 0.0), 0.0, 1.0).mean()
    return float(g_index), cell_index

def ascii_map(cell_index):
    chars = np.full(cell_index.shape, " ")
    chars[cell_index >= 0.9] = "█"
    chars[(cell_index >= 0.7) & (cell_index < 0.9)] = "▓"
    chars[(cell_index >= 0.5) & (cell_index < 0.7)] = "▒"
    chars[(cell_index >= 0.3) & (cell_index < 0.5)] = "░"
    chars[cell_index < 0.3] = "."
    return ["".join(row) for row in chars]

# ------------------------- EVENT SYSTEM --------------------------------
def trigger_random_event(resources, cooperation, config):
    """
    resources: (3, rows, cols)
    cooperation: (rows, cols)
    Returns (event_log, tech_log)
    """
    rows, cols = resources.shape[1], resources.shape[2]
    event_log = None
    if np.random.rand() < config["disaster_chance"]:
        choices = ["drought", "flood", "blackout", "locust"]
        event = np.random.choice(choices, p=[0.4, 0.2, 0.2, 0.2])
        r = np.random.randint(0, rows)
        c = np.random.randint(0, cols)
        radius = np.random.randint(1, max(2, min(rows, cols)//6))
        rr = np.arange(rows)[:, None]
        cc = np.arange(cols)[None, :]
        mask = (rr - r)**2 + (cc - c)**2 <= radius**2
        if event == "drought":
            resources[1][mask] *= np.random.uniform(0.4, 0.7)
            event_log = f"Drought near ({r},{c}) radius {radius}: water reduced"
        elif event == "flood":
            resources[1][mask] *= np.random.uniform(1.2, 1.6)
            resources[2][mask] *= np.random.uniform(0.7, 0.95)
            event_log = f"Flood near ({r},{c}) radius {radius}: water increased, food slightly damaged"
        elif event == "blackout":
            resources[0][mask] *= np.random.uniform(0.3, 0.6)
            event_log = f"Blackout near ({r},{c}) radius {radius}: energy sharply reduced"
        elif event == "locust":
            resources[2][mask] *= np.random.uniform(0.3, 0.6)
            event_log = f"Locust swarm near ({r},{c}) radius {radius}: food sharply reduced"

    tech_log = None
    if np.random.rand() < config["tech_progress_chance"]:
        boost = 1.0 + np.random.uniform(0.02, 0.1) * cooperation
        resources *= boost[np.newaxis, :, :]
        tech_log = "Global tech progress: small efficiency boost applied scaled by cooperation."

    return event_log, tech_log

# ------------------------- SIMULATOR CORE -------------------------------
def run_simulation(config):
    np.random.seed(config["seed"])
    rows, cols = config["grid_shape"]
    shape = (rows, cols)

    # initialize
    energy = np.random.normal(loc=config["initial_resource_scale"]["energy"], scale=80.0, size=shape)
    water = np.random.normal(loc=config["initial_resource_scale"]["water"], scale=150.0, size=shape)
    food = np.random.normal(loc=config["initial_resource_scale"]["food"], scale=120.0, size=shape)
    energy = np.clip(energy, a_min=10.0, a_max=None)
    water = np.clip(water, a_min=20.0, a_max=None)
    food = np.clip(food, a_min=15.0, a_max=None)

    population = np.random.poisson(lam=config["population_scale"], size=shape).astype(float)
    population = np.clip(population, a_min=1.0, a_max=None)

    cooperation = np.random.rand(*shape) * 0.6 + 0.2

    resources = stack_resources(energy, water, food)
    consumption_per_capita = np.array([config["consumption_per_capita"]["energy"],
                                       config["consumption_per_capita"]["water"],
                                       config["consumption_per_capita"]["food"]]).reshape(3,1,1)
    reg_rates = np.array([config["regeneration_rates"]["energy"],
                          config["regeneration_rates"]["water"],
                          config["regeneration_rates"]["food"]]).reshape(3,1,1)

    history = []
    os.makedirs(config["output_dir"], exist_ok=True)

    for it in range(1, config["iterations"] + 1):
        # 1) population growth (limited by food per capita)
        per_capita_food = resources[2] / (population + 1e-6)
        food_scarcity_factor = np.clip(per_capita_food / consumption_per_capita[2,0,0], 0.0, 2.0)
        growth = config["pop_growth_rate"] * (food_scarcity_factor - 0.5)
        population = population * (1.0 + growth)
        population = np.clip(population, a_min=0.5, a_max=None)

        # 2) consumption
        total_consumption = consumption_per_capita * population
        resources = resources - total_consumption
        resources = np.where(resources < 0.0, 0.0, resources)

        # 3) diffusion / redistribution
        for idx in range(resources.shape[0]):
            resources[idx] = diffuse_layer(resources[idx], cooperation, config["diffusion_strength"])
            resources[idx] = np.clip(resources[idx], 0.0, None)

        # 4) regeneration
        resources = resources + reg_rates * resources

        # 5) events & tech
        event_log, tech_log = trigger_random_event(resources, cooperation, config)

        # 6) sustainability
        g_index, cell_index = sustainability_index(resources, population, consumption_per_capita)
        history.append({
            "iteration": it,
            "global_index": g_index,
            "total_population": float(population.sum()),
            "total_energy": float(resources[0].sum()),
            "total_water": float(resources[1].sum()),
            "total_food": float(resources[2].sum()),
            "event": event_log,
            "tech": tech_log
        })

        # logging / ascii map
        if it % config["logging_interval"] == 0:
            print(dedent(f"""
            ========== Iteration {it} =========
            Global Sustainability Index: {g_index:.4f}
            Total Population: {population.sum():.0f}
            Total Energy: {resources[0].sum():.0f}
            Total Water: {resources[1].sum():.0f}
            Total Food: {resources[2].sum():.0f}
            Event: {event_log or 'None'}
            Tech: {tech_log or 'None'}
            ----------------------------------
            Local Sustainability Map (ASCII):"""))
            ascii_lines = ascii_map(cell_index)
            downsample = max(1, rows // 20)
            for r in range(0, rows, downsample):
                print("".join(ascii_lines[r][c] for c in range(0, cols, downsample)))
            print("="*40)

    # save outputs
    np.save(os.path.join(config["output_dir"], "final_resources.npy"), resources)
    np.save(os.path.join(config["output_dir"], "final_population.npy"), population)

    # save history as CSV
    csv_path = os.path.join(config["output_dir"], "history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration","global_index","total_population",
                                               "total_energy","total_water","total_food",
                                               "event","tech"])
        writer.writeheader()
        for row in history:
            writer.writerow({
                "iteration": row["iteration"],
                "global_index": f"{row['global_index']:.6f}",
                "total_population": int(round(row["total_population"])),
                "total_energy": int(round(row["total_energy"])),
                "total_water": int(round(row["total_water"])),
                "total_food": int(round(row["total_food"])),
                "event": row["event"] or "",
                "tech": row["tech"] or ""
            })

    # print concise summary
    print("\nSimulation finished. Summary of key metrics across iterations:")
    for h in history:
        print(f"Iter {h['iteration']:2d} | G-Index {h['global_index']:.3f} | Pop {h['total_population']:.0f} | "
              f"E {h['total_energy']:.0f} | W {h['total_water']:.0f} | F {h['total_food']:.0f} | "
              f"Event: {h['event'] or 'None'} | Tech: {h['tech'] or 'None'}")

    print(f"\nFinal arrays and history saved in directory: {config['output_dir']}")
    return history, resources, population

# ------------------------- CLI / ENTRYPOINT -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Global Resource Equilibrium Simulator (NumPy only)")
    p.add_argument("-i", "--iterations", type=int, default=DEFAULT_CONFIG["iterations"], help="Number of iterations")
    p.add_argument("-r", "--rows", type=int, default=DEFAULT_CONFIG["grid_shape"][0], help="Grid rows")
    p.add_argument("-c", "--cols", type=int, default=DEFAULT_CONFIG["grid_shape"][1], help="Grid cols")
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help="Random seed")
    p.add_argument("--out", type=str, default=DEFAULT_CONFIG["output_dir"], help="Output directory")
    p.add_argument("--no-logs", action="store_true", help="Suppress per-iteration ASCII logs")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()
    cfg["iterations"] = args.iterations
    cfg["grid_shape"] = (args.rows, args.cols)
    cfg["seed"] = args.seed
    cfg["output_dir"] = args.out
    cfg["logging_interval"] = 0 if args.no_logs else DEFAULT_CONFIG["logging_interval"]
    run_simulation(cfg)
