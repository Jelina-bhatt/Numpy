import numpy as np
import time

# City grid size
rows, cols = 10, 10
steps = 20  # Number of simulation steps

# Initialize grid (0 = empty, 1 = car)
city = np.zeros((rows, cols), dtype=int)

# Randomly place cars in the grid
num_cars = 20
car_positions = np.random.choice(rows*cols, num_cars, replace=False)
city[np.unravel_index(car_positions, city.shape)] = 1

def display(city):
    for row in city:
        print(" ".join("🚗" if cell == 1 else "⬜" for cell in row))
    print("\n" + "-"*30 + "\n")

# Simulation loop
for step in range(steps):
    print(f"Step {step+1}")
    display(city)

    # Cars randomly move (up, down, left, right)
    new_city = np.zeros_like(city)
    for r in range(rows):
        for c in range(cols):
            if city[r, c] == 1:
                move = np.random.choice(["up","down","left","right","stay"])
                new_r, new_c = r, c
                if move == "up" and r > 0: new_r -= 1
                elif move == "down" and r < rows-1: new_r += 1
                elif move == "left" and c > 0: new_c -= 1
                elif move == "right" and c < cols-1: new_c += 1
                # Place car if spot is empty
                if new_city[new_r, new_c] == 0:
                    new_city[new_r, new_c] = 1
                else:
                    new_city[r, c] = 1  # stay if blocked
    city = new_city
    time.sleep(0.5)
