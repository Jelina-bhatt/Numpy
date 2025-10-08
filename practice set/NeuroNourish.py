import numpy as np

# Define nutrient categories (per 100g)
# [calories, protein, carbs, fats, omega3, caffeine, sugar, vitamins]
nutrient_data = {
    "oats": [389, 17, 66, 7, 0.4, 0, 1, 0.8],
    "coffee": [2, 0.3, 0, 0, 0, 95, 0, 0],
    "salmon": [208, 20, 0, 13, 2.6, 0, 0, 0.7],
    "banana": [89, 1, 23, 0.3, 0, 0, 12, 1],
    "almonds": [579, 21, 22, 50, 0.1, 0, 4, 1.2],
}

# Cognitive weight matrix (arbitrary scale)
# brain_factors = [focus, calmness, alertness, memory_retention]
cognitive_weights = np.array([
    [0.3, 0.1, 0.4, 0.2],   # calories
    [0.2, 0.3, 0.1, 0.3],   # protein
    [0.1, 0.1, 0.3, 0.1],   # carbs
    [0.15, 0.4, 0.05, 0.25],# fats
    [0.4, 0.5, 0.2, 0.4],   # omega3
    [0.5, 0.1, 0.9, 0.3],   # caffeine
    [-0.3, -0.2, 0.4, -0.1],# sugar
    [0.25, 0.2, 0.1, 0.4]   # vitamins
])

def predict_cognitive_state(food_list):
    total_nutrients = np.zeros(8)
    for food in food_list:
        total_nutrients += nutrient_data[food]
    
    cognitive_state = total_nutrients @ cognitive_weights
    return np.round(cognitive_state / np.max(cognitive_state) * 100, 2)

# Test with different meals
meal_1 = ["oats", "banana", "coffee"]
meal_2 = ["salmon", "almonds", "coffee"]

print("Meal 1 Cognitive State [Focus, Calmness, Alertness, Memory]:")
print(predict_cognitive_state(meal_1))

print("\nMeal 2 Cognitive State [Focus, Calmness, Alertness, Memory]:")
print(predict_cognitive_state(meal_2))

# Find the most optimized meal combination
foods = list(nutrient_data.keys())
best_score, best_meal = 0, None

for i in range(len(foods)):
    for j in range(i + 1, len(foods)):
        meal = [foods[i], foods[j]]
        score = np.sum(predict_cognitive_state(meal))
        if score > best_score:
            best_score, best_meal = score, meal

print(f"\nBest Meal Combination: {best_meal} → Total Score: {best_score}")
