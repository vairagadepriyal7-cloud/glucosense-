import os
import pandas as pd

FOODS = pd.read_csv("foods.csv")
MY_MEALS = "my_meals.csv"
COLUMNS = ["time", "food", "servings", "carbs", "sugar", "calories", "protein", "fat", "fiber"]

def add_meal(food, servings, when):
    """Save one eaten food: name from foods.csv, number of servings, time eaten."""
    item = FOODS[FOODS["food"] == food].iloc[0]
    row = pd.DataFrame([{
        "time": when, "food": food, "servings": servings,
        "carbs": item["carbs"] * servings, "sugar": item["sugar"] * servings,
        "calories": item["calories"] * servings, "protein": item["protein"] * servings,
        "fat": item["fat"] * servings, "fiber": item["fiber"] * servings,
    }])
    row.to_csv(MY_MEALS, mode="a", header=not os.path.exists(MY_MEALS), index=False)

def load_my_meals():
    """Return all saved meals as a table (empty table if none yet)."""
    if not os.path.exists(MY_MEALS):
        return pd.DataFrame(columns=COLUMNS)
    meals = pd.read_csv(MY_MEALS)
    meals["time"] = pd.to_datetime(meals["time"])
    return meals