import numpy as np
import pandas as pd
from features import hrv_from_ibi, meal_features

def pick(columns, keywords):
    """Find the first column whose name contains one of the keywords."""
    for k in keywords:
        for c in columns:
            if k in c.lower():
                return c
    raise ValueError(f"No column with {keywords} in {list(columns)}")

def load_series(path, value_keywords):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    tcol = pick(df.columns, ["timestamp", "datetime", "time", "date"])
    vcol = pick(df.columns, value_keywords)
    out = pd.DataFrame({"time": pd.to_datetime(df[tcol], errors="coerce"),
                        "value": pd.to_numeric(df[vcol], errors="coerce")})
    return out.dropna().sort_values("time").reset_index(drop=True)

def optional_column(food, cols, keywords):
    """A column if it exists, otherwise zeros."""
    try:
        return pd.to_numeric(food[pick(cols, keywords)], errors="coerce")
    except ValueError:
        return pd.Series(0.0, index=food.index)

def load_meals(path):
    """Read a food log, whichever column names it uses."""
    food = pd.read_csv(path)
    food.columns = [c.strip().lower() for c in food.columns]
    cols = list(food.columns)
    if "time_begin" in cols:
        when = pd.to_datetime(food["time_begin"], errors="coerce")
    else:
        date_col = pick(cols, ["date"])
        time_col = pick(cols, ["time_of_day", "time"])
        when = pd.to_datetime(food[date_col].astype(str) + " " + food[time_col].astype(str),
                              errors="coerce")
    meals = pd.DataFrame({
        "time": when,
        "carbs": pd.to_numeric(food[pick(cols, ["total_carb", "carb"])], errors="coerce"),
        "sugar": pd.to_numeric(food[pick(cols, ["sugar"])], errors="coerce"),
        "calories": pd.to_numeric(food[pick(cols, ["calorie"])], errors="coerce"),
        "protein": pd.to_numeric(food[pick(cols, ["protein"])], errors="coerce"),
        "fat": optional_column(food, cols, ["total_fat", "fat"]),
        "fiber": optional_column(food, cols, ["dietary_fiber", "fiber"]),
    })
    return meals.dropna(subset=["time"]).fillna(0)

rows = []
for i in range(1, 17):
    pid = f"{i:03d}"
    folder = f"physionet/{pid}"
    try:
        glucose = load_series(f"{folder}/Dexcom_{pid}.csv", ["glucose", "value"])
        ibi = load_series(f"{folder}/IBI_{pid}.csv", ["ibi", "value"])
        meals = load_meals(f"{folder}/Food_Log_{pid}.csv")
    except Exception as e:
        print(f"Participant {pid}: SKIPPED, could not read a file ({e})")
        continue
    if len(meals) == 0:
        print(f"Participant {pid}: SKIPPED, food log has no usable times")
        continue
    if ibi["value"].median() < 10:
        ibi["value"] = ibi["value"] * 1000

    ibi_times = ibi["time"].to_numpy()
    ibi_values = ibi["value"].to_numpy()
    for t, g in zip(glucose["time"], glucose["value"]):
        start = np.searchsorted(ibi_times, np.datetime64(t - pd.Timedelta(minutes=5)), side="right")
        end = np.searchsorted(ibi_times, np.datetime64(t), side="right")
        hrv = hrv_from_ibi(ibi_values[start:end])
        if hrv is None:
            continue
        row = {"participant": pid, "time": t, "glucose": g}
        row.update(hrv)
        row.update(meal_features(meals, t))
        rows.append(row)
    print(f"Participant {pid}: done, total rows so far {len(rows)}")

data = pd.DataFrame(rows)
data.to_csv("training_data.csv", index=False)
print("Saved training_data.csv with", len(data), "rows")