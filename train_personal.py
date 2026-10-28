import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

data = pd.read_csv("training_data.csv", parse_dates=["time"])
data = data.sort_values(["participant", "time"]).reset_index(drop=True)

TIMING = [f"{k}_{lo}_{hi}" for k in ("carbs", "sugar")
          for lo, hi in [(0, 30), (30, 60), (60, 120), (120, 180), (180, 300)]]
MEAL = TIMING + ["calories_2h", "carbs_8h", "protein_2h", "mins_since_meal"]
PULSE = ["hr_mean", "sdnn", "rmssd", "pnn50"]
SETS = {"Meals only": MEAL, "Meals + pulse": MEAL + PULSE}
CAL_HOURS = 24

data["person_avg"] = data.groupby("participant")["glucose"].transform("mean")
data["deviation"] = data["glucose"] - data["person_avg"]

def make_model():
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=200, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1)

def scores(true, pred):
    err = pred - true
    return {"MAE": np.mean(np.abs(err)),
            "MARD %": np.mean(np.abs(err) / true) * 100,
            "Within 20 %": np.mean(np.abs(err) / true <= 0.20) * 100}

methods = ["Average of other people", "Your own day-1 average"] + list(SETS)
truths, after_meal, preds = [], [], {m: [] for m in methods}

for p in data["participant"].unique():
    train = data[data["participant"] != p]
    test = data[data["participant"] == p]
    cutoff = test["time"].min() + pd.Timedelta(hours=CAL_HOURS)
    cal, rest = test[test["time"] < cutoff], test[test["time"] >= cutoff]
    if len(cal) < 20 or len(rest) < 20:
        continue
    baseline = cal["glucose"].mean()
    truths.append(rest["glucose"].to_numpy())
    recent_carbs = rest["carbs_0_30"] + rest["carbs_30_60"] + rest["carbs_60_120"]
    after_meal.append((recent_carbs >= 20).to_numpy())
    preds["Average of other people"].append(np.full(len(rest), train["glucose"].mean()))
    preds["Your own day-1 average"].append(np.full(len(rest), baseline))
    for name, cols in SETS.items():
        model = make_model()
        model.fit(train[cols], train["deviation"])
        preds[name].append(baseline + model.predict(rest[cols]))
    print("Tested person", p)

true_all = np.concatenate(truths)
mask = np.concatenate(after_meal)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
overall = pd.DataFrame({m: scores(true_all, np.concatenate(preds[m])) for m in methods}).T.round(1)
meals_only = pd.DataFrame({m: scores(true_all[mask], np.concatenate(preds[m])[mask])
                           for m in methods}).T.round(1)
print("\nALL READINGS")
print(overall)
print(f"\nAFTER MEALS ONLY ({mask.sum()} readings within 2 h of a meal with 20 g+ carbs)")
print(meals_only)
overall.to_csv("personal_comparison.csv")
meals_only.to_csv("after_meal_comparison.csv")

# Save the better of the two models for the app
best = overall.loc[list(SETS), "MAE"].idxmin()
final_cols = SETS[best]
final = make_model()
final.fit(data[final_cols], data["deviation"])
joblib.dump(final, "glucose_model.pkl")
with open("model_info.json", "w") as f:
    json.dump({"features": final_cols, "target": "deviation", "model_set": best,
               "default_baseline": float(data["glucose"].mean()),
               "typical_error": float(overall.loc[best, "MAE"])}, f)
print(f"\nSaved the '{best}' model as glucose_model.pkl")