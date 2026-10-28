import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

data = pd.read_csv("training_data.csv", parse_dates=["time"])
data = data.sort_values(["participant", "time"]).reset_index(drop=True)

TIMING = [f"{k}_{lo}_{hi}" for k in ("carbs", "sugar")
          for lo, hi in [(0, 30), (30, 60), (60, 120), (120, 180), (180, 300)]]
PULSE = ["hr_mean", "sdnn", "rmssd", "pnn50"]
OLD = TIMING + ["calories_2h", "carbs_8h", "protein_2h", "mins_since_meal"] + PULSE
NEW = OLD + ["fat_2h", "fiber_2h", "carb_impact", "carb_impact_slow"]
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

methods = ["Average guess (day-1)", "Current model", "New meal features",
           "New features + personal scaling"]
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
    recent = rest["carbs_0_30"] + rest["carbs_30_60"] + rest["carbs_60_120"]
    after_meal.append((recent >= 20).to_numpy())

    preds["Average guess (day-1)"].append(np.full(len(rest), baseline))

    m_old = make_model().fit(train[OLD], train["deviation"])
    preds["Current model"].append(baseline + m_old.predict(rest[OLD]))

    m_new = make_model().fit(train[NEW], train["deviation"])
    change_rest = m_new.predict(rest[NEW])
    preds["New meal features"].append(baseline + change_rest)

    # Personal scaling: how strongly does THIS person react? Learned from day 1 only.
    change_cal = m_new.predict(cal[NEW])
    target_cal = cal["glucose"].to_numpy() - baseline
    denom = np.sum(change_cal ** 2)
    scale = float(np.clip(np.sum(change_cal * target_cal) / denom, 0.5, 2.0)) if denom > 0 else 1.0
    preds["New features + personal scaling"].append(baseline + scale * change_rest)
    print(f"Tested person {p} (personal scale {scale:.2f})")

true_all = np.concatenate(truths)
mask = np.concatenate(after_meal)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
overall = pd.DataFrame({m: scores(true_all, np.concatenate(preds[m])) for m in methods}).T.round(1)
meals_only = pd.DataFrame({m: scores(true_all[mask], np.concatenate(preds[m])[mask])
                           for m in methods}).T.round(1)
print("\nALL READINGS")
print(overall)
print(f"\nAFTER MEALS ONLY ({mask.sum()} readings)")
print(meals_only)
overall.to_csv("improved_comparison.csv")
meals_only.to_csv("improved_after_meal.csv")

# Keep the new features only if they actually helped
if overall.loc["New meal features", "MAE"] < overall.loc["Current model", "MAE"]:
    final_cols = NEW
    final = make_model().fit(data[final_cols], data["deviation"])
    joblib.dump(final, "glucose_model.pkl")
    with open("model_info.json", "w") as f:
        json.dump({"features": final_cols, "target": "deviation", "model_set": "New meal features",
                   "default_baseline": float(data["glucose"].mean()),
                   "typical_error": float(overall.loc["New meal features", "MAE"])}, f)
    print("\nNew features helped: saved the improved model.")
else:
    print("\nNew features did not help: your current model was kept.")