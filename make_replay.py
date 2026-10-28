import json
import numpy as np
import pandas as pd
import lightgbm as lgb

data = pd.read_csv("training_data.csv", parse_dates=["time"])
data = data.sort_values(["participant", "time"]).reset_index(drop=True)
cols = json.load(open("model_info.json"))["features"]

data["person_avg"] = data.groupby("participant")["glucose"].transform("mean")
data["deviation"] = data["glucose"] - data["person_avg"]

def make_model():
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=200, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1)

out = []
for p in data["participant"].unique():
    train = data[data["participant"] != p]          # model never sees this person
    test = data[data["participant"] == p]
    cutoff = test["time"].min() + pd.Timedelta(hours=24)
    cal, rest = test[test["time"] < cutoff], test[test["time"] >= cutoff].copy()
    if len(cal) < 20 or len(rest) < 20:
        continue
    baseline = cal["glucose"].mean()                # first day = calibration
    model = make_model().fit(train[cols], train["deviation"])

    # personal scaling learned from day 1 only
    change_cal = model.predict(cal[cols])
    target_cal = cal["glucose"].to_numpy() - baseline
    denom = np.sum(change_cal ** 2)
    scale = float(np.clip(np.sum(change_cal * target_cal) / denom, 0.5, 2.0)) if denom > 0 else 1.0

    rest["estimate"] = baseline + scale * model.predict(rest[cols])
    rest["average_guess"] = baseline
    out.append(rest[["participant", "time", "glucose", "estimate", "average_guess"]])
    print(f"Done person {p} (personal scale {scale:.2f})")

pd.concat(out).to_csv("replay_predictions.csv", index=False)
print("Saved replay_predictions.csv")