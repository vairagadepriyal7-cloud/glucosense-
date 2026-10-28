import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import LeaveOneGroupOut

data = pd.read_csv("training_data.csv")
y = data["glucose"].to_numpy()
groups = data["participant"]

MEAL = ["carbs_2h", "sugar_2h", "calories_2h", "carbs_8h", "sugar_8h", "calories_8h",
        "carbs_24h", "sugar_24h", "calories_24h", "protein_2h", "mins_since_meal", "hour"]
PULSE = ["hr_mean", "sdnn", "rmssd", "pnn50"]
SETS = {"Meals only": MEAL, "Pulse only": PULSE + ["hour"], "Meals + pulse": MEAL + PULSE}

def make_model():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                             min_child_samples=40, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1)

def scores(true, pred):
    err = pred - true
    return {"MAE": np.mean(np.abs(err)),
            "RMSE": np.sqrt(np.mean(err ** 2)),
            "MARD %": np.mean(np.abs(err) / true) * 100,
            "Within 20 %": np.mean(np.abs(err) / true <= 0.20) * 100}

results = {}
for name, cols in SETS.items():
    X = data[cols]
    pred = np.zeros(len(y))
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        model = make_model()
        model.fit(X.iloc[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X.iloc[test_idx])
    results[name] = scores(y, pred)
    print("Finished:", name)

results["Always guess the average"] = scores(y, np.full(len(y), y.mean()))
table = pd.DataFrame(results).T.round(1)
print(table)
table.to_csv("model_comparison.csv")

final_cols = SETS["Meals + pulse"]
final = make_model()
final.fit(data[final_cols], y)
joblib.dump(final, "glucose_model.pkl")
with open("model_info.json", "w") as f:
    json.dump({"features": final_cols,
               "typical_error": float(results["Meals + pulse"]["MAE"])}, f)

importance = pd.Series(final.feature_importances_, index=final_cols).sort_values(ascending=False)
print("\nMost useful features:\n", importance.head(8))
print("\nSaved glucose_model.pkl and model_info.json")