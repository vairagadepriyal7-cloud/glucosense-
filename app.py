import os, json, datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from record import record
from features import extract_features, meal_features
from meals import FOODS, add_meal, load_my_meals

USE_FAKE = True   # True = use data/fake.csv (no sensor). Change to False when the sensor is connected.

NAMES = {"carbs_0_30": "Carbs eaten 0-30 min ago", "carbs_30_60": "Carbs eaten 30-60 min ago",
         "carbs_60_120": "Carbs eaten 1-2 h ago", "carbs_120_180": "Carbs eaten 2-3 h ago",
         "carbs_180_300": "Carbs eaten 3-5 h ago", "sugar_0_30": "Sugar eaten 0-30 min ago",
         "sugar_30_60": "Sugar eaten 30-60 min ago", "sugar_60_120": "Sugar eaten 1-2 h ago",
         "sugar_120_180": "Sugar eaten 2-3 h ago", "sugar_180_300": "Sugar eaten 3-5 h ago",
         "mins_since_meal": "Time since last meal", "calories_2h": "Calories in last 2 h",
         "carbs_8h": "Carbs in last 8 h", "protein_2h": "Protein in last 2 h",
         "fat_2h": "Fat in last 2 h", "fiber_2h": "Fiber in last 2 h",
         "carb_impact": "Expected carb effect now (fast)",
         "carb_impact_slow": "Expected carb effect now (slow)",
         "hr_mean": "Heart rate", "sdnn": "Heart rate variability",
         "rmssd": "Beat-to-beat variability", "pnn50": "Large beat-to-beat changes"}

st.title("Non-invasive glucose estimator")
st.caption("Research prototype. Gives an estimate only. Not a medical device. "
           "Do not use it for medicine or treatment decisions.")
if USE_FAKE:
    st.info("Sensor not connected: the pulse reading uses a sample recording. "
            "The glucose estimate comes from the trained model and your logged meals.")

model = joblib.load("glucose_model.pkl")
info = json.load(open("model_info.json"))
FEATURES = info["features"]
CAL_FILE, READINGS = "calibration.json", "readings.csv"

def load_calibration():
    if os.path.exists(CAL_FILE):
        return json.load(open(CAL_FILE))
    return {"baseline": info["default_baseline"], "error": info["typical_error"], "calibrated": False}

meal_tab, measure_tab, calib_tab = st.tabs(["1. Log a meal", "2. Measure", "3. Calibrate"])

# ---------- Tab 1: meals ----------
with meal_tab:
    food = st.selectbox("Food", FOODS["food"])
    servings = st.number_input("Servings", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
    day = st.date_input("Date eaten", datetime.date.today())
    if "meal_time" not in st.session_state:
        st.session_state.meal_time = datetime.datetime.now().time().replace(second=0, microsecond=0)
    when = st.time_input("Time you started eating", key="meal_time")
    if st.button("Add meal"):
        add_meal(food, servings, datetime.datetime.combine(day, when))
        st.success(f"Added {servings} x {food}")
    st.write("Recent meals:")
    st.dataframe(load_my_meals().tail(10))

# ---------- Tab 2: measure ----------
with measure_tab:
    st.write("Sit still, rest your finger lightly on the sensor, then press Start.")
    if st.button("Start 60-second measurement"):
        if USE_FAKE:
            path = "data/fake.csv"
        else:
            with st.spinner("Measuring... keep your hand still"):
                path = record(60, "app")
        feats, message = extract_features(path)
        if feats is None:
            st.error(message)
        else:
            now = pd.Timestamp(datetime.datetime.now())
            row = meal_features(load_my_meals(), now)
            row.update(feats)
            X = pd.DataFrame([row])[FEATURES]
            change = float(model.predict(X)[0])          # rise or fall from your normal level
            cal = load_calibration()
            estimate = cal["baseline"] + change
            err = cal["error"]

            st.metric("Estimated glucose", f"{estimate:.0f} mg/dL",
                      f"{change:+.0f} vs your normal level")
            st.write(f"Likely range: {estimate - err:.0f} to {estimate + err:.0f} mg/dL")
            st.write(f"Heart rate: {feats['hr_mean']:.0f} beats per minute")
            if not cal.get("calibrated", False):
                st.info("Not calibrated yet: using an average normal level. "
                        "Save 5 glucometer readings in tab 3 to personalise it.")

            contrib = model.predict(X, pred_contrib=True)[0][:-1]
            reasons = pd.Series(contrib, index=[NAMES.get(f, f) for f in FEATURES])
            reasons = reasons[reasons.abs() > 0.5].sort_values(key=abs, ascending=False)
            st.write("Main reasons (mg/dL each clue adds or removes):")
            st.dataframe(reasons.head(5).round(1).rename("effect"))

            wave = -pd.read_csv(path)["ir"].iloc[500:1500].reset_index(drop=True)
            st.write("Your pulse wave (10 seconds):")
            st.line_chart(wave)

            st.session_state.last = {"time": now.isoformat(), "file": path, "change": change}
            st.info("Calibrating? Take a glucometer reading now and enter it in tab 3.")

# ---------- Tab 3: calibrate ----------
with calib_tab:
    cal = load_calibration()
    if cal.get("calibrated", False):
        st.success(f"Your normal level: {cal['baseline']:.0f} mg/dL (from {cal['pairs']} readings)")
    else:
        st.write("Not calibrated yet.")

    last = st.session_state.get("last")
    if last is None:
        st.write("Do a measurement in tab 2 first.")
    else:
        st.write(f"Last measurement at {last['time'][11:16]}.")
        true_value = st.number_input("Glucometer reading (mg/dL)", min_value=40, max_value=400, value=100)
        if st.button("Save this pair"):
            pair = pd.DataFrame([{**last, "glucometer": true_value}])
            pair.to_csv(READINGS, mode="a", header=not os.path.exists(READINGS), index=False)
            st.success("Saved.")

    if os.path.exists(READINGS):
        pairs = pd.read_csv(READINGS)
        st.write(f"Saved pairs: {len(pairs)}")
        st.dataframe(pairs[["time", "change", "glucometer"]].tail(10))
        if len(pairs) >= 5 and st.button("Update my calibration"):
            baseline = float((pairs["glucometer"] - pairs["change"]).mean())
            fitted = baseline + pairs["change"]
            err = max(10.0, float(np.mean(np.abs(fitted - pairs["glucometer"]))))
            json.dump({"baseline": baseline, "error": err, "calibrated": True,
                       "pairs": int(len(pairs))}, open(CAL_FILE, "w"))
            st.success(f"Calibration updated: your normal level is {baseline:.0f} mg/dL")