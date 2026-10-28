import numpy as np
import pandas as pd
import neurokit2 as nk

def hrv_from_ibi(ibi_ms):
    """Heart-rate numbers from the gaps between beats, in milliseconds."""
    ibi = np.asarray(ibi_ms, dtype=float)
    ibi = ibi[(ibi > 400) & (ibi < 1500)]          # keep realistic beats only
    if len(ibi) < 20:
        return None
    diffs = np.diff(ibi)
    return {
        "hr_mean": 60000 / ibi.mean(),
        "sdnn": ibi.std(),
        "rmssd": np.sqrt(np.mean(diffs ** 2)),
        "pnn50": np.mean(np.abs(diffs) > 50) * 100,
    }

def extract_features(csv_path):
    """Turn one recording file into a dictionary of features."""
    df = pd.read_csv(csv_path)
    fs = 1000 / np.median(np.diff(df["ms"]))       # readings per second (~100)
    df = df.iloc[int(5 * fs):]                     # skip the first 5 seconds
    ir = df["ir"].to_numpy(dtype=float)
    red = df["red"].to_numpy(dtype=float)

    if ir.mean() < 50000:
        return None, "No finger detected. Place your finger on the sensor."

    ir_clean = nk.ppg_clean(-ir, sampling_rate=fs)   # flip so beats become peaks
    red_clean = nk.ppg_clean(-red, sampling_rate=fs)
    peaks = nk.ppg_findpeaks(ir_clean, sampling_rate=fs)["PPG_Peaks"]
    ibi = np.diff(peaks) / fs * 1000

    feats = hrv_from_ibi(ibi)
    if feats is None:
        return None, "Too few clean heartbeats. Keep still and try again."
    if feats["sdnn"] > 150:
        return None, "Signal too irregular (did you move?). Try again."

    ac_ir = np.percentile(ir_clean, 95) - np.percentile(ir_clean, 5)
    ac_red = np.percentile(red_clean, 95) - np.percentile(red_clean, 5)
    feats["perfusion_ir"] = ac_ir / ir.mean() * 100
    feats["red_ir_ratio"] = (ac_red / red.mean()) / (ac_ir / ir.mean())
    return feats, "OK"

def meal_features(meals, now):
    """Meal numbers for the moment `now`. meals: columns time, carbs, sugar, calories, protein
    (and optionally fat, fiber)."""
    out = {}
    past = meals[meals["time"] <= now]
    if len(past) > 0:
        age = (now - past["time"]).dt.total_seconds() / 60    # minutes since each food
    else:
        age = pd.Series(dtype=float)

    def col(name):
        if name in past.columns:
            return past[name].astype(float)
        return pd.Series(0.0, index=past.index)

    # Totals over longer periods
    for hours in (2, 8, 24):
        recent = past[age < hours * 60]
        out[f"carbs_{hours}h"] = recent["carbs"].sum()
        out[f"sugar_{hours}h"] = recent["sugar"].sum()
        out[f"calories_{hours}h"] = recent["calories"].sum()
    out["protein_2h"] = past[age < 120]["protein"].sum()
    out["fat_2h"] = col("fat")[age < 120].sum()
    out["fiber_2h"] = col("fiber")[age < 120].sum()

    # When the carbs were eaten matters: glucose usually peaks 30-60 minutes after eating
    for lo, hi in [(0, 30), (30, 60), (60, 120), (120, 180), (180, 300)]:
        part = past[(age >= lo) & (age < hi)]
        out[f"carbs_{lo}_{hi}"] = part["carbs"].sum()
        out[f"sugar_{lo}_{hi}"] = part["sugar"].sum()

    # Expected effect of carbs right now, using a typical glucose-response curve
    t = age.to_numpy(dtype=float)
    net_carbs = (col("carbs") - col("fiber")).clip(lower=0).to_numpy()
    fast = (t / 45) * np.exp(1 - t / 45)      # peaks about 45 min after eating
    slow = (t / 90) * np.exp(1 - t / 90)      # slower (fatty) meals peak later
    fast[t > 300] = 0
    slow[t > 360] = 0
    out["carb_impact"] = float(np.sum(net_carbs * fast))
    out["carb_impact_slow"] = float(np.sum(net_carbs * slow))

    out["mins_since_meal"] = min(age.min(), 24 * 60) if len(past) > 0 else 24 * 60
    out["hour"] = now.hour + now.minute / 60
    return out

if __name__ == "__main__":
    import sys
    feats, message = extract_features(sys.argv[1])
    print(message)
    if feats:
        for name, value in feats.items():
            print(f"{name:>14}: {value:.2f}")