# GlucoSense: Estimating Glucose Trends from Meal and Pulse Data

GlucoSense is an experimental system that **estimates** glucose trends by combining
**pulse-derived features** (heart rate and heart-rate variability) with **meal logs**,
using a machine learning model. It also includes a low-cost sensor prototype
(ESP32 + MAX30102, about ₹700).

> ⚠️ **Not a medical device.** GlucoSense is an experimental research prototype.
> It estimates trends; it does not measure blood glucose and must not be used
> for medical decisions.

**Author:** Priyal Vairagade, 2nd-year B.E. Information Technology,
Prof. Ram Meghe Institute of Technology & Research (PRMITR), Badnera

---

## Results (leave-one-person-out, 15 subjects)

| Approach | MAE (mg/dL) ↓ | MARD ↓ | Within 20% ↑ |
|---|---|---|---|
| Average-guess baseline | 17.2 | 15.1% | 72.3% |
| Own day-1 average | 17.4 | 15.2% | 70.7% |
| **GlucoSense (final model)** | **16.4** | **14.4%** | **73.4%** |

- In the 2 hours after meals: **21.0 vs 22.5 mg/dL** (model vs baseline).
- The first version did **not** beat the baseline. Predicting each person's
  deviation from their own baseline, adding meal-timing features and adding
  personal calibration is what made the difference.

---

## Method

**Data:** [PhysioNet BIG IDEAs Lab Glycemic Variability and Wearable Device Data v1.1.3](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/)
(16 participants; 1 excluded due to a malformed food log, so 15 people and 18,971 readings).

**Features:**
- Heart rate and HRV from inter-beat intervals (NeuroKit2)
- Carbohydrates eaten 0–30, 30–60, 60–120, 120–180 and 180–300 minutes before each reading
- A carbohydrate impact curve (peaking at about 45 minutes), plus fat and fibre

**Model:** LightGBM regression predicting the deviation from each person's baseline,
with personal calibration from their first 24 hours.

**Evaluation and leakage prevention:**
- Leave-one-person-out cross-validation (always tested on an unseen person)
- Calibration uses only the test person's first 24 hours; evaluation uses only data after that
- Meal features use only food eaten *before* each reading

**Known limitation:** features were refined while looking at the same cross-validation
results, so a fully held-out test set would be needed for a final unbiased estimate.

---

## Hardware prototype

- **ESP32 DevKit** + **MAX30102** pulse sensor (I2C: SDA → D21, SCL → D22, VIN → 3V3)
- Arduino sketch in `arduino/glucose_sensor/` streams `millis,red,ir` at about 50 readings per second
- **Status:** sensor detected and streaming live red/IR pulse signals with tuned LED
  brightness. Python-side recording (`record.py`) and live estimation are in progress.

---

## Project structure

| File | Purpose |
|---|---|
| `build_dataset.py` | Builds the training table from PhysioNet files |
| `features.py` | HR/HRV and meal feature extraction |
| `train_improved.py` | Trains the model and runs leave-one-person-out evaluation |
| `app.py`, `pages/` | Streamlit app: meal logging, estimate, replay of real data |
| `meals.py`, `foods.csv` | Meal logging with Indian food nutrition values |
| `record.py` | Records pulse data from the ESP32 |
| `arduino/glucose_sensor/` | ESP32 + MAX30102 firmware |

## How to run

```bash
pip install -r requirements.txt
# 1. Download the PhysioNet dataset (link above) into a data/ folder
python build_dataset.py
python train_improved.py
streamlit run app.py
```

## Next steps
- Collect more participants, including people with diabetes
- Explore deep learning on raw pulse signals and self-supervised pretraining on unlabeled data
- Complete live sensor-to-app integration