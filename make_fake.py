import numpy as np, pandas as pd, os
os.makedirs("data", exist_ok=True)
t = np.arange(0, 60, 0.01)                                        # 60 s, 100 readings/s
phase = 2 * np.pi * np.cumsum(1.2 + 0.05 * np.random.randn(len(t))) * 0.01   # about 72 bpm
pulse = np.maximum(np.sin(phase), 0) ** 2                         # one bump per heartbeat
noise = np.random.randn(len(t))
pd.DataFrame({"ms": (t * 1000).astype(int),
              "red": (80000 - 500 * pulse + 20 * noise).astype(int),
              "ir": (100000 - 800 * pulse + 20 * noise).astype(int)}).to_csv("data/fake.csv", index=False)
print("Created data/fake.csv")