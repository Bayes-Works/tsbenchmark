"""Generate a synthetic weekly time series with canari.

Components:
    - Local trend (level + trend)
    - Fourier-form periodic component, period = 13 time steps
    - Fourier-form periodic component, period = 52 time steps
    - Autoregression with phi = 0.75 and process-noise std = 2
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from canari import Model
from canari.component import Autoregression, LocalTrend, Periodic

# Settings
SEED = 1
NUM_YEARS = 10
NUM_STEPS = 52 * NUM_YEARS
START_DATE = "2010-01-03"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

rng = np.random.default_rng(SEED)

# Components
local_trend = LocalTrend(
    mu_states=[10.0, 0.0],
    var_states=[1e-12, 1e-12],
    std_error=0.0,
)
# periodic_13 = Periodic(
#     period=13,
#     mu_states=[2.0, 0.0],
#     var_states=[1e-12, 1e-12],
#     std_error=0.0,
# )
# periodic_52 = Periodic(
#     period=52,
#     mu_states=[5.0, 0.0],
#     var_states=[1e-12, 1e-12],
#     std_error=0.0,
# )
autoregression = Autoregression(
    phi=0.9,
    std_error=0.5,
    mu_states=[0.0],
    var_states=[1e-12],
)

model = Model(local_trend, autoregression)

# Simulate from the state-space model: x_t = A x_{t-1} + w_t, y_t = C x_t
A = np.asarray(model.transition_matrix)
Q = np.asarray(model.process_noise_matrix)
C = np.asarray(model.observation_matrix).reshape(1, -1)
x = np.asarray(model.mu_states).reshape(-1, 1)

states = np.zeros((NUM_STEPS, A.shape[0]))
y = np.zeros(NUM_STEPS)
for t in range(NUM_STEPS):
    w = rng.multivariate_normal(np.zeros(A.shape[0]), Q).reshape(-1, 1)
    x = A @ x + w
    states[t] = x.ravel()
    y[t] = (C @ x).item()

# Save
dates = pd.date_range(start=START_DATE, periods=NUM_STEPS, freq="W")
df = pd.DataFrame({"date": dates, "y": y})
df_states = pd.DataFrame(states, columns=model.states_name)
df_states.insert(0, "date", dates)

os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(os.path.join(OUTPUT_DIR, "weekly_ts.csv"), index=False)
df_states.to_csv(os.path.join(OUTPUT_DIR, "weekly_ts_states.csv"), index=False)

# Plot
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dates, y, lw=1)
ax.set_xlabel("Date")
ax.set_ylabel("y")
ax.set_title("Synthetic weekly series: LT + AR(0.75, 0.5)")
fig.tight_layout()
plt.show()
