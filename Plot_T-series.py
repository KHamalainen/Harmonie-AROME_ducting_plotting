# ================================================================
# T_timeseries.py (FOUR MODEL CONFIGURATIONS, POINT-CSV VERSION)
#
# Reads:
#   - Observations
#   - Model CSV files generated from point-NC:
#         Model_data_Uto_2500L65.csv
#         Model_data_Uto_2500L90.csv
#         Model_data_Uto_750L65.csv
#         Model_data_Uto_750L90.csv
#
# Author: Karoliina H. (FMI), 2026
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------
LOC_NAME = "Utö"

MODEL_FILES = {
    "2500L65": "/dir/Model_data_Uto_2500L65.csv",
    "2500L90": "/dir/Model_data_Uto_2500L90.csv",
    "750L65":  "/dir/Model_data_Uto_750L65.csv",
    "750L90":  "/dir/Model_data_Uto_750L90.csv",
}

# Colors and styles EXACTLY as in original
MODEL_STYLE = {
    "2500L65": ("royalblue", "-"),
    "2500L90": ("royalblue", "--"),
    "750L65":  ("skyblue", "-"),
    "750L90":  ("skyblue", "--"),
}

MAST_COLOR = "pink"
AWS_COLOR  = "pink"

# ---------------------------------------------------------------
# LOAD OBSERVATIONS
# ---------------------------------------------------------------

# --- MAST 2–30 m ---
mast = pd.read_csv("/dir/Utö_30m_masto_testijakso.csv")
Tvars = ['TA #16 2m (degC)', 'TA #17 5m (degC)',
         'TA 10m (degC)', 'TA #2 20m (degC)', 'TA #3 30m (degC)']

mast["time"] = pd.to_datetime(mast["DATA_TIME"])
mast = mast.set_index("time")[Tvars]
mast.columns = ["T_2m","T_5m","T_10m","T_20m","T_30m"]
mast = mast.resample("H").first()

# --- MAST 59m ---
mast59 = pd.read_csv("/dir/Uto_T_RH_59m_1min.csv")
mast59["time"] = pd.to_datetime(mast59["Time"])
mast59 = mast59.set_index("time")[["T [degC]"]]
mast59.columns = ["T_59m"]
mast59 = mast59.resample("H").first()

# --- AWS 2m ---
aws = pd.read_csv("/dir/Utö_AWS_testijakso.csv",
                  parse_dates=["DATA_TIME"])
aws = aws.set_index("DATA_TIME")[["TA #2 (degC)"]]
aws.columns = ["AWS_2m"]
aws = aws.resample("10T").first()

# Merge OBS
OBS = mast.join(mast59, how="inner")
OBS = OBS.join(aws, how="left")

# ---------------------------------------------------------------
# LOAD ALL MODEL CSV FILES
# ---------------------------------------------------------------
models = {}

for exp, csv_file in MODEL_FILES.items():
    
    df = pd.read_csv(csv_file, parse_dates=["Time"])
    df = df.rename(columns={"Time": "time"})

    df = df.set_index("time")
    models[exp] = df  # store DF

# ---------------------------------------------------------------
# ALIGN TIME AXES (common intersection)
# ---------------------------------------------------------------
common_times = OBS.index

for exp in models:
    common_times = common_times.intersection(models[exp].index)

OBS2 = OBS.loc[common_times]
for exp in models:
    models[exp] = models[exp].loc[common_times]

# ---------------------------------------------------------------
# PLOT T2m, T30m, T59m
# ---------------------------------------------------------------
fig, axs = plt.subplots(3, figsize=(10,12), sharex=True)

# ---------------- T2m ----------------
axs[0].plot(OBS2.index, OBS2["T_2m"], label="Mast 2m", color=MAST_COLOR)
axs[0].plot(OBS2.index, OBS2["AWS_2m"], label="AWS 2m", color=AWS_COLOR, linestyle="--")

for exp in MODEL_FILES:
    color, style = MODEL_STYLE[exp]
    axs[0].plot(models[exp].index, models[exp]["T_2m"],
                label=exp, color=color, linestyle=style)

axs[0].set_ylabel("T2m [°C]")
axs[0].set_ylim(12,27)
axs[0].grid(True)
axs[0].legend()

# ---------------- T30m ----------------
axs[1].plot(OBS2.index, OBS2["T_30m"], label="Mast 30m", color=MAST_COLOR)

for exp in MODEL_FILES:
    color, style = MODEL_STYLE[exp]
    axs[1].plot(models[exp].index, models[exp]["T_30m"],
                label=exp, color=color, linestyle=style)

axs[1].set_ylabel("T30m [°C]")
axs[1].set_ylim(12,27)
axs[1].grid(True)
axs[1].legend()

# ---------------- T59m ----------------
axs[2].plot(OBS2.index, OBS2["T_59m"], label="Mast 59m", color=MAST_COLOR)

for exp in MODEL_FILES:
    color, style = MODEL_STYLE[exp]
    axs[2].plot(models[exp].index, models[exp]["T_59m"],
                label=exp, color=color, linestyle=style)

axs[2].set_ylabel("T59m [°C]")
axs[2].set_ylim(12,27)
axs[2].grid(True)
axs[2].legend()

# X-axis formatting
axs[-1].set_xlabel("Date")
for ax in axs:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(f"T_timeseries_{LOC_NAME}_4models.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: T_timeseries with four model configs.")

