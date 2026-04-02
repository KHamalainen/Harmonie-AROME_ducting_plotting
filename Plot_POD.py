#===============================================================
# Performance Diagram for Atmospheric Duct Detection (POD)
#
# This script plots a Performance Diagram (Success Ratio vs POD)
# for detecting negative refractivity gradients (dM/dh < 0)
# using four ATOS experiments (2500L65, 2500L90, 750L65, 750L90)
# at the Utö mast site.
#
# The diagram includes:
#   - POD (Probability of Detection)
#   - Success Ratio (SR = 1 - FAR)
#   - Bias isolines
#   - Threat Score (TS) isolines
#   - Experiment-wise colour coding
#   - Height-level markers for dM/dh layers
# --------------------------------------------------------------
# Inputs:
#   Statistics_Utö_*.csv   (computed from the NC-based pipeline)
# Output:
#   POD_Uto_new.png
#
#
# Author: Karoliina H. (FMI), 2026
#===============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

EXPERIMENTS = {
    "2500L65": "Statistics_Utö_2500L65.csv",
    "2500L90": "Statistics_Utö_2500L90.csv",
    "750L65":  "Statistics_Utö_750L65.csv",
    "750L90":  "Statistics_Utö_750L90.csv"
}

LEVEL_LABELS = {
    "dMdh_1": "2–5 m",
    "dMdh_2": "5–10 m",
    "dMdh_3": "10–20 m",
    "dMdh_4": "20–30 m",
    "dMdh_5": "30–59 m"
}

COLORS = {
    "2500L65": "royalblue",
    "2500L90": "skyblue",
    "750L65":  "mediumorchid",
    "750L90":  "hotpink"
}

MARKERS = ["o", "s", "^", "D", "P"]

# ------------------------------------------------------------
# Funktio: Read statistics
# ------------------------------------------------------------
def read_statistics(file_path):
    df = pd.read_csv(file_path)
    df = df[df["OBS_Level"].str.contains("dMdh")]
    df = df.sort_values("OBS_Level")
    df["SR"] = 1 - df["FAR"]

    return df

# ------------------------------------------------------------
# Performance Diagram
# ------------------------------------------------------------

x = np.linspace(0, 1, 500)
y = np.linspace(0, 1, 500)

# Bias
bias_lines = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 3, 5, 10]

# TS
ts_lines = [0.1, 0.165, 0.23, 0.285, 0.333, 0.375, 0.41, 0.444, 0.472]

plt.figure(figsize=(9, 8))

# ------------------------------------------------------------
# Bias
# ------------------------------------------------------------
for b in bias_lines:
    y_bias = b * x
    plt.plot(x, y_bias, color="lightgray", linestyle="-", linewidth=1)

    if b <= 1:
        plt.text(1.01, b, f"{b}", fontsize=9, va="center")
    else:
        xmax = 1 / b
        plt.text(xmax, 1.01, f"{b}", fontsize=9, ha="center")

# ------------------------------------------------------------
# TS
# ------------------------------------------------------------
for t in ts_lines:
    ts_curve = (t * x) / (x - t)
    ts_curve[x <= t] = np.nan
    plt.plot(x, ts_curve, color="#5a5a5a", linestyle="--", linewidth=1)

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------
for exp_name, filename in EXPERIMENTS.items():

    df = read_statistics(filename)
    color = COLORS[exp_name]

    for idx, row in df.iterrows():
        obs_level = row["OBS_Level"]
        label = LEVEL_LABELS.get(obs_level, obs_level)

        SR = row["SR"]
        POD = row["POD"]

        marker = MARKERS[list(LEVEL_LABELS.keys()).index(obs_level)]

        plt.scatter(
            SR, POD,
            color=color,
            marker=marker,
            s=110,
            label=f"{exp_name} – {label}"
        )

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Success Ratio (SR = 1 - FAR)", fontsize=16)
plt.ylabel("Probability of Detection (POD)", fontsize=16)
plt.title("Performance Diagram – Utö", fontsize=18, pad=20)

plt.grid(True, linestyle="--", alpha=0.4)

handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))

# Legend below the pic
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    fontsize=10,
    frameon=False,
    ncol=4
)

# More space for legend
plt.subplots_adjust(bottom=0.25)


plt.tight_layout()

plt.savefig("POD_Uto_new.png", dpi=300, bbox_inches="tight")
print("Saved POD_Uto_new.png")
