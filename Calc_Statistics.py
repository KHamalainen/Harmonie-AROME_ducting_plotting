# ================================================================
# Statistics_4models_from_NC.py  (FINAL, BUG-FIXED VERSION)
#
# ✅ Uses ORIGINAL logic of Kontingenssi_Utö_60m.py
# ✅ BUT fixes the original bug where FA and Miss columns were swapped
# ✅ Reads model dMdh directly from point_fc*.nc (hybrid dimension)
# ✅ Computes OBS dMdh from [2,5,10,20,30,59] m
# ✅ Pairs OBS levels with MODEL hybrid-level dMdh correctly
# ================================================================

import pandas as pd
import numpy as np
import xarray as xr
import os
from datetime import datetime

# ================================================================
# EXPERIMENTS
# ================================================================

EXPERIMENTS = [
    "2500L65",
    "2500L90",
    "750L65",
    "750L90"
]

BASE_PATH = "/ec/res4/scratch/fn7/DUCT/netCDF/new_files/"
LOC_NAME  = "Utö"

# ================================================================
# FUNCTIONS (exactly as original)
# ================================================================

def pressure(air_pressure, h_diff):
    return air_pressure + (1.225 * 9.81 * h_diff)/100

def e_s(T):
    return 0.01 * 611 * np.exp((17.27*T)/(237.3 + T))

def e(RH, es):
    return (RH * es)/100

def Ncalc(p,T,e):
    Tk = T + 273.15
    return 77.6*(p/Tk)+ 3.73e5*(e/(Tk*Tk))

def Mcalc(N,z):
    return N + 0.157*z

def calc_dMdh(M,h):
    return np.diff(M)/np.diff(h)

def extract_time(fname):
    base = fname.split("point_fc")[1][:10]   # YYYYMMDDHH
    lead = int(fname.split("+")[1].replace(".nc",""))
    return datetime.strptime(base,"%Y%m%d%H") + pd.Timedelta(hours=lead)

# ================================================================
# LOAD OBS DATA (exact original logic)
# ================================================================

PATH_M30 = "/home/fn7/Kanavoituminen/Utö_30m_masto_testijakso.csv"
PATH_M59 = "/home/fn7/Kanavoituminen/Uto_T_RH_59m_1min.csv"
PATH_SYN = "/home/fn7/Kanavoituminen/Utö_synop_testijakso.csv"

# --- Mast 2–30 m ---
mast = pd.read_csv(PATH_M30)
Tvars = ['TA #16 2m (degC)', 'TA #17 5m (degC)',
         'TA 10m (degC)', 'TA #2 20m (degC)', 'TA #3 30m (degC)']
RHvars = ['RH #16 2m (%)','RH #17 5m (%)','RH 10m (%)',
          'RH #2 20m (%)','RH #3 30m (%)']

mast["time"] = pd.to_datetime(mast["DATA_TIME"])
mast = mast.set_index("time")[Tvars + RHvars]
mast.columns = ["T_2m","T_5m","T_10m","T_20m","T_30m",
                "RH_2m","RH_5m","RH_10m","RH_20m","RH_30m"]
mast = mast.resample("H").mean()

# --- Mast 59 m ---
mast59 = pd.read_csv(PATH_M59)
mast59["time"] = pd.to_datetime(mast59["Time"])
mast59 = mast59.set_index("time")[["T [degC]","RH [%]"]]
mast59.columns = ["T_59m","RH_59m"]
mast59 = mast59.resample("H").mean()

# --- SYNOP ---
syn = pd.read_csv(PATH_SYN)
syn["time"] = pd.to_datetime(syn["DATA_TIME"])
syn = syn.set_index("time")[["PA0"]]
syn = syn.resample("H").mean()

# --- Pressure at heights ---
P = pd.concat([
    pressure(syn,  8),
    pressure(syn,  5),
    pressure(syn,  0),
    pressure(syn,-10),
    pressure(syn,-20),
    pressure(syn,-49)
], axis=1)
P.columns = ["P_2m","P_5m","P_10m","P_20m","P_30m","P_59m"]

# Combine OBS
OBS = mast.join(mast59, how="inner").join(P, how="inner")

# Compute OBS E, q, N, M
for h in [2,5,10,20,30,59]:
    T  = OBS[f"T_{h}m"]
    RH = OBS[f"RH_{h}m"]
    Pp = OBS[f"P_{h}m"]

    es = e_s(T)
    ev = e(RH,es)
    qv = (0.622*ev)/(Pp-(1-0.622)*ev)
    Nn = Ncalc(Pp,T,ev)
    Mm = Mcalc(Nn,h)

    OBS[f"E_{h}m"] = ev
    OBS[f"q_{h}m"] = qv
    OBS[f"N_{h}m"] = Nn
    OBS[f"M_{h}m"] = Mm

# OBS dMdh at fixed heights
OBS_heights = np.array([2,5,10,20,30,59])
Mprof = OBS[[f"M_{h}m" for h in OBS_heights]].values
dM_obs = calc_dMdh(Mprof, OBS_heights)

OBS["dMdh_1"] = dM_obs[:,0]
OBS["dMdh_2"] = dM_obs[:,1]
OBS["dMdh_3"] = dM_obs[:,2]
OBS["dMdh_4"] = dM_obs[:,3]
OBS["dMdh_5"] = dM_obs[:,4]


# ================================================================
# LOOP EXPERIMENTS
# ================================================================

for EX in EXPERIMENTS:

    print(f"\n=== PROCESSING {EX} ===")

    dirname = BASE_PATH + f"DUCT_{EX}/"
    files = sorted([f for f in os.listdir(dirname) if f.startswith("point_fc")],
                   key=extract_time)

    # Extract model dMdh from NC files
    rows = []

    for fname in files:
        ds = xr.open_dataset(dirname + fname)

        h = ds["h"].values        # hybrid height dimension only
        dmdh = ds["dMdh"].values  # shape: (hybrid)

        # Hybrid levels matching OBS pairing locations:
        model_heights = [5,10,20,30,59]
        idx = [int(np.argmin(np.abs(h-H))) for H in model_heights]

        row = {"time": extract_time(fname)}

        for H, level in zip(model_heights, idx):
            row[f"dMdh_{H}m"] = float(dmdh[level])

        rows.append(row)
        ds.close()

    model = pd.DataFrame(rows).set_index("time").sort_index()

    # Align OBS & MODEL
    COMMON = OBS.index.intersection(model.index)
    OBS2 = OBS.loc[COMMON]
    MOD2 = model.loc[COMMON]

    # ORIGINAL LEVEL PAIRS
    pairs = {
        "dMdh_1": "dMdh_5m",
        "dMdh_2": "dMdh_10m",
        "dMdh_3": "dMdh_20m",
        "dMdh_4": "dMdh_30m",
        "dMdh_5": "dMdh_59m",
    }

    results = []

    # Compute statistics
    for obs_col, mod_col in pairs.items():

        obs_arr = OBS2[obs_col].values
        mod_arr = MOD2[mod_col].values

        mask = ~(np.isnan(obs_arr)|np.isnan(mod_arr))
        obs_arr = obs_arr[mask]
        mod_arr = mod_arr[mask]

        obs_neg = obs_arr < 0
        mod_neg = mod_arr < 0

        TN = np.sum(obs_neg & mod_neg)
        FP = np.sum(~obs_neg & mod_neg)
        FN = np.sum(obs_neg & ~mod_neg)
        TP = np.sum(~obs_neg & ~mod_neg)

        # METRICS (same as original)
        POD  = TP/(TP+FN) if TP+FN>0 else np.nan
        FAR  = FP/(TP+FP) if TP+FP>0 else np.nan
        Bias = (TP+FP)/(TP+FN) if TP+FN>0 else np.nan
        PC   = (TP+TN)/(TP+TN+FP+FN)
        TS   = TP/(TP+FP+FN) if TP+FP+FN>0 else np.nan

        # ✅ FIXED ORDER: Hit, FA, Miss, CR (correct assignment)
        results.append([
            obs_col, mod_col, POD, FAR, Bias, PC, TS,
            TP,     # Hit
            FP,     # False Alarm
            FN,     # Miss
            TN      # Correct Negative
        ])

    # Save
    outname = f"Statistics_{LOC_NAME}_{EX}.csv"
    dfout = pd.DataFrame(results,
                         columns=["OBS_Level","Model_Level","POD","FAR","Bias","PC","TS",
                                  "Hit","FA","Miss","CR"])
    dfout.to_csv(outname, index=False)

    print(f"✅ Saved {outname}")

print("\n✅ ALL MODELS PROCESSED — ORIGINAL NC LOGIC + CORRECTED OUTPUT ORDER")
