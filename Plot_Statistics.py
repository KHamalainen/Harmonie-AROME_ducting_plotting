# ================================================================
# Statistics_4models.py — dMdh ONLY (Final Clean Version)
#
# Reads:
#   • Observations (mast 2–30m, mast 59m, synop pressure)
#   • Four model CSV files (Model_data_Uto_XXXX.csv)
#
# Computes categorical statistics **only for dMdh_1 ... dMdh_5**:
#     trapping = dMdh < 0
#
# Saves:
#     Statistics_Utö_2500L65.csv
#     Statistics_Utö_2500L90.csv
#     Statistics_Utö_750L65.csv
#     Statistics_Utö_750L90.csv
#
# ================================================================

import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# MODEL FILES
# ---------------------------------------------------------------
MODEL_FILES = {
    "2500L65": "/dir/Model_data_Uto_2500L65.csv",
    "2500L90": "/dir/Model_data_Uto_2500L90.csv",
    "750L65":  "/dir/Model_data_Uto_750L65.csv",
    "750L90":  "/dir/Model_data_Uto_750L90.csv",
}

LOC_NAME = "Utö"

# ---------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------
def pressure(air_pressure, h_diff):
    return air_pressure + (1.225 * 9.81 * h_diff)/100

def e_s(T):
    return 0.01*611*np.exp((17.27*T)/(237.3 + T))

def e(RH, es):
    return RH*es/100

def N_refrac(p,T,e):
    Tk = T + 273.15
    return 77.6*(p/Tk) + 3.73e5*(e/(Tk*Tk))

def M_mod(N,z):
    return N + 0.157*z

def specific_humidity(e,p):
    eps = 0.622
    return (eps*e)/(p-(1-eps)*e)

def contingency(obs_flag, mod_flag):
    """Compute categorical verification for boolean flags."""
    TP = np.sum((obs_flag==1) & (mod_flag==1))
    FP = np.sum((obs_flag==0) & (mod_flag==1))
    FN = np.sum((obs_flag==1) & (mod_flag==0))
    TN = np.sum((obs_flag==0) & (mod_flag==0))

    def safe(a,b): return np.nan if b==0 else a/b

    POD  = safe(TP, TP+FN)
    FAR  = safe(FP, TP+FP)
    Bias = safe(TP+FP, TP+FN)
    TS   = safe(TP, TP+FN+FP)
    PC   = safe(TP+TN, TP+TN+FP+FN)

    return POD, FAR, Bias, TS, PC, TP, FP, FN, TN


# ---------------------------------------------------------------
# LOAD OBSERVATIONS (all resampled to hourly)
# ---------------------------------------------------------------

PATH_M30 = "/dir/Utö_30m_masto_testijakso.csv"
PATH_M59 = "/dir/Uto_T_RH_59m_1min.csv"
PATH_SYN = "/dir/Utö_synop_testijakso.csv"

# --- Mast 2–30 m ---
mast = pd.read_csv(PATH_M30)
Tvars = ['TA #16 2m (degC)', 'TA #17 5m (degC)', 'TA 10m (degC)',
         'TA #2 20m (degC)', 'TA #3 30m (degC)']
RHvars = ['RH #16 2m (%)', 'RH #17 5m (%)', 'RH 10m (%)',
          'RH #2 20m (%)', 'RH #3 30m (%)']

mast["time"] = pd.to_datetime(mast["DATA_TIME"])
mast = mast.set_index("time")[Tvars + RHvars]
mast.columns = ["T_2m","T_5m","T_10m","T_20m","T_30m",
                "RH_2m","RH_5m","RH_10m","RH_20m","RH_30m"]
mast = mast.resample("H").first()

# --- Mast 59 m ---
mast59 = pd.read_csv(PATH_M59)
mast59["time"] = pd.to_datetime(mast59["Time"])
mast59 = mast59.set_index("time")[["T [degC]","RH [%]"]]
mast59.columns = ["T_59m","RH_59m"]
mast59 = mast59.resample("H").first()

# --- SYNOP pressure ---
syn = pd.read_csv(PATH_SYN)
syn["time"] = pd.to_datetime(syn["DATA_TIME"])
syn = syn.set_index("time")[["PA0"]].resample("H").first()

# --- Pressure at heights (exact original logic) ---
P = pd.concat([
    pressure(syn,  8),
    pressure(syn,  5),
    pressure(syn,  0),
    pressure(syn,-10),
    pressure(syn,-20),
    pressure(syn,-49)
], axis=1)
P.columns = ["P_2m","P_5m","P_10m","P_20m","P_30m","P_59m"]

# --- Combine OBS ---
OBS = mast.join(mast59, how="inner").join(P, how="inner")

# --- Compute E,q,N,M for OBS ---
for h in [2,5,10,20,30,59]:
    T  = OBS[f"T_{h}m"]
    RH = OBS[f"RH_{h}m"]
    Pp = OBS[f"P_{h}m"]

    es = e_s(T)
    ev = e(RH, es)
    qv = specific_humidity(ev, Pp)
    Nn = N_refrac(Pp, T, ev)
    Mm = M_mod(Nn, h)

    OBS[f"E_{h}m"] = ev
    OBS[f"q_{h}m"] = qv
    OBS[f"N_{h}m"] = Nn
    OBS[f"M_{h}m"] = Mm

# --- OBS: compute dMdh ---
heights = np.array([2,5,10,20,30,59])
Mprof = OBS[[f"M_{h}m" for h in heights]].values
dM = np.diff(Mprof, axis=1) / np.diff(heights)

OBS = pd.concat([
    OBS,
    pd.DataFrame(dM, index=OBS.index,
                 columns=[f"dMdh_{i}" for i in range(1,6)])
], axis=1)

# ---------------------------------------------------------------
# LOOP THROUGH ALL 4 MODEL CSV FILES
# ---------------------------------------------------------------
for EXP, model_csv in MODEL_FILES.items():

    print(f"\n=== PROCESSING MODEL {EXP} ===")

    # Load model CSV
    model = pd.read_csv(model_csv, parse_dates=["Time"])
    model = model.rename(columns={"Time":"time"})
    model = model.set_index("time")

    # Force exact hours
    model.index = model.index.floor("H")

    # Align with OBS
    COMMON = OBS.index.intersection(model.index)
    OBS2 = OBS.loc[COMMON]
    MOD2 = model.loc[COMMON]

    # Compute model dMdh using same heights
    M_cols = [f"M_{h}m" for h in heights]
    if not all(col in MOD2.columns for col in M_cols):
        print(f"⚠ Missing M-columns in model {EXP}, skipping.")
        continue

    Mprof_mod = MOD2[M_cols].values
    dM_mod = np.diff(Mprof_mod, axis=1) / np.diff(heights)

    for i in range(1,6):
        MOD2[f"dMdh_{i}"] = dM_mod[:,i-1]

    # dMdh columns only
    dmdh_cols = [f"dMdh_{i}" for i in range(1,6)]

    results = []

    for col in dmdh_cols:

        obs_arr = OBS2[col].values
        mod_arr = MOD2[col].values

        mask = ~(np.isnan(obs_arr) | np.isnan(mod_arr))
        obs_arr = obs_arr[mask]
        mod_arr = mod_arr[mask]

        # Classification: trapping = dMdh < 0
        obs_flag = (obs_arr < 0).astype(int)
        mod_flag = (mod_arr < 0).astype(int)

        POD, FAR, Bias, TS, PC, TP, FP, FN, TN = contingency(obs_flag, mod_flag)

        results.append([col, POD, FAR, Bias, TS, PC, TP, FP, FN, TN])

    # Save results
    outname = f"Statistics_{LOC_NAME}_{EXP}.csv"
    stats_df = pd.DataFrame(results, columns=[
        "Variable","POD","FAR","Bias","TS","PC",
        "Hits","FalseAlarms","Misses","CorrectNegatives"
    ])
    stats_df.to_csv(outname, index=False)

    print(f"✅ Saved: {outname}")

print("\n✅ All models processed — dMdh statistics only.")
