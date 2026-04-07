# T_vs_M and e_vs_M correlation analysis based ONLY on observational data.
# All mast levels (2–30 m + 59 m) unified to 2–30–60 m.
# Five gradient layers:
#     2–5 m, 5–10 m, 10–20 m, 20–30 m, 30–60 m
# 
# Plots:
#     • Left:  dT/dz vs dM/dz   (orange = where dT/dz > 0 & dM/dz < 0)
#     • Right: de/dz vs dM/dz   (blue   = where de/dz < 0 & dM/dz < 0)
# 
# Author: Karoliina H. (FMI), 2026
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# READ OBSERVATIONAL DATA
# ============================================================

mast = pd.read_csv('/dir/Utö_30m_masto_testijakso.csv')
synop = pd.read_csv('/dir/Utö_synop_testijakso.csv')
mast60 = pd.read_csv('/dir/Uto_T_RH_59m_1min.csv')

# --- Make datetime columns
mast['time'] = pd.to_datetime(mast['DATA_TIME'])
synop['time'] = pd.to_datetime(synop['DATA_TIME'])
mast60['time'] = pd.to_datetime(mast60['Time'])

mast = mast.set_index('time')
synop = synop.set_index('time')
mast60 = mast60.set_index('time')

# ============================================================
# SELECT 2–30 m VARIABLES
# ============================================================
Tvars = ['TA #16 2m (degC)', 'TA #17 5m (degC)', 'TA 10m (degC)',
         'TA #2 20m (degC)', 'TA #3 30m (degC)']
RHvars = ['RH #16 2m (%)', 'RH #17 5m (%)', 'RH 10m (%)',
          'RH #2 20m (%)', 'RH #3 30m (%)']

mast_T = mast[Tvars].copy()
mast_RH = mast[RHvars].copy()

mast_T.columns = ['T_2m','T_5m','T_10m','T_20m','T_30m']
mast_RH.columns = ['RH_2m','RH_5m','RH_10m','RH_20m','RH_30m']

# ============================================================
# SELECT 60 m VARIABLES
# ============================================================
mast60_T = mast60[['T [degC]']].rename(columns={'T [degC]':'T_60m'})
mast60_RH = mast60[['RH [%]']].rename(columns={'RH [%]':'RH_60m'})

# ============================================================
# PRESSURE AND MOISTURE COMPUTATIONS
# ============================================================
def pressure(p, hdiff): return p + (1.225*9.81*hdiff)/100
def e_s(T): return 0.01*611*np.exp((17.27*T)/(237.3+T))
def e_val(RH, es): return (RH*es)/100
def Ncalc(p,T,e):
    Tk = T + 273.15
    return 77.6*(p/Tk) + 3.73e5*(e/(Tk*Tk))
def Mcalc(N,z): return N + 0.157*z

# Compute pressures at heights 2,5,10,20,30,60 m
P = pd.DataFrame(index=synop.index)
P['P_2m']  = pressure(synop['PA0'],  8)
P['P_5m']  = pressure(synop['PA0'],  5)
P['P_10m'] = pressure(synop['PA0'],  0)
P['P_20m'] = pressure(synop['PA0'], -10)
P['P_30m'] = pressure(synop['PA0'], -20)
P['P_60m'] = pressure(synop['PA0'], -49)

# ============================================================
# COMPUTE e, N, M FOR ALL LEVELS
# ============================================================

# Merge T, RH with synop pressure
df_all = mast_T.join(mast_RH).join(mast60_T).join(mast60_RH).join(P, how='inner')

# Compute vapor pressures
for h in [2,5,10,20,30,60]:
    es = e_s(df_all[f'T_{h}m'])
    df_all[f'E_{h}m'] = e_val(df_all[f'RH_{h}m'], es)
    df_all[f'N_{h}m'] = Ncalc(df_all[f'P_{h}m'], df_all[f'T_{h}m'], df_all[f'E_{h}m'])
    df_all[f'M_{h}m'] = Mcalc(df_all[f'N_{h}m'], h)

# ============================================================
# GRADIENTS FOR 5 LAYERS
# ============================================================

# layer heights
z = np.array([2,5,10,20,30,60])

# Prepare gradient containers
for name in ["T","E","M"]:
    for i in range(1,6):
        df_all[f"d{name}dz_{i}"] = np.nan

# Compute gradients layer-by-layer
for i in range(5):
    z1 = z[i]
    z2 = z[i+1]
    Δz = z2 - z1
    for v in ["T","E","M"]:
        df_all[f"d{v}dz_{i+1}"] = (df_all[f"{v}_{z2}m"] - df_all[f"{v}_{z1}m"]) / Δz
      

# ============================================================
# CONDITIONS
# ============================================================

cond_T = []
cond_E = []
cond_ET = []

dT_all = []
dE_all = []
dM_all = []

for i in range(1, 6):
    T_i = df_all[f"dTdz_{i}"]
    E_i = df_all[f"dEdz_{i}"]
    M_i = df_all[f"dMdz_{i}"]

    # --- Conditions ---
    cond_T.append((T_i > 0) & (M_i < 0))
    cond_E.append((E_i < 0) & (M_i < 0))
    cond_ET.append((T_i > 0) & (E_i < 0) & (M_i < 0))

    # --- Data collection ---
    dT_all.append(T_i)
    dE_all.append(E_i)
    dM_all.append(M_i)

# --- Combine all layers into long vectors ---
cond_T_all  = pd.concat(cond_T,  ignore_index=True)
cond_E_all  = pd.concat(cond_E,  ignore_index=True)
cond_ET_all = pd.concat(cond_ET, ignore_index=True)

dT_all = pd.concat(dT_all, ignore_index=True)
dE_all = pd.concat(dE_all, ignore_index=True)
dM_all = pd.concat(dM_all, ignore_index=True)



# --- Print percentages for all conditions ---
T_pct  = cond_T_all.mean()  * 100
E_pct  = cond_E_all.mean()  * 100
ET_pct = cond_ET_all.mean() * 100

print(f"Fraction T-condition (dT/dz > 0 & dM/dz < 0): {T_pct:.2f}%")
print(f"Fraction E-condition (de/dz < 0 & dM/dz < 0): {E_pct:.2f}%")
print(f"Fraction T&E-condition (both true simultaneously): {ET_pct:.2f}%")



# ============================================================
# FINAL PLOTS
# ============================================================

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
afont = 14

# ------------------------------------------------------------
# LEFT: dT/dz vs dM/dz
# ------------------------------------------------------------
axs[0].scatter(dT_all, dM_all, color="gray", s=5, alpha=0.2, label="dM/dz < 0")

axs[0].scatter(
    dT_all[cond_T_all],
    dM_all[cond_T_all],
    s=5, color="orange", alpha=0.3,
    label="Where dT/dz > 0 & dM/dz < 0"
)

axs[0].set_title(
    f"Proportion where dT/dz > 0: {cond_T_all.mean()*100:.1f}%",
    fontsize=afont
)
axs[0].set_xlabel("dT/dz", fontsize=afont)
axs[0].set_ylabel("dM/dz", fontsize=afont)
axs[0].grid(True)
axs[0].set_xlim(-1,1)
axs[0].set_ylim(-1,1)
axs[0].legend()

# ------------------------------------------------------------
# RIGHT: de/dz vs dM/dz
# ------------------------------------------------------------
axs[1].scatter(dE_all, dM_all, color="gray", s=5, alpha=0.2, label="dM/dz < 0")

axs[1].scatter(
    dE_all[cond_E_all],
    dM_all[cond_E_all],
    s=5, color="royalblue", alpha=0.3,
    label="Where de/dz < 0 & dM/dz < 0"
)

axs[1].set_title(
    f"Proportion where de/dz < 0: {cond_E_all.mean()*100:.1f}%",
    fontsize=afont
)
axs[1].set_xlabel("de/dz", fontsize=afont)
axs[1].set_ylabel("dM/dz", fontsize=afont)
axs[1].grid(True)
axs[1].set_xlim(-1,1)
axs[1].set_ylim(-1,1)
axs[1].legend()

plt.tight_layout()
plt.savefig(f"T_and_e_correlation.png", dpi=200, bbox_inches="tight")
plt.close()

print("FIGURE CREATED!")
