# ================================================================
# Script for reading observational data and comparing it against 
# model point-forecast NetCDF files (point_fc*.nc), which 
# contain a single latitude–longitude point and hybrid levels.
#
# The script:
#   1. Loads OBS data (mast temperatures, RH, pressures)
#   2. Computes water vapor pressure, specific humidity, N, M, dM/dh
#   3. Loads MODEL DATA from point NetCDF files
#   4. Matches hybrid model heights to observation heights
#   5. Exports model time series to CSV
#   6. Produces scatter comparisons (OBS vs MODEL)
#
#
# Author: Karoliina H. (FMI), 2026
# ================================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import os
from datetime import datetime
from collections import Counter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FixedLocator

# ---------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------
EX = '2500L65'
EXP = f'DUCT_{EX}'

# Observation point (information only)
lat = 59.78094
lon = 21.35858

ofont = 26
afont = 22
tfont = 16


# ================================================================
# FUNCTIONS
# ================================================================

def pressure(air_pressure, h_diff):
    """Estimate pressure at a new height (hydrostatic approximation)."""
    return air_pressure + (1.225 * 9.81 * h_diff) / 100

def e_s(T):
    """Saturation vapor pressure (hPa). T in Celsius."""
    return 0.01 * 611 * np.exp((17.27 * T) / (237.3 + T))

def e(RH, es):
    """Water vapor pressure (hPa)."""
    return (RH * es) / 100

def N(p, T, e):
    """Radio refractivity approximation."""
    T = T + 273.15
    return 77.6*(p/T) + 3.73e5*(e/(T*T))

def M(N, z):
    """Modified refractivity."""
    return N + 0.157*z

def calc_dMdh(M, h):
    """Gradient of modified refractivity."""
    dM = np.diff(M)
    dh = np.diff(h)
    return dM/dh, dh

def extract_time(file_name):
    """Extract datetime and lead time from point_fc file format."""
    dtime_str = file_name[8:18]      # point_fcYYYYMMDDHH
    dtime = datetime.strptime(dtime_str, '%Y%m%d%H')
    lead_str = file_name.split('+')[-1].replace('.nc', '')
    return dtime, int(lead_str)

def specific_humidity(e, P):
    """Compute specific humidity from vapor pressure."""
    epsilon = 0.622
    return (epsilon*e)/(P - (1-epsilon)*e)


# ================================================================
# OBSERVATION DATA PROCESSING
# ================================================================

mast = pd.read_csv('/csv-files_path/Utö_30m_masto_testijakso.csv')
synop = pd.read_csv('/csv-files_path/Utö_synop_testijakso.csv')
mast60 = pd.read_csv('/csv-files_path/Uto_T_RH_59m_1min.csv')

# --- Mast data formatting ---
Tvars = ['TA #16 2m (degC)', 'TA #17 5m (degC)', 'TA 10m (degC)',
         'TA #2 20m (degC)', 'TA #3 30m (degC)']
RHvars = ['RH #16 2m (%)', 'RH #17 5m (%)', 'RH 10m (%)',
          'RH #2 20m (%)', 'RH #3 30m (%)']

T_data = mast[Tvars]
RH_data = mast[RHvars]
TIME_data = mast['DATA_TIME']

T_data.columns = ['T_2m','T_5m','T_10m','T_20m','T_30m']
RH_data.columns = ['RH_2m','RH_5m','RH_10m','RH_20m','RH_30m']

mast = pd.concat([TIME_data, T_data, RH_data], axis=1)
mast['time'] = pd.to_datetime(mast['DATA_TIME'])
mast_hourly = mast.resample('H', on='time').mean()

# --- 59m mast ---
mast60['time'] = pd.to_datetime(mast60["Time"])
mast60 = mast60[['time','T [degC]','RH [%]']]
mast60.columns = ['time','T_59m','RH_59m']
mast60 = mast60.set_index('time')
mast60_hourly = mast60.resample("H").mean()

# --- SYNOP pressure ---
synop['time'] = pd.to_datetime(synop["DATA_TIME"])
synop = synop[['time','PA0']].set_index('time').resample("H").mean()

# Pressures at heights
P_2m  = pressure(synop,  8)
P_5m  = pressure(synop,  5)
P_10m = pressure(synop,  0)
P_20m = pressure(synop,-10)
P_30m = pressure(synop,-20)
P_59m = pressure(synop,-49)

P_data = pd.concat([P_2m,P_5m,P_10m,P_20m,P_30m,P_59m],axis=1)
P_data.columns = ['P_2m','P_5m','P_10m','P_20m','P_30m','P_59m']

# Vapor pressures
ES_2m = e_s(mast_hourly['T_2m'])
ES_5m = e_s(mast_hourly['T_5m'])
ES_10m = e_s(mast_hourly['T_10m'])
ES_20m = e_s(mast_hourly['T_20m'])
ES_30m = e_s(mast_hourly['T_30m'])
ES_59m = e_s(mast60_hourly['T_59m'])

E_2m = e(mast_hourly['RH_2m'], ES_2m)
E_5m = e(mast_hourly['RH_5m'], ES_5m)
E_10m = e(mast_hourly['RH_10m'], ES_10m)
E_20m = e(mast_hourly['RH_20m'], ES_20m)
E_30m = e(mast_hourly['RH_30m'], ES_30m)
E_59m = e(mast60_hourly['RH_59m'], ES_59m)

E_data = pd.concat([E_2m,E_5m,E_10m,E_20m,E_30m,E_59m],axis=1)
E_data.columns = ['E_2m','E_5m','E_10m','E_20m','E_30m','E_59m']

# Specific humidity
q_data = pd.concat([
    specific_humidity(E_data['E_2m'], P_data['P_2m']),
    specific_humidity(E_data['E_5m'], P_data['P_5m']),
    specific_humidity(E_data['E_10m'],P_data['P_10m']),
    specific_humidity(E_data['E_20m'],P_data['P_20m']),
    specific_humidity(E_data['E_30m'],P_data['P_30m']),
    specific_humidity(E_data['E_59m'],P_data['P_59m'])
], axis=1)
q_data.columns=['q_2m','q_5m','q_10m','q_20m','q_30m','q_59m']

# Refractivity N
N_data = pd.concat([
    N(P_data['P_2m'], mast_hourly['T_2m'], E_2m),
    N(P_data['P_5m'], mast_hourly['T_5m'], E_5m),
    N(P_data['P_10m'],mast_hourly['T_10m'],E_10m),
    N(P_data['P_20m'],mast_hourly['T_20m'],E_20m),
    N(P_data['P_30m'],mast_hourly['T_30m'],E_30m),
    N(P_data['P_59m'],mast60_hourly['T_59m'],E_59m)
], axis=1)
N_data.columns=['N_2m','N_5m','N_10m','N_20m','N_30m','N_59m']

# Modified refractivity M
heights = np.array([2,5,10,20,30,59])
M_data = pd.concat([
    M(N_data['N_2m'],2),
    M(N_data['N_5m'],5),
    M(N_data['N_10m'],10),
    M(N_data['N_20m'],20),
    M(N_data['N_30m'],30),
    M(N_data['N_59m'],59)
], axis=1)
M_data.columns=['M_2m','M_5m','M_10m','M_20m','M_30m','M_59m']

# dM/dh
Mprof = M(N_data, heights)
dMdh, dh = calc_dMdh(Mprof, heights)

df_dMdh = pd.DataFrame(dMdh, columns=[f'dMdh_{i}' for i in range(1,6)])
df_dMdh['time'] = N_data.index
df_dMdh = df_dMdh.set_index('time')

# Combine all obs
OBSS = pd.concat([
    mast_hourly, mast60_hourly, P_data, E_data, N_data, M_data, df_dMdh, q_data
], axis=1)

# ================================================================
# MODEL DATA
# ================================================================

dir_path = f'/nc_file_path/{EXP}/'
all_files = os.listdir(dir_path)
file_paths = [f for f in all_files if f.startswith("point_fc") and f.endswith(".nc")]
file_paths = sorted(file_paths, key=lambda f: extract_time(f))

obs_heights = [2,5,10,20,30,59]
timestamps_list=[]
data_list=[]
h_values=[]

for file_path in file_paths:
    print("Processing:", dir_path+file_path)
    dset = xr.open_dataset(dir_path+file_path)

    timestamp = dset["time"].values
    timestamps_list.append(timestamp)

    # Hybrid levels
    h_arr = dset["h"].values
    if h_arr.ndim == 2:
        h_arr = h_arr[0,:]
    h_values.append(h_arr)

    row = {"Time": timestamp}

    # Match obs heights to hybrid levels
    match = {h: int(np.argmin(abs(h_arr - h))) for h in obs_heights}

    for hgt, lvl in match.items():
        row[f"T_{hgt}m"] = float(dset["t"].values[lvl] - 273.15)
        row[f"E_{hgt}m"] = float(dset["e"].values[lvl])
        row[f"P_{hgt}m"] = float(dset["p"].values[lvl]/100)
        row[f"M_{hgt}m"] = float(dset["M"].values[lvl])
        row[f"q_{hgt}m"] = float(dset["q"].values[lvl])
        row[f"U_{hgt}m"] = float(dset["u"].values[lvl])
        row[f"V_{hgt}m"] = float(dset["v"].values[lvl])

    dset.close()
    data_list.append(row)

# MODEL DATA → DataFrame
df = pd.DataFrame(data_list)
df.set_index("Time", inplace=True)
df.to_csv(f'Model_data_Uto_{EX}.csv')


# ================================================================
# SCATTER ANALYSIS
# ================================================================

tasoparit = {c: c for c in [
    'q_2m','q_5m','q_10m','q_20m','q_30m','q_59m',
    'M_2m','M_5m','M_10m','M_20m','M_30m','M_59m',
    'T_2m','T_5m','T_10m','T_20m','T_30m','T_59m',
    'E_2m','E_5m','E_10m','E_20m','E_30m','E_59m'
]}

color_map={'T':'red','E':'blue','q':'blue','M':'green'}

for obs_col, mod_col in tasoparit.items():

    if obs_col not in OBSS.columns or mod_col not in df.columns:
        continue

    obsU = OBSS[obs_col].values
    modelU = df[mod_col].values

    # Remove NaN pairs
    mask = ~(np.isnan(obsU) | np.isnan(modelU))
    obsU = obsU[mask]
    modelU = modelU[mask]

    # Color based on variable type
    key = obs_col.split('_')[0]
    c = color_map.get(key,'gray')

    plt.figure(figsize=(6,6))
    plt.scatter(obsU, modelU, alpha=0.6, color=c)

    # Axis limits
    if key=='T':
        plt.xlim(12,26); plt.ylim(12,26)
    elif key=='E':
        plt.xlim(8,26); plt.ylim(8,26)
    elif key=='M':
        plt.xlim(310,380); plt.ylim(310,380)
    elif key=='q':
        plt.xlim(0.005,0.016); plt.ylim(0.005,0.016)

    identity=np.linspace(-10,500,20)
    plt.plot(identity, identity, 'k--', lw=0.5)

    plt.xlabel(f"{obs_col} obs", fontsize=ofont)
    plt.ylabel(f"{mod_col} model", fontsize=ofont)
    plt.title(f"EXP={EX}", fontsize=afont)
    plt.grid(True)

    outname=f"/plots_path/{EXP}_scatter_{mod_col}.png"
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    plt.close()

print("END")
