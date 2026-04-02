# ================================================================
# Script for reading point-forecast NetCDF files produced from the
# MEPS/DUCT model output. These files contain only a single 
# latitude-longitude point and hybrid vertical levels.
#
# The script extracts vertical profiles of:
#   - Temperature (T)
#   - Water vapor pressure (e)
#   - Modified refractivity (M)
#   - Height (h)
#
# It then plots the atmospheric profiles at the chosen forecast
# time for multiple experiments and saves the resulting figures.
#
# Author: Karoliina H. (FMI), 2026
# ================================================================

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------
# User settings
# ---------------------------------------------------------------
mm = '07'          # Month
dd = '06'          # Day
hh = '012'         # Lead time (e.g., +012h)
hhh = '12:00:00'   # For title text
prefix = 'DUCT_'

# Experiments to be processed
EXP_names = ['2500L65', '2500L90']

# ---------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------
for EXP in EXP_names:

    # -----------------------------------------------------------
    # Construct the path to the point-forecast NetCDF file
    # -----------------------------------------------------------
    nc_file = (
        '/nc_input/file_path/' +
        prefix + EXP + '/point_fc2022' + mm + dd + '00+' + hh + '.nc'
    )

    print("Reading:", nc_file)
    data = nc.Dataset(nc_file)

    # -----------------------------------------------------------
    # Read coordinate metadata (scalars in point NetCDF files)
    # -----------------------------------------------------------
    lat = data.variables['latitude'][()]   # scalar latitude
    lon = data.variables['longitude'][()]  # scalar longitude

    # -----------------------------------------------------------
    # Read vertical profiles directly (no grid indexing)
    # Each variable is stored as (hybrid) or (time, hybrid)
    # -----------------------------------------------------------
    def read_profile(var):
        arr = data.variables[var][:]
        if arr.ndim == 2:
            return arr[0, :]   # (time, hybrid) -> take time index 0
        else:
            return arr[:]      # (hybrid)
    
    h_profile = read_profile('h')
    T_profile = read_profile('t')            # Temperature (K)
    e_profile = read_profile('e')            # Water vapor pressure (hPa)
    M_profile = read_profile('M')            # Modified refractivity (-)

    print("Profiles loaded. Number of hybrid levels:", h_profile.shape[0])

    # -----------------------------------------------------------
    # Create a figure with 3 profile panels
    # -----------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(7, 7), sharey=True)

    ofont = 22
    afont = 22
    ffont = 18

    # Temperature profile
    axs[0].plot(T_profile - 273.15, h_profile, color='red', marker='o', lw=3)
    axs[0].set_xlabel('T (°C)', fontsize=afont)
    axs[0].set_ylabel('Height (m)', fontsize=afont)
    axs[0].set_title('Temperature', fontsize=ffont)
    axs[0].set_xlim(15, 18)
    axs[0].grid(True)

    # Water vapor pressure
    axs[1].plot(e_profile, h_profile, color='blue', marker='o', lw=3)
    axs[1].set_xlabel('e (hPa)', fontsize=afont)
    axs[1].set_title('Water vapor\npressure', fontsize=ffont)
    axs[1].set_xlim(12, 15)
    axs[1].grid(True)

    # Modified refractivity
    axs[2].plot(M_profile, h_profile, color='green', marker='o', lw=3)
    axs[2].set_xlabel('M (-)', fontsize=afont)
    axs[2].set_title('Modified\nrefractivity', fontsize=ffont)
    axs[2].set_xlim(320, 350)
    axs[2].grid(True)

    # Shared y-axis settings
    for ax in axs:
        ax.set_ylim(0, 60)     # Limit to the lowest 60 m
        ax.tick_params(axis='both', labelsize=12)

    # Title
    fig.suptitle(
        f'Utö EXP={EXP} profile\n2022-{mm}-{dd} {hhh}',
        fontsize=ofont
    )

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # Save the figure
    # -----------------------------------------------------------
    savepath = (
        '/plots/output_path/Model_profile_' +
        EXP + '_2022' + mm + dd + hh + '.png'
    )
    fig.savefig(savepath, dpi=200, bbox_inches='tight')
    print("Saved:", savepath)
