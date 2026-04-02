# ================================================================
# Script for reading point-forecast netCDF files (already reduced 
# to a single grid point), extracting vertical profiles over time, 
# and producing meteogram-style plots for several experiments.
#
# Author: Karoliina H. (FMI), 2026
# ================================================================

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import re
import datetime
import matplotlib.dates as mdates
import locale
import copy
import matplotlib as mpl

# Levels
Levels = ['65','90']

# Horizontal resolutions
Reso = ['2500','750']

# Location metadata only (actual files contain one point)
Loc = "Utö"

# ----------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------

for LL in Levels:
    print(LL)
    for HR in Reso:

        exp = 'DUCT_' + str(HR) + 'L' + str(LL)
        exp_nimi = str(HR) + 'L' + str(LL)

        params = ['dMdh']    # test parameter

        for parameter_name in params:

            dir_path = '/netCDF/files/' + exp + '/'
            print(dir_path)

            fig, ax1 = plt.subplots(1, 1, figsize=(16, 4))

            parameter_values = []  # (time, hybrid)
            h_values = []          # (time, hybrid)
            timestamps_list = []
            hybrid_levels = None

            # List all files
            all_files = os.listdir(dir_path)

            # Only files ending with 001.nc
            file_paths = [file for file in all_files if file.endswith('.nc')]

            # Extract timestamps
            file_time = [
                re.search(r'point_fc(\d{4})(\d{2})(\d{2})00\+(\d{3})\.nc', fp).groups()
                for fp in file_paths
            ]

            file_times = [
                datetime.datetime(int(year), int(month), int(day)) +
                datetime.timedelta(hours=int(lead_time))
                for year, month, day, lead_time in file_time
            ]

            sorted_file_paths = [f for _, f in sorted(zip(file_times, file_paths))]

            # ----------------------------------------------------------
            # LOOP OVER POINT NETCDF FILES
            # ----------------------------------------------------------
            for file_path in sorted_file_paths:
                print(file_path)

                dset_nc = xr.open_dataset(dir_path + file_path)

                timestamp = dset_nc['time'].values
                timestamps_list.append(timestamp)

                # READ PARAMETER
                var_data = dset_nc[parameter_name]
                if var_data.ndim == 2:
                    var_data = var_data[0, :]   # remove time dim
                parameter_values.append(var_data.values)

                # READ HEIGHT FIELD
                h_data = dset_nc['h']
                if h_data.ndim == 2:
                    h_data = h_data[0, :]
                h_values.append(h_data.values)

                # Hybrid levels (static)
                if hybrid_levels is None:
                    hybrid_levels = dset_nc.hybrid.values

                dset_nc.close()

            # Convert data to arrays
            parameter_values = np.array(parameter_values)      # (time, 66)
            h_values = np.array(h_values)                      # (time, 66)
            timestamps = np.array(timestamps_list)             # (time,)

            # ----------------------------------------------------------
            # CORRECT HYBRID / HEIGHT ORIENTATION
            # ----------------------------------------------------------
            # Original code reversed hybrid -> surface first, top last
            hybrid_levels = hybrid_levels[::-1]

            # IMPORTANT:
            # h_values is (time, 66)
            # We want height for each hybrid level (but height does not change in time)
            H = h_values[0][::-1]     # (66,) reverse vertical axis exactly like original code

            # Z (data) must be (height, time)
            Z = parameter_values[:, ::-1].T   # reverse vertical axis, then transpose
                                              # result shape: (66, time)

            # Build 2D grids for pcolormesh
            Y = np.repeat(H[:, np.newaxis], Z.shape[1], axis=1)        # (66, time)
            X = np.repeat(timestamps[np.newaxis, :], Z.shape[0], axis=0)   # (66, time)

            print("timestamps:", timestamps)
            print(parameter_name, "Z shape", Z.shape)

            # ----------------------------------------------------------
            # COLOR MAPPING FOR PARAMETERS
            # ----------------------------------------------------------
            if parameter_name == "dMdh":
                # Mask only exact zeros → white
                masked_values = np.ma.masked_where(Z >= 0, Z)
 
                # Make a safe copy of the colormap before modifying it
                cmap = copy.copy(mpl.cm.get_cmap("Blues_r"))
                cmap.set_bad(color='white')


                vmin = -0.5
                vmax = 0
                norm = Normalize(vmin=vmin, vmax=vmax)

                kuva = ax1.pcolormesh(X, Y, masked_values, cmap=cmap, shading='auto', norm=norm)
                cbar = plt.colorbar(kuva, ax=ax1)
                cbar.set_label('M-gradient (dM/dh)', fontsize=14)

            # ----------------------------------------------------------
            # AXIS FORMAT
            # ----------------------------------------------------------
            myFmt = mdates.DateFormatter('%d.%b')
            locale.setlocale(locale.LC_ALL, "en_GB.utf8")
            ax1.xaxis.set_major_formatter(myFmt)
            ax1.tick_params(axis='x', labelsize=10)
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax1.invert_yaxis()

            plt.ylim(0, 60)
            plt.ylabel('Height (m)', fontsize=14)
            plt.title(f'{Loc}, EXP {exp_nimi}, {parameter_name}', fontsize=18)

            # ----------------------------------------------------------
            # SAVE OUTPUT
            # ----------------------------------------------------------
            output = '/plots/' + exp + '_' + parameter_name + '_meteogram_H.png'
            fig.savefig(output, dpi=300, bbox_inches="tight")
