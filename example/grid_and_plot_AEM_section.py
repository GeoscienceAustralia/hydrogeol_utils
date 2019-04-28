
# !/usr/bin/env python

# ===============================================================================
#    Copyright 2017 Geoscience Australia
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ===============================================================================

'''
Created on 16/4/2019
@author: Neil Symington

This script demonstrates a basic example of gridding and plotting a section. It stores the gridded data in memory
and returns a matplotlib plot

'''

import netCDF4
import matplotlib.pyplot as plt
import hydrogeol_utils.plotting_utils as plot_utils
import os

# Open netcdf files

ncdir ='\\\\prod.lan\\active\\proj\\futurex\\East_Kimberley\\Data\\Processed\\Geophysics\\AEM\\EK_nbc_inversions\\' \
       'OrdKeep_borehole_constrained\\netcdf'

# Open the file with the EM measurements
# Here we use the data response file provided by Niel Christensen
EM_path = os.path.join(ncdir,'OrdKeep2019_DataResp_cor2DLogOrd.nc')
EM_dataset = netCDF4.Dataset(EM_path)


# Open the file with the many layer model conductivity values
# The conductivity model was a 2d correlated borehole
# constrained inversion done by Niel Christensen

cond_path = os.path.join(ncdir,'OrdKeep2019_ModeExp_cor2DLogOrd.nc')
cond_dataset = netCDF4.Dataset(cond_path)

# Create an instance of plots for plotting the data
plots = plot_utils.ConductivitySectionPlot(cond_dataset, EM_dataset)


plots.conductivity_variables = ['conductivity', 'data_residual', 'tx_height_measured', 'depth_of_investigation']

plots.EM_variables  = ['data_values_by_low_moment_gate', 'data_values_by_high_moment_gate']

# Define the plot settings


plot_settings = {'figsize': (20, 11), 'dpi': 350}

panel_settings = {'panel_1': {'variable': 'data_values_by_high_moment_gate',
                              'plot_type': 'multi_line',
                              'panel_kwargs': {'title': 'high moment data',
                                               'ylabel': 'dB/dT (V/(A.turns.m^4))'},
                              'height_ratio': 4},

                  'panel_2': {'variable': 'data_values_by_low_moment_gate',
                              'plot_type': 'multi_line',
                              'panel_kwargs': {'title': 'low moment data',
                                               'ylabel': 'dB/dT (V/(A.turns.m^4))'},
                              'height_ratio': 4},

                  'panel_3': {'variable': 'data_residual',
                              'plot_type': 'line',
                              'panel_kwargs': {'title': 'data residual', 'color': 'black',
                                               'ylabel': 'data residual',
                                               'legend': False},
                              'height_ratio': 1},

                  'panel_4': {'variable': 'conductivity',
                              'plot_type': 'grid',
                              'panel_kwargs': {'title': 'AEM conductivity',
                                               'max_depth': 300, 'shade_doi': True, 'colourbar': True,
                                               'colourbar_label': 'Conductivity (S/m)',
                                               'log_plot': True, 'vmin': 0.001, 'vmax': 0.5,
                                               'cmap': 'jet', 'ylabel': 'elevation_(mAHD)'},
                              'height_ratio': 4}}

# Sections will be saved locally
outdir = r'..\example_sections'

if not os.path.exists(outdir):
    os.mkdir(outdir)

# Define the lines to plot
lines = [306102, 803901]

# Define grid resolution
xres, yres = 10.,2.

gridded_variables = plots.grid_variables(xres = xres, yres =yres, lines=lines,
                                          resampling_method = 'linear', save_hdf5 = False,
                                         return_dict = True)
# ITerate through lines
for line in lines:
    # Define filename

    outfile = os.path.join(outdir, str(line) + '.png')

    # Create figure using panel settings above
    fig, ax_array = plt.subplots(len(panel_settings), 1, sharex=True, figsize=(20, 11),
                                 gridspec_kw={'height_ratios':
                                                  plot_utils.unpack_plot_settings(panel_settings, 'height_ratio')})

    plot_utils.plot_conductivity_section(ax_array, gridded_variables[line], plot_settings, panel_settings,
                              save_fig=True, outfile=outfile)



