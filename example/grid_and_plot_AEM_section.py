
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
from hydrogeol_utils.plotting_utils import unpack_plot_settings
import os
import warnings
import gc


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

# Assign the conductivity model variables to plot
plots.conductivity_variables = ['conductivity', 'data_residual', 'tx_height_measured', 'depth_of_investigation']

# Assign the EM data variables to grid
plots.EM_variables  = ['data_values_by_low_moment_gate', 'data_values_by_high_moment_gate',
                       ]

# Sections will be saved locally
outdir = r'..\example_sections'

if not os.path.exists(outdir):
    os.mkdir(outdir)

# Define the lines to plot
lines = [304401,
       304501, 304601, 304701, 304801, 304901, 305001, 301701, 301801, 301901, 302001,
       302101, 302201,302301, 302401, 302501, 302601
       
       ]

# Define grid resolution
xres, zres = 20.,2.
                                    
# Define the plot settings
panel_settings = {'panel_1': {'variable': 'data_values_by_high_moment_gate',
                              'plot_type': 'multi_line',
                             'panel_kwargs': {'title': 'high moment data',
                                             'ylabel': 'dB/dT (V/(A.turns.m^4))'},
                             'height_ratio': 2},
                  
                 'panel_2': {'variable': 'data_values_by_low_moment_gate',
                             'plot_type': 'multi_line',
                             'panel_kwargs': {'title': 'low moment data',
                                              'ylabel': 'dB/dT (V/(A.turns.m^4))'},
                             'height_ratio': 2},
                  
                 'panel_3': {'variable': 'data_residual',
                             'plot_type': 'line',
                             'panel_kwargs': {'title': 'data residual', 'color': 'black',
                                              'ylabel': 'data residual',
                                              'legend': False},
                             'height_ratio': 0.5},
                 'panel_4': {'variable': 'tx_height_measured',
                             'plot_type': 'line',
                             'panel_kwargs': {'title': 'TX height', 'color': 'black',
                                              'ylabel': 'mAGL',
                                              'legend': False},
                             'height_ratio': 0.5},
                  
                 'panel_5': {'variable': 'conductivity',
                             'plot_type': 'grid',
                             'panel_kwargs': {'title': 'AEM conductivity',
                                              'max_depth': 200, 'shade_doi': True, 'colourbar': True,
                                              'colourbar_label': 'Conductivity (S/m)',
                                             'log_plot': True, 'vmin': 0.005, 'vmax': 0.5,
                                             'cmap': 'jet', 'ylabel': 'elevation_(mAHD)',
                                             'vertical_exaggeration': 10.},
                             'height_ratio': 2},
                             
                 'panel_6': {'variable': 'conductivity',
                             'plot_type': 'grid',
                             'panel_kwargs': {'title': 'AEM conductivity',
                                              'max_depth': 200, 'shade_doi': True, 'colourbar': True,
                                              'colourbar_label': 'Conductivity (S/m)',
                                             'log_plot': False, 'vmin': 0.0, 'vmax': 0.3,
                                             'cmap': 'jet', 'ylabel': 'elevation_(mAHD)',
                                             'vertical_exaggeration': 10.},
                             'height_ratio': 2}}                                      

height_ratios = unpack_plot_settings(panel_settings,'height_ratio') 
panel_kwargs = unpack_plot_settings(panel_settings, 'panel_kwargs')                     
                             
                         
# The plot settings can be tinkered with to get a better separation between panels
# This is only required when the vertical exaggeration is set explicitly                           
plot_settings = {'vertical_margin': 2.5,
                 'panel_vgap': 1.5, 'plot_width': 11.7 #A3 width
                }
   
# Iterate through lines
for line in lines:
    print(line)
    # Grid the variables and return the grids as dictionaries
    gridded_vars = plots.grid_variables(xres = xres, yres =zres,
                                        lines=line,return_dict = True)

    fig, ax_array = plt.subplots(len(panel_settings),1, sharex = True,
                                gridspec_kw={'height_ratios': height_ratios,
                                            'wspace':plot_settings['panel_vgap']})

    plot_objs = plot_utils.plot_conductivity_section(ax_array, gridded_vars[line], plot_settings,
                                         panel_settings, save_fig = False, )
     
    
    # Set the aspect based on the vertical exageration
    
    plot_utils.format_panels(ax_array, panel_settings, plot_settings)
            
    # Tighten the figure    
    
    fig.tight_layout()
            
    # Relative position of the colourbar
    x0, y0, width, height = [1.01, 0., 0.02, 1.
                            ]
    # Now we want to add the colourbar and coordinates axes
    for i, ax in enumerate(ax_array):
        try:
            if panel_kwargs[i]['colourbar']:
        
                # Add the colourbar to the plots that the user specifies
                plot_utils.add_colourbar(fig, ax_array[i], plot_objs[i], x0, y0, width, height, panel_kwargs[i])
        except KeyError:
            pass
        
    # Add coordinate axes to the bottom of the plot
    ax_pos = plot_utils.align_axes(ax_array)
    
    plot_utils.add_axis_coords('northing', gridded_vars[line]['northing'],
                               ax_array[-1], ax_pos[len(panel_settings) - 1], offset=-0.15)

    plot_utils.add_axis_coords('easting', gridded_vars[line]['easting'], ax_array[-1],
                               ax_pos[len(panel_settings) - 1], offset=-0.4)
    
    outfile = os.path.join(outdir, str(line) + '.png')
    
    plt.savefig(outfile, dpi=300)
    
    gridded_vars = None
    
    gc.collect()



