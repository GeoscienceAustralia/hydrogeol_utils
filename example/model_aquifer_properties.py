
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
"""
Created on 16/4/2019
@author: Neil Symington

This script demonstrates how to calculate aquifer transmissivities from geodata.

Eventually to be turned into a jupyter notebook

"""
from sqlite3 import dbapi2 as sqlite
import pandas as pd
import fiona
from shapely.geometry import Polygon, shape
from shapely import wkt
import numpy as np
from hydrogeol_utils import spatial_functions, AEM_utils, plotting_utils
from hydrogeol_utils import borehole_utils, SNMR_utils
from geophys_utils._netcdf_point_utils import NetCDFPointUtils
from geophys_utils._netcdf_line_utils import NetCDFLineUtils
import matplotlib.pyplot as plt
import os
import math
import sqlalchemy as db
from sqlalchemy import create_engine, event
import netCDF4
import rasterio
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
from hydrogeol_utils.db_utils import makeCon, closeCon

from hydrogeol_utils import grid_utils
import importlib
importlib.reload(spatial_functions)


# This function extracts the K profile using the SDR equation

def SDR_K(df, N = 1, C = 4000):
    '''
    df: dataframe containing SNMR inversion data
    N: empirical exponent for water content when estimating the water content
    C: empirical constant for estimating water content
    '''
    return C * df['Total_water_content'].values * df['T2*'].values

def point_within_bounds(x,y, bounds):
    if (bounds.left < x) & (bounds.right > x):
        if (bounds.bottom < y) & (bounds.top > y):
            return True
    return False

# First we want to define the grid onto which we will be calculating hydraulic
# properties. In this example we will work on the cenozoic aquifer

# Import the top and bottom of the aquifer. These surfaces were picked using a
# 3D interpretation package. In this example we will be working in depth space
# as most of our data are in depth


infile=(r"C:\Users\PCUser\Desktop\EK_data\Interp\KeepRiver_strat_grids\\"
"Cz_top_depth.tif")

cz_top_src = rasterio.open(infile)

infile = (r"C:\Users\PCUser\Desktop\EK_data\Interp\KeepRiver_strat_grids\\"
"Cz_base_depth.tif")

cz_base_src = rasterio.open(infile)

infile = (r"C:\Users\PCUser\Desktop\EK_data\Interp\KeepRiver_strat_grids\\"
"EK_lidar_50m_ave.tif")

topo_src = rasterio.open(infile)

# Open rasters as arrays
cz_base_arr = cz_base_src.read()[0]
cz_top_arr = cz_top_src.read()[0]
topo_arr = topo_src.read()[0]

# Remove nulls from topo array
topo_arr[topo_arr < -99.] = np.nan

cz_thickness = cz_top_arr + cz_base_arr # addition due to positive depths

topo_src.crs.to_proj4()

plt.imshow(topo_arr)

# Define the elevation of the base of the array
bot_arr= topo_arr - cz_base_arr

nrow, ncol = cz_base_arr.shape[0], cz_base_arr.shape[1]

xr, yr = np.abs(cz_base_src.transform.a), np.abs(cz_base_src.transform.e)


# Get the lower left coordinates for x and y-offset inputs into grid
xoff = cz_base_src.bounds.left
yoff = cz_base_src.bounds.bottom

# Create a structured grid
cz_grid = grid_utils.StructuredGrid(delr = np.array(ncol * [yr]),
                                    delc = np.array(nrow * [xr]),
                                    top=topo_arr,#
                                    botm=np.expand_dims(bot_arr, axis=0),
                                    proj4=cz_base_src.crs.to_proj4(),
                                    xoff = xoff,
                                    yoff = yoff,
                                    nlay = 1,
                                    lenuni=2, # units are metres
                                    nrow= nrow,
                                    ncol = ncol
                                    )

# Now lets begin by interpolating the AEM onto this grid
cz_grid

cz_grid.nrow
minx, maxx, miny, maxy = cz_grid.extent
cz_grid.extent

#Open the AEM data

infile=r"C:\Users\PCUser\Desktop\EK_data\AEM\OrdKeep2019_ModeExp_cor2DLogOrd.nc"

ek_cond = netCDF4.Dataset(infile)

# To utilise the geophys_utils for line data create a NetCDFLineUtils instance
cond_line_utils = NetCDFLineUtils(ek_cond)

cond_point_utils = NetCDFPointUtils(ek_cond)

# Display the lines for the conductivity mode
wkt, aem_coords = cond_point_utils.utm_coords(cond_point_utils.xycoords)


# Define gdal algorithm as string - see https://gdal.org/programs/gdal_grid.html
algorithm = 'invdist:power=2:radius1=250:radius2=250:max_points=15:'
algorithm += 'min_points=2:nodata=-32768.0'

grid_kwargs = {'conductivity': {'log_grid': True,
                                'gdal_algorithm': algorithm}}

# Currently this is too resource intense, need to make it more effecient
aem_grid = spatial_functions.grid_points_gdal(cond_point_utils,
                 grid_resolution = xr,
                 variables = 'conductivity',
                 reprojected_grid_bounds = (minx, miny, maxx, maxy),
                 grid_wkt = wkt,
                 point_step=8, # Only use every 4th point
                 grid_kwargs = grid_kwargs)

plt.imshow(aem_grid['conductivity'][0])
