#!/usr/bin/env python

#===============================================================================
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
#===============================================================================

'''
Created on 9/1/2019
@author: Neil Symington

Spatial functions used for various analysis
'''

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial.ckdtree import cKDTree
from geophys_utils._transect_utils import coords2distance
import rasterio
import math
import os
import gc
import tempfile
from osgeo import gdal
from rasterio import Affine
from rasterio.warp import reproject, Resampling
from hydrogeol_utils import misc_utils
from geophys_utils._crs_utils import transform_coords
import numpy.ma as ma
import matplotlib.pyplot as plt

def inverse_distance_weights(distance, power):
    """
    A function for an calculating array of weights from an array of distances
    using an inverse distance function

    :param dist: array of distances
    :param power: the power o ft
    :return:
    an array of weights
    """
    # In IDW, weights are 1 / distance^(power)
    weights = 1.0 / distance**power

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    return weights

def depth_to_thickness(depth):
    """
    Function for calculating thickness from depth array
    :param depth: an array of depths
    :return:
    a flat array of thicknesses with the last entry being a null
    """
    # Create a new thickness array
    thickness = np.nan*np.ones(shape=depth.shape,
                               dtype=np.float)
    # Iterate through the depth array
    if len(depth.shape) == 1:
        thickness[0:-1] = depth[1:] - depth[:-1]
        return thickness

    elif len(depth.shape) == 2:
        thickness[:, 0:-1] = depth[:, 1:] - depth[:, :-1]
        return thickness
    elif len(depth.shape) == 3:

        thickness[:-1,:,:] = depth[1:,:, :] - depth[:-1,:, :]
        return thickness



def interpolate_layered_model(df, parameter_columns, interval_columns, new_intervals):
    """
    A function that does interpolates model parameters
    from a layered model onto a new set of intervals for

    :param df: dataframe that is contains model parameters and depth intervals
    :param parameter_columns: sequence with column names for the parameters that
    are to be interpolated eg. ['Mobile_water_content', 'Bound_water_content']
    :param interval_columns: sequence with column names for existing depth intervals
    eg. ['Depth_from', 'Depth_to']
    :param new_intervals: dataframe with new intervals. Note that the new intervals
    need have the same column names as the interval_cols
    :return:
    dataframe with new intervals and interpolated parameter values
    """
    # Expand the new intervals and parameters so the top and bottom of layers
    # are represented in a single array
    intervals = np.sort(np.concatenate(tuple([df[c] for c in interval_columns])))

    # The variaous parameters will be added to a dictionary
    params = {}
    # If the parameter input is a string and not a list make it a list
    if isinstance(parameter_columns, ("".__class__, u"".__class__)):
        parameter_columns = [parameter_columns]

    # For each parameter, repeat each value to account for the depth from and depth
    # top
    for p in parameter_columns:
        # Create an array and add it to the dictionary
        params[p] = np.repeat(df[p].values, 2)

    # Now we add the new_intervals to the intervals and param arrays
    # To get the parameters we will do some basic interpolation using scipy

    x = intervals

    # Get just the first column of the new intervals
    xnew = new_intervals[new_intervals.columns[0]].values

    # Remove and values from the new intervals that are outsitde the
    # Range of the original
    xnew = xnew[(xnew > np.min(intervals)) & (xnew < np.max(intervals))]

    # These intervals will have all the old ones and the new ones together

    intervals = np.sort(np.concatenate((intervals, xnew)))

    # For each parameter interpolate and add the new values to the array
    for p in params.keys():
        y = params[p]
        f = interpolate.interp1d(x, y, kind='linear')

        # Find new parameters values
        ynew = f(intervals)

        # Add to the parameter array, we will sort later
        params[p] = ynew

    # Add the new intervals array to the params dictionary
    params['Depth'] = intervals

    # For ease of use we will create a dataframe for the manipulated profiles
    df_interp = pd.DataFrame(data=params)

    # Sort and reset the index
    # df_interp.sort_values(by='Depth', inplace=True)
    df_interp.reset_index(inplace=True, drop=True)

    # Cretae a thickness column
    df_interp['thickness'] = depth_to_thickness(df_interp['Depth'].values)

    # Now we round to 2 decimal places to avoid floating point errors
    df_interp = df_interp.round({'Depth': 2})

    # Create new columns for the interpolated parameter values
    for p in parameter_columns:
        new_intervals[p] = np.nan

    # Iterate through the new intervals and do an average
    for index, row in new_intervals.iterrows():
        # Find the upper and lower depths
        du, dl = row[new_intervals.columns[0]], row[new_intervals.columns[1]]

        # ROund du and dl
        du, dl = np.round(du,2), np.round(dl, 2)

        # Slice the data frame

        df_subset = df_interp[(df_interp['Depth'] >= du) & (df_interp['Depth'] < dl)]


        # Remove zero thickness layers which were only used for interpolation
        df_subset = df_subset[df_subset['thickness'] != 0]

        # Calculate weights based on normalised thickness
        df_subset['weights'] = 0.

        weights = df_subset['thickness']

        weights /= weights.sum()

        df_subset.at[weights.index, 'weights'] = weights


        # Iterate through the parameters and multiply by the corresponding weights
        for p in parameter_columns:
            new_intervals.at[index, p] = (df_subset['weights'] *
                                          df_subset[p]).sum()

    return new_intervals

def nearest_neighbours(points, coords, points_required = 1, max_distance = 250.):
    """

    :param points: array of points to find the nearest neighbour for
    :param coords: coordinates of points
    :param points_required: number of points to return
    :param max_distance: maximum search radius
    :return:
    """
    # Initialise tree instance
    kdtree = cKDTree(data=coords)

    # iterate throught the points and find the nearest neighbour
    distances, indices = kdtree.query(points, k=points_required,
                                      distance_upper_bound=max_distance)
    # Mask out infitnite distances in indices to avoid confusion
    mask = ~np.isfinite(distances)

    distances[mask] = np.nan

    return distances, indices


def interpolate_depth_data(df, parameter_columns, interval_column,
                           new_depths, kind='cubic'):
    """
    A function that interpolates depth data onto a new set of depths
    :param df: dataframe that is contains model parameters and depths
    :param parameter_columns: sequence with column names for the parameters that
    are to be interpolated eg. ['Conductivity', 'GAMMA_CALIBRATED]
    :param interval_column: string with column names for existing depth
    :param new_depths: a numpy array with new intervals. Note that the new intervals
    need have the same column names as the interval_cols
    :return:
    dataframe with new intervals and interpolated parameter values
    """
    # If the parameter input is a string and not a list make it a list
    if isinstance(parameter_columns, ("".__class__, u"".__class__)):
        parameter_columns = [parameter_columns]


    new_columns = parameter_columns.copy()

    new_columns.append(interval_column)

    new_df = pd.DataFrame(columns=new_columns)

    new_df[interval_column] = new_depths

    for item in parameter_columns:

        depths, values = df[interval_column].values, df[item].values

        # Use scipy 1d interpolator
        interp = interpolate.interp1d(depths, values, kind=kind)

        # Add to new dataframe
        new_df[item] = interp(new_depths)

    return new_df


def interpolate_depths_to_intervals(df, parameter_columns, new_depths,
                                    depth_column='Depth', how='mean',
                                    logspace = None):
    """
    A function that interpolates depth data onto a new set of depths

    :param df: dataframe that is contains model parameters and depths
    :param parameter_columns: sequence with column names for the parameters that
    are to be interpolated eg. ['Conductivity', 'GAMMA_CALIBRATED]
    :param depth_column: string with column names for existing depth values
    :param new_depths: a numpy array with new intervals. Note that the new intervals
    need have the same column names as the interval_cols
    :param how: method of averaging values from the interval
    ['mean', 'mode', 'median']

    :return:
    dataframe with new intervals and interpolated parameter values
    """
    # If the parameter input is a string and not a list make it a list
    if isinstance(parameter_columns, ("".__class__, u"".__class__)):
        parameter_columns = [parameter_columns]



    # Create a dataframe
    new_df = pd.DataFrame(columns = ['Depth_from', "Depth_to"],
                          data= new_depths)

    for item in parameter_columns:

        new_df[item] = np.nan

        for i, (index, row) in enumerate(new_df.iterrows()):
            # Get the depth top and bottow
            depth_from, depth_to = row.Depth_from, row.Depth_to

            # Subset the original dataframe
            df_interval = df.loc[(df[depth_column] >= depth_from) &
                                 (df[depth_column] <= depth_to)]
            if logspace:
                df_interval.at[:,item] = np.log10(df_intervals[item])


            if how == 'mean':
                new_df.at[index, item] = df_interval[item].mean()
            elif how == 'median':
                new_df.at[index, item] = df_interval[item].median()
            elif how == 'mode':
                new_df.at[index, item] = df_interval[item].mode()

            if logspace:
                new_df.at[index, item] = 10**(new_df.loc[index, item])


    return new_df


def resample_raster(infile, outfile, gridx, gridy, driver='GTiff',
                    null = -999, return_obj= False):
    """
    A function for resampling a raster onto a regular grid with the same
    crs

    :param infile: input raster path
    :param outfile: output raster path
    :param gridx: numpy array of grid x coordinates
    :param gridy: numpy array of grid y coordinates
    :param driver: rasterio driver
    :param null: value to be replaced by null in the grid

    """
    # Open
    src = rasterio.open(infile)

    # Extract data as an array
    arr = src.read()[0]

    # Get the affine
    aff = src.transform

    # Define the affine for the kriged dataset
    newaff = Affine(gridx[1] - gridx[0], 0, np.min(gridx),
                    0, gridy[1] - gridy[0], np.max(gridy))

    # Create new array with the grid
    # coordinates from the kriging
    newarr = np.empty(shape=(gridy.shape[0],
                             gridx.shape[0]))

    # Reproject
    reproject(
        arr, newarr,
        src_transform=aff,
        dst_transform=newaff,
        src_crs=src.crs,
        dst_crs=src.crs,
        resampling=Resampling.bilinear)

    src.close()

    # Do some post processing
    newarr[np.abs(newarr - null) < 0.0001] = np.nan

    # Hack implementation
    newarr[newarr < null] = np.nan



    src.close()

    # Create a new dataset
    new_dataset = rasterio.open(outfile, 'w', driver=driver,
                                height=newarr.shape[0], width=newarr.shape[1],
                                count=1, dtype=newarr.dtype,
                                crs=src.crs, transform=newaff)

    new_dataset.write(newarr, 1)
    if return_obj:
        return new_dataset
    else:
        new_dataset.close()



def resample_categorical_intervals(df, parameter_columns,
                                   interval_columns, new_intervals):
    # If the parameter input is a string and not a list make it a list
    if isinstance(parameter_columns, ("".__class__, u"".__class__)):
        parameter_columns = [parameter_columns]

    # Create a dataframe to add to
    df_resampled = pd.DataFrame(columns=interval_columns, data=new_intervals)

    for p in parameter_columns:

        df_resampled[p] = ''

        # Iterate through the new intervals
        for i, interval in enumerate(new_intervals):

            new_depth_from = interval[0]
            new_depth_to = interval[1]

            mask = (df[interval_columns[0]] < new_depth_to) & (df[interval_columns[1]] > new_depth_from)

            v = df[mask][p].mode()

            if len(v) == 1:

                df_resampled.at[i, p] = v.values[0]

            elif len(v) > 1:

                df_resampled.at[i, p] = 'transition'

    return df_resampled


def point_within_bounds(x,y, bounds):
    """
    Function for checking if a point is within a bounds
     from a raster
    :param x: x-coordiante
    :param y: y-coordinate
    :param bounds: raster bounds
    :return:
    boolean
    """
    if (bounds.left < x) & (bounds.right > x):
        if (bounds.bottom < y) & (bounds.top > y):
            return True
    return False

def interpolate_coordinates(utm_coordinates, new_utm, kind = 'nearest'):
    """

    :param utm_coordinates:
    :param new_utm:
    :param kind:
    :return:
    """

    distances = coords2distance(utm_coordinates)

    return interpolate.griddata(utm_coordinates, distances, new_utm,
                                      method=kind)


def grid_var(var_array, coordinates, grid_kwargs):
    """
    A function for gridding a variable array in gdal
    :param var_array: a column array of the variable to grid
    :param coordinates: utm coordinate array
    :param grid_kwargs: keyword argument dictionary for gridding using  gdal_grid
    :return:
    :param gridded: A gridded array with shape (height, width)
    :param geotransform: the gdal geotransform for the gridded array

    """
    if len(var_array.shape) == 1:
        var_array = var_array.reshape([-1,1])

    if grid_kwargs['log_grid']:
        var_array = np.log10(var_array)

    temp_a = np.concatenate((coordinates, var_array),
                            axis=1)

    # Output the points into a temporary csv

    temp_dir = r'C:\temp\tempfiles'
    # temp_dir = tempfile.gettempdir() # getting permissions error

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    os.chdir(temp_dir)

    temp_file = 'pts2grid_temp.csv'

    with open(temp_file, 'w') as f:
        f.write('x,y,z\n')
        # Write the array
        np.savetxt(f, temp_a, delimiter=',')
    # Write vrt
    vrt_file = misc_utils.write_vrt_from_csv(temp_file,
                                             'x', 'y', 'z')

    # Now grid using gdal_grid

    gridopt = gdal.GridOptions(format=grid_kwargs['format'],
                               outputType=gdal.GDT_Float32,
                               algorithm=grid_kwargs['gdal_algorithm'],
                               width=grid_kwargs['width'],
                               height=grid_kwargs['height'],
                               outputSRS=grid_kwargs['outputSRS'],# "EPSG:28352"
                               outputBounds=grid_kwargs['outputBounds'],
                               zfield='z',
                               layers=vrt_file.split('\\')[-1].split('.')[0])

   # Getting errors frmo tempdir
    outfile = 'temp_grid.tif'
    print("Gridding ", outfile)
    out_ds = gdal.Grid(destName=outfile,
                       srcDS=vrt_file, options=gridopt)
    print("Finished gridding ", outfile)
    geotransform = out_ds.GetGeoTransform()

    gridded = out_ds.ReadAsArray()

    # Currently the nulls are very annoyingly retunr zeros

    null = np.float(grid_kwargs['gdal_algorithm'].split('nodata=')[-1])

    gridded[gridded == null] = np.nan

    if grid_kwargs['log_grid']:
        gridded = 10 ** gridded

    # Remove trash
    temp_a = None
    out_ds = None

    gc.collect()

    # Delete the temporary files
    for file in [temp_file, vrt_file, outfile]:
        try:
            os.remove(file)
        except PermissionError:
            print('Permission error. Unable to delete ', file)

    return gridded, geotransform


def grid_points_gdal(cond_point_utils_inst, grid_resolution,
                     variables=None,
                     native_grid_bounds=None,
                     reprojected_grid_bounds=None,
                     grid_wkt=None,
                     point_step=1,
                     grid_kwargs=None,
                     depth_inds = None):
    '''
    Function that grids points in the cond_point_utils instance within a bounding box using gdal_grid.
    @parameter cond_point_utils_inst: instance of cond_point_utils from geophys_utils
    @parameter grid_resolution: cell size of regular grid in grid CRS units
    @parameter variables: Single variable name string or list of multiple variable name strings. Defaults to all point variables
    @parameter native_grid_bounds: Spatial bounding box of area to grid in native coordinates
    @parameter reprojected_grid_bounds: Spatial bounding box of area to grid in grid coordinates
    @parameter grid_wkt: WKT for grid coordinate reference system. Defaults to native CRS
    @parameter point_step: Sampling spacing for points. 1 (default) means every point, 2 means every second point, etc.
    @parameter depth inds: list of depth indices to grid.

    @return grid: dictionary with gridded data, geotransform and wkt
    '''
    assert not (
                native_grid_bounds and reprojected_grid_bounds), 'Either native_grid_bounds or reprojected_grid_bounds can be provided, but not both'
    # Grid all data variables if not specified
    variables = variables or cond_point_utils_inst.point_variables

    # Allow single variable to be given as a string
    single_var = (type(variables) == str)
    if single_var:
        variables = [variables]

    if native_grid_bounds:
        reprojected_grid_bounds = cond_point_utils_inst.get_reprojected_bounds(native_grid_bounds,
                                                                               cond_point_utils_inst.wkt,
                                                                               grid_wkt)
    elif reprojected_grid_bounds:
        native_grid_bounds = cond_point_utils_inst.get_reprojected_bounds(reprojected_grid_bounds,
                                                                          grid_wkt,
                                                                          cond_point_utils_inst.wkt)
    else:  # No reprojection required
        native_grid_bounds = cond_point_utils_inst.bounds
        reprojected_grid_bounds = cond_point_utils_inst.bounds

    # Determine spatial grid bounds rounded out to nearest GRID_RESOLUTION multiple
    pixel_centre_bounds = (round(math.floor(reprojected_grid_bounds[0] / grid_resolution) * grid_resolution, 6),
                           round(math.floor(reprojected_grid_bounds[1] / grid_resolution) * grid_resolution, 6),
                           round(math.floor(
                               reprojected_grid_bounds[2] / grid_resolution - 1.0) * grid_resolution + grid_resolution,
                                 6),
                           round(math.floor(
                               reprojected_grid_bounds[3] / grid_resolution - 1.0) * grid_resolution + grid_resolution,
                                 6)
                           )

    # Extend area for points an arbitrary two cells out beyond grid extents for nice interpolation at edges
    expanded_grid_bounds = [pixel_centre_bounds[0] - 2 * grid_resolution,
                            pixel_centre_bounds[1] - 2 * grid_resolution,
                            pixel_centre_bounds[2] + 2 * grid_resolution,
                            pixel_centre_bounds[3] + 2 * grid_resolution
                            ]

    expanded_grid_size = [expanded_grid_bounds[dim_index + 2] - expanded_grid_bounds[dim_index] for dim_index in
                          range(2)]

    # Get width and height of grid
    width = int(expanded_grid_size[0] / grid_resolution)
    height = int(expanded_grid_size[1] / grid_resolution)

    spatial_subset_mask = cond_point_utils_inst.get_spatial_mask(
        cond_point_utils_inst.get_reprojected_bounds(expanded_grid_bounds, grid_wkt, cond_point_utils_inst.wkt))

    # Skip points to reduce memory requirements
    # TODO: Implement function which grids spatial subsets.
    point_subset_mask = np.zeros(shape=(cond_point_utils_inst.netcdf_dataset.dimensions['point'].size,), dtype=bool)
    point_subset_mask[0:-1:point_step] = True
    point_subset_mask = np.logical_and(spatial_subset_mask, point_subset_mask)

    coordinates = cond_point_utils_inst.xycoords[point_subset_mask]
    # Reproject coordinates if required
    if grid_wkt is not None:
        # N.B: Be careful about XY vs YX coordinate order
        coordinates = np.array(transform_coords(coordinates[:], cond_point_utils_inst.wkt, grid_wkt))

    grid = {}
    for variable in [cond_point_utils_inst.netcdf_dataset.variables[var_name] for var_name in variables]:

        # If preferences are not given we grid using defaults
        if grid_kwargs is None:
            grid_kwargs = {}
            grid_kwargs[variable.name] = {}

        if not 'log_grid' in grid_kwargs:
            grid_kwargs[variable.name]['log_grid'] = False

        # Check for the algorithm in the grid kwargs
        if not 'gdal_algorithm' in grid_kwargs:
            s = 'invdist:power=2:radius1=250:'
            s += 'radius2=250:max_points=15:'
            s += 'min_points=2:nodata=-32768.0'
            # Defaut
            grid_kwargs[variable.name]['gdal_algorithm'] = s

        if not 'format' in grid_kwargs:
            # Defaut
            grid_kwargs[variable.name]['format'] = 'GTiff'

        if not 'outputType' in grid_kwargs:
            # Defaut
            grid_kwargs[variable.name]['outputType'] = gdal.GDT_Float32

        grid_kwargs[variable.name]['width'] = width
        grid_kwargs[variable.name]['height'] = height

        grid_kwargs[variable.name]['outputSRS'] = grid_wkt
        grid_kwargs[variable.name]['outputBounds'] = expanded_grid_bounds

        # For 2d arrays
        if len(variable.shape) == 2:

            if depth_inds is not None:
                nlayers = len(depth_inds)
            else:
                nlayers = variable.shape[1]
                depth_inds = np.arange(0,nlayers)

            a = np.nan * np.ones(shape=(nlayers, int(height),
                                        int(width)), dtype=np.float32)

            # Iterate through the layers
            for i, ind in enumerate(depth_inds):
                var_array = variable[point_subset_mask, ind].reshape([-1, 1])

                a[i], geotransform = grid_var(var_array, coordinates,
                                              grid_kwargs[variable.name])

        elif len(variable.shape) == 1:

            # Create a temporary array
            var_array = variable[point_subset_mask].reshape([-1, 1])

            a, geotransform = grid_var(var_array, coordinates,
                                       grid_kwargs[variable.name])

        grid[variable.name] = a

    grid['geotransform'] = geotransform
    grid['wkt'] = grid_wkt

    return grid
# Now we want to calculate a weighted average for the AEM

def weighted_average(layer_top_depth, variable, depth_top,
                    depth_bottom):
    """
    TODO add functoinality for part layers


    @parameter layer_top_depth: (1D) or 3d numpy array. If 1D array we assume a consistent layer
    depth discretisation. A 3d the array must be shaped (layer_top_depth, height, width) - TODO
    @parameter: a 3-dimensional numpy array with the variable to be averaged (e.g conductivity)
    of shape (layer_top_depth, height, width)
    @parameter depth_top: a 2D numpy array (width, height) with the layer top depth
    @parameter depth_bottom: a 2D numpy array (width, height) with the layer bottom depth
    """



    # If layer top depth is a 1d array make it 3d
    if len(layer_top_depth.shape) == 1:
        ncells = depth_top.flatten().shape
        layer_top_depth = np.tile(layer_top_depth, ncells).reshape((layer_top_depth.shape[0],
                                                                  depth_top.shape[0],
                                                                  depth_top.shape[1]),
                                                              order = 'F')
    # Otherwise assert correct shape
    else:
        assert len(layer_top_depth.shape) == 3

    # Create a layer_bottom_depth array

    layer_bottom_depth = np.nan*np.ones(shape = layer_top_depth.shape,
                                        dtype = layer_top_depth.dtype)

    layer_bottom_depth[:-1,:,:] = layer_top_depth[1:,:,:]

    # Repeat the depth top and bottom
    depth_top = np.repeat(depth_top[np.newaxis, :, :],
                          variable.shape[0], axis=0)

    depth_bottom = np.repeat(depth_bottom[np.newaxis, :, :],
                          variable.shape[0], axis=0)


    # calculate thickness
    thickness =  depth_to_thickness(layer_top_depth)

    # Clip slice the thickness and layer top depth arrays

    thickness = thickness[:variable.shape[0],:,:]
    layer_top_depth = layer_top_depth[:variable.shape[0],:,:]
    layer_bottom_depth = layer_bottom_depth[:variable.shape[0],:,:]

    # Now we want to find the indices for each grid cell that is outside
    # of the layer

    mask = np.greater(depth_top, layer_top_depth)

    mask = np.less(depth_bottom,layer_bottom_depth)

    mask+= np.isnan(depth_bottom)

    mask += np.isnan(variable)


    # Set thickness of AEM layers outside of stratigraphy to 0
    thickness[mask] = np.nan

    # Create a masked array for the variable

    var_mskd = ma.masked_array(variable,
                               mask = ~np.isnan(variable))

    # Calculate the weighted average by multiplying variable by
    # thickness and dividing by total thickness

    # Multiply the array and take the sum along the depth axis
    a1 =  var_mskd.data*thickness

    a2 = np.nansum(a1, axis = 0)

    # Divide by the sum of the thickness

    a3 = np.nansum(thickness, axis = 0)

    a3[a3 == 0] = np.nan
    # Make zeros into nan

    return np.divide(a2,a3)
