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
from rasterio import Affine
from rasterio.warp import reproject, Resampling


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

    new_columns = parameter_columns.append(interval_column)

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
                                    depth_column='Depth', how='mean'):
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
            if how == 'mean':
                new_df.at[index, item] = df_interval[item].mean()
            elif how == 'median':
                new_df.at[index, item] = df_interval[item].median()
            elif how == 'mode':
                new_df.at[index, item] = df_interval[item].mode()

    return new_df


def resample_raster(infile, outfile, gridx, gridy, driver='GTiff',
                    null = -999):
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

    src.close()

    # Create a new dataset
    new_dataset = rasterio.open(outfile, 'w', driver=driver,
                                height=newarr.shape[0], width=newarr.shape[1],
                                count=1, dtype=newarr.dtype,
                                crs=src.crs, transform=newaff)
    new_dataset.write(newarr, 1)

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