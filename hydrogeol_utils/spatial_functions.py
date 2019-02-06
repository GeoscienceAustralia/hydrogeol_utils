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
import collections

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
    :param depth: a flat array of depths
    :return:
    a flat array of thicknesses with the last entry being a null
    """
    # Create a new thickness array
    thickness = np.nan*np.ones(shape=depth.shape,
                               dtype=np.float)
    # Iterate through the depth array
    for i in range(len(depth)-1):
        thickness[i] = depth[i+1] - depth[i]
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

    for p in parameter_columns:
        # Create an array and add it to the dictionary
        params[p] = np.repeat(df[p].values, 2)

    # Now we add the new_intervals to the intervals and param arrays
    # To get the parameters we will do some basic interpolation using scipy

    x = intervals
    xnew = new_intervals[new_intervals.columns[0]].values
    xnew = xnew[(xnew > np.min(intervals)) & (xnew < np.max(intervals))]
    # Concatenate, we will sort at a later time
    intervals = np.concatenate((intervals, xnew))

    # For each parameter interpolate and add the new values to the array
    for p in params.keys():
        y = params[p]
        f = interpolate.interp1d(x, y, kind='linear')
        # Find new parameters values
        ynew = f(xnew)
        # Add to the parameter array, we will sort later
        params[p] = np.concatenate((params[p], ynew))

    # Add the new intervals array to the params dictionary
    params['Depth'] = intervals

    # For ease of use we will create a dataframe for the manipulated profiles
    df_interp = pd.DataFrame(data=params)

    # Sort and reset the index
    df_interp.sort_values(by='Depth', inplace=True)
    df_interp.reset_index(inplace=True, drop=True)

    # Cretae a thickness column
    df_interp['thickness'] = depth_to_thickness(df_interp['Depth'].values)

    # Now we round to 2 decimal places to avoid floating point errors
    df_interp = df_interp.round({'Depth': 2})
    for c in new_intervals.columns:
        new_intervals = new_intervals.round({c: 2})

    # Create new columns for the interpolated parameter values
    for p in parameter_columns:
        new_intervals[p + '_interpolated'] = np.nan

    # Iterate through the new intervals and do an average
    for index, row in new_intervals.iterrows():
        # Find the upper and lower depths
        du, dl = row[new_intervals.columns[0]], row[new_intervals.columns[1]]
        # Slice the data frame
        df_subset = df_interp[(df_interp['Depth'] >= du) & (df_interp['Depth'] <= dl)]

        # Remove zero thickness layers which were only used for interpolation
        df_subset = df_subset[df_subset['thickness'] != 0]

        # Calculate weights based on normalised thickness
        df_subset['weights'] = 0

        weights = 1.0 / df_subset['thickness']

        weights /= weights.sum()
        df_subset.at[weights.index, 'weights'] = weights

        # Iterate through the parameters and multiply by the corresponding weights
        for p in parameter_columns:
            new_intervals.at[index, p + '_interpolated'] = (df_subset['weights'] *
                                                            df_subset[p]).sum()

    return new_intervals

def nearest_neighbour(points, coords, points_required = 1, max_distance = 250.):
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
    return distances, indices

