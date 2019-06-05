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

These are functions used to process AEM data from netcdf objects or miscellaneous formats
'''

import numpy as np
from hydrogeol_utils import spatial_functions
import pandas as pd
from geophys_utils._transect_utils import coords2distance
from geophys_utils._netcdf_point_utils import NetCDFPointUtils
from geophys_utils._netcdf_line_utils import NetCDFLineUtils


# A function for getting the most representative conductivity profile given
# a set of distances, indices and corresponding AEM data
def extract_conductivity_profile(dataset, distances, indices,
                                 as_dataframe = False, mask_below_doi = True):
    """
    A function for finding the most representative AEM conductivity profile
    at a point in space given a set of distances. The inverse distance algorithm
    is used find a weighted average of the points using the distance

    :param dataset: a netcdf object containing the AEM conductivity line data
    :param distances: a flat array of distances from the point to the AEM fiducials
    :param indices: point indices for each point in the distances array
    :param as_dataframe: if True then the conductivity profile is returned as a
        pandas dataframe with Depth_from and Depth_to columns

    :return:
    array or dataframe with representative conductivity profile
    """

    # Now calculate weights based on the distances
    idws = spatial_functions.inverse_distance_weights(distances, 2)

    # The result will be a log10 conductivity array as averaging in conductivity
    # is typically done in the log space
    log_conductivity = np.zeros(shape= dataset['layer_top_depth'][:].shape[1],
                        dtype=np.float)

    doi = 0

    # Iteratively extract the conductivity profiles using the indices,
    # min_key and aem_keys


    for i, ind in enumerate(indices):
        # Get the logconductivity proile
        log_cond_profile = np.log10(dataset.variables['conductivity'][ind])

        # Now multiply it by its corresponding weight and add it to the log ond array
        log_conductivity += idws[i] * log_cond_profile

        # Now add this points doi to the doi variable
        if mask_below_doi:

            doi += idws[i] * dataset.variables['depth_of_investigation'][ind]

    # Now return the linear transform the conductivity array and reshape to a single column

    cond_profile = 10 ** log_cond_profile

    # Now mask below the depth of investigation

    if mask_below_doi:

        mask = dataset.variables['layer_top_depth'][indices[0]] < doi

        cond_profile = cond_profile[mask]


    # If specified in kwargs then return a dataframe
    if as_dataframe:

        # Now we need to create a depth from and depth too column
        depth_from = dataset.variables['layer_top_depth'][ind]

        depth_to = np.nan * np.ones(shape=depth_from.shape, dtype=np.float)
        depth_to[:-1] = depth_from[1:]

        # Round to 2 decimal places to correct for floating point precision errors
        depth_to = np.round(depth_to, 2)

        if mask_below_doi:

            depth_from = depth_from[mask]
            depth_to = depth_to[mask]

        # create and return a labelled dataframe
        return pd.DataFrame(data = {'Depth_from': depth_from,
                                    "Depth_to": depth_to,
                                    'conductivity': cond_profile})
    # Otherwise return a numpy array
    else:
        return cond_profile


def parse_gridded_conductivity_file(infile, header, null=1e-08,
                                    lex_sort = True, sort_inds = [3,0]):
    """
    Function for parsing the asci xyz files

    :param infile: path to an asci xyz file with gridded conductivity
    such as those supplied by CGI
    :param header: list of column names that should match keywords from
    geophys_utils
    :param null: float of null values
    :return:
    dictionary with flat arrays for each column in file and nulls
    replaced by np.nan

    """

    # TODO add some checks for the structure of the data

    # Save the data in a dictionary
    data = {}

    # Load the gridded data
    a = np.loadtxt(infile)

    # Replace null values with np.nan
    a[a == null] = np.nan

    # Lex sort the array on elevation then on easting
    # This is requried for the gridding function to work
    if lex_sort:
        a = a[np.lexsort((a[:, sort_inds[1]],
                          a[:, sort_inds[0]]))]
    for i, item in enumerate(header):

        data[item] = a[:,i]

    return data


# AS we know the structure of the data we can create a more usable
# where the variable (conductivity) is structured as a 2D
# array
def griddify_xyz(data):
    """
    Function for creating plot ready grids from the dictionary
    :param data: dictionary of data produced using parse_gridded_conductivity_file

    :return:
    dictionary with flat arrays
    """
    # Create a gridded data dictionary
    gridded_data = {}

    # Find the number of horizontal and vertical cells

    nvcells = len(np.unique(data['elevation']))
    nhcells = int(len(data['elevation']) / len(np.unique(data['elevation'])))

    # Create the grid

    cond = np.zeros(shape=(nvcells, nhcells),
                    dtype=data['conductivity'].dtype)

    # Now iterate through the conductivity and populate the array

    for i in range(nvcells):
        # Get the 2D array indices
        cond[i, :] = data['conductivity'][i * nhcells:(i + 1) * nhcells]

    # Replace the entry
    gridded_data['conductivity'] = cond

    # Add the east, northing and elevation top to the dictionary
    gridded_data['easting'] = data['easting'][:nhcells]
    gridded_data['northing'] = data['northing'][:nhcells]
    gridded_data['grid_elevations'] = np.unique(data['elevation'])

    # rotate or flip the grids if need be
    if gridded_data['easting'][0] > gridded_data['easting'][-1]:
        reverse_line = True
    else:
        reverse_line = False

    if gridded_data['grid_elevations'][0] < gridded_data['grid_elevations'][-1]:
        flip_grid = True
    else:
        flip_grid = False

    if reverse_line:
        gridded_data['conductivity'] = np.fliplr(gridded_data['conductivity'])
        gridded_data['easting'] = gridded_data['easting'][::-1]
        gridded_data['northing'] = gridded_data['northing'][::-1]

    if flip_grid:
        gridded_data['conductivity'] = np.flipud(gridded_data['conductivity'])
        gridded_data['grid_elevations'] = gridded_data['grid_elevations'][::-1]
    # Calculate the distance along the line

    utm_coordinates = np.hstack((gridded_data['easting'].reshape([-1, 1]),
                                 gridded_data['northing'].reshape([-1, 1])))

    gridded_data['grid_distances'] = coords2distance(utm_coordinates)

    # Estimate the ground elevation using the nan values

    elevation = np.zeros(shape=gridded_data['easting'].shape,
                         dtype=gridded_data['grid_elevations'].dtype)

    # Iterate through the cells and find the lowest elevation with data
    for i in range(gridded_data['conductivity'].shape[1]):
        try:
            idx = np.max(np.argwhere(np.isnan(gridded_data['conductivity'][:, i]))) + 1
            elevation[i] = gridded_data['grid_elevations'][idx]
        except ValueError:
            elevation[i] = np.max(gridded_data['grid_elevations'])

    # Add to dictionary
    gridded_data['elevation'] = elevation
    return gridded_data

def get_boundary_elevations(dataset):
    """

    :param dataset: netcdf AEM line dataset
    :return:
    an array of layer top elevations of the same shape as layer_top_depth
    """
    return np.repeat(dataset.variables['elevation'][:][:, np.newaxis],
                     dataset.variables['layer_top_depth'].shape[1],axis=1) - \
                    dataset.variables['layer_top_depth'][:]


# Function for reshaping arrays into xyz valu

def arr2xyz(utm_coords, nlayers, layer_top_elevations):
    # Tile coordinates
    utm_tiled = np.repeat(utm_coords, nlayers, axis=0)

    xyz = np.hstack((utm_tiled, layer_top_elevations.reshape([-1,1])))

    return xyz


def get_xyz_array(dataset, variables = None, lines = None, east_to_west = True):
    """
    If lines are supplied this is an iterator which yields line xyz value arrays. If none are supplied it yields
    on the whole dataset

    :param dataset: netcdf dataset
    :param variables: list of variables to add to xyz array
    :param lines: list of lines
    :param east_to_west: boolean flag whether to align lines in east west fashion
    :return:
    """

    if lines is not None:

        # Create a generator
        cond_line_util = NetCDFLineUtils(dataset)

        line_gen = cond_line_util.get_lines(line_numbers=lines)

        for _ in lines:

           # Yield variables for the line
            line_vars = next(line_gen)[1]

            utm_coords = cond_line_util.utm_coords(line_vars['coordinates'])[1]

            # Now repeat the utm coordinates so it can be joined into an x,y,z array

            nlayers = line_vars['layer_top_depth'].shape[1]

            # Get the elevation of the layer boundaries

            layer_top_elevations = np.repeat(line_vars['elevation'][:, np.newaxis],
                             nlayers, axis=1) - line_vars['layer_top_depth']

            # Turn these arrays into a xyz array
            layer_boundaries = arr2xyz(utm_coords, nlayers, layer_top_elevations)

            # Now retrieve the additional variable point attribute

            if variables is not None:

                for v in variables:
                    # Retrieve array from the line variable dictionary
                    a = line_vars[v].reshape([-1, 1])

                    layer_boundaries = np.hstack((layer_boundaries, a))

            if east_to_west:

                if layer_boundaries[0,0] > layer_boundaries[-1,0]:

                    layer_boundaries = np.flipud(layer_boundaries)

            yield layer_boundaries
    # Otherwise we apply this on whole dataset
    else:

        # Create a point generator
        cond_point_util = NetCDFPointUtils(dataset)
        # Get all coords
        utm_coords = cond_point_util.utm_coords(cond_point_util.get_xy_coord_values())[1]

        # Get the layer_top_elevations
        layer_top_elevations = get_boundary_elevations(dataset)

        nlayers = dataset.variables['layer_top_depth'].shape[1]

        # Turn these arrays into a xyz array
        layer_boundaries = arr2xyz(utm_coords, nlayers, layer_top_elevations)

        if variables is not None:

            for v in variables:
                # Retrieve array from the line variable dictionary
                a = dataset.variables[v][:].reshape([-1, 1])

                layer_boundaries = np.hstack((layer_boundaries, a))

        yield layer_boundaries




