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
Created on 16/1/2019
@author: Neil Symington

These are functions used to visualise hydrogeological data
'''

import netCDF4
import math
from math import log10, floor, pow
import os
import collections
import gc
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import numpy as np
from geophys_utils._netcdf_line_utils import NetCDFLineUtils
from geophys_utils._transect_utils import coords2distance
from hydrogeol_utils import spatial_functions
from geophys_utils import get_spatial_ref_from_wkt
import h5py

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as mPolygon
from matplotlib.collections import LineCollection
import matplotlib.image as mpimg

from skimage.transform import resize

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt



class ConductivitySectionPlot:
    """
    VerticalSectionPlot class for functions for creating vertical section plots from netcdf file
    """

    def __init__(self,
                 netCDFConductivityDataset = None,
                 netCDFemDataset = None):
        """
        :param netCDFConductivityDataset: netcdf line dataset with
         conductivity model
        :param netCDFemDataset: netcdf line dataset with
         EM measurements
        """
        if netCDFConductivityDataset is not None:
            if not self.testNetCDFDataset(netCDFConductivityDataset):
                raise ValueError("Input datafile is not netCDF4 format")
            else:
                self.conductivity_model = netCDFConductivityDataset
                self.condLineUtils = NetCDFLineUtils(self.conductivity_model)
                self.conductivity_variables = []
        else:
            self.conductivity_model = None

        # If datafile is given then check it is a netcdf file

        if netCDFemDataset is not None:
            if not self.testNetCDFDataset(netCDFemDataset):
                raise ValueError("Input datafile is not netCDF4 format")
            else:
                self.EM_data = netCDFemDataset
                self.dataLineUtils = NetCDFLineUtils(self.EM_data)
                self.EM_variables = []
        else:
            self.EM_data = None


    def save_dict_to_hdf5(self, fname, dictionary):
        """
        Save a dictionary to hdf5
        """
        f = h5py.File(fname, "w")

        for key in dictionary.keys():
            dset = f.create_dataset(key, data=dictionary[key])
        f.close()

    def testNetCDFDataset(self, netCDF_dataset):
        """
        A  function to test if correctly if file is formatted netCDF4 file

        :param netCDF_dataset: netCDF4 dataset
        :return:

        True if correct, False if not
        """

        return netCDF_dataset.__class__ == netCDF4._netCDF4.Dataset

    def interpolate_data_coordinates(self, line, var_dict, gridding_params):
        """

        :param line:
        :param var_dict:
        :param gridding_params:
        :return:
        """
        # Create a dictionary into whcih to write interpolated coordinates
        interpolated = {}

        # Define coordinates
        utm_coordinates = self.dataLineUtils.utm_coords(var_dict['coordinates'])[1]

        if utm_coordinates[0, 0] > utm_coordinates[-1, 0]:
            var_dict['reverse_line'] = True
        else:
            var_dict['reverse_line'] = False

        # Find distance along the line
        distances = coords2distance(utm_coordinates)
        var_dict['distances'] = distances

        # Calculate 'grid' distances

        var_dict['grid_distances'] = np.arange(distances[0], distances[-1], gridding_params['xres'])

        # Interpolate the two coordinate variables
        interp1d = interpolate_1d_vars(['easting', 'northing'],
                                       var_dict, gridding_params['resampling_method'])

        for var in ['easting', 'northing']:
            # Generator yields the interpolated variable array
            interpolated[var] = next(interp1d)

        return interpolated, var_dict


    def grid_conductivity_variables(self, line, cond_var_dict, gridding_params, smoothed = False):

        """

        :param line:
        :param cond_var_dict:
        :return:
        """

        # Create an empty dictionary
        interpolated = {}

        # If the line is west to east we want to reverse the coord
        # array and flag it

        # Define coordinates
        utm_coordinates = self.condLineUtils.utm_coords(cond_var_dict['coordinates'])[1]


        # Add the flag to the dictionary
        if utm_coordinates[0, 0] > utm_coordinates[-1, 0]:
            cond_var_dict['reverse_line'] = True
        else:
            cond_var_dict['reverse_line'] = False

        # Add distance array to dictionary
        cond_var_dict['distances'] = coords2distance(utm_coordinates)

        # Add number of layers to the array
        cond_var_dict['nlayers'] = self.conductivity_model.dimensions['layer'].size

        # Interpolate 2D and 1D variables

        vars_2d = [v for v in self.conductivity_variables if cond_var_dict[v].ndim == 2]
        vars_1d = [v for v in self.conductivity_variables if cond_var_dict[v].ndim == 1]

        # Generator for inteprolating 2D variables from the vars_2d list
        if not smoothed:
            interp2d = interpolate_2d_vars_true(vars_2d, cond_var_dict, gridding_params['xres'],
                                       gridding_params['yres'])
        else:
            interp2d = interpolate_2d_vars_smooth(vars_2d, cond_var_dict, gridding_params['xres'],
                                           gridding_params['yres'], gridding_params['layer_subdivisions'],
                                           gridding_params['resampling_method'])

        for var in vars_2d:
            # Generator yields the interpolated variable array
            interpolated[var], cond_var_dict = next(interp2d)

        # Add grid distances and elevations to the interpolated dictionary
        interpolated['grid_distances'] = cond_var_dict['grid_distances']
        interpolated['grid_elevations'] = cond_var_dict['grid_elevations']

        # Generator for inteprolating 1D variables from the vars_1d list
        interp1d = interpolate_1d_vars(vars_1d, cond_var_dict, gridding_params['resampling_method'])

        for var in vars_1d:
            # Generator yields the interpolated variable array
            interpolated[var] = next(interp1d)

        return interpolated



    def grid_variables(self, xres, yres, lines,
                       layer_subdivisions = None, resampling_method = 'linear',
                       smoothed = False, save_hdf5 = False, hdf5_dir = None,
                       overwrite_hdf5 = True, return_dict = True):
        """
        A function for interpolating 1D and 2d variables onto a vertical grid
        cells size xres, yres
        :param xres: Float horizontal cell size along the line
        :param yres: Float vertical cell size
        :param lines: int single line or list of lines to be gridded
        :param layer_subdivisions:
        :param resampling_method: str or int, optional - from scipy gridata
        :param save_hdf5: Boolean parameter indicating whether interpolated variables
         get saved as hdf or no
        :param hdf5_dir: path of directory into which the hdf5 files are saved
        :param overwrite_hdf5: Boolean parameter referring to if the user wants to
         overwrite any pre-existing files
        :param return_dict: Boolean parameter indicating if a dictionary is returned or not
        :return:
        dictionary with interpolated variables as numpy arrays
        """

        # Create a line utils for each object if the objects exist
        if self.conductivity_model is not None:
            # Flag for if dta was included in the plot section initialisation
            plot_cond = True
            # Add key variables if they aren't in the list to grid
            for item in ['easting', 'northing', 'elevation', 'layer_top_depth']:
                if item not in self.conductivity_variables:
                    self.conductivity_variables.append(item)
        else:
            plot_cond = False

        if self.EM_data is not None:
            # Flag for if dta was included in the plot section initialisation
            plot_dat = True

        else:
            plot_dat = False

        # If line is not in an array like object then put it in a list
        if type(lines) == int:
            lines = [lines]
        elif isinstance(lines ,(list, tuple, np.ndarray)):
            pass
        else:
            raise ValueError("Check lines variable.")

        # First create generators for returning coordinates and variables for the lines

        if plot_cond:
            cond_lines= self.condLineUtils.get_lines(line_numbers=lines,
                                                variables=self.conductivity_variables)
        if plot_dat:
            dat_lines = self.dataLineUtils.get_lines(line_numbers=lines,
                                                variables=self.EM_variables)

        # Interpolated results will be added to a dictionary
        interpolated = {}

        # Create a gridding parameters dictionary

        gridding_params = {'xres': xres, 'yres': yres, 'layer_subdivisions': layer_subdivisions,
                           'resampling_method': resampling_method}

        # Iterate through the lines
        for i in range(len(lines)):

            # Extract the variables and coordinates for the line in question
            if plot_cond:

                line_no, cond_var_dict = next(cond_lines)

                cond_var_dict['utm_coordinates'] = self.condLineUtils.utm_coords(cond_var_dict['coordinates'])[1]

                interpolated[line_no] =  self.grid_conductivity_variables(line_no, cond_var_dict,
                                                                          gridding_params, smoothed=smoothed)

            if plot_dat:
                # Extract variables from the data
                line_no, data_var_dict = next(dat_lines)

                data_var_dict['utm_coordinates'] = self.dataLineUtils.utm_coords(data_var_dict['coordinates'])[1]

                # If the conductivity variables have not been plotted then we need to interpolate the coordinates

                if not plot_cond:

                    interpolated[line_no], data_var_dict = self.interpolate_data_coordinates(line_no,data_var_dict,
                                                                                        gridding_params)

                interpolated_utm = np.hstack((interpolated[line_no]['easting'].reshape([-1, 1]),
                                              interpolated[line_no]['northing'].reshape([-1, 1])))

                # Generator for interpolating data variables from the data variables list
                interp_dat = interpolate_data(self.EM_variables, data_var_dict, interpolated_utm,
                                              resampling_method)

                for var in self.EM_variables:

                    interpolated[line_no][var] = next(interp_dat)

            # Save to hdf5 file if the keyword is passed
            if save_hdf5:
                fname = os.path.join(hdf5_dir, str(line_no) + '.hdf5')
                if overwrite_hdf5:
                    self.save_dict_to_hdf5(fname, interpolated[line_no])
                else:
                    if os.path.exists(fname):
                        print("File ", fname, " already exists")
                    else:
                        self.save_dict_to_hdf5(fname, interpolated[line_no])

            # Many lines may fill up memory so if the dictionary is not being returned then
            # we garbage collect
            if not return_dict:

                del interpolated[line_no]

                # Collect the garbage
                gc.collect()

        if return_dict:
            return interpolated

def save_dict_to_hdf5(fname, dictionary):
    """
    Save a dictionary to hdf5
    """
    f = h5py.File(fname, "w")
    for key in dictionary.keys():
        dset = f.create_dataset(key, data=dictionary[key])
    f.close()

def purge_invalid_elevations(var_grid, grid_y, min_elevation_grid,
                                 max_elevation_grid, yres):
    """
    Function for purging interpolated values that sit above the maximum or below the minimum elevation
    :param var_grid:
    :param grid_y:
    :param min_elevation_grid:
    :param max_elevation_grid:
    :param yres:
    :return:
    """
    # Iterate through the
    for x_index in range(var_grid.shape[1]):
        # Get indices which are below the minimum elevation
        min_elevation_indices = np.where(grid_y[:,x_index] < min_elevation_grid[x_index] + yres)[0]

        try:
            var_grid[min_elevation_indices, x_index] = np.NaN
        except:
            pass
        # Get indices which are above the maximum elevation
        max_elevation_indices = np.where(grid_y[:,x_index] > max_elevation_grid[x_index] - yres)[0]

        try:
            var_grid[max_elevation_indices, x_index] = np.NaN
        except:
            pass

    return var_grid


def interpolate_2d_vars_true(vars_2d, var_dict, xres, yres):
    """
    Generator to interpolate 2d variables (i.e conductivity, uncertainty)

    :param vars_2d:
    :param var_dict:
    :param xres:
    :param yres:
    :return:
    """

    nlayers = var_dict['nlayers']

    # Get the thickness of the layers

    layer_thicknesses = spatial_functions.depth_to_thickness(var_dict['layer_top_depth'])

    # Give the bottom layer a thickness of 20 metres

    layer_thicknesses[:,-1] = 20.

    # Get the vertical limits, note guard against dummy values > 800m

    elevations = var_dict['elevation']

    # Guard against dummy values which are deeper than 900 metres

    max_depth = np.max(var_dict['layer_top_depth'][var_dict['layer_top_depth'] < 900.])

    vlimits = [np.min(elevations) - max_depth,
               np.max(elevations) + 5]

    # Get the horizontal limits

    distances = var_dict['distances']

    hlimits = [np.min(distances), np.max(distances)]

    # Get the x and y dimension coordinates

    xres = np.float(xres)
    yres = np.float(yres)

    grid_y, grid_x = np.mgrid[vlimits[1]:vlimits[0]:-yres,
                     hlimits[0]:hlimits[1]:xres]

    grid_distances = grid_x[0]

    grid_elevations = grid_y[:, 0]

    # Add to the variable dictionary

    var_dict['grid_elevations'] = grid_elevations

    var_dict['grid_distances'] = grid_distances

    # Interpolate the elevation

    f = interp1d(distances, elevations)

    max_elevation = f(grid_distances)

    # Interpolate the layer thicknesses

    grid_thicknesses = np.nan*np.ones(shape = (grid_distances.shape[0],
                                               grid_elevations.shape[0]),
                                      dtype = layer_thicknesses.dtype)

    for j in range(layer_thicknesses.shape[1]):
        # Guard against nans

        if not np.isnan(layer_thicknesses[:,j]).any():
            # Grid in log10 space
            layer_thickness = np.log10(layer_thicknesses[:, j])
            f = interp1d(distances, layer_thickness)
            grid_thicknesses[:,j] = f(grid_distances)

    # Tranform back to linear space
    grid_thicknesses = 10**grid_thicknesses

    # Interpolate the variables

    # Iterate through variables and interpolate onto new grid
    for var in vars_2d:

        interpolated_var = np.nan*np.ones(grid_thicknesses.shape,
                                          dtype = var_dict[var].dtype)

        # For conductivity we interpolate in log10 space

        point_var = var_dict[var]

        new_var = np.ones(shape = (len(grid_distances),
                                   nlayers))

        if var == 'conductivity':

            point_var = np.log10(point_var)

        for j in range(point_var.shape[1]):

            f = interp1d(distances, point_var[:,j])
            new_var[:, j] = f(grid_distances)

        if var == 'conductivity':

            new_var = 10**(new_var)

        # Now we need to place the 2d variables on the new grid
        for i in range(grid_distances.shape[0]):
            dtop = 0.
            for j in range(nlayers - 1):
                # Get the thickness
                thick = grid_thicknesses[i,j]
                # Find the elevation top and bottom
                etop = max_elevation[i] - dtop
                ebot = etop - thick
                # Get the indices for this elevation range
                j_ind = np.where((etop >= grid_elevations) & (ebot <= grid_elevations))
                # Populate the section
                interpolated_var[i, j_ind] = new_var[i,j]
                # Update the depth top
                dtop += thick

        # Reverse the grid if it is west to east

        if var_dict['reverse_line']:

            interpolated_var = np.flipud(interpolated_var)

        # We also want to transpose the grid so the up elevations are up

        interpolated_var = interpolated_var.T

        # Yield the generator and the dictionary with added variables
        yield interpolated_var, var_dict


def interpolate_2d_vars_smooth(vars_2d, var_dict, xres, yres,
                        layer_subdivisions, resampling_method):
    """
    Generator to interpolate 2d variables (i.e conductivity, uncertainty). This function is not currently used but
    produces a smoother model than

    :param vars_2d:
    :param var_dict:
    :param xres:
    :param yres:
    :param layer_subdivisions:
    :param resampling_method:
    :return:
    """

    nlayers = var_dict['nlayers']

    # Create array for the top elevation of each point and layer
    layer_top_elevations = (np.repeat(var_dict['elevation'][:, np.newaxis],
                                      nlayers, axis=1) - var_dict['layer_top_depth'])

    # Create array for the top elevation of each sublayer (i.e. layers divided into sublayers given the
    # layer_subdivisions parameters
    sublayer_elevations = np.ones(shape=(layer_top_elevations.shape[0],
                                         layer_top_elevations.shape[1] * layer_subdivisions),
                                  dtype=layer_top_elevations.dtype) * np.NaN

    # Create complete 2D grid of sub-layer point distances for every point/layer - needed for interpolation
    point_distances = np.ones(shape=(layer_top_elevations.shape[0],
                                     layer_top_elevations.shape[1] * layer_subdivisions),
                              dtype=layer_top_elevations.dtype) * np.NaN

    # Populate the point distances array

    for depth_index in range(point_distances.shape[1]):
        point_distances[:, depth_index] = var_dict['distances']

    # Iterate through points in elevation array
    for point_index in range(layer_top_elevations.shape[0]):
        # Iterate through layers
        for layer_index in range(layer_top_elevations.shape[1]):
            # Calculate layer thickness
            try:
                layer_thickness = layer_top_elevations[point_index, layer_index] - \
                                  layer_top_elevations[point_index, layer_index + 1]
            # Break if on bottom layer which has infinite thikness
            except IndexError:
                break

            # Iterate through the sub-layers
            for i in range(layer_subdivisions):
                # Get sublayer index
                sublayer_index = layer_index * layer_subdivisions + i

                sublayer_elevations[point_index,sublayer_index]=layer_top_elevations[point_index, layer_index] - \
                                                                i * layer_thickness /layer_subdivisions


    # Create an empty dictionary for the sublayer variables
    subvar_dict = {}

    # iterate through the variables and create a sublayer array for each
    for var in vars_2d:
        subvar_dict[var] = np.repeat(var_dict[var], layer_subdivisions, axis=1)

    # Obtain good data mask -- is this required??
    good_data_mask = ~np.isnan(sublayer_elevations)

    # Discard invalid points and store distance/elevation coordinates in dense 2D array
    point_distance_elevation = np.ones(shape=(np.count_nonzero(good_data_mask), 2),
                                       dtype=layer_top_elevations.dtype) * np.NaN

    point_distance_elevation[:, 0] = point_distances[good_data_mask]
    point_distance_elevation[:, 1] = sublayer_elevations[good_data_mask]

    # Compute distance range for bitmap
    distance_range = (math.floor(min(point_distance_elevation[:, 0]) / 10.0) * 10.0,
                      math.ceil(max(point_distance_elevation[:, 0]) / 10.0) * 10.0)


    # Compute elevation range for bitmap
    elevation_range = (math.floor(min(point_distance_elevation[:, 1]) / 10.0) * 10.0,
                       math.ceil(max(point_distance_elevation[:, 1]) / 10.0) * 10.0)

    xres = np.float(xres)
    yres = np.float(yres)

    grid_y, grid_x = np.mgrid[elevation_range[1]:elevation_range[0]:-yres,
                     distance_range[0]:distance_range[1]:xres]

    grid_distances = grid_x[0]

    grid_elevations = grid_y[:, 0]

    # Mask below the maximum depth
    max_depth = np.max(var_dict['layer_top_depth'][point_index]) + 50

    min_elevations = var_dict['elevation'] - max_depth * np.ones(np.shape(var_dict['layer_top_depth'][:, -1]))

    # Compute interpolated 1D array of minimum valid elevation values for each X
    min_elevation_grid = griddata(point_distances[:, 0], min_elevations, grid_distances,
                                  method=resampling_method)

    # Compute interpolated 1D array of maximum valid elevation values for each X

    max_elevation_grid = griddata(point_distances[:, 0],
                                  var_dict['elevation'],
                                  grid_x[0], method=resampling_method)

    # Add important variables to the cond_vars_dict

    var_dict['grid_elevations'] = grid_elevations

    var_dict['grid_distances'] = grid_distances


    # Iterate through variables and interpolate onto new grid
    for var in vars_2d:

        # Discard invalid variable points

        point_vars = subvar_dict[var][good_data_mask]

        var_grid = griddata(point_distance_elevation[:, ::-1],
                            point_vars, (grid_y, grid_x),
                            method=resampling_method)

        interpolated_var = purge_invalid_elevations(var_grid, grid_y, min_elevation_grid,
                                                    max_elevation_grid, yres)

        # Reverse the grid if it is west to east

        if var_dict['reverse_line']:

            interpolated_var = np.fliplr(interpolated_var)

        # Yield the generator and the dictionary with added variables
        yield interpolated_var, var_dict

def interpolate_1d_vars(vars_1D, var_dict, resampling_method='linear'):
    """
    Interpolate the 1D variables onto regular distance axes

    """
    # Iterate through the 1D variables, interpolate them onto the distances that were used for
    # the 2D variable gridding and add it to the dictionary

    for var in vars_1D:

        varray = griddata(var_dict['distances'],
                          var_dict[var], var_dict['grid_distances'],
                          method=resampling_method)

        # Reverse the grid if it is west to east

        if var_dict['reverse_line']:
            varray = varray[::-1]

        yield varray

def interpolate_data(data_variables, var_dict, interpolated_utm,
                     resampling_method='linear'):
    """

    :param data_variables: variables from netCDF4 dataset to interpolate
    :param var_dict: dictionary with the arrays for each variable
    :param interpolated_utm: utm corrdinates onto which to interpolate the line data
    :param resampling_method:
    :return:
    """

    # Define coordinates
    utm_coordinates = var_dict['utm_coordinates']

    # Add distance array to dictionary
    distances = coords2distance(utm_coordinates)

    # Now we want to find the equivalent line distance of the data based on the
    # gridded coordinates

    interpolated_distances = griddata(utm_coordinates, distances, interpolated_utm,
                                      method='nearest')

    # Now extract the data variable, interpolate them and add them to the dictionary

    for var in data_variables:

        # Create an empty array for interpolation

        arr = var_dict[var]

        interp_arr = np.zeros(shape=(np.shape(interpolated_distances)[0], np.shape(arr)[0]),
                              dtype=var_dict[var].dtype)

        # Interpolate each column separately

        for i in range(len(arr[0])):

            vals = arr[:, i]

            interp_arr[:, i] = griddata(distances, vals, interpolated_distances,
                                        method=resampling_method)

        # Add to the dictionary

        yield interp_arr



def unpack_plot_settings(panel_dict, entry):
    """

    :param panel_dict:
    :param entry:
    :return:
    """

    return [panel_dict[key][entry] for key in ['panel_' + str(i + 1) for i in range(len(panel_dict))]]

# Pull data from h5py object to a dictionary
def extract_hdf5_data(f, plot_vars):
    """

    :param f: hdf5 file
    :param plot_vars:
    :return:
    dictionary with interpolated datasets
    """

    datasets = {}

    for item in f.values():
        if item.name[1:] in plot_vars:
            datasets[item.name[1:]] = item.value
        # We also need to know easting, northing, doi, elevations and grid elevations
        if item.name[1:] == 'easting':
            datasets['easting'] = item.value
        if item.name[1:] == 'northing':
            datasets['northing'] = item.value
        if item.name[1:] == 'grid_elevations':
            datasets['grid_elevations'] = item.value
        if item.name[1:] == 'depth_of_investigation':
            datasets['depth_of_investigation'] = item.value
        if item.name[1:] == 'elevation':
            datasets['elevation'] = item.value
        if item.name[1:] == 'grid_distances':
            datasets['grid_distances'] = item.value
        if item.name[1:] == 'flm_layer_top_depth':
            datasets['flm_layer_top_depth'] = item.value

    return datasets

def plot_grid(ax, gridded_variables, variable, panel_kwargs):
    """

    :param gridded_variables:
    :param variables:
    :param panel_kwargs:
    :return:
    """

    # Define extents based on kwarg max depth

    try:
        min_elevation = np.min(gridded_variables['elevation']) - panel_kwargs['max_depth']
        ax.set_ylim(min_elevation)

    except KeyError:

        min_elevation = gridded_variables['grid_elevations'][-1]

    max_extent = gridded_variables['grid_elevations'][0] + 10

    extent = (gridded_variables['grid_distances'][0], gridded_variables['grid_distances'][-1],
              gridded_variables['grid_elevations'][-1], max_extent)

    ax.set_ylim(min_elevation, gridded_variables['grid_elevations'][0] + 40)

    # Define stretch
    # Flag for a logarithmic stretch

    try:
        log_stretch = panel_kwargs['log_plot']

    except KeyError:
        log_stretch = False  # False unless otherwise specified


    if log_stretch:
        # Tranform the plot data
        data = np.log10(gridded_variables[variable])

    else:
        data = gridded_variables[variable]
        # set automatic stretch values in case vmin and vmax aren't specified
        vmin, vmax = 0, 0.5

    # Define vmin an vmax if specified
    if 'vmin' in panel_kwargs.keys():
        vmin = panel_kwargs['vmin']
    if 'vmax' in panel_kwargs.keys():
        vmax = panel_kwargs['vmax']

    if log_stretch:
        vmin, vmax = np.log10(vmin), np.log10(vmax)

    # Define cmap if it is specified
    if 'cmap' in panel_kwargs.keys():
        cmap = panel_kwargs['cmap']

    else:
        cmap = 'jet'
    # Plot data
    im = ax.imshow(data, vmin=vmin, vmax=vmax,
                   extent=extent,
                   aspect='auto',
                   cmap=cmap)

    # Plot the elevation as a line over the section
    line_x = np.linspace(gridded_variables['grid_distances'][0], gridded_variables['grid_distances'][-1],
                         np.shape(gridded_variables[variable])[1])

    ax.plot(line_x,  gridded_variables['elevation'], 'k')

    # To remove gridded values that stick above this line we will fill the sky in as white
    ax.fill_between(line_x, max_extent * np.ones(np.shape(line_x)),
                    gridded_variables['elevation'], interpolate=True, color='white', alpha=1)

    try:
        if panel_kwargs['colourbar']:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if log_stretch:
                cb.ax.set_yticklabels([round(10 ** x, 4) for x in cb.get_ticks()])
            if 'colourbar_label' in panel_kwargs.keys():
                cb_label = panel_kwargs['colourbar_label']
                cb.set_label(cb_label, fontsize=10)
                cb.ax.tick_params(labelsize=10)
            else:
                pass
        else:
            pass
    except KeyError:
        pass
    # Add ylabel
    try:
        ylabel  = panel_kwargs['ylabel']
        ax.set_ylabel(ylabel)
    except KeyError:
        pass

    # PLot depth of investigation and make area underneath more transparent if desired
    if panel_kwargs['shade_doi']:
        eoi = gridded_variables['elevation'] - gridded_variables['depth_of_investigation']

        ax.plot(line_x, eoi, 'k')

        grid_base = gridded_variables['grid_elevations'][-1]

        # Shade the belwo doi areas

        ax.fill_between(line_x, eoi, grid_base, interpolate=True, color='white', alpha=0.5)

def plot_single_line(ax, gridded_variables, variable, panel_kwargs):
    """

    :param ax:
    :param gridded_variables:
    :param variables:
    :param panel_kwargs:
    :return:
    """
    # Define the array

    data = gridded_variables[variable]

    if 'colour' in panel_kwargs.keys():
        colour = panel_kwargs['colour']
    else:
        colour = 'black'


    ax.plot(gridded_variables['grid_distances'], data, colour)

    # Extract ymin and ymax if specified, otherwise assign based on the range with the line dataset
    if 'ymin' in panel_kwargs.keys():
        ymin = panel_kwargs['ymin']
    else:
        ymin = np.min(data) - 0.1 * np.min(data)

    if 'ymax' in panel_kwargs.keys():
        ymax = panel_kwargs['ymax']
    else:
        ymax = np.max(data) - 0.1 * np.max(data)

    ax.set_ylim(bottom=ymin, top=ymax, auto=False)

    try:
        ylabel  = panel_kwargs['ylabel']
        ax.set_ylabel(ylabel)
    except KeyError:
        pass

    try:
        if panel_kwargs['legend']:
            ax.legend()
    except KeyError:
        pass

def plot_multilines_data(ax, gridded_variables, variable, panel_kwargs):
    # Define the data

    data = gridded_variables[variable]

    try:
        colour = panel_kwargs["colour"]
        linewidth = panel_kwargs["linewidth"]
    except KeyError:
        colour = 'k'
        linewidth = 1


    for i, col in enumerate(data.T):
        ax.plot(gridded_variables['grid_distances'], data.T[i],
                color = colour, linewidth = linewidth)
        ax.set_yscale('log')
    try:
        ylabel = panel_kwargs['ylabel']
        ax.set_ylabel(ylabel)
    except KeyError:
        pass


def add_axis_coords(axis_label, array,
                    axis_above, axis_position, offset=-0.15):
    """
    Funtion for adding a coordinate axis to the bottom of the plot

    :param axis_label:
    :param array:
    :param axis_above:
    :param axis_position:
    :param offset:
    :return:
    """
    new_ax = axis_above.twiny()

    new_ax.set_xlabel(axis_label)

    new_ax.set_position(axis_position)
    new_ax.xaxis.set_ticks_position("bottom")
    new_ax.xaxis.set_label_position("bottom")
    # Offset the twin axis below the host
    new_ax.spines["bottom"].set_position(("axes", offset))

    # Turn on the frame for the twin axis, but then hide all
    # but the bottom spine
    new_ax.set_frame_on(True)
    new_ax.patch.set_visible(False)

    new_ax.spines["bottom"].set_visible(True)

    # Get tick locations from the old axis

    new_tick_locations = np.array(np.arange(0, 1.1, 0.1))

    new_ax.set_xticks(new_tick_locations)

    # Find the ticks to label

    new_x = griddata(np.linspace(0, 1, num=len(array)), array,
                     new_tick_locations)

    new_ax.set_xticklabels([str(int(x)) for x in new_x])

def align_axes(ax_array):
    """
    Function for aligning the axes and adding easting and northing to the bottom
    :param ax_array:
    :param add_easting:
    :param add_northing:
    :return:
    """
    # Dictionary for defining axis positions
    ax_pos = {}

    # Iterate through the axes and get position
    for i, ax in enumerate(ax_array):
        ax_pos[i] = ax.get_position()

    x0 = np.min([x.x0 for x in ax_pos.values()])
    ax_width = np.min([x.width for x in ax_pos.values()])

    for i, ax in enumerate(ax_array):
        ax.set_position([x0, ax_pos[i].y0, ax_width, ax_pos[i].height])

    return ax_pos


def plot_conductivity_section(ax_array, gridded_variables, plot_settings, panel_settings,
                              save_fig=False,  outfile=None):
    """

    :param gridded_variables:
    :param plot_settings:
    :param panel_settings:
    :param save_fig:
    :param outfile:
    :return:
    """

    # Unpack the panel settings

    variables = unpack_plot_settings(panel_settings,
                                          'variable')
    panel_kwargs = unpack_plot_settings(panel_settings,
                                             'panel_kwargs')

    plot_type = unpack_plot_settings(panel_settings,
                                          'plot_type')


    # Iterate through the axes and plot
    for i, ax in enumerate(ax_array):

        if 'title' in panel_kwargs:
            ax.set_title(panel_kwargs['title'])
        else:
            ax.set_title(' '.join([variables[i].replace('_', ' '), 'plot']))

        if plot_type[i] == 'grid':

            # PLot the grid
            plot_grid(ax, gridded_variables, variables[i], panel_kwargs[i])

        elif plot_type[i] == 'multi_line':

            plot_multilines_data(ax, gridded_variables, variables[i], panel_kwargs[i])

        elif plot_type[i] == 'line':
            plot_single_line(ax, gridded_variables, variables[i], panel_kwargs[i])

    # Now iterate through all of the axes and move them to align with the minimum
    # right hand margin seen for all axes

    ax_pos = align_axes(ax_array)

    # Add axis with northing at the bottom of the plot

    add_axis_coords('northing', gridded_variables['northing'], ax_array[-1], ax_pos[i], offset=-0.2)

    add_axis_coords('easting', gridded_variables['easting'], ax_array[-1], ax_pos[i], offset=-0.45)

    if save_fig:
        # If the dpi is set extract it from the plot settings
        if 'dpi' in plot_settings:
            dpi = plot_settings['dpi']
        else:
            dpi= 300
        plt.savefig(outfile, dpi=dpi)
        plt.close()


def plot_conductivity_section_from_hdf5file(ax_array, path, plot_settings, panel_settings, save_fig = False,
                                                outfile = None):
    """
    Function for plotting a vertical section from an hdf5 file

    :param path: path to hdf5 file
    :param plot_settings:
    :param panel_settings:
    :param save_fig:
    :param outfile:
    :return:
    """

    # Open hdf5 file
    f = h5py.File(path, 'r')

    # Extract the key datasets from the file

    plot_vars = unpack_plot_settings(panel_settings, 'variable')

    gridded_variables = extract_hdf5_data(f, plot_vars)

    plot_conductivity_section(ax_array, gridded_variables, plot_settings, panel_settings,
                              save_fig=save_fig, outfile=outfile)

def add_1d_layered_model(ax, df, gridded_variables, plot_variable, xy_columns, cmap='plasma_r',
                         colour_stretch=[0, 0.2], max_distance=200., stick_thickness=150.):

    # Get the coordinates of the section
    utm_coords = np.hstack((gridded_variables['easting'].reshape([-1, 1]),
                            gridded_variables['northing'].reshape([-1, 1])))

    # Find the nearest neighbours within the maximum distance
    d, i = spatial_functions.nearest_neighbours(df[xy_columns].values,
                                               utm_coords,
                                               points_required=1,
                                               max_distance=200.)

    # Add the minimum distance to the dataframe and remove nulls (i.e. those
    # that have a distance greater than the maximum allowable as denoted by a value
    # that is greater thant the length of the xy coordinates
    df['min_index'] = i

    df = df[df['min_index'] < len(utm_coords)]

    # Create an elevation from, to and distance along the line using the elevation and
    # distance along the line of the nearest neighbour

    df.loc[:, 'Elevation_from'] = gridded_variables['elevation'][df['min_index']] - df['Depth_to']
    df.loc[:, 'Elevation_to'] = gridded_variables['elevation'][df['min_index']] - df['Depth_from']
    df.loc[:, 'dist_along_line'] = gridded_variables['grid_distances'][df['min_index']]

    # Now we will define the colour stretch for water content based on the plasma colourbar
    vmin, vmax = colour_stretch[0], colour_stretch[1]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Iterate through the elevation intervls and add them to the axis
    for index, row in df.iterrows():
        # Define variables from the dataframe row
        elevation_from = row['Elevation_from']
        thickness = row['Elevation_to'] - elevation_from
        distance_along_line = row['dist_along_line']
        variable = row[plot_variable]
        # Add them to the axis
        rect = Rectangle((distance_along_line, elevation_from), stick_thickness, thickness,
                         edgecolor='k', facecolor=m.to_rgba(variable))
        ax.add_patch(rect)


def add_downhole_log_data(ax, df, gridded_variables, plot_variable, xy_columns, cmap='jet',
                          colour_stretch=[0, 0.2], log_stretch=False, max_distance=200., stick_thickness=150.):

    # Get the coordinates of the section
    utm_coords = np.hstack((gridded_variables['easting'].reshape([-1, 1]),
                            gridded_variables['northing'].reshape([-1, 1])))

    # Find the nearest neighbours within the maximum distance
    d, i = spatial_functions.nearest_neighbours(df[xy_columns].values,
                                               utm_coords,
                                               points_required=1,
                                               max_distance=max_distance)

    # Add the minimum distance to the dataframe and remove nulls (i.e. those
    # that have a distance greater than the maximum allowable as denoted by a value
    # that is greater thant the length of the xy coordinates
    df['min_index'] = i

    df = df[df['min_index'] < len(utm_coords)]

    # Kill the function if the downhole logs is not within the max distance
    if len(df) == 0:
        return None


    # Create an elevation from, to and distance along the line using the elevation and
    # distance along the line of the nearest neighbour

    df.loc[:, 'Elevation_from'] = gridded_variables['elevation'][df['min_index']] - df['Depth']

    # Create a fake elevation to column

    elevation_to = np.nan * np.zeros(len(df), dtype=np.float)

    elevation_to[:-1] = df['Elevation_from'].values[1:]

    df.loc[:, 'Elevation_to'] = elevation_to

    df.loc[:, 'dist_along_line'] = gridded_variables['grid_distances'][df['min_index']]

    # Now we will define the colour stretch
    vmin, vmax = colour_stretch[0], colour_stretch[1]
    if not log_stretch:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Iterate through the elevation intervls and add them to the axis
    for index, row in df.iterrows():
        # Define variables from the dataframe row
        elevation_from = row['Elevation_from']
        thickness = row['Elevation_to'] - elevation_from
        distance_along_line = row['dist_along_line']
        variable = row[plot_variable]
        # Add them to the axis
        rect = Rectangle((distance_along_line, elevation_from), stick_thickness, thickness,
                         edgecolor=None, facecolor=m.to_rgba(variable))
        ax.add_patch(rect)

    # Add the outline
    thickness = df['Elevation_from'].max() - df['Elevation_to'].min()

    rect = Rectangle((distance_along_line, df['Elevation_from'].min()), stick_thickness,
                     thickness, edgecolor='k', facecolor='none')

    ax.add_patch(rect)

def add_custom_colourbar(ax, cmap, vmin, vmax, xlabel):
    """
    Function for adding a custom gradient based colour bar to a matplotlib axis
    :param ax: axis created for colourbar
    :param cmap: string - matplotlib colour stretch
    :param vmin: float - minimium data value
    :param vmax: float - maximum data value
    :param xlabel: string - label for the x-axis
    """
    # Define the discretisation
    disc = 25
    # Create a grid that
    m= np.expand_dims(np.linspace(vmin,vmax,disc),axis=0)
    # Grid
    ax.imshow(m, interpolation='bicubic', cmap=cmap,
              extent=(vmin,vmax,0,vmax*0.1))
    # Set the ticks
    ax.set_yticks(np.arange(0))
    ax.set_xticks([vmin, vmax])
    # Set the axis label
    ax.set_xlabel(xlabel)


def plot_1D_layered_model(ax, profile, depth_top, doi=None, log_plot=True):
    """

    :param ax: matplotlib axis
    :param profile: flat numpy array with layered values
    :param depth_top: flat numpy array with layer top values
    :param doi: float of depth of investigation
    :param log_plot: boolean: if True conductivity gets displayed in log space
    :return:
    matplotlib axis
    """
    # First we want to expand the axes to get the layered
    # effect on the plot

    prof_expanded = np.zeros(shape=2 * len(profile) + 1,
                             dtype=np.float)

    prof_expanded[1:] = np.repeat(profile, 2)

    depth_expanded = (np.max(depth_top) + 10) * np.ones(shape=len(prof_expanded),
                                                        dtype=np.float)

    depth_expanded[:-1] = np.repeat(depth_top, 2)

    # PLot
    ax.plot(prof_expanded, depth_expanded)

    plt.gca().invert_yaxis()

    # Add depth of investigation if provided
    if doi is not None:
        ax.hlines(doi, 0, np.max(prof_expanded),
                  color='green', linestyles='dotted',
                  label='DOI')

        ax.legend()

    if log_plot:
        ax.set_xscale('log')
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
    return ax


def plot_downhole_log(ax, values, depth, log_plot=True,
                              color='k', label = ''):
    """

    :param ax: matplotlib axis
    :param values: downhole log values
    :param depth: downhole log depth
    :param logplot: boolean: if True conductivity gets displayed in log space
    :param color: matplotlib colour code
    :return:
    """
    ax.plot(values, depth, color=color, label=label)

    if log_plot:
        ax.set_xscale('log')

    ax.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')

    return ax


def plot_point_dataset(utm_coords,
                       utm_wkt,
                       variable,
                       utm_bbox=None,
                       colourbar_label=None,
                       plot_title=None,
                       colour_scheme='binary',
                       point_size=10,
                       point_step=1
                       ):
    '''
    Function to plot data points on a map.
    @author: Andrew Turner & Alex Ip
    @param utm_coords: coordiante array shape of (,2)
    @param utm_wkt: well known text code for utm coordinates
    @param variable numpy array of variable to plot
    @param utm_bbox: UTM Bounding box of form [xmin, ymin, xmax, ymax] or None for all points. Default=None
    @param colourbar_label:
    @param plot_title: String to prefix before dataset title. Default=None for dataset title or dataset basename
    @param colour_scheme: String specifying colour scheme for data points. Default='binary'
    @param point_size: Point size for data points. Default=10
    @param point_step: Point step between plotted points - used to skip points in dense datasets. Default=1
    '''

    def rescale_array(input_np_array, new_range_min=0, new_range_max=1):
        old_min = input_np_array.min()
        old_range = input_np_array.max() - old_min
        new_range = new_range_max - new_range_min

        scaled_np_array = ((input_np_array - old_min) / old_range * new_range) + new_range_min

        return scaled_np_array

    utm_zone = get_spatial_ref_from_wkt(utm_wkt).GetUTMZone()  # -ve for Southern Hemisphere
    southern_hemisphere = (utm_zone < 0)
    utm_zone = abs(utm_zone)
    projection = ccrs.UTM(zone=utm_zone,
                          southern_hemisphere=southern_hemisphere)
    print('utm_zone = {}'.format(utm_zone))

    # Set geographic range of plot
    if utm_bbox is None:
        utm_bbox = [
            np.min(utm_coords[:, 0]),
            np.min(utm_coords[:, 1]),
            np.max(utm_coords[:, 0]),
            np.max(utm_coords[:, 1])
        ]
        spatial_mask = np.ones(shape=variable.shape, dtype='Bool')
    else:
        spatial_mask = np.logical_and(np.logical_and((utm_bbox[0] <= utm_coords[:, 0]),
                                                     (utm_coords[:, 0] <= utm_bbox[2])),
                                      np.logical_and((utm_bbox[1] <= utm_coords[:, 1]),
                                                     (utm_coords[:, 1] <= utm_bbox[3]))
                                      )
        utm_coords = utm_coords[spatial_mask]

    print('{} points in UTM bounding box: {}'.format(np.count_nonzero(spatial_mask),
                                                     utm_bbox))

    colour_array = rescale_array(variable[spatial_mask], 0, 1)

    fig = plt.figure(figsize=(30, 30))

    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.set_title(plot_title)

    # map_image = cimgt.OSM() # https://www.openstreetmap.org/about
    # map_image = cimgt.StamenTerrain() # http://maps.stamen.com/
    map_image = cimgt.QuadtreeTiles()
    ax.add_image(map_image, 10)

    # Compute and set regular tick spacing
    range_x = utm_bbox[2] - utm_bbox[0]
    range_y = utm_bbox[3] - utm_bbox[1]
    x_increment = pow(10.0, floor(log10(range_x))) / 2
    y_increment = pow(10.0, floor(log10(range_y))) / 2
    x_ticks = np.arange((utm_bbox[0] // x_increment + 1) * x_increment,
                        utm_bbox[2], x_increment)
    y_ticks = np.arange((utm_bbox[1] // y_increment + 1) * y_increment,
                        utm_bbox[3], y_increment)
    plt.xticks(x_ticks, rotation=45)
    plt.yticks(y_ticks)

    # set the x and y axis labels
    plt.xlabel("Eastings (m)", rotation=0, labelpad=20)
    plt.ylabel("Northings (m)", rotation=90, labelpad=20)

    # See link for possible colourmap schemes:
    # https://matplotlib.org/examples/color/colormaps_reference.html
    cm = plt.cm.get_cmap(colour_scheme)

    # build a scatter plot of the specified data, define marker,
    # spatial reference system, and the chosen colour map type
    sc = ax.scatter(utm_coords[::point_step, 0],
                    utm_coords[::point_step, 1],
                    marker='o',
                    c=colour_array[::point_step],
                    s=point_size,
                    alpha=0.9,
                    transform=projection,
                    cmap=cm
                    )

    # set the colour bar ticks and labels
    cb = plt.colorbar(sc, ticks=[0, 1])
    cb.ax.set_yticklabels([str(np.min(variable[spatial_mask])),
                           str(np.max(variable[spatial_mask]))])
    if colourbar_label is not None:
        cb.set_label(colourbar_label)

    plt.show()


def getMaxDepth(data):
    """
    A quick helper function to loop through a dict of dataframes and extract the largest (deepest) depth value
    Will hopefully be deprecated in the future when a drilled depth field is added to header

    :param: data, a dict of dataframes to be checked for max depth

    :return: floating point number of the biggest depth value in the input data
    """
    # A place to store all the depths extracted from the dataframes
    depth_data = []
    for table in data.keys():
        # All the possible column names that store depth data
        # Not all depth columns will be in each table, hence the try/except statements
        if table == 'aem':
            continue
        for column in ['Depth_from', 'Depth_to', 'Depth']:
            try:
                depth_data.append(data[table][column].values)
            except KeyError:
                continue
    # this will still have negative values for above ground construction
    depths = np.concatenate(depth_data)
    return depths.max()


def getGLElevation(header):
    """
    Quick function to extract the ground elevation from the header
    """
    return header.loc[0, 'Ground_elevation_mAHD']


def axisBuilder(axis_name, data):
    """
    Function to call the relevant drawing function based on the input request
    """
    if axis_name == 'cond':
        return drawDownHoleConds(data['indgam'])
    if axis_name == 'gamma':
        return drawGamma(data['indgam'])
    if axis_name == 'nmr':
        return drawNMR(data['javelin'])
    if axis_name == 'lith':
        return drawLith(data['lithology'])
    if axis_name == 'construction':
        return drawConstruction(data['construction'])
    if axis_name == 'EC':
        return drawPoreFluidEC(data['porefluid'])
    if axis_name == 'pH':
        return drawPoreFluidpH(data['porefluid'])
    if axis_name == 'magsus':
        return drawMagSus(data['magsus'])
    if axis_name == 'AEM':
        return drawAEMConds(data['aem'])


def getLastSWL(waterlevels):
    """
    A function to extract the datetime and level of the most recent waterlevel on record for that hole
    """
    # Sort all waterlevels by date
    waterlevels = waterlevels.sort_values(['Date'])
    # Extract the last water level
    last_waterlevel = waterlevels['Depth'].iloc[-1]
    # Extract the last timestamp
    last_wl_datetime = waterlevels['Date'].iloc[-1]
    return last_waterlevel, last_wl_datetime


def remove_inner_ticklabels(fig):
    """
    A function to strip off the tick marks and labels from any axis that is not clear to the left
    """
    for ax in fig.axes:
        try:
            ax.label_outer()
        except:
            pass


def make_header_table(header, values_per_row=2):
    '''
    Function to turn the first row of pandas Dataframe into a table for display in a matplotlib figure


    :param: header, a pandas DataFrame, only the first row will be used
    :param: values_per_row, defines how many key/value pairs are in each returned row. Default is 2, ie 4 columns in table

    :return:
    A 2 dimensional list with key/value pairs in adjacent cells.
    Width of table is defined as input parameter.
    Length of table adapts to number of columns in input header dataframe


    NOTE: should be rewritten to use simple reshaping of np.array
    '''

    def my_round(arg, dps):
        '''
        Quick rounding function that rounds a float to the requested precision and returns it as a string
        '''
        if isinstance(arg, float):
            return str(round(arg, dps))
        else:
            return arg

    # Convert header dataframe into pd.Series type
    s_header = header.iloc[0]
    # Clean up columns that we don't want displayed in table
    s_header = s_header.drop(['geom', 'geometry'])
    # Create a list with the desired numbers of rows, each row is an empty list at this point
    table_vals = [[]] * math.ceil(len(s_header) / values_per_row)

    # Iterate over series
    for i, (key, val) in enumerate(s_header.iteritems()):
        # Calculate the row that the values will be stored in
        row = (i - 1) // values_per_row
        # Add the new values (as a list) to the existing row (also a list)
        table_vals[row] = table_vals[row] + [my_round(key, 4), my_round(val, 4)]
    # If the length of the row isn't filled yet
    if len(table_vals[-1]) != values_per_row * 2:
        # Extend it by
        table_vals[-1].extend([''] * (values_per_row * 2 - len(table_vals[-1])))
    return table_vals


def drawConstruction(construction):
    """
    The function to draw bore construction onto an axes object.
    This is a simplified version of some much more complex code.
    This version only deals with lining and screens in a single casing string.
    Casing protectors are not drawn, and nested piezos are drawn wrong.

    :param: construction, a dataframe containing the construction data of the borehole

    :return:
    matplotlib axes object with the appropriate shapes drawn onto it
    """
    # only interested in drawing lining or screen, so cut the rest
    construction = construction[construction['Construction_type'].isin(['lining', 'inlet'])]
    # only interested in drawing below ground level, so remove above ground values
    construction.loc[construction[construction['Depth_from'] < 0].index, 'Depth_from'] = 0

    # create a set of axes and format accordingly
    plt.plot()
    ax = plt.gca()
    ax.set_xlim([-0.1, 1.1])
    ax.set_xlabel('Construction')
    ax.set_xticks([])

    for _, ctype in construction.iterrows():
        # define width of construction as a % of axes width
        left = 0.2
        right = 0.8
        top = ctype['Depth_from']
        bottom = ctype['Depth_to']

        if ctype['Construction_type'] == 'lining':
            casing = LineCollection([[[left, top], [left, bottom]], [[right, top], [right, bottom]]], color='black')
            ax.add_collection(casing)
        if ctype['Construction_type'] == 'inlet':
            screen = mPolygon([[left, top], [right, top], [right, bottom], [left, bottom]],
                              closed=True, hatch='---', edgecolor='black', linestyle='solid',
                              facecolor='white')
            ax.add_patch(screen)
    return ax


def drawLith(lithology):
    """
    Function to draw the lithology of a borehole.
    This function relies very heavily on the lookup table defined in the nested function buildLithPatches
    There is definite need to make this a more comprehensive lookup table, probably configured via spreadsheet.
    See \\prod.lan\active\proj\futurex\Common\ScriptsAndTools\Borehole_Data_Consolidation_CompositeLogs\Scripts\lithologydisplaymapping.xlsx

    :param: lithology, a dataframe containing the lithology intervals for the borehole
    :return:
    matplotlib axes object with the lithology drawn on as coloured, patterned polygons
    """

    def buildLithPatches(lithology):
        lithsymbols = {'sandstone': {'facecolor': 'yellow', 'edgecolor': 'None', 'linestyle': 'None', 'hatch': '.'},
                       'sand': {'facecolor': 'yellow', 'edgecolor': 'None', 'linestyle': 'None', 'hatch': '.'},
                       'conglomerate': {'facecolor': 'yellow', 'edgecolor': 'None', 'linestyle': 'None', 'hatch': 'o'},
                       'siltstone': {'facecolor': 'green', 'edgecolor': 'None', 'linestyle': 'None', 'hatch': '-'},
                       'clay': {'facecolor': 'lightgrey', 'edgecolor': 'None', 'linestyle': 'None', 'hatch': '-'},
                       'shale': {'facecolor': 'grey', 'edgecolor': 'None', 'linestyle': 'None', 'hatch': '-'},
                       'mudstone': {'facecolor': 'lightgrey', 'edgecolor': 'None', 'linestyle': 'None', 'hatch': '-'},
                       'soil': {'facecolor': 'brown', 'edgecolor': 'None', 'linestyle': 'None', 'hatch': ''},
                       'unknown': {'facecolor': 'lightgrey', 'edgecolor': 'black', 'linestyle': '-', 'hatch': ''},
                       'silty sand': {'facecolor': 'khaki', 'edgecolor': 'None', 'linestyle': 'None', 'hatch': ''},
                       'granite': {'facecolor': 'pink', 'edgecolor': 'None', 'linestyle': 'None', 'hatch': '+'}}

        patches = []
        labels = []
        drawn_lithtypes = []

        for _, lith_row in lithology.iterrows():
            # Always want the lithology to fill the axes
            left = 0
            right = 1
            # Update the top and bottom for each row
            top = lith_row['Depth_from']
            bottom = lith_row['Depth_to']
            # Extract lithology info
            lith = lith_row['Lithology_type']
            # Apply lithology to lookup
            if lith not in lithsymbols.keys():
                simp_lith = 'unknown'
            else:
                simp_lith = lith
            # Don't want to double up on legend, so if lithology has already been drawn, keep track of this
            if simp_lith not in drawn_lithtypes:
                drawn_lithtypes.append(simp_lith)
                # Add the patch to the patch collection
            patches.append(mPolygon([[left, top], [right, top], [right, bottom], [left, bottom]],
                                    closed=True, **lithsymbols[simp_lith]))
            # If the lithology isn't in the lookup table, add a label for what the actual lithology is
            if simp_lith == 'unknown':
                labels.append([0.05, (bottom + top) / 2, lith_row['Lithology_type']])
        # Define the legend for the lithology
        leg_patches = [mpatches.Patch(color=lithsymbols[simp_lith]['facecolor'], hatch=lithsymbols[simp_lith]['hatch'],
                                      label=simp_lith) for simp_lith in drawn_lithtypes]

        return patches, labels, leg_patches

    # Setup the axes as required
    plt.plot()
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_xlabel([])
    ax.set_xlim([0, 1])
    ax.set_xlabel('Lithology')

    # Get the required inputs from the nested function
    polys, labels, leg_patches = buildLithPatches(lithology)
    # Need to loop through as for some reason ax.add_patch_collection() wasn't working
    for poly in polys:
        ax.add_patch(poly)
    # Add the labels
    for x, y, s in labels:
        ax.text(x, y, s)
    ax.legend(handles=leg_patches, loc='lower center')

    return ax


def drawNMR(javelin):
    """
    Function to draw the downhole NMR (aka Javelin) inversion results for a single borehole

    Water content results are typically drawn as stacked area charts, with clay bound water first
    and progressively freer water stacked ontop. This has been done by drawing polygons defined
    by the origin vertical axis, the clay content, the clay + capilliary water content and the
    total water content.

    :param: javelin, a DataFrame with the calculated water content results and depth stored
    :return:
    matplotlib axes function with the water contents plotted as polygons
    """
    # make coordinate pairs for each quantity
    clay = list(zip(javelin['Clay_water_content'], javelin['Depth']))
    capillary = list(zip(javelin['Capillary_water_content'], javelin['Depth']))
    total = list(zip(javelin['Total_water_content'], javelin['Depth']))
    # sum clay and capillary to give the inside edge x values for mobile
    clay_cap_sum = list(
        zip(javelin['Capillary_water_content'].values + javelin['Clay_water_content'].values, javelin['Depth']))

    # make polygons for each quantity
    # The clay bound water polygon is defined by the vertical axes, and the calculated clay bound water content by depth
    p1 = mpatches.Polygon([[0, clay[0][1]]] + clay + [[0, clay[-1][1]]], closed=True, color='lightblue')
    # The capillary bound water polygon is defined by the clay
    p2 = mpatches.Polygon(clay + clay_cap_sum[::-1], closed=True, color='blue')
    p3 = mpatches.Polygon(clay_cap_sum + total[::-1], closed=True, color='darkblue')

    nmr_legend = collections.OrderedDict(
        (('clay bound water', 'lightblue'), ('capillary bound water', 'blue'), ('free water', 'darkblue')))
    leg_patches = [mpatches.Patch(color=value, label=key) for key, value in nmr_legend.items()]

    plt.plot()
    ax = plt.gca()
    ax.add_patch(p1)
    ax.add_patch(p2)
    ax.add_patch(p3)
    ax.set_xlabel('Water Fraction')
    ax.grid(True)
    ax.xaxis.tick_top()
    ax.set_xlim([0.5, 0])
    ax.legend(handles=leg_patches, loc='lower center')
    return ax


def drawDownHoleConds(indgam):
    plt.plot(indgam['Apparent_conductivity'], indgam['Depth'], label='Conductivity', linestyle='-', color='blue')
    ax = plt.gca()
    ax.set_xlabel('Conductivity (S/m)')
    ax.set_xscale('log')
    ax.grid(True)
    ax.xaxis.tick_top()

    return ax


def drawAEMConds(aem):
    tops = list(zip(aem['Bulk_conductivity'], aem['Depth_from']))
    bots = list(zip(aem['Bulk_conductivity'], aem['Depth_to']))
    coords = []
    for i in range(len(tops)):
        coords.append(tops[i])
        coords.append(bots[i])
    coords = np.array(coords)
    plt.plot(coords[:, 0], coords[:, 1], '-')
    ax = plt.gca()
    ax.set_xlabel('Conductivity (S/m)')
    ax.set_xscale('log')
    ax.grid(True)
    ax.xaxis.tick_top()

    return ax


def drawGamma(indgam):
    if indgam['GR'].notna().any():
        gam_col = 'GR'
        gam_unit = 'API'
    else:
        gam_col = 'Gamma_calibrated'
        gam_unit = 'counts per second'
    plt.plot(indgam[gam_col], indgam['Depth'], label=gam_col, linestyle='-', color='red')
    ax = plt.gca()
    ax.set_xlabel('Natural Gamma Ray ({})'.format(gam_col, gam_unit))
    ax.grid(True)
    ax.xaxis.tick_top()
    return ax


def drawPoreFluidpH(porefluid):
    plt.plot()
    ax = plt.gca()
    ax.grid(True)
    ax.set_xlabel('Porefluid pH (pH)')
    ax.plot(porefluid['pH'], porefluid['Depth'], marker='.')
    ax.xaxis.tick_top()
    return ax


def drawPoreFluidEC(porefluid):
    plt.plot()
    ax = plt.gca()
    ax.grid(True)
    ax.set_xlabel('Porefluid EC (S/m)')
    ax.plot(porefluid['EC'], porefluid['Depth'], marker='.')
    ax.xaxis.tick_top()
    return ax


def drawMagSus(magsus):
    plt.plot()
    ax = plt.gca()
    ax.grid(True)
    ax.set_xlabel('Magnetic Susceptibiliy')
    ax.plot(magsus['Magnetic_susceptibility'], magsus['Depth'], marker='.')
    ax.xaxis.tick_top()
    return ax


def drawCompLog(data, output_path=None):
    header = data['header']

    # load GA and EFTF logos for placement on the logs
    ga = mpimg.imread(
        r'\\prod.lan\active\proj\futurex\Common\ScriptsAndTools\Borehole_Data_Consolidation_CompositeLogs\StandardInputs\ga-logo.jpg')
    new_height = [int(dim / 2) for dim in ga.shape[0:2]][0:2]
    ga = resize(ga, new_height)
    eftf = mpimg.imread(
        r'\\prod.lan\active\proj\futurex\Common\ScriptsAndTools\Borehole_Data_Consolidation_CompositeLogs\StandardInputs\eftf-logo.png')

    # booleans for sectioning the code later
    hasConductivity = bool(header.loc[0, 'Induction_acquired'])
    hasGamma = bool(header.loc[0, 'Gamma_acquired'])
    hasLith = bool(header.loc[0, 'Lithology_available'])
    hasNMRLogs = bool(header.loc[0, 'Javelin_acquired'])
    hasConstructionLogs = bool(header.loc[0, 'Construction_available'])
    hasPoreWaterChem = bool(header.loc[0, 'EC_pH_acquired'])
    hasWL = bool(header.loc[0, 'SWL_available'])
    hasTimeSeries = hasWL and len(data['waterlevels']) > 2
    hasAEMConductivity = bool(header.loc[0, 'AEM_conductivity_available'])
    hasMagSus = bool(header.loc[0, 'MagSus_available'])

    # key parameters for during plotting
    hole_name = header.loc[0, 'Borehole_name']
    max_depth = math.ceil(getMaxDepth(data))
    metres_per_inch = 5
    figlength = 4.5 + max_depth / metres_per_inch
    elevation = getGLElevation(data['header'])
    if hasWL:
        swl, swl_time = getLastSWL(data['waterlevels'])
    # row ratios in the gridspec
    header_height = 2.5
    if hasTimeSeries:
        timelog_height = 2
    else:
        timelog_height = 0
    depthlog_height = figlength - (header_height - timelog_height)

    if hasTimeSeries:
        height_ratios = [header_height, depthlog_height, timelog_height]
    else:
        height_ratios = [header_height, depthlog_height]
    nrows = len(height_ratios)

    # column ratios in the gridspec
    # the order of these boolean evaluations dictates the order of the axes from left to right
    width_ratios = []
    chart_col_order = []
    if hasGamma:
        width_ratios.append(3)
        chart_col_order.append('gamma')
    if hasConductivity:
        width_ratios.append(3)
        chart_col_order.append('cond')
    if hasAEMConductivity:
        width_ratios.append(3)
        chart_col_order.append('AEM')
    if hasNMRLogs:
        width_ratios.append(3)
        chart_col_order.append('nmr')
    if hasLith:
        width_ratios.append(2)
        chart_col_order.append('lith')
    if hasConstructionLogs:
        width_ratios.append(1)
        chart_col_order.append('construction')
    if hasPoreWaterChem:
        width_ratios.append(2)
        width_ratios.append(2)
        chart_col_order.append('EC')
        chart_col_order.append('pH')
    if hasMagSus:
        width_ratios.append(2)
        chart_col_order.append('magsus')

    # defining the figure size
    figwidth = max(8, int(sum(width_ratios) * (2 / 3)))
    figsize = [figwidth, figlength]

    width_ratios = width_ratios if len(width_ratios) > 0 else [1]
    ncols = len(width_ratios)
    SWLlabelaxis = int(ncols / 2)
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, width_ratios=width_ratios, height_ratios=height_ratios)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(hole_name + ' Composite Log', size=22)

    # the code to add the images so they display well when saved to a file!
    fig.figimage(ga, xo=0.3 * ga.shape[0], yo=fig.bbox.ymax - 1.5 * ga.shape[0])
    fig.figimage(eftf, xo=fig.bbox.xmax - 3 * eftf.shape[0], yo=fig.bbox.ymax - 2 * eftf.shape[0])

    axt = fig.add_subplot(gs[0, :])
    table = plt.table(cellText=make_header_table(header), loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    axt.axis('off')

    axs = []
    for i, key in enumerate(chart_col_order):
        if i == 0:
            ax = fig.add_subplot(gs[1, i])
            ax.set_ylabel('Depth (m)')
            ax.set_ylim([max_depth, 0])  # sets the range for the logs and inverts the axes
        else:
            ax = fig.add_subplot(gs[1, i], sharey=axs[0])
            # pinched from
            # #https://stackoverflow.com/questions/20416609/remove-the-x-axis-ticks-while-keeping-the-grids-matplotlib
            # don't really understand what it does
            for tic in ax.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
        ax = axisBuilder(key, data)
        ax.xaxis.set_label_position('top')
        if hasWL:
            # draw a blue line across the depth logs at the last known standing water level
            # add the line
            ax.axhline(y=swl, color='darkblue')

        axs.append(ax)

    # if hasWL:
    # # create the label that should be printed below the line
    # swl_label = 'DTW @ ' + str(swl_time) + ' was\n' + str(round(swl,3)) + ' m below surface'
    # # add the label as text to the middle axes after the loop, so it spreads over multiple axes easily
    # axs[SWLlabelaxis].text(x = ax.get_xlim()[0], y = swl - 0.5, s = swl_label,
    # bbox=dict(facecolor='white', alpha = 0.5, zorder = -1))

    #     set up the AHD axis
    ax0 = axs[0].twinx()
    ax0.spines['right'].set_position(('axes', -0.42))
    ax0.set_ylim([elevation - max_depth, elevation])
    ax0.set_ylabel('Elevation (m AHD)', labelpad=-40)

    if hasTimeSeries:
        ax7 = fig.add_subplot(gs[2, :])
        sorted_swl = data['waterlevels'].sort_values('Date')
        ax7.plot(sorted_swl['Date'], sorted_swl['Depth'], marker='.', linestyle='-', color='blue')
        ax7.set_ylabel('Depth To Water (m Below Ground Level)')

    fig.subplots_adjust(wspace=0)
    remove_inner_ticklabels(fig)
    if output_path is not None:
        plt.savefig(output_path + '.svg')
        plt.savefig(output_path + '.png')
    else:
        return fig, axs