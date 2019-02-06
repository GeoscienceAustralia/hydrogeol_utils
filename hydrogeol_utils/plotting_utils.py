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
import os
import gc
from scipy.interpolate import griddata
import numpy as np
from geophys_utils._netcdf_line_utils import NetCDFLineUtils
from geophys_utils._transect_utils import coords2distance
from hydrogeol_utils import spatial_functions
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.patches import Rectangle


class ConductivitySectionPlot:
    """
    VerticalSectionPlot class for functions for creating vertical section plots
    """

    def __init__(self,
                 netCDFConductivityDataset,
                 netCDFemDataset = None):
        """
        :param netCDFConductivityDataset: netcdf line dataset with
         conductivity model
        :param netCDFemDataset: netcdf line dataset with
         EM measurements
        """
        if not self.testNetCDFDataset(netCDFConductivityDataset):
            raise ValueError("Input datafile is not netCDF4 format")

        # If datafile is given then check it is a netcdf file

        if netCDFemDataset is not None:
            if not self.testNetCDFDataset(netCDFemDataset):
                raise ValueError("Input datafile is not netCDF4 format")
            else:
                self.EM_data =  netCDFemDataset

        self.conductivity_model = netCDFConductivityDataset

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

    def purge_invalid_elevations(self, var_grid, grid_y, min_elevation_grid,
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


    def interpolate_2d_vars(self, vars_2d, var_dict, xres, yres,
                           layer_subdivisions, resampling_method):
        """
        Generator to interpolate 2d variables (i.e conductivity, uncertainty)

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

            interpolated_var = self.purge_invalid_elevations(var_grid,
                                                             grid_y,
                                                             min_elevation_grid,
                                                             max_elevation_grid,
                                                             yres)

             # Reverse the grid if it is west to east

            if var_dict['reverse_line']:

                interpolated_var = np.fliplr(interpolated_var)

            # Yield the generator and the dictionary with added variables
            yield interpolated_var, var_dict

    def interpolate_1d_vars(self, vars_1D, var_dict, resampling_method='linear'):
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

    def interpolate_data(self, data_variables, var_dict, interpolated_utm,
                         resampling_method='linear'):
        """

        :param data_variables: variables from netCDF4 dataset to interpolate
        :param var_dict: victionary with the arrays for each variable
        :param interpolated_utm: utm corrdinates onto which to interpolate the line data
        :param resampling_method:
        :return:
        """

        # Define coordinates
        utm_coordinates = var_dict['coordinates']

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

    def grid_vars(self, xres, yres, lines, conductivity_variables,
                     data_variables = None,
                     layer_subdivisions = 4, resampling_method = 'linear',
                     save_hdf5 = False, hdf5_dir = None,
                     overwrite_hdf5 = True, return_dict = True):
        """
        A function for interpolating 1D and 2d variables onto a vertical grid
        cells size xres, yres
        :param xres: Float horizontal cell size along the line
        :param yres: Float vertical cell size
        :param lines: int single line or list of lines to be gridded
        :param conductivity_variables: list of variables from the conductivity
         model
        :param data_variables: list of variables from the EM data
        :param layer_subdivisions:
        :param resampling_method: str or int, optional
         Specifies the kind of interpolation as a string (‘linear’, ‘nearest’,
         ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’, where ‘zero’,
         ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of
         zeroth, first, second or third order; ‘previous’ and ‘next’ simply
         return the previous or next value of the point) or as an integer specifying
         the order of the spline interpolator to use. Default is ‘linear’.
        :param save_hdf5: Boolean parameter indicating whether interpolated variables
         get saved as hdf or no
        :param hdf5_dir: path of directory into which the hdf5 files are saved
        :param overwrite_hdf5: Boolean parameter referring to if the user wants to
         overwrite any pre-existing files
        :param return_dict: Boolean parameter indicating if a dictionary is returned or not
        :return:
        dictionary with interpolated variables as numpy arrays
        """

        # Create a line utils for each object

        condLineUtils = NetCDFLineUtils(self.conductivity_model)

        try:
            dataLineUtils = NetCDFLineUtils(self.EM_data)
            # Flag for if dta was included in the plot section initialisation
            plot_dat = True
        except AttributeError:
            plot_dat = False

        # If line is not in an array like object then put it in a list
        if type(lines) == int:
            lines = [lines]
        elif isinstance(lines ,(list, tuple, np.ndarray)):
            pass
        else:
            raise ValueError("Check lines variable.")

        # We need easting, northing, elevation and layer top depth to be include so add them to the variable list if not
        for item in ['easting', 'northing', 'elevation', 'layer_top_depth']:
            if item not in conductivity_variables:
                conductivity_variables.append(item)

        # First create generators for returning coordinates and variables for the
        # lines


        cond_lines= condLineUtils.get_lines(line_numbers=lines,
                                            variables=conductivity_variables)
        if plot_dat:
            dat_lines = dataLineUtils.get_lines(line_numbers=lines,
                                                variables=data_variables)
        # Interpolated results will be added to a dicitonary
        interpolated = {}

        # Iterate through the lines
        for i in range(len(lines)):
            # Extract the variables and coordinates for the line in question
            line_no, cond_var_dict = next(cond_lines)

            # Add an entry to the dictionary for this line
            interpolated[line_no] = {}

            # If the line is west to east we want to reverse the coord
            # array and flag it

            # Define coordinates
            utm_coordinates = condLineUtils.utm_coords(cond_var_dict['coordinates'])[1]

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

            vars_2d = [v for v in conductivity_variables if cond_var_dict[v].ndim == 2]
            vars_1d = [v for v in conductivity_variables if cond_var_dict[v].ndim == 1]

            # Interpolate 2d variables and save them into the interpolated dictionary



            # Generator for inteprolating 2D variables from the vars_2d list
            interp2d = self.interpolate_2d_vars(vars_2d, cond_var_dict, xres, yres, layer_subdivisions,
                                                resampling_method)

            for var in vars_2d:
                # Generator yields the interpolated variable array
                interpolated[line_no][var], cond_var_dict = next(interp2d)

            # Add grid distances and elevations to the interpolated dictionary
            interpolated[line_no]['grid_distances'] = cond_var_dict['grid_distances']
            interpolated[line_no]['grid_elevations'] = cond_var_dict['grid_elevations']

            # Generator for inteprolating 1D variables from the vars_1d list
            interp1d = self.interpolate_1d_vars(vars_1d, cond_var_dict, resampling_method)

            for var in vars_1d:
                # Generator yields the interpolated variable array
                interpolated[line_no][var] = next(interp1d)

            if plot_dat:
                # Extract variables from the data
                line_no, data_var_dict = next(dat_lines)

                # For interpolating the EM data we need the interpolated easting and northing
                # values

                interpolated_utm = np.hstack((interpolated[line_no]['easting'].reshape([-1, 1]),
                                         interpolated[line_no]['northing'].reshape([-1, 1])))

                # Generator for interpolating data variables from the data variables list
                interp_dat = self.interpolate_data(data_variables, data_var_dict, interpolated_utm,
                                                    resampling_method)

                for var in data_variables:
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

                # Many lines may fil up memory so if the dictionary is not being returned then
                # we garbage collect
                if not return_dict:
                    del interpolated[line_no]

                    # Collect the garbage
                    gc.collect()
        if return_dict:
            return interpolated

    def unpack_plot_settings(self, panel_dict, entry):
        """

        :param panel_dict:
        :param entry:
        :return:
        """

        return [panel_dict[key][entry] for key in ['panel_' + str(i + 1) for i in range(len(panel_dict))]]

    # Pull data from h5py object to a dictionary
    def extract_hdf5_data(self, f, plot_vars):
        """

        :param f: hdf5 file path
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

    def plot_grid(self, ax, gridded_variables, variable, panel_kwargs):
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

    def plot_single_line(self, ax, gridded_variables, variable, panel_kwargs):
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

    def plot_multilines_data(self, ax, gridded_variables, variable, panel_kwargs):
        # Define the data

        data = gridded_variables[variable]

        for i, col in enumerate(data.T):
            ax.plot(gridded_variables['grid_distances'], data.T[i])
            ax.set_yscale('log')
        try:
            ylabel = panel_kwargs['ylabel']
            ax.set_ylabel(ylabel)
        except KeyError:
            pass


    def add_axis_coords(self, axis_label, array,
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

    def plot_conductivity_section(self, ax_array, gridded_variables, plot_settings, panel_settings,
                                   save_fig=False,  outfile=None):
        """

        :param gridded_variables:
        :param plot_settings:
        :param panel_settings:
        :param save_fig:
        :param outfile:
        :return:
        """

        # Find the hlen and vlen
        hlen =  gridded_variables['grid_distances'][-1]
        vlen =  gridded_variables['grid_elevations'][0] -  gridded_variables['grid_elevations'][-1]


        # Dictionary for defining axis positions
        ax_pos = {}

        # Unpack the panel settings

        variables = self.unpack_plot_settings(panel_settings,
                                                 'variable')
        panel_kwargs = self.unpack_plot_settings(panel_settings,
                                                 'panel_kwargs')
        plot_type = self.unpack_plot_settings(panel_settings,
                                                 'plot_type')


        # Iterate through the axes and plot
        for i, ax in enumerate(ax_array):

            if 'title' in panel_kwargs:
                ax.set_title(panel_kwargs['title'])
            else:
                ax.set_title(' '.join([variables[i].replace('_', ' '), 'plot']))

            if plot_type[i] == 'grid':

                # PLot the grid
                self.plot_grid(ax, gridded_variables, variables[i], panel_kwargs[i])
                ax_pos[i] = ax.get_position()

            elif plot_type[i] == 'multi_line':

                self.plot_multilines_data(ax, gridded_variables, variables[i], panel_kwargs[i])
                ax_pos[i] = ax.get_position()

            elif plot_type[i] == 'line':
                self.plot_single_line(ax, gridded_variables, variables[i], panel_kwargs[i])
                ax_pos[i] = ax.get_position()

        # Now iterate through all of the axes and move them to align with the minimum
        # right hand margin seen for all axes

        x0 = np.min([x.x0 for x in ax_pos.values()])
        ax_width = np.min([x.width for x in ax_pos.values()])

        for i, ax in enumerate(ax_array):
            ax.set_position([x0, ax_pos[i].y0, ax_width, ax_pos[i].height])

        # Add axis with northing at the bottom of the plot

        self.add_axis_coords('northing', gridded_variables['northing'], ax_array[-1], ax_pos[i], offset=-0.2)

        self.add_axis_coords('easting', gridded_variables['easting'], ax_array[-1], ax_pos[i], offset=-0.45)

        if save_fig:
            # If the dpi is set extract it from the plot settings
            if 'dpi' in plot_settings:
                dpi = plot_settings['dpi']
            else:
                dpi= 300
            plt.savefig(outfile, dpi=dpi)
            plt.close()


    def plot_conductivity_section_from_hdf5file(self, ax_arry, path, plot_settings, panel_settings, save_fig = False,
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

        plot_vars = self.unpack_plot_settings(panel_settings, 'variable')

        gridded_variables = self.extract_hdf5_data(f, plot_vars)

        self.plot_conductivity_section(ax_array, gridded_variables, plot_settings, panel_settings,
                                   save_fig=save_fig, outfile=outfile)

    def add_sticks(self, ax, df, gridded_variables, plot_variable, xy_columns, cmap='plasma_r',
                   colour_stretch=[0, 0.2], max_distance=200., stick_thickness=150.):

            # Get the coordinates of the section
            utm_coords = np.hstack((gridded_variables['easting'].reshape([-1, 1]),
                                    gridded_variables['northing'].reshape([-1, 1])))

            # Find the nearest neighbours within the maximum distance
            d, i = spatial_functions.nearest_neighbour(df[xy_columns].values,
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

    def add_custom_colourbar(self, ax, cmap, vmin, vmax, xlabel):
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

