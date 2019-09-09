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
from hydrogeol_utils import spatial_functions, misc_utils
import pandas as pd
import re, os
from geophys_utils._transect_utils import coords2distance
from geophys_utils._netcdf_point_utils import NetCDFPointUtils
from geophys_utils._netcdf_line_utils import NetCDFLineUtils
from collections import OrderedDict
from math import ceil, log10


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
    # Ensure that the indices and distances are arrays
    if not isinstance(distances ,(list, tuple, np.ndarray)):
        distances = np.array([distances])
    if not isinstance(indices ,(list, tuple, np.ndarray)):
        indices = np.array([indices])

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
def griddify_xyz(data, var = 'conductivity'):
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
                    dtype=data[var].dtype)

    # Now iterate through the conductivity and populate the array

    for i in range(nvcells):
        # Get the 2D array indices
        cond[i, :] = data[var][i * nhcells:(i + 1) * nhcells]

    # Replace the entry
    gridded_data[var] = cond

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
        gridded_data[var] = np.fliplr(gridded_data[var])
        gridded_data['easting'] = gridded_data['easting'][::-1]
        gridded_data['northing'] = gridded_data['northing'][::-1]

    if flip_grid:
        gridded_data[var] = np.flipud(gridded_data[var])
        gridded_data['grid_elevations'] = gridded_data['grid_elevations'][::-1]
    # Calculate the distance along the line

    utm_coordinates = np.hstack((gridded_data['easting'].reshape([-1, 1]),
                                 gridded_data['northing'].reshape([-1, 1])))

    gridded_data['grid_distances'] = coords2distance(utm_coordinates)

    # Estimate the ground elevation using the nan values

    elevation = np.zeros(shape=gridded_data['easting'].shape,
                         dtype=gridded_data['grid_elevations'].dtype)

    # Iterate through the cells and find the lowest elevation with data
    for i in range(gridded_data[var].shape[1]):
        try:
            idx = np.max(np.argwhere(np.isnan(gridded_data[var][:, i]))) + 1
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


def parse_wb_file(fname):
    """
    A function for parsing the workbench file
    :param fname: workbench file
    :return:
    """
    key_terms = ["INFO", "COORDINATE SYSTEM", "DATA TYPE", "NODE NAME(S)",
                 "DUMMY", "DATA UNIT", "NUMBER OF GATES"]

    # The data will be written into a dictionary
    inversion = {}

    # Open file
    with open(fname, 'r') as f:
        # Create string into which to add the header
        s = ''
        # Iterate through each line
        for line in f:
            # This character distinguishes header from data
            if line[0] == '/':
                # Flag header line
                header = True
            else:
                header = False
            # If it is a header, add the line to the string
            if header:

                s += line
            # Otherwise extract the data
            if not header:
                # Split the header on line breaks
                L = s.split('\n')
                # Assign the dictionary with the h
                inversion['header'] = L[-2][1:].strip().split()

                # Get data as structured array
                inversion['data'] = np.genfromtxt(f, dtype = None,
                                                  names = inversion['header'])

                break


    # Now we want to find the key word and add them as entries into the dictionary
    for item in key_terms:
        for i, entry in enumerate(L):
            if item in entry:
                # Add the following line to the dictionary
                inversion[item] = L[i + 1][1:].strip()

    # Add the array header to the dictionary
    inversion['header'] = L[-2][1:].strip().split()
    return inversion

# Functions for parsing the stm files used for ga_aem stmfiles
# See https://github.com/GeoscienceAustralia/ga-aem/tree/master/examples/bhmar-skytem/stmfiles for examples

# Class for extracting regular expressions from the stm files
class _RegExLib:
    """Set up regular expressions"""
    # use https://regexper.com to visualise these if required
    _reg_begin = re.compile(r'(.*) Begin\n')
    _reg_end = re.compile(r'(.*) End\n')
    _reg_param = re.compile(r'(.*) = (.*)\n')

    __slots__ = ['begin', 'end', 'param']

    def __init__(self, line):
        # check whether line has a positive match with all of the regular expressions
        self.begin = self._reg_begin.match(line)
        self.end = self._reg_end.match(line)
        self.param = self._reg_param.match(line)



# Define the blocks from the stm files

blocks = {'Transmitter': ['NumberOfTurns', 'PeakCurrent', 'LoopArea',
                          'BaseFrequency', 'WaveformDigitisingFrequency',
                          'WaveFormCurrent'],
          'Receiver': ['NumberOfWindows', 'WindowWeightingScheme',
                       'WindowTimes', 'CutOffFrequency', 'Order'],

          'ForwardModelling': ['ModellingLoopRadius', 'OutputType',
                               'SaveDiagnosticFiles', 'XOutputScaling',
                               'YOutputScaling', 'ZOutputScaling',
                               'SecondaryFieldNormalisation',
                               'FrequenciesPerDecade',
                               'NumberOfAbsiccaInHankelTransformEvaluation']}
# Class for AEM systems

class AEM_System:

    def __init__(self, name, dual_moment=True):
        """

        :param name: string: system name
        :param dual_moment: boolean, is the system fual moment (i.e. syktem
        """

        self.name = name

        if dual_moment:
            self.LM = {'Transmitter': {}, 'Receiver': {}, 'ForwardModelling': {}}
            self.HM = {'Transmitter': {}, 'Receiver': {}, 'ForwardModelling': {}}

    def parse_stm_file(self, infile, moment):

        # Save the results into a dictionary

        parameters = {}
        # Extract file line by line
        with open(infile, 'r') as f:
            # Yield the lines from the file
            line = next(f)
            while line:
                reg_match = _RegExLib(line)

                if reg_match.begin:
                    key = reg_match.begin.group(1).strip()

                    if key == "WaveFormCurrent":
                        a = misc_utils.block_to_array(f)
                        parameters[key] = a

                    if key == "WindowTimes":
                        a = misc_utils.block_to_array(f)
                        parameters[key] = a

                if reg_match.param:
                    key = reg_match.param.group(1).strip()
                    val = reg_match.param.group(2).strip()

                    if misc_utils.RepresentsInt(val):
                        val = int(val)

                    elif misc_utils.RepresentsFloat(val):
                        val = float(val)

                    elif key == "CutOffFrequency":

                        val = np.array([int(x) for x in val.split()])

                    if not key.startswith(r'//'):
                        parameters[key] = val

                line = next(f, None)

        for item in blocks.keys():
            for entry in blocks[item]:
                if moment == "HM":
                    self.HM[item][entry] = parameters[entry]
                elif moment == "LM":
                    self.LM[item][entry] = parameters[entry]


# These functions assist with writing adhoc arrays and metadata into aseg-gdf
# compliant data and .dfn files
def write_record2dfn_file(dfn_file,
                          rt,
                          name,
                          aseg_gdf_format,
                          definition=None,
                          defn=None,
                          st='RECD'):
    '''
    Helper function to write line to .dfn file.
    @param dfn_file: output file for DEFN line
    @param rt: value for "RT=<rt>" portion of DEFN line, e.g. '' or 'PROJ'
    @param name: Name of DEFN
    @param aseg_gdf_format: ASEG-GDF output format, e.g. 'I5', 'D12.1' or 'A30'
    @param definition=None: Definition string
    @param defn=None: New value of DEFN number. Defaults to self.defn+1
    @param st: value for "RT=<rt>" portion of DEFN line. Default = 'RECD'

    @return line: output line
    '''

    line = 'DEFN {defn} ST={st},RT={rt}; {name}'.format(defn=defn,
                                                        st=st,
                                                        rt=rt,
                                                        name=name,
                                                        )

    if aseg_gdf_format:
        line +=  ': {aseg_gdf_format}'.format(aseg_gdf_format=aseg_gdf_format)

    if definition:
        line += ': ' + definition

    dfn_file.write(line + '\n')
    return line

def write_dfn_file(dfn_out_path, field_definitions):
    '''
    Helper function to output .dfn file
    '''
    def write_defns(dfn_file, field_definitions):
        """
        Helper function to write multiple DEFN lines
        """
        defn = 1 # reset DEFN number
        for field_definition in field_definitions:
            short_name = field_definition['short_name']

            optional_attribute_list = []

            units = field_definition.get('units')
            if units:
                optional_attribute_list.append('UNITS={units}'.format(units=units))

            fill_value = field_definition.get('fill_value')
            if fill_value is not None:
                optional_attribute_list.append('NULL=' + field_definition['python_format'].format(fill_value).strip())

            long_name = field_definition.get('long_name')
            if long_name:
                optional_attribute_list.append('NAME={long_name}'.format(long_name=long_name))

            if optional_attribute_list:
                definition = ' , '.join(optional_attribute_list)
            else:
                definition = None
            if defn == 1:
                rt = 'DATA'
            else:
                rt = ''
            write_record2dfn_file(dfn_file,
                                       rt=rt,
                                       name=short_name,
                                       aseg_gdf_format=field_definition['format'],
                                       definition=definition,
                                       defn=defn
                                       )
            defn+=1


        # Write 'END DEFN'
        write_record2dfn_file(dfn_file,
                                   rt='',
                                   name='END DEFN',
                                   aseg_gdf_format=''
                                   )

        return # End of function write_defns
    # Create, write and close .dat file

    dfn_file = open(dfn_out_path, 'w')
    dfn_file.write('DEFN   ST=RECD,RT=COMM;RT:A4;COMMENTS:A76\n') # TODO: Check this first line

    write_defns(dfn_file, field_definitions)

    dfn_file.close()

# Approximate maximum number of significant decimal figures for each signed datatype
SIG_FIGS = OrderedDict([('uint8', 4), # 128
                        ('uint16', 10), # 32768
                        ('uint32', 19), # 2147483648 - should be 9, but made 10 because int64 is unsupported
                        ('uint64', 30), # 9223372036854775808 - Not supported in netCDF3 or netCDF4-Classic
                        ('int8', 2), # 128
                        ('int16', 4), # 32768
                        ('int32', 10), # 2147483648 - should be 9, but made 10 because int64 is unsupported
                        ('int64', 19), # 9223372036854775808 - Not supported in netCDF3 or netCDF4-Classic
                        # https://en.wikipedia.org/wiki/Floating-point_arithmetic#IEEE_754:_floating_point_in_modern_computers
                        ('float32', 7), # 7.2
                        ('float64', 35) # 15.9 - should be 16, but made 35 to support unrealistic precision specifications
                        ]
                       )


ASEG_DTYPE_CODE_MAPPING = {'uint8': 'I',
                           'uint16': 'I',
                           'uint32': 'I',
                           'uint64': 'I',
                           'int8': 'I',
                           'int16': 'I',
                           'int32': 'I',
                           'int64': 'I',
                           'float32': 'F', # real in exponent form
                           'float64': 'F', # double precision real in exponent form
                           'str': 'A'
                           }

def decode_aseg_gdf_format(aseg_gdf_format):
    '''
    Function to decode ASEG-GDF format string
    @param aseg_gdf_format: ASEG-GDF format string

    @return columns: Number of columns (i.e. 1 for 1D data, or read from format string for 2D data)
    @return aseg_dtype_code: ASEG-GDF data type character, e.g. "F" or "I"
    @return width_specifier:  Width of field in number of characters read from format string
    @return decimal_places: Number of fractional digits read from format string
    '''
    if not aseg_gdf_format:
        raise BaseException('No ASEG-GDF format string to decode')

    match = re.match('(\d+)*(\w)(\d+)\.*(\d+)*', aseg_gdf_format)

    if not match:
        raise BaseException('Invalid ASEG-GDF format string {}'.format(aseg_gdf_format))

    columns = int(match.group(1)) if match.group(1) is not None else 1
    aseg_dtype_code = match.group(2).upper()
    width_specifier = int(match.group(3))
    decimal_places = int(match.group(4)) if match.group(4) is not None else 0

    logger.debug('aseg_gdf_format: {}, columns: {}, aseg_dtype_code: {}, width_specifier: {}, decimal_places: {}'.format(aseg_gdf_format,
                                                                                                                      columns,
                                                                                                                      aseg_dtype_code,
                                                                                                                      width_specifier,
                                                                                                                      decimal_places
                                                                                                                      )
                 )
    return columns, aseg_dtype_code, width_specifier, decimal_places

def aseg_gdf_format2dtype(aseg_gdf_format):
    '''
    Function to return Python data type string and other precision information from ASEG-GDF format string
    @param aseg_gdf_format: ASEG-GDF format string

    @return dtype: Data type string, e.g. int8 or float32
    @return columns: Number of columns (i.e. 1 for 1D data, or read from format string for 2D data)
    @return width_specifier:  Width of field in number of characters read from format string
    @return decimal_places: Number of fractional digits read from format string
    '''
    columns, aseg_dtype_code, width_specifier, decimal_places = decode_aseg_gdf_format(aseg_gdf_format)
    dtype = None # Initially unknown datatype

    # Determine type and size for required significant figures
    # Integer type - N.B: Only signed types available
    if aseg_dtype_code == 'I':
        assert not decimal_places, 'Integer format cannot be defined with fractional digits'
        for test_dtype, sig_figs in SIG_FIGS.items():
            if test_dtype.startswith('int') and sig_figs >= width_specifier:
                dtype = test_dtype
                break
        assert dtype, 'Invalid width_specifier of {}'.format(width_specifier)

    # Floating point type - use approximate sig. figs. to determine length
    #TODO: Remove 'A' after string field handling has been properly implemented
    elif aseg_dtype_code in ['D', 'E', 'F']: # Floating point
        for test_dtype, sig_figs in SIG_FIGS.items():
            if test_dtype.startswith('float') and sig_figs >= width_specifier-2: # Allow for sign and decimal place
                dtype = test_dtype
                break
        assert dtype, 'Invalid floating point format of {}.{}'.format(width_specifier, decimal_places)

    elif aseg_dtype_code == 'A':
        assert not decimal_places, 'String format cannot be defined with fractional digits'
        dtype = '<U{}'.format(width_specifier) # Numpy fixed-length string type

    else:
        raise BaseException('Unhandled ASEG-GDF dtype code {}'.format(aseg_dtype_code))

    logger.debug('aseg_dtype_code: {}, columns: {}, width_specifier: {}, decimal_places: {}'.format(dtype,
                                                                                                 columns,
                                                                                                 width_specifier,
                                                                                                 decimal_places
                                                                                                 )
                 )
    return dtype, columns, width_specifier, decimal_places


def variable2aseg_gdf_format(array_variable, decimal_places=None):
    '''
    Function to return ASEG-GDF format string and other info from data array or netCDF array variable
    @param array_variable: data array or netCDF array variable
    @param decimal_places: Number of decimal places to respect, or None for value derived from datatype and values

    @return aseg_gdf_format: ASEG-GDF format string
    @return dtype: Data type string, e.g. int8 or float32
    @return columns: Number of columns (i.e. 1 for 1D data, or second dimension size for 2D data)
    @return width_specifier: Width of field in number of characters
    @return decimal_places: Number of fractional digits (derived from datatype sig. figs - width_specifier)
    @param python_format: Python Formatter string for fixed-width output
    '''
    if len(array_variable.shape) <= 1: # 1D variable or scalar
        columns = 1
    elif len(array_variable.shape) == 2: # 2D variable
        columns = array_variable.shape[1]
    else:
        raise BaseException('Unable to handle arrays with dimensionality > 2')

    data_array = array_variable[:]

    # Try to determine the dtype string from the variable and data_array type

    if not len(array_variable.shape): # Scalar
        dtype = type(data_array).__name__
        if dtype == 'str':
            width_specifier = len(data_array) + 1
            decimal_places = 0
        elif dtype == 'ndarray': # Single-element array
            dtype = str(array_variable.dtype)
            data = np.asscalar(data_array)

            sig_figs = SIG_FIGS[dtype] + 1 # Look up approximate significant figures and add 1
            sign_width = 1 if data < 0 else 0
            integer_digits = ceil(log10(np.abs(data) + 1.0))
    else: # Array
        dtype = str(array_variable.dtype)
        print(dtype)
        if dtype in ['str', "<class 'str'>"]: # String array or string array variable
            dtype = 'str'
            width_specifier = max([len(string.strip()) for string in data_array]) + 1
            decimal_places = 0

        else:  # Numeric datatype array
            # Include fill value if required
            if type(data_array) == np.ma.core.MaskedArray:
                logger.debug('Array is masked. Including fill value.')
                data_array = data_array.data

            sig_figs = SIG_FIGS[dtype] + 1 # Look up approximate significant figures and add 1
            sign_width = 1 if np.nanmin(data_array) < 0 else 0
            integer_digits = ceil(log10(np.nanmax(np.abs(data_array)) + 1.0))

    aseg_dtype_code = ASEG_DTYPE_CODE_MAPPING.get(dtype)
    # Here we decide if we want to use exponential or floating point form
    # Rather ad hoc
    if aseg_dtype_code == 'F':
        print(np.abs(np.nanmin(data_array)))
        print(np.log10(np.abs(np.nanmax(data_array)/np.nanmin(data_array))))
        ### TODO fix up the errors that this si throwing

        if np.abs(np.nanmin(data_array)) < 10**-2: # small numbers
            if dtype  == 'float64':
                aseg_dtype_code = 'D'
            elif dtype  == 'float32':
                aseg_dtype = 'E'
        elif np.log10(np.abs(np.nanmax(data_array)/np.nanmin(data_array))) > 2.5:
            if dtype  == 'float64':
                aseg_dtype_code = 'D'
            elif dtype  == 'float32':
                aseg_dtype = 'E'
        print(aseg_dtype_code)
    assert aseg_dtype_code, 'Unhandled dtype {}'.format(dtype)

    if aseg_dtype_code == 'I': # Integer
        decimal_places = 0
        width_specifier = integer_digits + sign_width + 1
        aseg_gdf_format = 'I{}'.format(width_specifier)
        python_format = '{' + ':>{:d}.{:d}f'.format(width_specifier, decimal_places) + '}'

    elif aseg_dtype_code in ['F', 'D', 'E']: # Floating point
        # If array_variable is a netCDF variable with a "format" attribute, use stored format string to determine decimal_places
        if decimal_places is not None:
            decimal_places = min(decimal_places, abs(sig_figs-integer_digits))

        elif hasattr(array_variable, 'aseg_gdf_format'):
            _columns, _aseg_dtype_code, _integer_digits, decimal_places = decode_aseg_gdf_format(array_variable.aseg_gdf_format)
            decimal_places = min(decimal_places, abs(sig_figs-integer_digits))
            logger.debug('decimal_places set to {} from variable attribute aseg_gdf_format {}'.format(decimal_places, array_variable.aseg_gdf_format))
        else: # No aseg_gdf_format variable attribute
            decimal_places = abs(sig_figs-integer_digits) # Allow for full precision of datatype
            logger.debug('decimal_places set to {} from sig_figs {} and integer_digits {}'.format(decimal_places, sig_figs, integer_digits))

        width_specifier = min(sign_width + integer_digits + decimal_places + 2,
                              sign_width + sig_figs + 2
                              )
        if aseg_dtype_code == 'D':
            width_specifier += 4

        aseg_gdf_format = '{}{}.{}'.format(aseg_dtype_code, width_specifier, decimal_places)
        if aseg_dtype_code == 'F': # Floating point notation
            python_format = '{' + ':>{:d}.{:d}f'.format(width_specifier, decimal_places) + '}' # Add 1 to width for decimal point
        else: # Exponential notation for 'D' or 'E'
            python_format = '{' + ':>{:d}.{:d}E'.format(width_specifier, decimal_places) + '}' # Add 1 to width for decimal point

    elif aseg_dtype_code == 'A': # String
        if hasattr(array_variable, 'aseg_gdf_format'):
            _columns, _aseg_dtype_code, width_specifier, decimal_places = decode_aseg_gdf_format(array_variable.aseg_gdf_format)
            aseg_gdf_format = array_variable.aseg_gdf_format
        else:
            aseg_gdf_format = 'A{}'.format(width_specifier)

        python_format = '{' + ':>{:d}s'.format(width_specifier) + '}'
    else:
        raise BaseException('Unhandled ASEG-GDF dtype code {}'.format(aseg_dtype_code))

    # Pre-pend column count to start of aseg_gdf_format
    if columns > 1:
        aseg_gdf_format = '{}{}'.format(columns, aseg_gdf_format)

    return aseg_gdf_format, dtype, columns, width_specifier, decimal_places, python_format
