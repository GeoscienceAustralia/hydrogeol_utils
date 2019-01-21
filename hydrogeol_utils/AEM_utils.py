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

These are functions used to process AEM data from netcdf objects
'''

import numpy as np
from hydrogeol_utils import spatial_functions
import pandas as pd


# A function for getting the most representative conductivity profile given
# a set of distances, indices and corresponding AEM data
def extract_conductivity_profile(dataset, distances, indices,
                                 as_dataframe = False):
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
    idws = spatial_functions.inverse_distance_weights(distances, 1)

    # The result will be a log10 conductivity array as averaging in conductivity
    # is typically done in the log space
    log_cond = np.zeros(shape=
                        dataset['layer_top_depth'][:].shape[1],
                        dtype=np.float)

    # Iteratively extract the conductivity profiles using the indices,
    # min_key and aem_keys

    for i, ind in enumerate(indices):
        # Get the logconductivity proile
        log_cond_profile = np.log10(dataset.variables['conductivity'][ind])

        # Now multiply it by its corresponding weight and add it to the log ond array
        log_cond_profile += idws[i] * log_cond_profile

    # Now return the linear transform the conductivity array and reshape to a single column

    cond_profile = 10 ** log_cond_profile.reshape([-1, 1])


    # If specified in kwargs then return a dataframe
    if as_dataframe:
        # Now we need to create a depth from and depth too column
        depth_from = dataset.variables['layer_top_depth'][ind].reshape([-1, 1])

        depth_to = np.nan * np.ones(shape=depth_from.shape, dtype=np.float)
        depth_to[:-1, :] = depth_from[1::]
        # Round to 2 decimal places to correct for floating point precision errors
        depth_to = np.round(depth_to, 2)

        # Stack the arrays
        profile = np.hstack((depth_from, depth_to, cond_profile))
        # create and return a labelled dataframe
        return pd.DataFrame(columns=['Depth_from','Depth_to',
                                     'conductivity'],data=profile)
    # Otherwise return a numpy array
    else:
        return cond_profile
