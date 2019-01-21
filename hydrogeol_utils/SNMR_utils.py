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

These are functions used to process SNMR data from  spatialite database
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def choose_snmr_site_acquisition(df_acquisitions,
                                 pulse_sequence_criteria,
                                 pulse_length_criteria="max"):
    """

    :param df_acquisitions: dataframe extracted from database acquisition
    :param pulse_sequence_criteria: ordered list with most to least preferred
    eg ['FID', 'CPMG', 'T1']
    :param pulse_length_criteria: 'min' or 'max' depending on if the larger
    or smaller pulse length is prioritised
    :return:
    """

    # Our acquisition ids are appended to an empty list
    acqu_ids = []

    # Iterate through the site ids and pick sites based on our
    for item in df_acquisitions.site_id.unique():

        # Our first criterion is to find the rows with our given site id
        criterion1 = df_acquisitions.site_id.map(lambda x: x == item)

        # If there is only one acquisition we will use it regardless of how it was acquired
        if len(df_acquisitions[criterion1]) == 1:

            acqu_ids.append(int(df_acquisitions[df_acquisitions.site_id == item].index.values))

            # If there are more than one acuqisitions at the site we revert to other criterios
        elif len(df_acquisitions[criterion1]) > 1:

            # Apply the pulse sequence criteria
            for sequ in pulse_sequence_criteria:

                criterion2 = df_acquisitions[criterion1]['pulse_sequence'].map(lambda x: x == sequ)
                # If we find of our priority acquisition then we break the loop
                if criterion2.any():
                    break

            # Apply pulse sequence criteria
            # Extract all pulse widths
            pulse_lengths = df_acquisitions[criterion1][criterion2]['pulse_length'].unique()

            if pulse_length_criteria == 'min':
                criterion3 = df_acquisitions[criterion1][criterion2]['pulse_length'].map(
                    lambda x: x == np.max(pulse_lengths))
            elif pulse_length_criteria == 'max':
                criterion3 = df_acquisitions[criterion1][criterion2]['pulse_length'].map(
                    lambda x: x == np.max(pulse_lengths))

            # Append the index to the acquisition ids list

            acqu_ids.append(int(df_acquisitions[criterion1][criterion2][criterion3].index.values))
        # If there are no acquisitions then pass
        elif len(df_acquisitions[criterion1]) == 0:
            pass

    # Return the indices
    return acqu_ids

# This function extracts the K profile using the SDR equation

def sdr_k_profile(df, N=1, C=4000):
    """
    Function to calculate hydraulic conductivity (K) using the
    Schlumberger-Doll equation


    :param df: dataframe with SNMR inversion data from database.
     Must contain the 'Total water content' and 'T2*' columns
    :param N:  empirical exponent for water content when
    estimating the water content
    :param C: empirical constant for estimating water content
    :return:
    hydraulic conductivity profile as array
    """

    return C * np.power(df['Total_water_content'].values,
                        N) * np.square(df['T2*'].values)


def tc_k_profile(mobile_water_content, total_water_content,
                 N=2, C=4000):
    """
    Function for calculating hydraulic conductivity using the TC equation

    :param mobile_water_content: flat array with mobile water content
    values
    :param total_water_content: flat array with total water content
    values
    :param N: empirical exponent for water content when estimating
     the water content
    :param C: empirical constant for estimating water content
    :return:
    hydraulic conductivity profile as array
    """

    k = C * np.power(total_water_content, N) * np.divide(
        np.power(total_water_content, N),
        np.subtract(np.power(total_water_content, N),
                    np.power(mobile_water_content, N)))
    return k

def extract_snmr_inversions(acquisition_ids, connection, mask_below_doi=True):

    """
    A function for exrtracting SNMR inversions from the SNMR database
    using acquisitions ids within a spatial query

    :param acquisition_ids: primary key indices for the acquisition table
    :param connection:  sql database connection
    :param mask_below_doi: boolean, whether or not to remove all values below the doi
    :return:
    """

    # Create a SQL query
    placeholder = '?'  # For SQLite. See DBAPI paramstyle.
    placeholders = ', '.join(placeholder * len(acquisition_ids))
    query = 'SELECT * FROM inverse_models WHERE acquisition_id IN (%s)' % placeholders
    df_inversions = pd.read_sql_query(query, connection, index_col='table_id',
                                      params=tuple(acquisition_ids))

    # Return this dataframe unless the user want to mask below the depth of investigation
    if not mask_below_doi:
        return df_inversions
    # Otherwise we create a mask based on the doi in the invere_model_metadata table
    else:
        query = 'SELECT * FROM inverse_model_metadata WHERE acquisition_id IN (%s)' % placeholders
        df_doi = pd.read_sql_query(query, connection, index_col='table_id',
                                   params=tuple(acquisition_ids))
        # Note that since there is an issue with the inversion _id column in the inverison
        # table we are using the acquisition _id column
        df_merged = df_inversions.merge(df_doi, on='acquisition_id')

        # Now purge all rows where dethe depth from column is greater than
        # the doi
        condition = np.where(df_merged['Depth_from'] <= df_merged['Depth_of_Investigation'])

        return df_inversions.iloc[condition]

def plot_profile(ax, df, doi= None):
    """
    Function for plotting SNMR profiles similarly to the GMR inversion
    software. This function allows customised plots and importantly
    the ability to include the doi.

    :param ax: matplotlib axis
    :param df: individual inversion dataframe
    :param doi: depth of investigation
    :return:
    matplotlib axis with profile plotted
    """


    # invert the y axix so that depth in a positive number
    #plt.gca().invert_yaxis()

    # define plot data using pandas series names
    y = df['Depth_from'].values
    Total = df['Total_water_content'].values * 100
    Mobile = df['Mobile_water_content'].values * 100

    # set the range on the x axis so all the plots are the same scale
    ax.set_xlim([0, np.max(Total) + 5])

    # Plot the data
    ax.fill_betweenx(y, 0, Total,
                     label='Bound H2O', facecolor='pink')
    ax.fill_betweenx(y, 0, Mobile,
                     label='Mobile H2O', facecolor='blue')
    # plot lines
    ax.plot(Total, y, 'k-', linewidth=0.5)
    ax.plot(Mobile, y, 'k-', linewidth=0.5)

    # make legend
    ax.legend(fontsize=6)

    # define axis names
    ax.set_ylabel('Depth (upper) m')
    ax.set_xlabel('Water content %')

    # Add the depth of investigation if provided

    if doi is not None:

        ax.hlines(doi, 0, np.max(Total) + 5,
                  color='green', linestyles='dotted')
        ax.text(np.max(Total) - 5, (doi - 1),
                'Depth of investigation', fontsize=6)

    return ax
