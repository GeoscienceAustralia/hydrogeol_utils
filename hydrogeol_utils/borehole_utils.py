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
Created on 21/3/2019
@author: Neil Symington

These are functions used to process borehole data. These are mostly based on boreholes stored within a spatialite
database and processed using pandas dataframes or series
'''
import pandas as pd
from shapely import wkt

def extract_by_primary_key(df, primary_keys,
                           primary_key_column = 'borehole_id'):
    """
    
    :param df: pandas dataframe
    :param primary_keys: sequence of primary key values
    :param primary_key_column:
    :return:
    subset of data in pandas dataframe
    """

    # If the parameter input is an integer and not a list make it a list
    if isinstance(primary_keys, (int, float, str)):
        primary_keys = [primary_keys]

    # Create a mask based on the primary key
    mask = df[primary_key_column].isin(primary_keys)
    #Apply mask
    df_subset = df[mask]

    if len( df_subset) > 0:
        return  df_subset
    # If it is empty return none
    else:
        return None


# Now load the various datasets

def extract_sql_with_primary_key(table_name, columns, connection, primary_keys,
                                 primary_key_column = 'borehole_id', verbose = True):
    """
    A function for creating a sql query using table names, columns,
    and primary key values

    :param table_name: string: sql table name
    :param columns: sequence of column names
    :param connection: sql alchemy connection to database
    :param primary_keys: sequence of primary keys
    :param primary_key_column: column of primary key
    :return:
    pandas dataframe with extracted data
    """
    # Create a string of placeholders
    st_key = ','.join(str(x) for x in primary_keys)

    # Create query
    query = "select t."

    if columns == 'all':
        query+='*'
    else:
        cols = ", t.".join(columns)
        query += cols
    query += " from "
    query += table_name
    query += " t where t."
    query += primary_key_column
    query += " in ({});".format(st_key)

    if verbose:
        print(query)


    # Execute query
    return pd.read_sql(query, connection)

def extract_boreholes_within_geometry(table_name, connection, geometry, columns = 'all', verbose = True):
    """

    :param table_name: borehole tableanme from sql database
    :param connection: sql alchemy connection to database
    :param geometry: wkt of geometry with same coordinate reference sysetm
    :param columns: column name (defaults to all)
    :return:
    dataframe with subset of boreholes
    """
    # Check geometry is a string

    assert isinstance(geometry, str), "Check that geometry is in well known text"

    query = "Select b."

    if columns == "all":
        query += "* "
    # If columns are specified then iteratively add them to the sql query
    else:
        query += ', b.'.join(columns)
        query += ' '

    # add from statement
    query += "from "
    query += table_name
    query += " b "
    # add where statement and spatialite function
    query += " where within(b.geom,GeomFromText('{}'));"

    query = query.format(geometry)

    if verbose:
        print(query)

    # Extract as pandas dataframe
    df = pd.read_sql(query, connection)

    # Drop the useless bianry geometry and create a shapely object from the geomnetry
    # column

    df.drop(columns = ['geom'], inplace=True)

    df['geometry'] = [wkt.loads(x) for x in df['geometry']]


    return df


def extract_all_boredata_by_simple_query(con, primary_key, primary_key_column='borehole_id', tables='all'):
    """
    A function for extracting all the data in the boreholes spatialite data schema associated with a single borehole.

    :param con: The connection object to the borehole spatialite database
    :primary_key: The primary key the borehole of interest
    :primary_key_column: The name of the column used for the SQL query that generates the result
    :tables: A list of tables that should be queried to extract the borehole data from. Defaults to all

    :returns: A dict of dataframes, one for each table queried
    """

    def makeHeader_Query(bhid):
        return "select * from borehole where {} = {}".format(primary_key_column, bhid)

    def makeConstruction_Query(bhid):
        return "select * from borehole_construction where {} = {}".format(primary_key_column, bhid)

    def makeLithology_Query(bhid):
        return "select * from borehole_lithology where {} = {}".format(primary_key_column, bhid)

    def makeJavelin_Query(bhid):
        return "select * from boreholeNMR_data where {} = {}".format(primary_key_column, bhid)

    # The two hylogger queries are commented out as the tables don't currently exist in the borehole database
    # Double check the borehole_id column name (previously was Borehole_ENO) which would break this code
    # def makeHyLogChips_Query(bhid):
    #     return "select * from Hylogging_data_from_chips where {} = {}".format(primary_key_column, bhid)

    # def makeHyLogCore_Query(bhid):
    #     return "select * from Hylogging_data_from_core where {} = {}".format(primary_key_column, bhid)

    def makeIndGam_Query(bhid):
        return "select * from induction_gamma_data where {} = {}".format(primary_key_column, bhid)

    def makePoreFluid_Query(bhid):
        return "select * from pore_fluid_EC_pH where {} = {}".format(primary_key_column, bhid)

    def makeWaterLevel_Query(bhid):
        return "select * from standing_water_level where {} = {}".format(primary_key_column, bhid)

    def makeAEMConductivty_Query(bhid):
        return "select * from representative_AEM_bulk_conductivity where {} = {}".format(primary_key_column, bhid)

    def makeMagSus_Query(bhid):
        return "select * from magnetic_susceptibility where {} = {}".format(primary_key_column, bhid)

    data_gathering_functions = {'header': makeHeader_Query,
                                'construction': makeConstruction_Query,
                                'lithology': makeLithology_Query,
                                'javelin': makeJavelin_Query,
                                #                             'hylogchips' : makeHyLogChips_Query,
                                #                             'hylogcore' : makeHyLogCore_Query,
                                'indgam': makeIndGam_Query,
                                'porefluid': makePoreFluid_Query,
                                'waterlevels': makeWaterLevel_Query,
                                'aem': makeAEMConductivty_Query,
                                'magsus': makeMagSus_Query}
    if tables != 'all':
        # If a valid subset of tables is requested, update dict accordingly
        if set(tables).issubset(data_gathering_functions.keys()):
            data_gathering_functions = {tablename: data_gathering_functions[tablename] for tablename in tables}
        else:
            # If not, create a list of the invalid table names requested, format it nicely, and return an error
            bad_names = str([tablename for tablename in tables if tablename not in data_gathering_functions.keys()])[
                        1:-1]
            raise KeyError(
                'The following requested table names did not exactly match with table names in the requested database. {}'.format(
                    bad_names))

    # Loop through all the tables requested and if the length of the query result > 0, save the result.
    # This is essentially the same as checking the hasData flags in the header table, but saves some typing
    data = {}
    for name, function in data_gathering_functions.items():
        df = pd.read_sql_query(function(primary_key), con)
        if len(df) == 0:
            continue
        else:
            data[name] = df
    return data
