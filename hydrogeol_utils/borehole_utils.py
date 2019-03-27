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

    ### TODO fix columns bug

    return df