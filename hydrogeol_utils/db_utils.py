import os
import shutil
import sqlite3
import pandas as pd

spatiallite_path = r'C:\Users\U19955\Desktop\mod_spatialite-4.3.0a-win-amd64'
os.environ['PATH'] = spatiallite_path + ';' + os.environ['PATH']


def makeCon(db_path, use_actual = False):
    '''
    Wrapper about standard connection to SpatiaLite database code
    
    The use_actual flag allows the user to connect to the original file if in place
    modificaitons are required, otherwise, to avoid the concurrent user limitations
    of sqlite databases, this code copies the spatialite database to a new file with
    a suffix of the users name, and then creates a connection to that database.
    
    This means that the user is always accessing the latest version of the database.
    
    Note: should be used inconjunction with the partner closeCon() function
    
    :param: db_path, the path to the spatialite database you wish to connect to
    
    :return:
    con, the database connection object
    '''
    if not use_actual:
        userid = os.getlogin()
        tmp_db_path = db_path.replace('.sqlite','_{}.sqlite'.format(userid))
        shutil.copyfile(db_path, tmp_db_path)
        con = sqlite3.connect(tmp_db_path)
    else:
        con = sqlite3.connect(db_path)   
    con.enable_load_extension(True)
    con.load_extension("mod_spatialite")
    cur = con.cursor()
    cur.execute("SELECT InitSpatialMetaData(1);")
    if not use_actual:
        print('Connected to {}\nTemporary working copy created.'.format(db_path))
    else:
        print('Connected to {}'.format(db_path))
    return con

def closeCon(con, db_path = None):
    '''
    Wrapper about standard close connection to SpatiaLite database code
    
    This code closes the connection, and then, if a path is specified, deletes the
    temporary copy of the database that was created as part of the makeCon() function.
    
    
    :param: con, the connection object wishing to be closed
    :param: db_path, the path to the spatialite database you are connected to
    
    :return:
    None
    '''
    con.close()
    if db_path:
        userid = os.getlogin()
        tmp_db_path = db_path.replace('.sqlite','_{}.sqlite'.format(userid))
        os.remove(tmp_db_path)
        print('Connection to {} is closed. Temporary working copy removed.'.format(db_path))
    else:
        print('Connection closed')
    return
    
def listTables(con, all = False):
    '''
    Simple wrapper to list all the tables in a given database.
    This has been setup to ignore the background tables in Spatialite.
    
    :param: con, a connection to a spatialite database
    :param: all, a flag to allow the user to get all the tables, or just the non-spatialite specific ones
    
    :return:
    A list of all the table names
    '''
    tables = [table[0] for table in con.cursor().execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    exclude_list = ['spatial_ref_sys',
                     'spatialite_history',
                     'sqlite_sequence',
                     'geometry_columns',
                     'spatial_ref_sys_aux',
                     'views_geometry_columns',
                     'virts_geometry_columns',
                     'geometry_columns_statistics',
                     'views_geometry_columns_statistics',
                     'virts_geometry_columns_statistics',
                     'geometry_columns_field_infos',
                     'views_geometry_columns_field_infos',
                     'virts_geometry_columns_field_infos',
                     'geometry_columns_time',
                     'geometry_columns_auth',
                     'views_geometry_columns_auth',
                     'virts_geometry_columns_auth',
                     'sql_statements_log',
                     'SpatialIndex',
                     'ElementaryGeometries']
    if not all:
        tables = [table for table in tables if table not in exclude_list]
    return tables

def listColumns(table_name, con):
    '''
    Returns a python list containing all the columns within a single spatialite table
    
    :param: table_name, the name of the table of interest
    :param: con, the connection to the spatialite database
    
    :return:
    A list of all the columns in the requested table
    '''
    if table_name not in listTables(con):
        raise NameError('"{}" not a valid table name for specified connection'.format(table_name))
    else:
        cols = [row[1] for row in con.cursor().execute("pragma table_info('{}')".format(table_name)).fetchall()]
        return cols
    
def describeTable(table_name, con):
    '''
    A function to extract table info from a spatialite database
    
    :param: table_name, the name of the table of interest
    :param: con, the connection to the spatialite database
    
    :return:
    A pandas DataFrame containing the details about each column
    '''
    if table_name not in listTables(con):
        raise NameError('"{}" not a valid table name for specified connection'.format(table_name))
    else:
        desc = pd.DataFrame(con.cursor().execute("pragma table_info('{}')".format(table_name)).fetchall())
        desc.columns = ['column_num', 'column_name', 'data_type', 'is_nullable','default_value','is_primary_key']
        desc = desc.set_index('column_num', drop = True)
        return desc