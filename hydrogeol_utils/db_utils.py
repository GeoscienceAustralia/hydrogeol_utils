import os
import shutil
import sqlite3

spatiallite_path = r'C:\Users\U19955\Desktop\mod_spatialite-4.3.0a-win-amd64'
os.environ['PATH'] = spatiallite_path + ';' + os.environ['PATH']


def makeCon(db_path):
    '''
    Wrapper about standard connection to SpatiaLite database code
    
    To avoid the concurrent user limitations of sqlite databases, this code copies the
    spatialite database to a new file with a suffix of the users name, and then creates
    a connection to that database.
    
    This means that the user is always accessing the latest version of the database.
    
    Note: should always be used inconjunction with the partner closeCon() function
    
    :param: db_path, the path to the spatialite database you wish to connect to
    
    :return:
    con, the database connection object
    '''
    userid = os.getlogin()
    tmp_db_path = db_path.replace('.sqlite','_{}.sqlite'.format(userid))
    shutil.copyfile(db_path, tmp_db_path)
    con = sqlite3.connect(tmp_db_path)
    con.enable_load_extension(True)
    con.load_extension("mod_spatialite")
    cur = con.cursor()
    cur.execute("SELECT InitSpatialMetaData(1);")
    print('Connected to {}. Temporary working copy created.'.format(db_path))
    return con

def closeCon(con, db_path):
    '''
    Wrapper about standard close connection to SpatiaLite database code
    
    This code closes the connection, and then deletes the temporary copy of the database
    that was created as part of the makeCon() funciton.
    
    
    Note: should always be used inconjunction with the partner makeCon() function
    
    :param: con, the connection object wishing to be closed
    :param: db_path, the path to the spatialite database you wish to connect to
    
    :return:
    None
    '''
    con.close()
    userid = os.getlogin()
    tmp_db_path = db_path.replace('.sqlite','_{}.sqlite'.format(userid))
    os.remove(tmp_db_path)
    print('Connection to {} is closed. Temporary working copy removed.'.format(db_path))
    return