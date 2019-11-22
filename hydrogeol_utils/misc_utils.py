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
Created on 4/6/2019
@author: Neil Symington

Miscellaneous, often used functions
'''

import numpy as np

def search_dictionary(kword_dict, word):
    """
    A search function for finding if a keword
    exists as a list element within a dictionary entry

    @param kword_dict: dictionary with keywords
    @word: string
    returns
    the key for the entry if it exists or None if not
    """
    word = word.lower()
    for k in kword_dict:
        for v in kword_dict[k]:
            if word in v:
                return k
    return None

def return_floats(string):
    try:
        return [float(x) for x in string.split()]
    except ValueError:
        return None


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def RepresentsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def block_to_array(file):
    """
    Function for return blocks of floats from text files to a numpy array
    :param file:
    :return:
    """
    line = next(file)
    L = []
    while return_floats(line) is not None:
        L.append(return_floats(line))
        line = next(file)
    return np.array(L)

def write_vrt_from_csv(infile, x, y, z):
    """
    A function for writing a vrt file for a csv point dataset

    """
    assert infile.endswith('.csv')
    s = "<OGRVRTDataSource>\n"
    s+='    <OGRVRTLayer name="{}">\n'.format(infile.split('.')[0])
    s+='        <SrcDataSource>'
    s+= infile
    s+= '</SrcDataSource>\n'
    s+= '        <GeometryType>wkbPoint</GeometryType>\n'
    s+= '        <GeometryField separator="," encoding="PointFromColumns" x="{}" y="{}" z="{}"/>\n'.format(x,y,z)
    s+= '        </OGRVRTLayer>\n'
    s+= '    </OGRVRTDataSource>'
    vrt_path = infile.replace('csv', 'vrt')
    with open(vrt_path, 'w') as f:
        f.write(s)
    return vrt_path