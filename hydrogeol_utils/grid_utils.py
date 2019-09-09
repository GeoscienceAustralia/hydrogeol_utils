#!/usr/bin/env python

#Copyright (c) 2014,
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.

#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.

#* Neither the name of flopy nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on 19/8/2019
@author: Neil Symington

Gridding classes and functions.

This is mostly adapted from the grid base class
from flopy (https://github.com/modflowpy/flopy). The licence above is from
flopy (https://github.com/modflowpy/flopy/blob/develop/LICENSE)
'''
import numpy as np
import copy

# From https://github.com/modflowpy/flopy/blob/develop/flopy/utils/geometry.py
def rotate(x, y, xoff, yoff, angrot_radians):
    """
    Given x and y array-like values calculate the rotation about an
    arbitrary origin and then return the rotated coordinates.

    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    xrot = xoff + np.cos(angrot_radians) * \
           (x - xoff) - np.sin(angrot_radians) * \
           (y - yoff)
    yrot = yoff + np.sin(angrot_radians) * \
           (x - xoff) + np.cos(angrot_radians) * \
           (y - yoff)

    return xrot, yrot

class CachedData(object):
    def __init__(self, data):
        self._data = data
        self.out_of_date = False

    @property
    def data_nocopy(self):
        return self._data

    @property
    def data(self):
        return copy.deepcopy(self._data)

    def update_data(self, data):
        self._data = data
        self.out_of_date = False

class Grid(object):
    """
    Base class for a structured or unstructured model grid

    Parameters
    ----------
    grid_type : enumeration
        type of model grid ('structured', 'vertex_layered',
        'vertex_unlayered')
    top : ndarray(np.float)
        top elevations of cells in topmost layer
    botm : ndarray(np.float)
        bottom elevations of all cells
    idomain : ndarray(np.int)
        ibound/idomain value for each cell
    lenuni : ndarray(np.int)
        model length units
    origin_loc : str
        Corner of the model grid that is the model origin
        'ul' (upper left corner) or 'll' (lower left corner)
    origin_x : float
        x coordinate of the origin point (lower left corner of model grid)
        in the spatial reference coordinate system
    origin_y : float
        y coordinate of the origin point (lower left corner of model grid)
        in the spatial reference coordinate system
    rotation : float
        rotation angle of model grid, as it is rotated around the origin point

    Properties
    ----------
    grid_type : enumeration
        type of model grid ('structured', 'vertex_layered',
        'vertex_unlayered')
    top : ndarray(np.float)
        top elevations of cells in topmost layer
    botm : ndarray(np.float)
        bottom elevations of all cells
    idomain : ndarray(np.int)
        ibound/idomain value for each cell
    proj4 : proj4 SpatialReference
        spatial reference locates the grid in a coordinate system
    epsg : epsg SpatialReference
        spatial reference locates the grid in a coordinate system
    lenuni : int
        modflow lenuni parameter
    origin_x : float
        x coordinate of the origin point in the spatial reference coordinate
        system
    origin_y : float
        y coordinate of the origin point in the spatial reference coordinate
        system
    rotation : float
        rotation angle of model grid, as it is rotated around the origin point
    xgrid : ndarray
        returns numpy meshgrid of x edges in reference frame defined by
        point_type
    ygrid : ndarray
        returns numpy meshgrid of y edges in reference frame defined by
        point_type
    zgrid : ndarray
        returns numpy meshgrid of z edges in reference frame defined by
        point_type
    xcenters : ndarray
        returns x coordinate of cell centers
    ycenters : ndarray
        returns y coordinate of cell centers
    ycenters : ndarray
        returns z coordinate of cell centers
    xyzgrid : [ndarray, ndarray, ndarray]
        returns the location of grid edges of all model cells. if the model
        grid contains spatial reference information, the grid edges are in the
        coordinate system provided by the spatial reference information.
        returns a list of three ndarrays for the x, y, and z coordinates
    xyzcellcenters : [ndarray, ndarray, ndarray]
        returns the cell centers of all model cells in the model grid.  if
        the model grid contains spatial reference information, the cell centers
        are in the coordinate system provided by the spatial reference
        information. otherwise the cell centers are based on a 0,0 location
        for the upper left corner of the model grid. returns a list of three
        ndarrays for the x, y, and z coordinates

    Methods
    ----------
    get_coords(x, y)
        transform point or array of points x, y from model coordinates to
        spatial coordinates
    grid_lines : (point_type=PointType.spatialxyz) : list
        returns the model grid lines in a list.  each line is returned as a
        list containing two tuples in the format [(x1,y1), (x2,y2)] where
        x1,y1 and x2,y2 are the endpoints of the line.
    xyvertices : (point_type) : ndarray
        1D array of x and y coordinates of cell vertices for whole grid
        (single layer) in C-style (row-major) order
        (same as np.ravel())
    intersect(x, y, local)
        returns the row and column of the grid that the x, y point is in

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """
    def __init__(self, grid_type=None, top=None, botm=None, idomain=None,
                 lenuni=2, epsg=None, proj4=None, prj=None, xoff=0.0,
                  yoff=0.0, ArithmeticErrorangrot=0.0,angrot=0.0):
        lenunits = {0: "undefined", 1: "feet", 2: "meters", 3: "centimeters"}
        LENUNI = {"u": 0, "f": 1, "m": 2, "c": 3}
        self.use_ref_coords = True
        self._grid_type = grid_type
        self._top = top
        self._botm = botm
        self._idomain = idomain

        if lenuni is None:
            lenuni = 0
        elif isinstance(lenuni, str):
            lenuni = LENUNI[lenuni.lower()[0]]
        self._lenuni = lenuni

        self._units = lenunits[self._lenuni]
        self._epsg = epsg
        self._proj4 = proj4
        self._prj = prj
        self._xoff = xoff
        self._yoff = yoff
        if angrot is None:
            angrot = 0.0
        self._angrot = angrot
        self._cache_dict = {}
        self._copy_cache = True

    ###################################
    # access to basic grid properties
    ###################################
    def __repr__(self):
        s = "xll:{0:<.10G}; yll:{1:<.10G}; rotation:{2:<G}; ". \
            format(self.xoffset, self.yoffset, self.angrot)
        s += "proj4_str:{0}; ".format(self.proj4)
        s += "units:{0}; ".format(self.units)
        s += "lenuni:{0}; ".format(self.lenuni)
        return s

    @property
    def grid_type(self):
        return self._grid_type

    @property
    def xoffset(self):
        return self._xoff

    @property
    def yoffset(self):
        return self._yoff

    @property
    def angrot(self):
        return self._angrot

    @property
    def angrot_radians(self):
        return self._angrot * np.pi / 180.

    @property
    def epsg(self):
        return self._epsg

    @epsg.setter
    def epsg(self, epsg):
        self._epsg = epsg

    @property
    def proj4(self):
        proj4 = None
        if self._proj4 is not None:
            if "epsg" in self._proj4.lower():
                if "init" not in self._proj4.lower():
                    proj4 = "+init=" + self._proj4
                else:
                    proj4 = self._proj4
                # set the epsg if proj4 specifies it
                tmp = [i for i in self._proj4.split() if
                       'epsg' in i.lower()]
                self._epsg = int(tmp[0].split(':')[1])
            else:
                proj4 = self._proj4
        elif self.epsg is not None:
            proj4 = '+init=epsg:{}'.format(self.epsg)
        return proj4

    @proj4.setter
    def proj4(self, proj4):
        self._proj4 = proj4

    @property
    def prj(self):
        return self._prj

    @prj.setter
    def prj(self, prj):
        self._proj4 = prj

    @property
    def top(self):
        return self._top

    @property
    def botm(self):
        return self._botm

    @property
    def top_botm(self):
        new_top = np.expand_dims(self._top, 0)
        return np.concatenate((new_top, self._botm), axis=0)

    @property
    def units(self):
        return self._units

    @property
    def lenuni(self):
        return self._lenuni

    @property
    def idomain(self):
        return self._idomain

    @property
    def shape(self):
        raise NotImplementedError(
            'must define extent in child '
            'class to use this base class')

    @property
    def extent(self):
        raise NotImplementedError(
            'must define extent in child '
            'class to use this base class')

    @property
    def grid_lines(self):
        raise NotImplementedError(
            'must define get_cellcenters in child '
            'class to use this base class')

    @property
    def xcellcenters(self):
        return self.xyzcellcenters[0]

    @property
    def ycellcenters(self):
        return self.xyzcellcenters[1]

    @property
    def zcellcenters(self):
        return self.xyzcellcenters[2]

    @property
    def xyzcellcenters(self):
        raise NotImplementedError(
            'must define get_cellcenters in child '
            'class to use this base class')

    @property
    def xvertices(self):
        return self.xyzvertices[0]

    @property
    def yvertices(self):
        return self.xyzvertices[1]

    @property
    def zvertices(self):
        return self.xyzvertices[2]

    @property
    def xyzvertices(self):
        raise NotImplementedError(
            'must define xyzgrid in child '
            'class to use this base class')


    def get_coords(self, x, y):
        """
        Given x and y array-like values, apply rotation, scale and offset,
        to convert them from model coordinates to real-world coordinates.
        """
        if isinstance(x, list):
            x = np.array(x)
            y = np.array(y)
        if not np.isscalar(x):
            x, y = x.copy(), y.copy()

        x += self._xoff
        y += self._yoff
        return rotate(x, y, self._xoff, self._yoff,
                               self.angrot_radians)

    def get_local_coords(self, x, y):
        """
        Given x and y array-like values, apply rotation, scale and offset,
        to convert them from real-world coordinates to model coordinates.
        """
        if isinstance(x, list):
            x = np.array(x)
            y = np.array(y)
        if not np.isscalar(x):
            x, y = x.copy(), y.copy()

        x, y = rotate(x, y, self._xoff, self._yoff,
                               -self.angrot_radians)
        x -= self._xoff
        y -= self._yoff

        return x, y

    def intersect(self, x, y, local=False, forgive=False):
        if not local:
            return self.get_local_coords(x, y)
        else:
            return x, y

    def set_coord_info(self, xoff=0.0, yoff=0.0, angrot=0.0, epsg=None,
                       proj4=None, merge_coord_info=True):
        if merge_coord_info:
            if xoff is None:
                xoff = self._xoff
            if yoff is None:
                yoff = self._yoff
            if angrot is None:
                angrot = self._angrot
            if epsg is None:
                epsg = self._epsg
            if proj4 is None:
                proj4 = self._proj4

        self._xoff = xoff
        self._yoff = yoff
        self._angrot = angrot
        self._epsg = epsg
        self._proj4 = proj4
        self._require_cache_updates()

    # Internal
    def _xul_to_xll(self, xul, angrot=None):
        yext = self.xyedges[1][0]
        if angrot is not None:
            return xul + (np.sin(angrot * np.pi / 180) * yext)
        else:
            return xul + (np.sin(self.angrot_radians) * yext)

    def _yul_to_yll(self, yul, angrot=None):
        yext = self.xyedges[1][0]
        if angrot is not None:
            return yul - (np.cos(angrot * np.pi / 180) * yext)
        else:
            return yul - (np.cos(self.angrot_radians) * yext)

    def _set_sr_coord_info(self, sr):
        self._xoff = sr.xll
        self._yoff = sr.yll
        self._angrot = sr.rotation
        self._epsg = sr.epsg
        self._proj4 = sr.proj4_str
        self._require_cache_updates()

    def _require_cache_updates(self):
        for cache_data in self._cache_dict.values():
            cache_data.out_of_date = True

    @property
    def _has_ref_coordinates(self):
        return self._xoff != 0.0 or self._yoff != 0.0 or self._angrot != 0.0

    def _load_settings(self, d):
        self._xoff = d.xul

    def _zcoords(self):
        if self.top is not None and self.botm is not None:
            zcenters = []
            top_3d = np.expand_dims(self.top, 0)
            zbdryelevs = np.concatenate((top_3d, self.botm), axis=0)

            for ix in range(1, len(zbdryelevs)):
                zcenters.append((zbdryelevs[ix - 1] + zbdryelevs[ix]) / 2.)
        else:
            zbdryelevs = None
            zcenters = None
        return zbdryelevs, zcenters

class StructuredGrid(Grid):
    """
    class for a structured model grid

    Parameters
    ----------
    delc
        delc array
    delr
        delr array

    Properties
    ----------
    nlay
        returns the number of model layers
    nrow
        returns the number of model rows
    ncol
        returns the number of model columns
    delc
        returns the delc array
    delr
        returns the delr array
    xyedges
        returns x-location points for the edges of the model grid and
        y-location points for the edges of the model grid

    Methods
    ----------
    get_cell_vertices(i, j)
        returns vertices for a single cell at row, column i, j.
    """
    def __init__(self, delc=None, delr=None, top=None, botm=None, idomain=None,
                 lenuni=2, epsg=None, proj4=None, prj=None, xoff=0.0,
                 yoff=0.0, angrot=0.0, nlay=None, nrow=None, ncol=None):
        super(StructuredGrid, self).__init__('structured', top, botm, idomain,
                                             lenuni, epsg, proj4, prj, xoff,
                                             yoff, angrot)
        self.__delc = delc
        self.__delr = delr
        if delc is not None:
            self.__nrow = len(delc)
        else:
            self.__nrow = nrow
        if delr is not None:
            self.__ncol = len(delr)
        else:
            self.__ncol = ncol
        if top is not None:
            assert self.__nrow * self.__ncol == len(np.ravel(top))
        if botm is not None:
            assert self.__nrow * self.__ncol == len(np.ravel(botm[0]))

            self.__nlay = len(botm)
        else:
            self.__nlay = nlay

    ####################
    # Properties
    ####################
    @property
    def nlay(self):
        return self.__nlay

    @property
    def nrow(self):
        return self.__nrow

    @property
    def ncol(self):
        return self.__ncol

    @property
    def shape(self):
        return self.__nlay, self.__nrow, self.__ncol

    @property
    def extent(self):
        self._copy_cache = False
        xyzgrid = self.xyzvertices
        self._copy_cache = True
        return (np.min(xyzgrid[0]), np.max(xyzgrid[0]),
                np.min(xyzgrid[1]), np.max(xyzgrid[1]))

    @property
    def delc(self):
        return self.__delc

    @property
    def delr(self):
        return self.__delr

    @property
    def xyzvertices(self):
        """
        """
        cache_index = 'xyzgrid'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            xedge = np.concatenate(([0.], np.add.accumulate(self.__delr)))
            length_y = np.add.reduce(self.__delc)
            yedge = np.concatenate(([length_y], length_y -
                                    np.add.accumulate(self.delc)))
            xgrid, ygrid = np.meshgrid(xedge, yedge)
            zgrid, zcenter = self._zcoords()
            if self._has_ref_coordinates:
                # transform x and y
                pass
            xgrid, ygrid = self.get_coords(xgrid, ygrid)
            if zgrid is not None:
                self._cache_dict[cache_index] = \
                    CachedData([xgrid, ygrid, zgrid])
            else:
                self._cache_dict[cache_index] = \
                    CachedData([xgrid, ygrid])

        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def xyedges(self):
        cache_index = 'xyedges'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            xedge = np.concatenate(([0.], np.add.accumulate(self.__delr)))
            length_y = np.add.reduce(self.__delc)
            yedge = np.concatenate(([length_y], length_y -
                                    np.add.accumulate(self.delc)))
            self._cache_dict[cache_index] = \
                CachedData([xedge, yedge])
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def xyzcellcenters(self):
        """
        Return a list of two numpy one-dimensional float array one with
        the cell center x coordinate and the other with the cell center y
        coordinate for every row in the grid in model space -
        not offset of rotated, with the cell center y coordinate.
        """
        cache_index = 'cellcenters'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            # get x centers
            x = np.add.accumulate(self.__delr) - 0.5 * self.delr
            # get y centers
            Ly = np.add.reduce(self.__delc)
            y = Ly - (np.add.accumulate(self.__delc) - 0.5 *
                      self.__delc)
            x_mesh, y_mesh = np.meshgrid(x, y)
            if self.__nlay is not None:
                # get z centers
                z = np.empty((self.__nlay, self.__nrow, self.__ncol))
                z[0, :, :] = (self._top[:, :] + self._botm[0, :, :]) / 2.
                for l in range(1, self.__nlay):
                    z[l, :, :] = (self._botm[l - 1, :, :] +
                                  self._botm[l, :, :]) / 2.
            else:
                z = None
            if self._has_ref_coordinates:
                # transform x and y
                x_mesh, y_mesh = self.get_coords(x_mesh, y_mesh)
            # store in cache
            self._cache_dict[cache_index] = CachedData([x_mesh, y_mesh, z])
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def grid_lines(self):
        """
            Get the grid lines as a list

        """
        # get edges initially in model coordinates
        use_ref_coords = self.use_ref_coords
        self.use_ref_coords = False
        xyedges = self.xyedges
        self.use_ref_coords = use_ref_coords

        xmin = xyedges[0][0]
        xmax = xyedges[0][-1]
        ymin = xyedges[1][-1]
        ymax = xyedges[1][0]
        lines = []
        # Vertical lines
        for j in range(self.ncol + 1):
            x0 = xyedges[0][j]
            x1 = x0
            y0 = ymin
            y1 = ymax
            lines.append([(x0, y0), (x1, y1)])

        # horizontal lines
        for i in range(self.nrow + 1):
            x0 = xmin
            x1 = xmax
            y0 = xyedges[1][i]
            y1 = y0
            lines.append([(x0, y0), (x1, y1)])

        if self._has_ref_coordinates:
            lines_trans = []
            for ln in lines:
                lines_trans.append([self.get_coords(*ln[0]),
                                    self.get_coords(*ln[1])])
            return lines_trans
        return lines

    ###############
    ### Methods ###
    ###############
    def intersect(self, x, y, local=False, forgive=False):
        """
        Get the row and column of a point with coordinates x and y

        When the point is on the edge of two cells, the cell with the lowest
        row or column is returned.

        Parameters
        ----------
        x : float
            The x-coordinate of the requested point
        y : float
            The y-coordinate of the requested point
        local: bool (optional)
            If True, x and y are in local coordinates (defaults to False)
        forgive: bool (optional)
            Forgive x,y arguments that fall outside the model grid and
            return NaNs instead (defaults to False - will throw exception)

        Returns
        -------
        row : int
            The row number
        col : int
            The column number

        """
        # transform x and y to local coordinates
        x, y = super(StructuredGrid, self).intersect(x, y, local, forgive)

        # get the cell edges in local coordinates
        xe, ye = self.xyedges

        xcomp = x > xe
        if np.all(xcomp) or not np.any(xcomp):
            if forgive:
                col = np.nan
            else:
                raise Exception(
                    'x, y point given is outside of the model area')
        else:
            col = np.where(xcomp)[0][-1]

        ycomp = y < ye
        if np.all(ycomp) or not np.any(ycomp):
            if forgive:
                row = np.nan
            else:
                raise Exception(
                    'x, y point given is outside of the model area')
        else:
            row = np.where(ycomp)[0][-1]
        if np.any(np.isnan([row, col])):
            row = col = np.nan
        return row, col

    def _cell_vert_list(self, i, j):
        """Get vertices for a single cell or sequence of i, j locations."""
        self._copy_cache = False
        pts = []
        xgrid, ygrid = self.xvertices, self.yvertices
        pts.append([xgrid[i, j], ygrid[i, j]])
        pts.append([xgrid[i + 1, j], ygrid[i + 1, j]])
        pts.append([xgrid[i + 1, j + 1], ygrid[i + 1, j + 1]])
        pts.append([xgrid[i, j + 1], ygrid[i, j + 1]])
        pts.append([xgrid[i, j], ygrid[i, j]])
        self._copy_cache = True
        if np.isscalar(i):
            return pts
        else:
            vrts = np.array(pts).transpose([2, 0, 1])
            return [v.tolist() for v in vrts]

    def get_cell_vertices(self, i, j):
        """
        Method to get a set of cell vertices for a single cell
            used in the Shapefile export utilities
        :param i: (int) cell row number
        :param j: (int) cell column number
        :return: list of x,y cell vertices
        """
        self._copy_cache = False
        cell_verts = [(self.xvertices[i, j], self.yvertices[i, j]),
                      (self.xvertices[i, j+1], self.yvertices[i, j+1]),
                      (self.xvertices[i+1, j+1], self.yvertices[i+1, j+1]),
                      (self.xvertices[i+1, j], self.yvertices[i+1, j]),]
        self._copy_cache = True
        return cell_verts


    # Importing
    @classmethod
    def from_gridspec(cls, gridspec_file, lenuni=0):
        f = open(gridspec_file, 'r')
        raw = f.readline().strip().split()
        nrow = int(raw[0])
        ncol = int(raw[1])
        raw = f.readline().strip().split()
        xul, yul, rot = float(raw[0]), float(raw[1]), float(raw[2])
        delr = []
        j = 0
        while j < ncol:
            raw = f.readline().strip().split()
            for r in raw:
                if '*' in r:
                    rraw = r.split('*')
                    for n in range(int(rraw[0])):
                        delr.append(float(rraw[1]))
                        j += 1
                else:
                    delr.append(float(r))
                    j += 1
        delc = []
        i = 0
        while i < nrow:
            raw = f.readline().strip().split()
            for r in raw:
                if '*' in r:
                    rraw = r.split('*')
                    for n in range(int(rraw[0])):
                        delc.append(float(rraw[1]))
                        i += 1
                else:
                    delc.append(float(r))
                    i += 1
        f.close()
        grd = cls(np.array(delc), np.array(delr), lenuni=lenuni)
        xll = grd._xul_to_xll(xul)
        yll = grd._yul_to_yll(yul)
        cls.set_coord_info(xoff=xll, yoff=yll, angrot=rot)
        return cls
