#hydrogeol_utils
A github repository containing tools for the processing, integration, interpretation and visualisation of important high performance hydrogeological data. Much of this work focusses on using Airborne EM data with borehole and surface NMR. AEM data used here is stored in netCDF files (see https://github.com/GeoscienceAustralia/geophys_utils for this format) and acessed using utilities from geophys_utils. Vector datasets are stored in spatialite databsaes and accessed using sql or swql based utility functions. Raster data are accessed using the rasterio package.

The master branch is the most well testest code while Neil_dev is really more of a working space. There are some gems in there but also a lot of poorly commented garbage that may not work even for me.

#Installation
My recommendation for installation is to use the anaconda package manager (https://www.anaconda.com/distribution/) to create a virtual environment as this package has a lot of dependencies which can be difficult to manage. My preferred approach is to use the anaconda command line. Here are some anaconda commands that will create an environment called hydrogeol_utils and install the dependencies.
'''
conda create -n hydrogeol_utils
conda activate hydrogeol_utils
conda install netCDF4 numpy h5py matplotlib pandas gdal scipy shapely numexpr owslib 
conda install rasterio pyyaml scikit-image cartopy sqlalchemy
pip install lasio
'''
Now we clone hydrogeol_utils and geophys_utils. This should be done into a sensible local folder. The path should be added to the anaconda environment automatically.
'''
git clone https://github.com/GeoscienceAustralia/geophys_utils.git
cd geophys_utils/
pip install -e .
cd ..
git clone https://github.com/GeoscienceAustralia/hydrogeol_utils.git
cd hydrogeol_utils
git checkout neil-dev
pip install -e .
'''
Contact
If anything in here breaks or is unclear, please feel free to email me at neil.symington@ga.gov.au . I do my best to keep everything usable but the time I have to curate this is very limited.
