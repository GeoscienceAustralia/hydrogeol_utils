# hydrogeol_utils
A github repository containing tools for the processing, integration, interpretation and visualisation of important high performance hydrogeological data.

My recommendation for installation is to use the anaconda package manager (https://www.anaconda.com/distribution/) to create a virtual environment as this package has a lot of dependencies which can be difficult to manage. My preferred approach is to use the anaconda command line. Here are some anaconda commands that will create an environment called hydrogeol_utils and install the key packages. This assumes git is installed and that you are starting from a directory into which you are going to save local copies of github repositories

'''
conda create -n hydrogeol_utils
conda activate hydrogeol_utils
conda install netCDF4 numpy h5py matplotlib pandas gdal scipy shapely numexpr owslib rasterio pyyaml scikit-image
pip install lasio
git clone https://github.com/GeoscienceAustralia/geophys_utils.git
cd geophys_utils/
pip install -e .
cd ..
git clone git@github.com:GeoscienceAustralia/hydrogeol_utils.git
cd hydrogeol_utils
git checkout neil-dev
pip install -e .
'''
