from hydrogeol_utils import AEM_utils
import pandas as pd
import numpy as np
import math
import yaml
import importlib


infile = r"C:\Users\PCUser\Desktop\NSC_data\data\AEM\HE\WB_Exported_Data\HE_raw_all.xyz"

EM_data = AEM_utils.parse_wb_file(infile)

# Find the aseg-gdf-data tyoe
np.unique(EM_data['data']['LINE_NO']).min()

EM_data['data'].dtype
# Pandas is far  better for manipulating structured data than numpy so create
# a pandas dataframe from EM_data

df = pd.DataFrame(EM_data['data'])

# Force all dbdt columns to 64-bit floats and remove nulls using the dummy value
# from the header

for c in df.columns:
    if c.startswith('DBDT') and df[c].dtype!=np.float64:
        print(c)
        df.at[:,c] = df[c].astype(np.float64)
# Replace dummy with nulls for floating point columns

df[df == np.float(EM_data['DUMMY'])] = np.nan


# We need to do some additional processing on this data in order to get it
# consistent

# sort by timestamp
df.drop_duplicates(subset = 'TIMESTAMP', inplace=True)
df.sort_values(by=['TIMESTAMP'], inplace=True)
df.reset_index(inplace = True)


# Check that the timestamp is increasing and that the moments are interleaved
for index, row in df.iterrows():
    if index != 0:

        if df.loc[index, 'CHANNEL_NO'] == df.loc[index - 1, 'CHANNEL_NO']:
             assert df.loc[index, 'LINE_NO'] != df.loc[index - 1, 'LINE_NO']


# Create low and high moment channel current columns
df['curr_lm'], df['curr_hm'] = df['CURRENT'].values, df['CURRENT'].values
# Assign nulls based on the indices
lm_inds, hm_inds = df[df['CHANNEL_NO'] == 1].index, df[df['CHANNEL_NO'] == 2].index
df.at[lm_inds, 'curr_hm'] = np.nan
df.at[hm_inds, 'curr_lm'] = np.nan


# So lets create a single dataframe which has both channels in a row by
# interpolating time, distance, elevation ad

new_cols = ['TIMESTAMP', 'LINE_NO', 'UTMX', 'UTMY', 'ELEVATION',  'TX_ALTITUDE',
            'TILT_X', 'TILT_Y', 'CHANNEL_NO', 'curr_lm', 'curr_hm']

for i in range(1,27):
    new_cols.append('DBDT_Ch1GT' + str(i))
    new_cols.append('DBDT_STD_Ch1GT' + str(i))
for i in range(1,39):
    new_cols.append('DBDT_Ch2GT' + str(i))
    new_cols.append('DBDT_STD_Ch2GT' + str(i))

# To do the interpolation we are going to split the data by lines

lines = df['LINE_NO'].unique()

line_dfs = {}

for item in lines:
    print(item)
    # query the dataframe by line and create a new dataframe
    df_lin = df[df['LINE_NO'] == item]
    df_lin = df_lin.loc[:,new_cols]
    # Check if odd length
    if len(df_lin)%2 != 0:
        line_dfs[item] = df_lin.iloc[:-1,:]
    else:
        line_dfs[item] = df_lin
averaged_df = []

for item in line_dfs:
    print(item)
    df_lin = line_dfs[item]
    df_ave = pd.DataFrame(columns = new_cols)
    for i in range(int(len(df_lin)/2)):
        df_averaged = df_lin.iloc[2*i:2*i+2,:].mean()
        df_ave = df_ave.append(df_averaged, ignore_index =True)
    averaged_df.append(df_ave)

df_interpolated = pd.concat(averaged_df)


df_interpolated.at[:,'LINE_NO'] = df_interpolated['LINE_NO'].astype(np.int32)

# Add the otehr important columns
df_interpolated['GA_project'] = np.int16(1294)
df_interpolated['Job_No'] = np.int16(10021)

# Add a made up fiducial
df_interpolated['fiducial'] = np.float64(1000000.00 + (df_interpolated.index)/2)

# Now we add a longitude and latitude column


df_interpolated.dtypes

from pyproj import Proj, transform

inProj = Proj(init='epsg:28352')
outProj = Proj(init='epsg:4283')
x1,y1 = df_interpolated['UTMX'].values, df_interpolated['UTMY'].values
x2,y2 = transform(inProj,outProj,x1,y1)
df_interpolated['lon'], df_interpolated['lat'] = np.float64(x2), np.float64(y2)

df_interpolated['lon'].max()

# Open yaml file

with open("hydrogeol_utils\\workbench_header.yml", 'r') as f:
    try:
        wb_header = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# Map the column names in df_interpolated to the standard names within the
# header file

col2hdr = {}

df_interpolated.columns

for item in df_interpolated.columns:
    if item.startswith('DBDT'):
        kword = item.split('GT')[0] + 'GT'
    else:
        kword = item
    if kword in wb_header['workbench_header_translation'].keys():
        col2hdr[kword] = wb_header['workbench_header_translation'][kword]
col2hdr

col2hdr.keys()
# Write a list for how many decimal places to give each variable (None for
# integers)
#df_interpolated[df_interpolated == -999.999] = np.nan
#df_copy[df_copy == -999.999] = np.nan

df_copy = df_interpolated.copy()
col2dp = {'GA_project': None, 'Job_No': None, 'fiducial': 2, 'TIMESTAMP': 7,
          'LINE_NO': None,'UTMX': 2, 'UTMY': 2,
          'lon':6, 'lat':6, 'ELEVATION':2, 'TX_ALTITUDE':2, 'TILT_X':2,
          'TILT_Y':2, 'curr_lm':2 , 'curr_hm':2, 'DBDT_Ch1GT':3,
           'DBDT_Ch2GT': 3, 'DBDT_STD_Ch1GT':3,
          'DBDT_STD_Ch2GT': 3}
#df_interpolated = df_copy.copy()
field_definitions = []

df_interpolated['LINE_NO'].unique().min()


for item in col2dp.keys():
    short_name = col2hdr[item].get('short_name')
    long_name = col2hdr[item].get('long_name')
    units = col2hdr[item].get('units')
    print(short_name)
    # Make allowances for multidimensional variables
    if item.startswith('DBDT'):
        prefix = item.split('GT')[0] + 'GT'
        cols = [x for x in df_interpolated.columns if x.startswith(prefix)]
    else:
        cols = item
   # Extract the array
    arr = df_interpolated.loc[:,cols].values
    dp = col2dp[item]

    # Returns aseg_gdf_format, dtype, columns, width_specifier,
    # decimal_places, python_format
    asg_fmt,dt,c,widspec,dp,pyfmt = AEM_utils.variable2aseg_gdf_format(arr,
                                                             decimal_places=dp)
    field_definition = {'variable_name': item,
                        'short_name': short_name,
                        'long_name': long_name,
                        'units': units,
                        'dtype': dt,
                        'columns': c,
                        'format': asg_fmt,
                        'width_specifier': widspec,
                        'decimal_places': dp,
                        'python_format': pyfmt
                        }
    # Add fill values fro the data columns
    if item.startswith('DBDT'):
        null = -9999.
        # Ensure compatibility with aseg-gdf output
        field_definition['fill_value'] = float(field_definition['python_format'].format(null).strip())

    field_definitions.append(field_definition)

field_definitions[6]

field_definitions[9].update({'python_format':'{:>11.2f}'})

field_definitions[10]
importlib.reload(AEM_utils)
dfn_outpath = r'C:\temp\EK_EM_workbench.dfn'
AEM_utils.write_dfn_file(dfn_outpath, field_definitions)

# Now we write the data into a file
field_definitions[10].update({'python_format': '{:>6.2f}'})
# Start by replacing nans with fill values
for field_definition in field_definitions:
    var_name = field_definition.get('variable_name')
    null_value = field_definition.get('fill_value')
    if field_definition.get('columns') > 1:
        for i in range(1, field_definition.get('columns')+1):
            key = var_name+str(i)
            if null_value is not None:
                df_interpolated[key].fillna(null_value, inplace=True)

# Finally we add our comment column which is 'DATA'

df_interpolated['COMM'] = 'DATA'

# Reorder the array to the same shape as the .dfn file

output_dt = [('COMM', '{:>4s}')]

for field_definition in field_definitions:
    var_name = field_definition.get('variable_name')
    if field_definition.get('columns') > 1:
        for i in range(1, field_definition.get('columns')+1):
            key = var_name+str(i)
            output_dt.append((key, field_definition.get('python_format')))
    else:
        key= var_name
        output_dt.append((key, field_definition.get('python_format')))

cols = [x[0] for x in output_dt]

df_output = pd.DataFrame(columns = cols, dtype = str)

for i, c in enumerate(output_dt):
    key = c[0]
    new_col = [c[1].format(x) for x in df_interpolated[key].values]
    df_output[key] = new_col

outfile = r'C:\temp\EK_EM_workbench_temp.dat'

# Note use a pipe so we can easily delete later
df_output.to_csv(outfile, sep = '|', index = False, header = False)


with open(outfile, 'r') as inf:
    s = inf.read()

new_s = s.replace('|','')


# Reomve the final
if new_s[-1:] == '\n':
    new_s = new_s[:-1]

new_outfile = r'C:\temp\EK_EM_workbench.dat'

with open(new_outfile, 'w') as f:
    f.write(new_s)
