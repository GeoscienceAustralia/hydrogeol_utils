import numpy as np
import pandas as pd

infile = r"C:\Users\PCUser\Desktop\SSC_data\AEM_data\LCI\Altjere\Altejere_LCI_hdr.csv"

# Get the header from a file
fields = pd.read_csv(infile)
fields.columns

infile = r"C:\Users\PCUser\Desktop\SSC_data\AEM_data\LCI\Altjere\Altejere_LCI.dat"

a = np.genfromtxt(infile, dtype = None,
                                  names = list(fields.columns))

# Create a dataframe

df = pd.DataFrame(a)
# Create depth from fields

for i in range(1,31):
    df['Layer_top_depth_' + str(i)] = df['DTM_AHD'] - df['Layer_top_elev_' + str(i)]

outfile =  r"C:\Users\PCUser\Desktop\SSC_data\AEM_data\LCI\Altjere\Altejere_LCI_hdr.csv"

df.to_csv(outfile)
