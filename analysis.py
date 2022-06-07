from glob import glob

import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

import rasterio as rio

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import plotly.graph_objects as go

# define how floating point errors should be handled
np.seterr(divide='ignore', invalid='ignore')

###########################################################
#  READ DATA
###########################################################


arr_st = rio.open('C:\\Users\\marit_na7iraq\\Downloads\\samples\\samples\\images\\patches\\patches\\LC08_L1TP_173030_20200817_20200817_01_RT_p00450.tif').read()
arr_st.sort()

############################################################
#  Visualize Bands
############################################################

ep.plot_bands(arr_st, cmap='gist_earth', figsize=(20, 12), cols=6, cbar=False)

#############################################################
#  Create RGB Composite Image
#############################################################

rgb = ep.plot_rgb(arr_st, rgb=(3, 2, 1), figsize=(10, 16))
plt.show()

rgb2 = ep.plot_rgb(arr_st, rgb=(3, 2, 1), stretch=True, str_clip=0.2, figsize=(10, 16))
plt.show()

colors = ['tomato', 'navy', 'MediumSpringGreen', 'lightblue', 'orange', 'blue',
          'maroon', 'purple', 'yellow', 'olive', 'brown', 'cyan']

#############################################################
#  Histograms
#############################################################

ep.hist(arr_st,
        colors=colors,
        title=[f'Band-{i}' for i in range(1, arr_st.shape[0]+1)],
        cols=3,
        alpha=0.5,
        figsize=(12, 10))

plt.show()
