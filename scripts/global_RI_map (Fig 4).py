# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow, 
# and Ehsan Foroumandi (eforoumandi@crimson.ua.edu), PhD Candidate
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on October 18, 2024
#
# This script visualizes the start locations of rapid intensification (RI) events
# of tropical cyclones on a global map using the Robinson projection. The script
# uses a scatter plot to represent the starting wind speed of each event, color-coded
# to highlight intensity variations across different regions. The dataset used contains
# geographical coordinates and the wind speeds of these events.
#
# Outputs:
# - A PDF file ('globe_HI_dist.pdf') displaying the global distribution of RI start
#   points, color-coded by wind speed.
#
# For a detailed description of the methodologies and further insights, please refer to:
# Radfar, S., Foroumandi, E., Moftakhari, H., Moradkhani, H., Foltz, G., and Sen Gupta, A. (2024). 
# Global predictability of marine heatwave induced rapid intensification of tropical cyclones. Earth’s Future.
#
# Disclaimer:
# This script is intended for research and educational purposes only. It is provided 'as is' 
# without warranty of any kind, express or implied. The developers assume no responsibility for 
# errors or omissions in this script. No liability is assumed for damages resulting from the use 
# of the information contained herein.
#
# -----------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

# Load the dataset containing RI events
df = pd.read_csv('../intensifications.csv')

# Set up the figure and the basemap with 'robin' projection for a global view
fig = plt.figure(figsize=(10, 6))
m = Basemap(projection='robin', lon_0=0)

# Extract wind speed feature for color coding
feature = df['start_wind_speed']

# Define color map and normalization based on wind speed values
cmap = plt.cm.get_cmap('Reds')
norm = mcolors.Normalize(vmin=feature.min(), vmax=feature.max())

# Convert latitude and longitude to map projection coordinates
x, y = m(df['lon_start'].values, df['lat_start'].values)

# Plot RI start locations as a scatter plot
sc = m.scatter(x, y, c=feature, cmap='Reds', norm=norm, s=20, edgecolor='none')

# Add map features
m.drawmapboundary(fill_color='#C8E8FF')
m.fillcontinents(color='lightgrey', lake_color='#C8E8FF')
m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 0], color='gray', dashes=[1, 3])
m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1], color='gray', dashes=[1, 3])
m.drawcoastlines()
m.drawcountries()

# Create a color bar inset to show wind speed intensity
ax = plt.gca()
cbar_ax = inset_axes(ax, width="25%", height="3%", loc='lower left', bbox_to_anchor=(0.37, 0.06, 1, 1),
                     bbox_transform=ax.transAxes)
cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cbar.set_label('RI start wind speed (kt)')

# Position color bar label on top
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')

# Save the figure as a PDF file
plt.savefig('../globe_HI_dist.pdf', format='pdf')

plt.show()