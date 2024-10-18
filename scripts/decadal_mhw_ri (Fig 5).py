# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow, 
# and Ehsan Foroumandi (eforoumandi@crimson.ua.edu), PhD Candidate
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on October 18, 2024
#
# This script processes and visualizes the rapid intensification (RI) events 
# of tropical cyclones within various basins using historical data. It filters 
# the data based on basin locations and timeframes, normalizes RI events by 
# the number of tropical cyclones (TCs), and plots the RI frequency per decade, 
# latitude, and longitude for each basin.
#
# Outputs:
# - Bar charts showing RI frequency normalized by the number of TCs per decade, 
#   latitude, and longitude for each basin. The plots highlight patterns in RI 
#   occurrence over time and across spatial dimensions.
# - The plots are saved as PNG files named '{basin_name}_RI_plot.png', suitable 
#   for use in reports or presentations.
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
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data_mhw_ri = pd.read_csv('../T26_MHW_RI_input.csv')
data_mhw_ri['ISO_TIME'] = pd.to_datetime(data_mhw_ri['ISO_TIME'], errors='coerce')
data_mhw_ri['Year'] = data_mhw_ri['ISO_TIME'].dt.year
data_mhw_ri = data_mhw_ri[(data_mhw_ri['RI'] == 1)]

# Define the basins
basins = {
    'East Pacific': {'range': (0, 40, -180, -100), 'color': '#DD6D67'},
    'North Atlantic': {'range': (0, 50, -100, -20), 'color': '#67C4CA'},
    'Northwest Pacific': {'range': (0, 50, 100, 180), 'color': '#67CA86'},
    'North Indian': {'range': (0, 30, 45, 100), 'color': '#CA67AB'},
    'Southwest Indian': {'range': (-40, 0, 30, 90), 'color': '#677ACA'},
    'Australian': {'range': (-40, 0, 90, 180), 'color': '#9E67CA'},
    'East Australian': {'range': (-40, 0, -180, -140), 'color': '#CAB767'}
}

# Function to process and plot data for each basin
def plot_basin(basin_name, basin_info):
    # Filter the data for the current basin
    basin_data = data_mhw_ri[
        (data_mhw_ri['HI_LAT'] >= basin_info['range'][0]) &
        (data_mhw_ri['HI_LAT'] <= basin_info['range'][1]) &
        (data_mhw_ri['HI_LON'] >= basin_info['range'][2]) &
        (data_mhw_ri['HI_LON'] <= basin_info['range'][3])
    ].copy()
    
    unique_tcs = basin_data.drop_duplicates(subset=['SEASON', 'NAME'])
    unique_ri_events = basin_data.drop_duplicates(subset=['SEASON', 'HI_LAT', 'HI_LON'])
    
    # Define bins
    decade_bins = np.arange(1980, 2040, 10)
    decade_labels = [f"{i}'s" for i in range(1980, 2030, 10)]
    lat_bins = np.arange(basin_info['range'][0], basin_info['range'][1] + 10, 10)
    lon_bins = np.arange(basin_info['range'][2], basin_info['range'][3] + 10, 10)
    
    # Assign bins
    unique_ri_events['Decade'] = pd.cut(unique_ri_events['Year'], bins=decade_bins, labels=decade_labels, right=False).copy()
    unique_ri_events['Lat_bin'] = pd.cut(unique_ri_events['HI_LAT'], bins=lat_bins).copy()
    unique_ri_events['Lon_bin'] = pd.cut(unique_ri_events['HI_LON'], bins=lon_bins).copy()
    
    unique_tcs['Decade'] = pd.cut(unique_tcs['Year'], bins=decade_bins, labels=decade_labels, right=False).copy()
    unique_tcs['Lat_bin'] = pd.cut(unique_tcs['HI_LAT'], bins=lat_bins).copy()
    unique_tcs['Lon_bin'] = pd.cut(unique_tcs['HI_LON'], bins=lon_bins).copy()
    
    # Count unique TCs by bins
    tc_counts_by_decade = unique_tcs.groupby('Decade').size()
    tc_counts_by_lat = unique_tcs.groupby('Lat_bin').size()
    tc_counts_by_lon = unique_tcs.groupby('Lon_bin').size()
    
    # Count unique RI events by bins
    ri_counts_by_decade = unique_ri_events.groupby('Decade').size()
    ri_counts_by_lat = unique_ri_events.groupby('Lat_bin').size()
    ri_counts_by_lon = unique_ri_events.groupby('Lon_bin').size()
    
    # Normalize RI counts by unique TC counts
    normalized_decade = ri_counts_by_decade / tc_counts_by_decade / 10
    normalized_lat = ri_counts_by_lat / tc_counts_by_lat / 4.3
    normalized_lon = ri_counts_by_lon / tc_counts_by_lon / 4.3
    
    # Replace NaN values with zero in the normalized counts
    normalized_decade = normalized_decade.fillna(0)
    normalized_lat = normalized_lat.fillna(0)
    normalized_lon = normalized_lon.fillna(0)
    
    # Initialize figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(25, 10))
    
    # Set consistent font sizes
    title_fontsize = 46
    axis_label_fontsize = 36
    ticks_fontsize = 24
    
    # Uniform labelpad
    x_labelpad = 30
    y_labelpad = 10
    
    # Title for the entire plot
    fig.suptitle(basin_name, fontsize=title_fontsize, fontweight='bold', color=basin_info['color'])
    
    # Decade plot
    axes[0].bar(normalized_decade.index, normalized_decade, color=basin_info['color'])
    axes[0].set_ylabel('RI per TC per year', fontsize=axis_label_fontsize, labelpad=y_labelpad)
    axes[0].tick_params(axis='x', labelrotation=90, labelsize=ticks_fontsize)
    axes[0].tick_params(axis='y', labelsize=ticks_fontsize)
    axes[0].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=2)
    axes[0].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5, zorder=2)
    axes[0].set_axisbelow(True)
    
    # Latitude plot
    axes[1].barh(y=[str(x) for x in normalized_lat.index.categories], width=normalized_lat, color=basin_info['color'])
    axes[1].set_ylabel('Latitudinal band (°N)', fontsize=axis_label_fontsize, labelpad=y_labelpad)
    axes[1].tick_params(axis='x', labelrotation=90, labelsize=ticks_fontsize)
    axes[1].tick_params(axis='y', labelsize=ticks_fontsize)
    axes[1].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5, zorder=2)
    axes[1].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=2)
    axes[1].set_axisbelow(True)
    
    # Longitude plot
    lon_labels = [f"({int(interval.left)}, {int(interval.right)}]" for interval in normalized_lon.index.categories]
    axes[2].bar(lon_labels, normalized_lon, color=basin_info['color'])
    axes[2].set_ylabel('RI per TC per decade', fontsize=axis_label_fontsize, labelpad=y_labelpad)
    axes[2].tick_params(axis='x', labelrotation=90, labelsize=ticks_fontsize)
    axes[2].tick_params(axis='y', labelsize=ticks_fontsize)
    axes[2].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5, zorder=2)
    axes[2].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=2)
    axes[2].set_axisbelow(True)
    
    # Hide original x-axis labels
    for ax in axes:
        ax.set_xlabel('')
        ax.xaxis.label.set_visible(False)
    
    # Manually add x-axis titles using text, positioned consistently for all plots
    axes[0].text(0.5, -0.37, 'Decade', transform=axes[0].transAxes, fontsize=axis_label_fontsize, ha='center', va='top')
    axes[1].text(0.5, -0.37, 'RI per TC per decade', transform=axes[1].transAxes, fontsize=axis_label_fontsize, ha='center', va='top')
    axes[2].text(0.5, -0.37, 'Longitudinal band (°E)', transform=axes[2].transAxes, fontsize=axis_label_fontsize, ha='center', va='top')
    
    # Adjust subplots
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    
    # Save the plot as a PNG file, ensuring it fits within the bounds of the figure
    plt.savefig(f'../plots/{basin_name}_RI_plot.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()

# Iterate over each basin and plot
for basin_name, basin_info in basins.items():
    plot_basin(basin_name, basin_info)