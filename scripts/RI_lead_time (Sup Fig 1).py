# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow, 
# and Ehsan Foroumandi (eforoumandi@crimson.ua.edu), PhD Candidate
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on October 18, 2024
#
# This script reads a dataset containing the time differences (hr_diff) for reaching 
# a 30-knot increase in wind speed and visualizes the distribution of these events 
# over different time spans. It bins the time differences, calculates the frequency 
# of events as percentages, and plots a normalized bar chart.
#
# Outputs:
# - A bar chart showing the normalized percentage of events for each time span 
#   (hours required for a 30-knot increase), which helps in understanding the 
#   forecast lead time of the proposed ML model.
# - The chart is saved as a PDF file titled 'normalized_hr_diff_plot.pdf', suitable 
#   for inclusion in reports or presentations.
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
import seaborn as sns
import numpy as np

# Read the dataset
file_path = '../pro_intensifications.csv'
data = pd.read_csv(file_path)

# Adjust the hr_diff values for binning
bins = np.arange(0, 27, 3)
data['hr_diff'] = np.digitize(data['hr_diff'], bins) * 3
data['hr_diff'] = np.where(data['hr_diff'] > 24, 24, data['hr_diff'])

# Calculate the histogram
counts, _ = np.histogram(data['hr_diff'], bins=bins)
bin_centers = bins[:-1] + 1.5

# Normalize the counts to percentages
normalized_counts = (counts / counts.sum()) * 100

# Plotting the normalized bar chart
plt.figure(figsize=(10, 6))
colors = sns.color_palette('Set2_r', len(bin_centers))
plt.bar(bin_centers, normalized_counts, width=3, align='center', edgecolor='black', color=colors)

# Customizing the plot
plt.xlabel('Time to reach 30 knots increase [hrs]')
plt.ylabel('Frequency [%]')
plt.title('Normalized number of events in each RI time span')
plt.xticks(np.arange(3, 25, 3))
plt.ylim(0, max(normalized_counts) * 1.1)
plt.xlim(0, 24.1)
plt.grid(True, linestyle='--', linewidth=0.5)

# Removing the plot border
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Save the plot as a PDF file
plt.savefig('../normalized_hr_diff_plot.pdf', format='pdf')

plt.show()