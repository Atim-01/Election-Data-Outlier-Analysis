#!/usr/bin/env python
# coding: utf-8

# In[27]:


import os
import numpy as np
import pandas as pd


# In[28]:


pd.set_option('display.max_columns', None) #To ensure that all the data is display when sample is viewed.


# In[29]:


# Load the Excel file
data = pd.read_excel("C:\\Users\\Administrator\\Downloads\\anambra_l_l.xlsx")


# In[30]:


data.shape


# In[31]:


data.describe()


# In[32]:


data.info()


# In[33]:


data.sample(15)


# In[34]:


# Removing all duplicate rows
duplicate_rows = data[data.duplicated()]
print(duplicate_rows)


# In[35]:


data['APC'].unique()


# In[36]:


data['NNPP'].unique()


# In[37]:


data['LP'].unique()


# In[38]:


df = data.copy()
df.head(2)


# **Step 1: Neighbour Identification**
# 
# I'll use geodesic distance to identify neighbouring polling units within a predefined radius.

# In[39]:


from geopy.distance import geodesic

# Function to find neighbouring polling units within a certain radius (using 1 km)
def find_neighbours(polling_unit, radius=1):
    neighbours = []
    for index, row in df.iterrows():
        if polling_unit.name != index:  # Exclude the polling unit itself
            distance = geodesic((polling_unit['Latitude'], polling_unit['Longitude']), (row['Latitude'], row['Longitude'])).km
            if distance <= radius:
                neighbours.append(index)
    return neighbours

# Add a column for neighbours
df['neighbours'] = df.apply(lambda row: find_neighbours(row), axis=1)

#save the data to a csv file
df.to_csv('Anambra_Election_Data_With_Outlier_Scores.csv', index=False)


# **Step 2: Outlier Score Calculation**
# 
# Calculate the outlier score for each party in each polling unit based on the votes compared to its neighbours.

# In[40]:


# Function to calculate outlier score
def calculate_outlier_score(polling_unit, neighbours, party_columns):
    scores = {}
    for party in party_columns:
        neighbour_votes = df.loc[neighbours, party].values
        if len(neighbour_votes) > 0:
            mean_neighbour_votes = np.mean(neighbour_votes)
            scores[party] = abs(polling_unit[party] - mean_neighbour_votes)
        else:
            scores[party] = 0  # If no neighbours, score is 0
    return scores

# Define the columns that contain the votes for each party
party_columns = ['APC', 'LP', 'PDP', 'NNPP'] 

# Add outlier scores to the dataset
df['outlier_scores'] = df.apply(lambda row: calculate_outlier_score(row, row['neighbours'], party_columns), axis=1)

# Expand outlier scores into separate columns
outlier_scores_df = pd.DataFrame(df['outlier_scores'].tolist(), index=df.index)

# Remove existing outlier score columns if they exist
for party in party_columns:
    outlier_col = f"{party}_outlier_score"
    if outlier_col in df.columns:
        df.drop(columns=[outlier_col], inplace=True)

# Join the new outlier score columns
df = df.join(outlier_scores_df.add_suffix('_outlier_score'))

# Save the updated dataset as an Excel file
df.to_excel('Anambra_Polling_Units_Sorted_By_Outlier_Scores.xlsx', index=False)


# In[41]:


df


# **Map of the entire state**

# In[42]:


import folium

# Create a map
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=10)

# Add polling units to the map
for index, row in df.iterrows():
    folium.Marker(
        [row['Latitude'], row['Longitude']],
        popup=f"Outlier Scores: {row[[f'{party}_outlier_score' for party in party_columns]].to_dict()}",
        tooltip=row['Address']
    ).add_to(m)

# Save the map
m.save('Map_For_Polling_Units.html')


# **Visual Result of the Outlier Score Column of Each Party**

# In[43]:


import matplotlib.pyplot as plt

# Plot histogram of outlier scores for a specific party
plt.hist(df['APC_outlier_score'], bins=50, alpha=0.7, label='APC')
plt.hist(df['LP_outlier_score'], bins=50, alpha=0.7, label='LP')
plt.hist(df['PDP_outlier_score'], bins=50, alpha=0.7, label='PDP')
plt.hist(df['NNPP_outlier_score'], bins=50, alpha=0.7, label='NNPP')
plt.xlabel('Outlier Score')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Outlier Scores by Party')
plt.tight_layout()  # Ensures all labels are visible
plt.savefig('Outlier_Scores_Histogram.png')  # Save the plot as PNG
plt.show()


# In[44]:


df


# In[46]:


# To see the stats of the calculation

print(df[['APC_outlier_score', 'LP_outlier_score', 'PDP_outlier_score', 'NNPP_outlier_score']].describe())


# **Visualisation of The Histograms of Outlier Scores for Each Party**

# In[22]:


import matplotlib.pyplot as plt

# Plot histograms of outlier scores for each party
plt.figure(figsize=(12, 10))  # Adjust figure size as needed

plt.subplot(2, 2, 1)
plt.hist(df['APC_outlier_score'], bins=50, color='blue', alpha=0.7)
plt.title('APC Outlier Scores')
plt.xlabel('Outlier Score')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(df['LP_outlier_score'], bins=50, color='green', alpha=0.7)
plt.title('LP Outlier Scores')
plt.xlabel('Outlier Score')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.hist(df['PDP_outlier_score'], bins=50, color='red', alpha=0.7)
plt.title('PDP Outlier Scores')
plt.xlabel('Outlier Score')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.hist(df['NNPP_outlier_score'], bins=50, color='purple', alpha=0.7)
plt.title('NNPP Outlier Scores')
plt.xlabel('Outlier Score')
plt.ylabel('Frequency')

plt.tight_layout()

# Save the combined figure as a PNG file
plt.savefig('combined_histograms.png')

plt.show()


# - APC_outlier_score:
# The average outlier score for APC across all polling units is approximately 0.64. This suggests that, on average, APC's votes in each polling unit do not deviate significantly from the average votes of its neighboring units.
# 
# - LP_outlier_score:
# The average outlier score for LP is approximately 35.15. This indicates that, on average, LP's votes in polling units show a larger deviation from the average votes of neighboring units compared to APC.
# 
# - PDP_outlier_score:
# The average outlier score for PDP is approximately 0.97. This means that, on average, PDP's votes in each polling unit exhibit slight deviations from the average votes of its neighboring units, but these deviations are relatively small.
# 
# - NNPP_outlier_score:
# The average outlier score for NNPP is approximately 0.53. This indicates that NNPP's votes in polling units, on average, also show minor deviations from the average votes of neighboring units.
# 
# 
# **Insights:** LP has a notably higher mean outlier score compared to APC, PDP, and NNPP, suggesting that LP's voting patterns might be more varied or influenced in certain polling units.
# 

# **Step 3: Sorting and Reporting**
# 
# Sort the dataset by outlier scores and prepare the report.

# In[47]:


# Sort the dataset by outlier scores for each party
sorted_df = df.sort_values(by=[f'{party}_outlier_score' for party in party_columns], ascending=False)

# Select top 3 outliers for each party
top_outliers = sorted_df.head(3)
top_outliers


# **DEDUCTION FROM THE OVERALL DATA**

# The top three polling units with the highest outlier scores in Anambra State are Community Secondary School Odekpe I, Okoti Market Square I, and Community Secondary School Odekpe II. These units exhibit significant deviations in voting patterns compared to their neighbors. 
# 
# Community Secondary School Odekpe I has the highest outlier score for APC at 85 and LP at 61, indicating unusually high or low votes for these parties. 
# 
# Okoti Market Square I shows a substantial outlier score for LP at 107 and for PDP at 0.5, reflecting atypical voting behaviors. 
# 
# Community Secondary School Odekpe II also presents notable outlier scores for APC and LP, though to a lesser extent. 
# 
# These discrepancies suggest that these polling units experienced voting patterns that differ markedly from surrounding areas, warranting further investigation into possible reasons for these anomalies.
# 

# **Visualizing the Results**
# 
# You can use maps and charts to visualize the results and include them in your report.

#  **Bar Charts for Outlier Scores**

# In[48]:


import matplotlib.pyplot as plt

# Sample DataFrame for top outliers with abbreviated labels
top_outliers = pd.DataFrame({
    'Address': ['C.S.S Odekpe I', 'Okoti Mkt Sqr I', 'C.S.S Odekpe II'],
    'APC_outlier_score': [85.0, 42.5, 42.5],
    'LP_outlier_score': [61.0, 107.0, 46.0],
    'PDP_outlier_score': [1.0, 0.5, 0.5],
    'NNPP_outlier_score': [0.5, 0.5, 1.0]
})

# Plot bar charts
parties = ['APC', 'LP', 'PDP', 'NNPP']
addresses = top_outliers['Address']

plt.figure(figsize=(12, 8))

for i, party in enumerate(parties):
    plt.subplot(2, 2, i+1)
    plt.bar(addresses, top_outliers[f'{party}_outlier_score'], color=['blue', 'green', 'red', 'purple'][i])
    plt.title(f'{party} Outlier Scores')
    plt.xlabel('Polling Units')
    plt.ylabel('Outlier Score')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('top_outliers_bar_charts.png')  # Save the figure
plt.show()


# In[25]:


import folium

# Sample DataFrame for top outliers
top_outliers = pd.DataFrame({
    'Address': ['Community Secondary School Odekpe I', 'Okoti Market Square I', 'Community Secondary School Odekpe II'],
    'Latitude': [6.065541, 6.071943, 6.065541],
    'Longitude': [6.743125, 6.743981, 6.743125],
    'APC_outlier_score': [85.0, 42.5, 42.5],
    'LP_outlier_score': [61.0, 107.0, 46.0],
    'PDP_outlier_score': [1.0, 0.5, 0.5],
    'NNPP_outlier_score': [0.5, 0.5, 1.0]
})

# Create a map centered around the average coordinates of the outliers
m = folium.Map(location=[top_outliers['Latitude'].mean(), top_outliers['Longitude'].mean()], zoom_start=13)

# Add markers for each polling unit
for _, row in top_outliers.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=(
            f"Address: {row['Address']}<br>"
            f"APC Outlier Score: {row['APC_outlier_score']}<br>"
            f"LP Outlier Score: {row['LP_outlier_score']}<br>"
            f"PDP Outlier Score: {row['PDP_outlier_score']}<br>"
            f"NNPP Outlier Score: {row['NNPP_outlier_score']}"
        ),
        tooltip=row['Address']
    ).add_to(m)

# Save the map as an HTML file
m.save('top_outliers_map.html')


# In[ ]:


import folium
from folium.plugins import MarkerCluster
import pandas as pd

# Sample DataFrame for top outliers
top_outliers = pd.DataFrame({
    'Address': ['Community Secondary School Odekpe I', 'Okoti Market Square I', 'Community Secondary School Odekpe II'],
    'Latitude': [6.065541, 6.071943, 6.065541],
    'Longitude': [6.743125, 6.743981, 6.743125],
    'APC_outlier_score': [85.0, 42.5, 42.5],
    'LP_outlier_score': [61.0, 107.0, 46.0],
    'PDP_outlier_score': [1.0, 0.5, 0.5],
    'NNPP_outlier_score': [0.5, 0.5, 1.0]
})

# Create a map centered around the average coordinates of the outliers
m = folium.Map(location=[top_outliers['Latitude'].mean(), top_outliers['Longitude'].mean()], zoom_start=13)

# Create a marker cluster
marker_cluster = MarkerCluster().add_to(m)

# Add markers for each polling unit to the cluster
for _, row in top_outliers.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=(
            f"Address: {row['Address']}<br>"
            f"APC Outlier Score: {row['APC_outlier_score']}<br>"
            f"LP Outlier Score: {row['LP_outlier_score']}<br>"
            f"PDP Outlier Score: {row['PDP_outlier_score']}<br>"
            f"NNPP Outlier Score: {row['NNPP_outlier_score']}"
        ),
        tooltip=row['Address']
    ).add_to(marker_cluster)

# Save the map as an HTML file
m.save('top_outliers_map.html')


# In[ ]:


import folium
import pandas as pd

# Sample DataFrame for top outliers
top_outliers = pd.DataFrame({
    'Address': ['Community Secondary School Odekpe I', 'Okoti Market Square I', 'Community Secondary School Odekpe II'],
    'Latitude': [6.065541, 6.071943, 6.065541],
    'Longitude': [6.743125, 6.743981, 6.743125],
    'APC_outlier_score': [85.0, 42.5, 42.5],
    'LP_outlier_score': [61.0, 107.0, 46.0],
    'PDP_outlier_score': [1.0, 0.5, 0.5],
    'NNPP_outlier_score': [0.5, 0.5, 1.0]
})

# Create a map centered around the average coordinates of the outliers
m = folium.Map(location=[top_outliers['Latitude'].mean(), top_outliers['Longitude'].mean()], zoom_start=13)

# Keep track of coordinates to slightly offset duplicates
seen_coords = {}

# Add markers for each polling unit
for _, row in top_outliers.iterrows():
    coord = (row['Latitude'], row['Longitude'])
    if coord in seen_coords:
        # Slightly offset the coordinates if they have been seen before
        offset_lat, offset_lon = seen_coords[coord]
        offset_lat += 0.0001
        offset_lon += 0.0001
        seen_coords[coord] = (offset_lat, offset_lon)
        coord = (offset_lat, offset_lon)
    else:
        seen_coords[coord] = coord

    folium.Marker(
        location=coord,
        popup=(
            f"Address: {row['Address']}<br>"
            f"APC Outlier Score: {row['APC_outlier_score']}<br>"
            f"LP Outlier Score: {row['LP_outlier_score']}<br>"
            f"PDP Outlier Score: {row['PDP_outlier_score']}<br>"
            f"NNPP Outlier Score: {row['NNPP_outlier_score']}"
        ),
        tooltip=row['Address']
    ).add_to(m)

# Save the map as an HTML file
m.save('top_outliers_map.html')


# In[ ]:




