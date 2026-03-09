

CSV_PATH = "C:\\Users\\chamu\\OneDrive\\Desktop\\particum\\chicago_crime_full.csv"

import os
size_gb = os.path.getsize(CSV_PATH) / (1024**3)
print(f"File size: {size_gb:.2f} GB")

import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(CSV_PATH)

# Display the first few rows of the DataFrame
print("First 5 rows of the DataFrame:")
print(df.head())

print("\nDataFrame Info:")
df.info()


print("\nMissing values per column:")
print(df.isnull().sum())

# Calculate the median latitude and longitude for each district
median_lat_per_district = df.groupby('district')['latitude'].transform('median')
median_lon_per_district = df.groupby('district')['longitude'].transform('median')

# Fill missing latitude and longitude values using district medians
df['latitude'] = df['latitude'].fillna(median_lat_per_district)
df['longitude'] = df['longitude'].fillna(median_lon_per_district)

# For any districts that might still have missing values (e.g., if an entire district had missing lat/lon),
# fall back to the global median (which we already calculated and used implicitly via previous steps)
# Or, if this is a fresh run, one might fall back to global median after district-level imputation.
# As per the current state, `latitude` and `longitude` should already be fully imputed globally,
# so this step would primarily refine an already imputed dataset or handle initial state if rerun.

print("Missing values after district-wise imputation:")
print(df[['latitude', 'longitude']].isnull().sum())


# Drop rows with any remaining missing values
df.dropna(inplace=True)

# Verify that all missing values have been handled
print("Missing values after dropping final rows:")
print(df.isnull().sum())


print(
"\nDescriptive statistics for numerical columns:")
print(df.describe())


print("\nUnique values and counts for 'primary_type':")
print(df['primary_type'].value_counts())

df[df["primary_type"] == "DECEPTIVE PRACTICE"]["description"].value_counts().head(10)

df[df["primary_type"] == "RITUALISM"]["description"].value_counts().head(10)

df[df["primary_type"] == "OFFENSE INVOLVING CHILDREN"]["description"].value_counts().head(10)

# Standardize the rape category (Chicago has two labels for this due to historical changes)
df['primary_type'] = df['primary_type'].replace('CRIM SEXUAL ASSAULT', 'CRIMINAL SEXUAL ASSAULT')

# Define the list of violent crime types as specified by the user
violent_crime_types = [
    "CRIMINAL SEXUAL ASSAULT",
    "ASSAULT"
    "SEX OFFENSE",
    "STALKING",
    "KIDNAPPING",
    "ROBBERY",
    "BATTERY",
    "HOMICIDE",
    "ARSON",
    "HUMAN TRAFFICKING",
    "CRIMINAL TRESPASS",
    "OFFENSE INVOLVING CHILDREN"
]

# Create a new DataFrame containing only the violent crimes
violent_crimes_df = df[df['primary_type'].isin(violent_crime_types)].copy()

violent_crimes_path = "violent_crimes.csv"
violent_crimes_df.to_csv(violent_crimes_path, index=False)

print(f"New dataset 'violent_crimes_df' created with {len(violent_crimes_df)} rows.")
print(f"Violent crimes data saved to: {violent_crimes_path}")
print("First 5 rows of the violent crimes dataset:")
print(violent_crimes_df.info())

print("\nUnique values and counts for 'arrest':")
print(df['arrest'].value_counts())


print("\nUnique values and counts for 'domestic':")
print(df['domestic'].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

print("Generating histograms for numerical columns:")

# Set up the figure size
plt.figure(figsize=(15, 10))

# Histogram for 'beat'
plt.subplot(2, 2, 1)
sns.histplot(df['beat'].dropna(), bins=30, kde=True)
plt.title('Distribution of Beat')
plt.xlabel('Beat')
plt.ylabel('Frequency')

# Histogram for 'district'
plt.subplot(2, 2, 2)
sns.histplot(df['district'].dropna(), bins=30, kde=True)
plt.title('Distribution of District')
plt.xlabel('District')
plt.ylabel('Frequency')

# Histogram for 'latitude'
plt.subplot(2, 2, 3)
sns.histplot(df['latitude'].dropna(), bins=30, kde=True)
plt.title('Distribution of Latitude')
plt.xlabel('Latitude')
plt.ylabel('Frequency')

# Histogram for 'longitude'
plt.subplot(2, 2, 4)
sns.histplot(df['longitude'].dropna(), bins=30, kde=True)
plt.title('Distribution of Longitude')
plt.xlabel('Longitude')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig("plots/distribution_numerical_plots.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()


print("Generating bar plots for categorical columns:")

plt.figure(figsize=(18, 6))

# Bar plot for top 10 'primary_type'
plt.subplot(1, 3, 1)
top_10_primary_types = df['primary_type'].value_counts().head(10)
sns.barplot(x=top_10_primary_types.index, y=top_10_primary_types.values, palette='viridis')
plt.title('Top 10 Primary Crime Types')
plt.xlabel('Primary Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# Bar plot for 'arrest'
plt.subplot(1, 3, 2)
sns.countplot(x='arrest', data=df, palette='cividis')
plt.title('Distribution of Arrests')
plt.xlabel('Arrest')
plt.ylabel('Count')

# Bar plot for 'domestic'
plt.subplot(1, 3, 3)
sns.countplot(x='domestic', data=df, palette='magma')
plt.title('Distribution of Domestic Incidents')
plt.xlabel('Domestic')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig("plots/distribution_categorical_plots.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()

print("\nUnique values and counts for 'location_description':")
print(df['location_description'].value_counts())


print("\nTop 10 unique values and counts for 'description':")
print(violent_crimes_df['description'].value_counts().head(10))


# Create a crosstabulation of primary_type and location_description
crosstab_df = pd.crosstab(df['primary_type'], df['location_description'])

# Get the top 10 primary crime types and top 10 location descriptions
top_10_primary_types = df['primary_type'].value_counts().head(10).index
top_10_locations = df['location_description'].value_counts().head(10).index

# Filter the crosstab to include only the top 10 crime types and top 10 locations
filtered_crosstab = crosstab_df.loc[top_10_primary_types, top_10_locations]

plt.figure(figsize=(14, 10))
sns.heatmap(filtered_crosstab, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
plt.title('Relationship between Top 10 Primary Crime Types and Top 10 Location Descriptions')
plt.xlabel('Location Description')
plt.ylabel('Primary Crime Type')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/primary_type_location_heatmap.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()

plt.figure(figsize=(15, 7))
top_15_primary_types = violent_crimes_df['primary_type'].value_counts().head(15)
sns.barplot(x=top_15_primary_types.index, y=top_15_primary_types.values, hue=top_15_primary_types.index, palette='viridis', legend=False)
plt.title('Top 15 Primary Crime Types Distribution')
plt.xlabel('Primary Type')
plt.ylabel('Count')
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.savefig("plots/primary_type_distribution.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()


# THEN CLOSE (important for memory)
plt.close()

plt.figure(figsize=(12, 6))
top_10_locations = violent_crimes_df['location_description'].value_counts().head(10)
sns.barplot(x=top_10_locations.index, y=top_10_locations.values, hue=top_10_locations.index, palette='viridis', legend=False)
plt.title('Top 10 Crime Location Descriptions')
plt.xlabel('Location Description')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("plots/location_description_distribution.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()

import matplotlib.pyplot as plt
import seaborn as sns

print("Generating correlation heatmap for numerical columns:")

# Select only numerical columns for correlation calculation
numerical_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Numerical Columns')
plt.savefig("plots/correlation_heatmap.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()

import matplotlib.pyplot as plt
import seaborn as sns

print("Generating correlation heatmap for numerical columns:")

# Select only numerical columns for correlation calculation
numerical_df = violent_crimes_df.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Numerical Columns')
plt.savefig("plots/correlation_heatmap.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()

print(f"Shape of DataFrame before filtering: {violent_crimes_df.shape}")

# Convert 'date' column to datetime objects with the specified format
violent_crimes_df['date'] = pd.to_datetime(violent_crimes_df['date'], format='%Y-%m-%dT%H:%M:%S.000', errors='coerce')

# Find the most recent date in the dataset, ignoring NaT values
most_recent_date = violent_crimes_df['date'].max()

# Calculate the cutoff date (10 years prior to the most recent date)
cutoff_date = most_recent_date - pd.DateOffset(years=10)

# Filter the DataFrame to include only records from the last 10 years
violent_crimes_df = violent_crimes_df[violent_crimes_df['date'] >= cutoff_date].copy()

print(f"Shape of DataFrame after filtering: {df.shape}")

# Display the first few rows of the filtered DataFrame to verify
print("\nFirst 5 rows of the filtered DataFrame:")
print(violent_crimes_df.head())

print("\nDataFrame Info after date filtering:")
violent_crimes_df.info()

print("\nMissing values after date filtering:")
print(violent_crimes_df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Extract the year from the 'date' column
violent_crimes_df['year'] = violent_crimes_df['date'].dt.year

# 2. Extract the month from the 'date' column
violent_crimes_df['month'] = violent_crimes_df['date'].dt.month

# 3. Extract the day of the week from the 'date' column
violent_crimes_df['day_of_week'] = violent_crimes_df['date'].dt.day_name()

# 4. Calculate the crime counts per year
yearly_crime_counts = violent_crimes_df['year'].value_counts().sort_index()

# 5. Create a bar plot for yearly crime counts
plt.figure(figsize=(12, 6))
sns.barplot(x=yearly_crime_counts.index, y=yearly_crime_counts.values, hue=yearly_crime_counts.index, palette='viridis', legend=False)
plt.title('Yearly Crime Trends')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/yearly_crime_trends.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()

# 6. Calculate the crime counts per month
monthly_crime_counts = violent_crimes_df['month'].value_counts().sort_index()

# 7. Create a bar plot for monthly crime counts
plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_crime_counts.index, y=monthly_crime_counts.values, hue=monthly_crime_counts.index, palette='magma', legend=False)
plt.title('Monthly Crime Trends')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/monthly_crime_trends.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()

print("Temporal features extracted and crime trends plotted.")

print("Generating bar plots for top categorical columns from filtered data:")

# 1. Calculate the value counts for the 'primary_type' column and get the top 10
top_10_primary_types_filtered = violent_crimes_df['primary_type'].value_counts().head(10)

# 2. Create a bar plot for these top 10 primary crime types
plt.figure(figsize=(15, 7))
sns.barplot(x=top_10_primary_types_filtered.index, y=top_10_primary_types_filtered.values, hue=top_10_primary_types_filtered.index, palette='viridis', legend=False)
plt.title('Top 10 Primary Crime Types Distribution (Filtered Data)')
plt.xlabel('Primary Type')
plt.ylabel('Count')
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.savefig("plots/top_10_primary_types_filtered.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()

import matplotlib.pyplot as plt
import seaborn as sns

# 3. Calculate the value counts for the 'location_description' column and get the top 10
top_10_locations_filtered = violent_crimes_df['location_description'].value_counts().head(10)

# 4. Create a bar plot for these top 10 crime location descriptions
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_locations_filtered.index, y=top_10_locations_filtered.values, hue=top_10_locations_filtered.index, palette='viridis', legend=False)
plt.title('Top 10 Crime Location Descriptions (Filtered Data)')
plt.xlabel('Location Description')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("plots/top_10_location_descriptions_filtered.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()

print("Generating heatmap for relationship between top crime types and locations (filtered data):")

# 5. Create a crosstabulation between 'primary_type' and 'location_description'
crosstab_df_filtered = pd.crosstab(violent_crimes_df['primary_type'], violent_crimes_df['location_description'])

# 6. Filter crosstab_df_filtered to include only the top 10 crime types and top 10 locations
filtered_crosstab_heatmap = crosstab_df_filtered.loc[top_10_primary_types_filtered.index, top_10_locations_filtered.index]

# 7. Generate a heatmap of the filtered crosstabulation
plt.figure(figsize=(14, 10))
sns.heatmap(filtered_crosstab_heatmap, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
plt.title('Relationship between Top 10 Primary Crime Types and Top 10 Location Descriptions (Filtered Data)')
plt.xlabel('Location Description')
plt.ylabel('Primary Crime Type')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/filtered_crosstab_heatmap.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()

print("Preparing features for clustering...")

# 1. Define numerical features to include
numerical_cols = ['latitude', 'longitude', 'beat', 'district']

# 2. Extract the actual crime type names and location names for the top 10 categories
top_primary_types_list = top_10_primary_types_filtered.index.tolist()
top_locations_list = top_10_locations_filtered.index.tolist()

# 3. Create a new DataFrame containing only the selected numerical features
features_for_clustering_df = violent_crimes_df[numerical_cols].copy()

# 4. Apply one-hot encoding to the selected categorical features for top 10 categories
# For 'primary_type'
for p_type in top_primary_types_list:
    features_for_clustering_df[f'primary_type_{p_type}'] = (violent_crimes_df['primary_type'] == p_type).astype(int)

# For 'location_description'
for loc in top_locations_list:
    features_for_clustering_df[f'location_description_{loc}'] = (violent_crimes_df['location_description'] == loc).astype(int)

# 5. No explicit concatenation needed as we added columns directly to features_for_clustering_df.
# Ensure all original rows are retained and aligned by index.

# Display the first few rows and the shape of the new DataFrame to verify the feature preparation
print("\nFirst 5 rows of the features_for_clustering_df:")
print(features_for_clustering_df.head())
print(f"\nShape of features_for_clustering_df: {features_for_clustering_df.shape}")

print("Preparing features for clustering using pd.get_dummies...")

# 1. Define numerical features to include
numerical_cols = ['latitude', 'longitude', 'beat', 'district']

# 2. Extract the actual crime type names and location names for the top 10 categories
top_primary_types_list = top_10_primary_types_filtered.index.tolist()
top_locations_list = top_10_locations_filtered.index.tolist()

# 3. Create a new DataFrame containing only the selected numerical features
numerical_features_df = violent_crimes_df[numerical_cols].copy()

# 4. Apply one-hot encoding using pd.get_dummies() for the selected categorical features
# Generate all dummy columns first
primary_type_dummies = pd.get_dummies(violent_crimes_df['primary_type'], prefix='primary_type')
location_description_dummies = pd.get_dummies(violent_crimes_df['location_description'], prefix='location_description')

# Filter to include only the top 10 categories for each
# Construct column names for the top categories to select from the dummy variables
top_primary_type_cols = [f'primary_type_{p_type}' for p_type in top_primary_types_list]
top_location_cols = [f'location_description_{loc}' for loc in top_locations_list]

# Select only the columns corresponding to the top categories
primary_type_ohe_filtered = primary_type_dummies[top_primary_type_cols]
location_description_ohe_filtered = location_description_dummies[top_location_cols]

# 5. Concatenate the numerical features DataFrame and the one-hot encoded categorical features DataFrames
features_for_clustering_df = pd.concat([
    numerical_features_df,
    primary_type_ohe_filtered,
    location_description_ohe_filtered
], axis=1)

# 6. Display the first few rows and the shape of the new DataFrame to verify the feature preparation
print("\nFirst 5 rows of the features_for_clustering_df:")
print(features_for_clustering_df.head())
print(f"\nShape of features_for_clustering_df: {features_for_clustering_df.shape}")

from sklearn.preprocessing import StandardScaler

print("Scaling features for clustering...")

# 1. Instantiate a StandardScaler object
scaler = StandardScaler()

# 2. Fit the scaler to features_for_clustering_df and transform the data
scaled_features = scaler.fit_transform(features_for_clustering_df)

# Convert the scaled data back to a DataFrame for easier inspection, preserving column names
scaled_features_df = pd.DataFrame(scaled_features, columns=features_for_clustering_df.columns, index=features_for_clustering_df.index)

# 3. Display the first few rows of the scaled data and its shape
print("\nFirst 5 rows of the scaled_features_df:")
print(scaled_features_df.head())
print(f"\nShape of scaled_features_df: {scaled_features_df.shape}")

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

print("Applying Elbow Method to determine optimal K...")

# 1. Define a range of cluster numbers to test
k_values = range(1, 11) # Testing from 1 to 10 clusters

# 2. Initialize an empty list to store the Within-Cluster Sum of Squares
wcss = []

# 3. Loop through each k in k_values
for k in k_values:
    # a. Instantiate a KMeans object with the current k, random_state=42, and n_init='auto'
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init='auto')

    # b. Fit the KMeans model to the scaled_features_df
    kmeans_model.fit(scaled_features_df)

    # c. Append the inertia_ attribute (WCSS) of the fitted model to the wcss list
    wcss.append(kmeans_model.inertia_)

# 4. Create a plot with k_values on the x-axis and wcss on the y-axis
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss, marker='o', linestyle='--')

# 5. Add a title to the plot and labels for the x and y axes
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)

# 6. Display the plot
plt.savefig("plots/elbow_method_plot.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()

print("Elbow method plot generated. Look for the 'elbow point' to determine optimal K.")

from sklearn.cluster import KMeans

print("Applying K-Means clustering...")

# 1. Choose an optimal number of clusters (K) based on the Elbow Method plot.
# From the plot, K=4 seems to be a reasonable elbow point.
optimal_k = 4

# 2. Instantiate a KMeans object with the chosen n_clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')

# 3. Fit the KMeans model to the scaled_features_df
kmeans.fit(scaled_features_df)

# 4. Assign the resulting cluster labels to a new column named 'cluster_label' in the original DataFrame
violent_crimes_df['cluster_label'] = kmeans.labels_

print(f"K-Means clustering applied with K={optimal_k} clusters.")

# 5. Display the value counts of the cluster_label column to check the distribution
print("\nDistribution of records per cluster:")
print(violent_crimes_df['cluster_label'].value_counts())

print("Analyzing cluster characteristics...")

# Add the cluster_label to features_for_clustering_df
features_for_clustering_df['cluster_label'] = violent_crimes_df['cluster_label']

# Group by 'cluster_label' and calculate the mean of the features
cluster_profiles = features_for_clustering_df.groupby('cluster_label').mean()

# Remove 'cluster_label' from features_for_clustering_df to keep it consistent for future steps
features_for_clustering_df.drop(columns=['cluster_label'], inplace=True);

print("\nCluster Profiles (Mean values of features per cluster):")
print(cluster_profiles)

print("\nVisualizing clusters on a scatter plot...")

# Create a scatter plot of 'longitude' vs 'latitude' colored by 'cluster_label'
plt.figure(figsize=(18, 15)) # Increased figure size for better visibility
sns.scatterplot(
    x='longitude',
    y='latitude',
    hue='cluster_label',
    data=violent_crimes_df,
    palette='viridis',
    s=10, # Adjust point size for better visualization with large datasets
    alpha=0.6 # Adjust alpha for better visibility of overlapping points
)

# Set x and y axis limits to zoom in on the cluster areas
plt.xlim(-88, -87.5)
plt.ylim(41.6, 42.1)

plt.title('Crime Clusters by Latitude and Longitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/clusters_spatial_visualization.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()

# 1. Define a dictionary named target_case
target_case = {
    'id': 10000000,
    'date': '2026-02-11 00:00:00',
    'primary_type': 'OFFENSE INVOLVING CHILDREN',
    'description': 'CHILD ABDUCTION',
    'location_description': 'ALLEY',
    'latitude': 41.794173,
    'longitude': -87.703576,
    'arrest': False,
    'domestic': False,
    'beat': 923,
    'district': 9
}

# 2. Convert the 'date' column in the df DataFrame to datetime objects
violent_crimes_df['date'] = pd.to_datetime(violent_crimes_df['date'], format='%Y-%m-%dT%H:%M:%S.000', errors='coerce')

# 3. Identify the most recent date in the df DataFrame
most_recent_date = violent_crimes_df['date'].max()

# 4. Calculate the cutoff_date by subtracting 24 years from the most_recent_date
cutoff_date = most_recent_date - pd.DateOffset(years=24)

# 6. Filter the df DataFrame to include only records where the 'date' is >= cutoff_date
violent_crimes_df = violent_crimes_df[violent_crimes_df['date'] >= cutoff_date].copy()

# 7. Remove the target_case from the df DataFrame if its 'id' matches any record
violent_crimes_df = violent_crimes_df[violent_crimes_df['id'] != target_case['id']].copy()

print(f"Shape of DataFrame after filtering and removing target_case: {violent_crimes_df.shape}")

# 9. Display the first few rows of the filtered df DataFrame
print("\nFirst 5 rows of the filtered DataFrame (historical_data):")
print(violent_crimes_df.head())

import numpy as np

# 1. Extract the 'hour' from the 'date' column for the target_case
# Ensure target_case['date'] is a datetime object
target_case['date'] = pd.to_datetime(target_case['date'])
target_case['hour'] = target_case['date'].hour

print(f"Target Case Hour: {target_case['hour']}")

# 2. Extract the 'hour' from the 'date' column of the df DataFrame
# Ensure df['date'] is already datetime; if not, convert it
if not pd.api.types.is_datetime64_any_dtype(violent_crimes_df['date']):
    violent_crimes_df['date'] = pd.to_datetime(violent_crimes_df['date'])
violent_crimes_df['hour'] = violent_crimes_df['date'].dt.hour

print("First 5 rows with 'hour' extracted for historical data:")
print(violent_crimes_df[['date', 'hour']].head())

# 3. Define a Python function circular_time_distance(hour1, hour2)
def circular_time_distance(hour1, hour2):
    # Calculate the absolute difference
    diff = abs(hour1 - hour2)
    # Return the shortest distance in a 24-hour cycle
    return min(diff, 24 - diff)

# 4. Apply this circular_time_distance function to calculate the temporal distance
violent_crimes_df['temporal_distance'] = violent_crimes_df['hour'].apply(lambda h: circular_time_distance(target_case['hour'], h))

print("\nFirst 5 rows with 'temporal_distance' calculated:")
print(violent_crimes_df[['date', 'hour', 'temporal_distance']].head())

print(f"Temporal features (hour and circular distance) added to the DataFrame. Shape: {violent_crimes_df.shape}")

print("Installing 'haversine' library...")
print("Installation complete.")

from haversine import haversine, Unit

print("Calculating spatial distance and proximity...")

# 2. Define a function, calculate_haversine_distance
def calculate_haversine_distance(row, target_lat, target_lon):
    # Ensure coordinates are floats for haversine calculation
    case_coords = (row['latitude'], row['longitude'])
    target_coords = (target_lat, target_lon)
    return haversine(target_coords, case_coords, unit=Unit.KILOMETERS)

# 3. Apply this function to each row of the df DataFrame
violent_crimes_df['spatial_distance'] = violent_crimes_df.apply(
    lambda row: calculate_haversine_distance(
        row, target_case['latitude'], target_case['longitude']
    ), axis=1
)

# 4. Define another function, calculate_spatial_proximity
def calculate_spatial_proximity(distance):
    return 1 / (1 + distance)

# 5. Apply this function to the 'spatial_distance' column
violent_crimes_df['spatial_proximity'] = violent_crimes_df['spatial_distance'].apply(calculate_spatial_proximity)

# 6. Display the first 5 rows of the DataFrame, showing the new columns
print("\nFirst 5 rows with 'spatial_distance' and 'spatial_proximity' calculated:")
print(violent_crimes_df[['id', 'latitude', 'longitude', 'spatial_distance', 'spatial_proximity']].head())
print(f"\nShape of DataFrame after adding spatial features: {violent_crimes_df.shape}")

print("Combining 'description' and 'location_description' into 'behavioral_signature'...")

# 2. Create 'behavioral_signature' for df
violent_crimes_df['behavioral_signature'] = (
    violent_crimes_df['description'].fillna('') + ' ' + violent_crimes_df['location_description'].fillna('')
).str.strip()

# 3. Create 'behavioral_signature' for target_case
target_case_description = target_case.get('description', '')
target_case_location_description = target_case.get('location_description', '')
target_case['behavioral_signature'] = (
    str(target_case_description) + ' ' + str(target_case_location_description)
).strip()

print("First 5 rows with 'behavioral_signature' for historical data:")
print(violent_crimes_df[['description', 'location_description', 'behavioral_signature']].head())
print(f"\nTarget Case Behavioral Signature: {target_case['behavioral_signature']}")
print(f"Shape of DataFrame after adding 'behavioral_signature': {violent_crimes_df.shape}")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Combining 'description' and 'location_description' into 'behavioral_signature'...")

# 2. Create 'behavioral_signature' for df
violent_crimes_df['behavioral_signature'] = (
    violent_crimes_df['description'].fillna('') + ' ' + violent_crimes_df['location_description'].fillna('')
).str.strip()

# 3. Create 'behavioral_signature' for target_case
target_case_description = target_case.get('description', '')
target_case_location_description = target_case.get('location_description', '')
target_case['behavioral_signature'] = (
    str(target_case_description) + ' ' + str(target_case_location_description)
).strip()

print("First 5 rows with 'behavioral_signature' for historical data:")
print(violent_crimes_df[['description', 'location_description', 'behavioral_signature']].head())

# Fix for KeyError: 'behavior_signature' - Corrected key name
print(f"\nTarget Case Behavioral Signature: {target_case['behavioral_signature']}")

print("\nApplying TF-IDF and calculating cosine similarity...")

# 4. Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# 5. Fit and transform the behavioral_signature of the target_case
target_case_tfidf = tfidf_vectorizer.fit_transform([target_case['behavioral_signature']])

# 6. Transform the behavioral_signature column of the df DataFrame
df_tfidf = tfidf_vectorizer.transform(violent_crimes_df['behavioral_signature'])

# 7. Calculate the cosine similarity
violent_crimes_df['behavioral_similarity'] = cosine_similarity(df_tfidf, target_case_tfidf).flatten()

# 9. Print the first 5 rows of the df DataFrame, showing the new columns
print("\nFirst 5 rows with 'behavioral_signature' and 'behavioral_similarity' calculated:")
print(violent_crimes_df[['description', 'location_description', 'behavioral_signature', 'behavioral_similarity']].head())

# 10. Print the shape of the df DataFrame
print(f"\nShape of DataFrame after adding 'behavioral_similarity': {violent_crimes_df.shape}")

print("Developing Linkage Scoring Function...")

# 1. Define a function calculate_temporal_consistency
def calculate_temporal_consistency(temporal_distance):
    return 1 / (1 + temporal_distance)

# 2. Apply the calculate_temporal_consistency function to the 'temporal_distance' column
violent_crimes_df['temporal_consistency'] = violent_crimes_df['temporal_distance'].apply(calculate_temporal_consistency)

# 3. Define a function calculate_linkage_score
def calculate_linkage_score(row, w_behavioral=0.5, w_spatial=0.3, w_temporal=0.2):
    # Ensure weights sum to 1, or normalize if they don't.
    # For this task, assuming they already sum to 1.
    return (
        row['behavioral_similarity'] * w_behavioral +
        row['spatial_proximity'] * w_spatial +
        row['temporal_consistency'] * w_temporal
    )

# 4. Apply the calculate_linkage_score function to each row of the df DataFrame
violent_crimes_df['linkage_score'] = violent_crimes_df.apply(calculate_linkage_score, axis=1)

# 5. Display the first 5 rows of the df DataFrame, showing the new columns
print("\nFirst 5 rows with 'temporal_consistency' and 'linkage_score' calculated:")
print(violent_crimes_df[['id', 'temporal_distance', 'temporal_consistency', 'behavioral_similarity', 'spatial_proximity', 'linkage_score']].head())

print(f"\nShape of DataFrame after adding linkage scores: {violent_crimes_df.shape}")

print("Identifying Top 5 Most Linkable Cases...")

# 1. Sort the df DataFrame by the 'linkage_score' column in descending order
sorted_df = violent_crimes_df.sort_values(by='linkage_score', ascending=False)

# 2. Select the top 10 rows from the sorted DataFrame
top_10_linked_cases = sorted_df.head(10)

# 3. Print the top_5_linked_cases DataFrame to display the results
print("\nTop 10 Most Linkable Historical Cases:")
print(top_10_linked_cases[['id', 'primary_type', 'description', 'location_description', 'date', 'linkage_score']])

import matplotlib.pyplot as plt
import seaborn as sns

print("Generating geographic scatter plot for linked cases...")

plt.figure(figsize=(12, 10))

# 1. Plot all historical cases (background)
sns.scatterplot(
    x=violent_crimes_df['longitude'],
    y=violent_crimes_df['latitude'],
    color='lightgray',
    alpha=0.2,
    s=10,
    label='Other Historical Cases'
)

# 2. Overlay the top 10 linked cases
sns.scatterplot(
    x=top_10_linked_cases['longitude'],
    y=top_10_linked_cases['latitude'],
    color='red',
    marker='o',
    s=50,
    label='Top 10 Linked Cases',
    edgecolor='black',
    linewidth=1
)

# 3. Overlay the target case
plt.scatter(
    target_case['longitude'],
    target_case['latitude'],
    color='blue',
    marker='*', # Using a star marker for the target case
    s=50, # Larger size to highlight
    label='Target Case',
    edgecolor='black',
    linewidth=1.5,
    zorder=5 # Ensure target case is on top
)

plt.xlim(-88, -87.5)
plt.ylim(41.6, 42.1)

# Add plot enhancements
plt.title('Target Case vs. Top 5 Linked Historical Cases',
            fontsize=16)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.legend(title='Case Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("plots/target_vs_linked_cases.png", dpi=300, bbox_inches="tight")

# THEN SHOW
plt.show()

# THEN CLOSE (important for memory)
plt.close()

print("Geographic scatter plot generated.")

print("Installing 'folium' library...")
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "folium"])
print("Installation complete.")
import pandas as pd

print("Preparing data for mapping...")

# 1. Convert the target_case dictionary into a pandas DataFrame named target_case_df.
#    Ensure that the 'date' column is converted to datetime objects within this DataFrame.
# target_case was redefined in the previous step, ensuring its 'date' is still a string or datetime.
if isinstance(target_case['date'], str):
    target_case_df = pd.DataFrame([target_case])
    target_case_df['date'] = pd.to_datetime(target_case_df['date'])
else:
    # If target_case['date'] is already a datetime object (from previous cell's modifications)
    target_case_df = pd.DataFrame([target_case])


# 2. Add a new column named 'link_group' to target_case_df and assign it the value 'Target'.
target_case_df['link_group'] = 'Target'

# 3. Add a new column named 'link_group' to the top_5_linked_cases DataFrame and assign it the value 'Linked'.
top_10_linked_cases_for_map = top_10_linked_cases.copy() # Create a copy to avoid SettingWithCopyWarning
top_10_linked_cases_for_map['link_group'] = 'Linked'

# 4. Define a list of common columns required for mapping
common_cols = ['id', 'date', 'primary_type', 'latitude', 'longitude', 'link_group']

# 5. Select only these common columns from both target_case_df and top_5_linked_cases
target_case_filtered_for_map = target_case_df[common_cols]
linked_cases_filtered_for_map = top_10_linked_cases_for_map[common_cols]

# 6. Concatenate the filtered target_case_df and top_5_linked_cases into a single DataFrame called linked_cases_df.
linked_cases_df = pd.concat([target_case_filtered_for_map, linked_cases_filtered_for_map], ignore_index=True)

# 7. Display the first few rows of linked_cases_df and its shape to verify the combined data.
print("\nFirst 10 rows of linked_cases_df:")
print(linked_cases_df.head(10))
print(f"\nShape of linked_cases_df: {linked_cases_df.shape}")

import folium
import webbrowser
import os

print("Initializing Folium Map...")

mean_lat = linked_cases_df['latitude'].mean()
mean_lon = linked_cases_df['longitude'].mean()

m = folium.Map(location=[mean_lat, mean_lon], zoom_start=13, tiles='CartoDB positron')

print("Adding markers to the map...")

for index, row in linked_cases_df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    case_id = row['id']
    primary_type = row['primary_type']
    case_date = row['date'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['date']) else 'N/A'
    link_group = row['link_group']

    if link_group == 'Target':
        marker_color = 'blue'
        popup_text = f"<b>Target Case: {case_id}</b><br>Type: {primary_type}<br>Date: {case_date}"
    else:
        marker_color = 'red'
        popup_text = f"<b>Linked Case: {case_id}</b><br>Type: {primary_type}<br>Date: {case_date}"

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color=marker_color)
    ).add_to(m)

map_path = os.path.abspath("crime_linkage_map.html")
m.save(map_path)

print(f"Map exported to: {map_path}")
webbrowser.open("file://" + map_path)