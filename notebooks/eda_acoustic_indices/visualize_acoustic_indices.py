""" Visualize acoustic indices """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load general data
media = pd.read_csv("../../data_preparation/media.csv")[["mediaID", "deploymentID", "timestamp"]]
deployments = pd.read_csv("../../data_preparation/deployments.csv")[["deploymentID", "locationID"]]
# remove deployments PAH08 and PAH24
deployments = deployments[~deployments["deploymentID"].isin(["PAH08", "PAH24"])]

#%% Section 1: Visualize acoustic indices for a selected deployment
# Plot a selected deployment and acoustic index
deployment = "PAH05"
indx = "ACI"  # Acoustic index to visualize
df = pd.read_csv(f"../../acoustic_indices/indices_{deployment}.csv")
df = df.merge(media, on="mediaID").merge(deployments, on="deploymentID")

# Plot value per day
plt.figure(figsize=(10, 5))
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df[indx].resample("D").mean().plot()
plt.title(f"{indx} for deployment {deployment}")
plt.xlabel("Date")
plt.ylabel(indx)
plt.show()

# Plot value per hour of the day with variance using seaborn
plt.figure(figsize=(10, 5))
df["hour"] = df.index.hour
sns.violinplot(x="hour", y=indx, data=df)
plt.title(f"{indx} by hour of the day for deployment {deployment}")
plt.xlabel("Hour of the day")
plt.ylabel(indx)
plt.show()

#%% Section 2: Compare acoustic indices across multiple deployments
deployments_to_compare = deployments["deploymentID"]
indx = "NDSI"  # Acoustic index to visualize

# Load and concatenate data for selected deployments
dfs = []
for dep in deployments_to_compare:
    df = pd.read_csv(f"../../acoustic_indices/indices_{dep}.csv")
    df = df.merge(media, on="mediaID").merge(deployments, on="deploymentID")
    df["deploymentID"] = dep  # Add deployment ID for comparison
    dfs.append(df)
combined_df = pd.concat(dfs)

# Transform RMS values into decibels (dB) for better visualization

combined_df["RMS_dB"] = 20 * np.log10(combined_df["RMS"] + 1e-6)  # Add small value to avoid log(0)


# Plot boxplot comparing the selected acoustic index across deployments ordered by median value. Do not show outliers.
plt.figure(figsize=(10, 5))
order = combined_df.groupby("locationID")[indx].median().sort_values().index
sns.boxplot(x="locationID", y=indx, data=combined_df, order=order, showfliers=False)
plt.title(f"Comparison of {indx} across deployments")
plt.xlabel("Location ID")
plt.ylabel(indx)
plt.xticks(rotation=45)
plt.show()

# Plot correlation heatmap for all acoustic indices across deployments
acoustic_indices = [col for col in combined_df.columns if col not in ["mediaID", "deploymentID", "timestamp", "locationID"]]
plt.figure(figsize=(12, 10))
corr = combined_df[acoustic_indices].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Acoustic Indices")
plt.show()