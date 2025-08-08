# Anomaly Detection Overview
These notes summarize how the anomaly detection module works, particularly for identifying irregular patterns in distributor transactions.

# Purpose
The system is designed to detect unusual transaction behavior among distributors. It leverages a combination of DBSCAN (Density-Based Spatial Clustering of Applications with Noise) and Isolation Forest to identify outliers.
The features analyzed include operational timing, transaction success rates, and consistency of activity.

## Key Features Analyzed
 - Business hours activity — transactions between 09:00 and 17:00
 - Success rate — proportion of successful       transactions
 - Total transaction volume — overall activity level
 - Weekend transaction percentage — proportion of activity occurring on weekends
 - Daily variation — fluctuations in transaction counts from day to day
 - Average transactions per day — overall baseline activity
 - Distributors with fewer than 10 total transactions are excluded from analysis to ensure statistical reliability.

## DBSCAN — How It Works
Concept: DBSCAN groups distributors with similar patterns based on feature similarity and flags points that don’t belong to any group as anomalies.

## Parameters Used:

- eps = 0.8 — maximum distance between two points for them to be considered neighbors

- min_samples = 3 — minimum number of neighboring points required to form a dense region

## Algorithm Steps:

1) Select an unvisited point at random.
2) Find all points within the eps radius    (neighborhood).
3) If at least min_samples points are found → mark as a core point.
4) Expand the cluster from this core point by recursively including all reachable points.
5) Points that cannot be assigned to any cluster are labeled as outliers.

# Analogy: Imagine dropping stones into a pond:
 - Areas where ripples overlap represent dense clusters.
 - Isolated ripples represent anomalies.
 - Pros: Does not require specifying the number of clusters in advance.
 - Cons: Sensitive to eps and min_samples — tuning is essential.

## Interpreting the Plots
# Transaction Pattern Plots:

 - Yellow shaded area = typical business hours
 - Each line = transaction activity for a specific day
 - Spikes indicate transaction bursts
 - Cluster Plots (via PCA projection):
 - Colored points = normal distributors
 - Red ✖ = anomalies
 - PCA reduces high-dimensional feature space to 2D for visualization

## What Constitutes an Anomaly
 - High concentration of transactions outside standard business hours
 - Extremely high or low success rates compared to peers
 - Significant variability in day-to-day transaction counts
 - Patterns inconsistent with typical distributor activity
 - Note: If too many or too few anomalies are detected, adjust eps and min_samples to calibrate sensitivity.

