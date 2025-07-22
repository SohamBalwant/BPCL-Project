import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class AnomalyDetector:
    def __init__(self, eps=0.8, min_samples=3, contamination=0.01):
        self.eps = eps
        self.min_samples = min_samples
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

    def prepare_features(self, df):
        """Extract features for each distributor with improved metrics."""
        distributor_features = []

        for dist_code in df["DistCode"].unique():
            if dist_code == 0:
                continue

            dist_data = df[
                df["DistCode"] == dist_code
            ].copy()  # Make a copy of the data

            # Skip distributors with too few transactions
            if len(dist_data) < 5:
                continue

            dist_data["InsertedOn"] = pd.to_datetime(
                dist_data["InsertedOn"], errors="coerce"
            )

            # Calculate business hours percentage
            business_hours = dist_data[dist_data["InsertedOn"].dt.hour.between(9, 17)]
            business_hours_pct = len(business_hours) / len(dist_data) * 100

            # Calculate success rates
            success_rate = (
                len(dist_data[dist_data["eKYCStatus"].isin([1, 2])])
                / len(dist_data)
                * 100
            )

            # Calculate variance in daily patterns
            daily_patterns = []
            for day in range(7):
                day_data = dist_data[dist_data["InsertedOn"].dt.dayofweek == day]
                hourly_pattern = (
                    day_data.groupby(day_data["InsertedOn"].dt.hour)
                    .size()
                    .reindex(range(24), fill_value=0)
                )
                if len(day_data) > 0:
                    hourly_pattern = hourly_pattern / len(day_data) * 100
                daily_patterns.append(hourly_pattern)
            pattern_variance = np.var([pattern.sum() for pattern in daily_patterns])

            features = {
                "dist_code": dist_code,
                "business_hours_pct": business_hours_pct,
                "success_rate": success_rate,
                "transactions": len(dist_data),
                "weekend_pct": len(
                    dist_data[dist_data["InsertedOn"].dt.dayofweek.isin([5, 6])]
                )
                / len(dist_data)
                * 100,
                "pattern_variance": pattern_variance,
                "avg_daily_transactions": len(dist_data)
                / len(dist_data["InsertedOn"].dt.date.unique()),
                "unique_days": len(dist_data["InsertedOn"].dt.date.unique()),
            }

            distributor_features.append(features)

        return pd.DataFrame(distributor_features)
    
     def detect_anomalies(self, df):
        """Detect anomalies with DBSCAN and Isolation Forest."""
        feature_df = self.prepare_features(df)
        feature_df = feature_df[feature_df["transactions"] >= 10]
        dist_codes = feature_df["dist_code"]

        # Select relevant features
        features_for_detection = [
            "business_hours_pct",
            "success_rate",
            "transactions",
            "weekend_pct",
            "pattern_variance",
            "avg_daily_transactions",
        ]
        
        X = self.scaler.fit_transform(feature_df[features_for_detection])

        # DBSCAN
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        dbscan_labels = dbscan.fit_predict(X)

        # Isolation Forest
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        iso_forest_labels = iso_forest.fit_predict(X)

        # Combine anomalies
        anomalies = dist_codes[
            (dbscan_labels == -1) & (iso_forest_labels == -1)
        ].tolist()

        results = {
            "feature_matrix": X,
            "pca_matrix": self.pca.fit_transform(X),
            "dist_codes": dist_codes,
            "dbscan_labels": dbscan_labels,
            "iforest_labels": iso_forest_labels,
            "anomalies": anomalies,
            "feature_df": feature_df,
            "features_for_detection": features_for_detection,
        }

        return results

    def visualize_results(self, results):
        """Visualize DBSCAN clusters and PCA scatter plots."""
        X_pca = results["pca_matrix"]
        anomalies = results["anomalies"]
        dbscan_labels = results["dbscan_labels"]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot DBSCAN clusters
        unique_labels = np.unique(dbscan_labels)
        print(f"Unique DBSCAN labels: {unique_labels}")
        print(f"Number of points in each cluster:")

        for label in unique_labels:
            print(f"Label {label}: {np.sum(dbscan_labels == label)}")
            if label == -1:
                # Plot anomalies (noise points)
                ax.scatter(
                    X_pca[dbscan_labels == label, 0],
                    X_pca[dbscan_labels == label, 1],
                    c="red",
                    label="Anomalies",
                    marker="x",
                )
            else:
                # Plot clusters
                ax.scatter(
                    X_pca[dbscan_labels == label, 0],
                    X_pca[dbscan_labels == label, 1],
                    label=f"Cluster {label}",
                )
        print(f"eps: {self.eps}, min_samples: {self.min_samples}")

        ax.set_title("DBSCAN Clustering with Anomalies")
        ax.set_xlabel("First Principal Component")
        ax.set_ylabel("Second Principal Component")
        ax.legend()
        plt.savefig("dbscan-results.png")
        plt.show()
        return fig

    def generate_anomalous_report(self, df, anomalies, results):
        """Generate detailed report for anomalous distributors with specific anomalous features."""
        report = []

        features_for_detection = results["features_for_detection"]

        for dist_code in anomalies:
            dist_data = df[df["DistCode"] == dist_code]
            dist_data["InsertedOn"] = pd.to_datetime(dist_data["InsertedOn"], errors="coerce")
            dist_data["week"] = dist_data["InsertedOn"].dt.isocalendar().week
            dist_data["day_of_week"] = dist_data["InsertedOn"].dt.dayofweek
            dist_data["hour"] = dist_data["InsertedOn"].dt.hour

            # Get the distributor's feature values
            dist_features = results["feature_df"][results["feature_df"]["dist_code"] == dist_code]

            # Identify anomalous features
            anomalous_features = []
            for feature in features_for_detection:
                # Calculate z-score
                z_score = (
                    dist_features[feature] - results["feature_df"][feature].mean()
                ) / results["feature_df"][feature].std()

                # Handle success rate separately
                if feature == "success_rate":
                    # Only flag low success rates as anomalous (below 70%)
                    if dist_features[feature].values[0] < 70:
                        if abs(z_score.values[0]) > 2:  # If z-score is extreme for low success rates
                            anomalous_features.append(
                                f"{feature}: {dist_features[feature].values[0]} (z-score: {z_score.values[0]:.2f})"
                            )
                else:
                    # Flag other features as anomalous based on the z-score
                    for score in z_score:
                        if abs(score) > 2:  # Threshold for anomaly (adjust as needed)
                            anomalous_features.append(
                                f"{feature}: {dist_features[feature].values[0]} (z-score: {score:.2f})"
                            )

            # Filter out anomalies flagged only for success_rate
            if len(anomalous_features) == 1 and "success_rate" in anomalous_features[0]:
                continue  # Skip this entry if it is flagged only for success_rate

            # Extract anomalous time periods (if applicable)
            anomalous_times = None
            if "business_hours_pct" in anomalous_features:
                if dist_features["business_hours_pct"].values[0] > 90:
                    anomalous_times = "High activity outside of business hours"
                else:
                    anomalous_times = "Low activity during business hours"
            elif "weekend_pct" in anomalous_features:
                if dist_features["weekend_pct"].values[0] > 50:
                    anomalous_times = "High activity on weekends"
                else:
                    anomalous_times = "Low activity on weekdays"

            report.append(
                {
                    "DistCode": dist_code,
                    "AnomalousFeatures": ", ".join(anomalous_features),
                    "AnomalousTimes": anomalous_times,
                }
            )

        return pd.DataFrame(report)
