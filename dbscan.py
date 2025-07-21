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