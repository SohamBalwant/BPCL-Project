import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def preprocess_data(file_path):
    df = pd.read_csv(file_path, delimiter=",")

    # Strip spaces from all string values
    df = df.apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Remove rows where InsertedOn is '0'
    df = df[df["InsertedOn"] != "0"]

    df["InsertedOn"] = pd.to_datetime(df["InsertedOn"], errors="coerce")

    df = df.dropna(subset=["InsertedOn"])

    df["InsertedOnHour"] = df["InsertedOn"].dt.hour
    df["InsertedOnDate"] = pd.to_datetime(df["InsertedOn"])
    df["InsertedOnDayOfWeek"] = df["InsertedOnDate"].dt.day_name()

    df["LastUpdatedOn"] = pd.to_datetime(df["LastUpdatedOn"])
    df["LastUpdatedOnHour"] = df["LastUpdatedOn"].dt.hour
    df["LastUpdatedOnDate"] = pd.to_datetime(df["LastUpdatedOn"])
    df["LastUpdatedOnDayOfWeek"] = df["LastUpdatedOnDate"].dt.day_name()

    # Calculate TransactionCount
    df_count = df.groupby(["DistCode", "InsertedOnHour", "InsertedOnDayOfWeek"]).size().reset_index(name="TransactionCount")

    # Merge TransactionCount back to the original dataframe
    df = df.merge(df_count, on=["DistCode", "InsertedOnHour", "InsertedOnDayOfWeek"], how="left")

    return df



def plot_distributor_df(dfs, days_order):
    for df in dfs:
        df["InsertedOnDayOfWeek"] = pd.Categorical(
            df["InsertedOnDayOfWeek"], categories=days_order, ordered=True
        )
        dist_code = df["DistCode"].tolist()[0]

        # x axis hour of day, y-axis transaction count
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="InsertedOnHour", y="TransactionCount", data=df)
        sns.stripplot(
            x="InsertedOnHour",
            y="TransactionCount",
            data=df,
            jitter=True,
            color=".3",
            linewidth=1,
        )
        plt.title("Transaction Count by Hour of Day (Box and Individual Points)")
        plt.xlabel("Hour of Day")
        plt.ylabel("Transaction Count")
        plt.xticks(range(24))  # Ensure all hours are shown
        plt.grid(axis="y", alpha=0.3)  # Add subtle gridlines
        plt.tight_layout()
        plt.savefig(f"./graphs/hourofday/hod-{dist_code}-iqr.png")

        # x-axis day of week, y-axis transaction count
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="InsertedOnDayOfWeek", y="TransactionCount", data=df)
        sns.stripplot(
            x="InsertedOnDayOfWeek",
            y="TransactionCount",
            data=df,
            jitter=True,
            color=".3",
            linewidth=1,
        )
        plt.title("Transaction Count by Day of Week (Box and Individual Points)")
        plt.xlabel("Day of Week")
        plt.ylabel("Transaction Count")
        plt.grid(axis="y", alpha=0.3)  # Add subtle gridlines
        plt.tight_layout()  # Prevent labels from getting cut
        plt.savefig(f"./graphs/dayofweek/dow-{dist_code}-iqr.png")


def create_detailed_anomaly_analysis(plot_data, dist_code, days_order):
    """
    Create detailed visualization for a single distributor
    """
    dist_data = plot_data[plot_data['DistCode'] == dist_code].copy()
    
    # Calculate z-scores
    dist_data['zscore'] = (dist_data['TransactionCount'] - dist_data['TransactionCount'].mean()) / dist_data['TransactionCount'].std()
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3)
    
  
