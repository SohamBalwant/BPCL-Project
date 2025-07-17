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
     
    # 1. Transactions by Hour with Day of Week
    ax1 = fig.add_subplot(gs[0, 0])
    for day in days_order:
        day_data = dist_data[dist_data['InsertedOnDayOfWeek'] == day]
        ax1.scatter(day_data['InsertedOnHour'], day_data['TransactionCount'], 
                   label=day, alpha=0.5)
    ax1.set_title(f'Distributor {dist_code}: Hourly Transaction Patterns')
    ax1.set_xlabel('Hour of Day (0-23)')
    ax1.set_ylabel('Number of Transactions per Hour')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Distribution of Transactions with Anomalies
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(data=dist_data, x='TransactionCount', bins=30, color='gray', ax=ax2)
    sns.rugplot(data=dist_data[dist_data['is_anomaly']], x='TransactionCount', 
                color='red', ax=ax2, height=0.1)
    ax2.set_title('Distribution of Hourly Transaction Volumes\n(Red marks = Anomalies)')
    ax2.set_xlabel('Number of Transactions per Hour')
    ax2.set_ylabel('Frequency')
    
    # 3. Z-score plot
    ax3 = fig.add_subplot(gs[0, 2])
    sns.scatterplot(data=dist_data, x='zscore', y='TransactionCount', 
                    hue='is_anomaly', palette={True: 'red', False: 'blue'}, 
                    ax=ax3)
    ax3.set_title('Deviation from Normal Pattern\nRed = Anomalous Hours')
    ax3.set_xlabel('Standard Deviations from Mean')
    ax3.set_ylabel('Number of Transactions per Hour')
    
    
    # 4. Heatmap
    ax4 = fig.add_subplot(gs[1, :2])
    pivot_data = dist_data.pivot_table(
        values='TransactionCount', 
        index='InsertedOnDayOfWeek', 
        columns='InsertedOnHour', 
        aggfunc='mean'
    ).reindex(days_order)
    
    sns.heatmap(pivot_data, cmap='YlOrRd', ax=ax4, fmt='.0f',
                cbar_kws={'label': 'Average Transactions per Hour'})
    ax4.set_title('Average Hourly Transactions by Day of Week')
    ax4.set_xlabel('Hour of Day (0-23)')

    
    # 5. Summary statistics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Calculate peak hours
    peak_hour = dist_data.groupby('InsertedOnHour')['TransactionCount'].mean().idxmax()
    peak_day = dist_data.groupby('InsertedOnDayOfWeek')['TransactionCount'].mean().idxmax()
    
    summary_stats = [
        f"Distributor Code: {dist_code}",
        f"Total Transactions: {dist_data['TransactionCount'].sum():,}",
        f"Average Transactions per Hour: {dist_data['TransactionCount'].mean():.1f}",
        f"Busiest Hour: {peak_hour}:00",
        f"Busiest Day: {peak_day}",
        f"Maximum Transactions in One Hour: {dist_data['TransactionCount'].max()}",
        f"Number of Anomalous Hours: {dist_data['is_anomaly'].sum()}",
        f"Percent of Hours Flagged: {(dist_data['is_anomaly'].mean() * 100):.1f}%",
        "\nHow to Read This Analysis:",
        "1. Each point represents one hour's transactions",
        "2. Red points are unusual transaction patterns",
        "3. Heatmap shows typical patterns by day/hour",
        "4. Z-score shows how unusual each hour is",
        "\nAnomalies are flagged when:",
        "• Transaction count is unusually high/low for that time",
        "• Pattern doesn't match typical day/hour pattern",
        "• Volume deviates significantly from average"
    ]
    ax5.text(0, 1, '\n'.join(summary_stats), va='top', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def analyze_anomalies(file_path, top_n=20):
    """
    Main function to analyze anomalies in transaction data
    """
    # Preprocessing
    df = preprocess_data(file_path)
    df_hour = (
        df.groupby(["DistCode", "InsertedOnHour", "InsertedOnDayOfWeek"])
        .size()
        .reset_index(name="TransactionCount")
    )

    print(df_hour)
    
  # Get top distributors
    top_dists = df_hour.groupby('DistCode')['TransactionCount'].sum() \
                       .sort_values(ascending=False).head(top_n).index
    
    # Prepare data
    plot_data = df_hour[df_hour['DistCode'].isin(top_dists)].copy()
    
    # Add features
    days_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    
    plot_data["HourSin"] = np.sin(2 * np.pi * plot_data["InsertedOnHour"] / 24)
    plot_data["HourCos"] = np.cos(2 * np.pi * plot_data["InsertedOnHour"] / 24)
    plot_data["DayNum"] = plot_data["InsertedOnDayOfWeek"].map(
        lambda x: days_order.index(x)
    )

    # Detect anomalies for each distributor
    features = ['TransactionCount', 'HourSin', 'HourCos', 'DayNum']
    anomalies_by_dist = {}
    
    for dist in top_dists:
        dist_data = plot_data[plot_data['DistCode'] == dist].copy()
        X = dist_data[features]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso_forest.fit_predict(X_scaled)
        plot_data.loc[plot_data['DistCode'] == dist, 'is_anomaly'] = predictions == -1
        
        # Store anomalies
        anomalies = dist_data[predictions == -1].copy()
        anomalies_by_dist[dist] = anomalies

    # 1. Overview Plot (Violin + Strip plot)
    plt.figure(figsize=(15, 8))
    sns.violinplot(data=plot_data, x='DistCode', y='TransactionCount', 
                  color='lightgray', inner=None)
    sns.stripplot(data=plot_data, x='DistCode', y='TransactionCount',
                 hue='is_anomaly', palette={False: 'blue', True: 'red'},
                 size=4, alpha=0.6)
    plt.xticks(rotation=45)
    plt.title('Transaction Distribution by Distributor with Anomalies')
    plt.xlabel('Distributor Code')
    plt.ylabel('Transaction Count')
    plt.tight_layout()
    plt.show() 

# 2. Detailed Analysis for Top 5 Distributors
    for dist_code in list(top_dists)[:5]:  # Analyze top 5 distributors
        create_detailed_anomaly_analysis(plot_data, dist_code, days_order)

    return anomalies_by_dist, plot_data


def flag_anomalies_isolation_forest(top_n=20):
    df = preprocess_data("./cleaned-failure.csv")

    # # df_date = df.groupby(['DistCode', 'InsertedOnDate']).size().reset_index(name='TransactionCount')
    df_hour = (
        df.groupby(["DistCode", "InsertedOnHour", "InsertedOnDayOfWeek"])
        .size()
        .reset_index(name="TransactionCount")
    )

    total_transactions = (
        df_hour.groupby("DistCode")["TransactionCount"].sum().reset_index()
    )

    sorted_transactions = total_transactions.sort_values(
        by="TransactionCount", ascending=False
    )

    dist_dfs = []
    for index, row in sorted_transactions.iterrows():
        dist_code = row["DistCode"]
        rd_df = df_hour[df_hour["DistCode"] == dist_code]
        dist_dfs.append(rd_df)
        if len(dist_dfs) == 10:
            break

    days_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    df_hour["HourSin"] = np.sin(2 * np.pi * df_hour["InsertedOnHour"] / 24)
    df_hour["HourCos"] = np.cos(2 * np.pi * df_hour["InsertedOnHour"] / 24)
    df_hour["DayNum"] = df_hour["InsertedOnDayOfWeek"].map(
        lambda x: days_order.index(x)
    )

    plot_distributor_df(dist_dfs, days_order)

    print(df_hour)

    # get top distributors
    top_dists = df_hour.groupby('DistCode')['TransactionCount'].sum() \
                       .sort_values(ascending=False).head(top_n).index
    
    # filter the data
    plot_data = df_hour[df_hour['DistCode'].isin(top_dists)].copy()
    
    # features wed use
    features = ['TransactionCount', 'HourSin', 'HourCos', 'DayNum']
    
    # detect anomalies for each distributor
    for dist in top_dists:
        dist_data = plot_data[plot_data['DistCode'] == dist]
        X = dist_data[features]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        plot_data.loc[plot_data['DistCode'] == dist, 'is_anomaly'] = \
            iso_forest.fit_predict(X_scaled) == -1
    
    
    plt.figure(figsize=(15, 8))
    
    # violin plot for distribution
    sns.violinplot(data=plot_data, x='DistCode', y='TransactionCount', 
                  color='lightgray', inner=None)
    
    # Overlay strip plot with color-coded anomalies
    sns.stripplot(data=plot_data, x='DistCode', y='TransactionCount',
                 hue='is_anomaly', palette={False: 'blue', True: 'red'},
                 size=4, alpha=0.6)
    
    plt.xticks(rotation=45)
    plt.title('Transaction Distribution by Distributor with Anomalies')
    plt.xlabel('Distributor Code')
    plt.ylabel('Transaction Count')
    plt.tight_layout()
    plt.show()


# analyze_anomalies("./cleaned-failure.csv")
flag_anomalies_isolation_forest()

