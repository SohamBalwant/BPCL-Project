import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def preprocess_data(file_path):
    # read csv file into dataframe
    df = pd.read_csv(file_path, delimiter=",")

    # strip spaces from string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # convert timestamps to datetime and handle errors
    df["InsertedOn"] = pd.to_datetime(df["InsertedOn"], errors="coerce")
    df["LastUpdatedOn"] = pd.to_datetime(df["LastUpdatedOn"], errors="coerce")

    # drop rows with invalid timestamps
    df = df.dropna(subset=["InsertedOn", "LastUpdatedOn"])

    # extract datetime features
    df["InsertedOnHour"] = df["InsertedOn"].dt.hour
    df["InsertedOnDayOfWeek"] = df["InsertedOn"].dt.day_name()
    df["InsertedOnWeek"] = df["InsertedOn"].dt.isocalendar().week

    # group by distributor and datetime features to get transaction count
    df_count = df.groupby(["DistCode", "InsertedOnHour", "InsertedOnDayOfWeek"]).size().reset_index(name="TransactionCount")

    # merge transaction count back to the original dataframe
    df = df.merge(df_count, on=["DistCode", "InsertedOnHour", "InsertedOnDayOfWeek"], how="left")

    return df

def calculate_expected_pattern(df, weeks):
    """calculate expected transaction patterns."""
    if weeks:
        previous_weeks_data = df[df["InsertedOnWeek"].isin(weeks)]
        expected_pattern = previous_weeks_data.groupby("InsertedOnHour")["TransactionCount"].mean()
    else:
        # default bell curve pattern for hours
        hours = np.arange(24)
        mu, sigma = 13, 4  # peak at 1 PM
        expected_pattern = np.exp(-((hours - mu) ** 2) / (2 * sigma ** 2))
        expected_pattern = pd.Series(expected_pattern, index=hours)

    return expected_pattern / expected_pattern.max()  # normalize pattern

def detect_individual_anomalies(df):
    """detect individual anomalous transactions."""
    df["OffHours"] = ~df["InsertedOnHour"].between(8, 18)

    # identify potential off-hour spikes
    off_hour_transactions = df[df["OffHours"]]

    # standardize transaction counts for anomaly detection
    features = off_hour_transactions[["TransactionCount"]]
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    off_hour_transactions["is_anomaly"] = iso_forest.fit_predict(X) == -1

    anomalies = off_hour_transactions[off_hour_transactions["is_anomaly"]]

    return anomalies

def detect_anomalies(df, top_n=5):
    """detect anomalies in transaction patterns."""
    # get top distributors by total transaction count
    top_dists = df.groupby("DistCode")["TransactionCount"].sum().nlargest(top_n).index

    anomalies = []

    for dist_code in top_dists:
        dist_data = df[df["DistCode"] == dist_code]
        weeks = sorted(dist_data["InsertedOnWeek"].unique())

        for week in weeks:
            week_data = dist_data[dist_data["InsertedOnWeek"] == week]
            prev_weeks = weeks[:weeks.index(week)]

            expected_pattern = calculate_expected_pattern(dist_data, prev_weeks)

            for day in week_data["InsertedOnDayOfWeek"].unique():
                day_data = week_data[week_data["InsertedOnDayOfWeek"] == day]
                
                actual_pattern = day_data.groupby("InsertedOnHour")["TransactionCount"].sum()
                actual_pattern = actual_pattern / actual_pattern.max() if actual_pattern.max() != 0 else 0

                # calculate deviation for business hours
                business_hours = np.arange(8, 18)
                actual_values = actual_pattern.reindex(business_hours, fill_value=0).to_numpy()
                expected_values = expected_pattern.reindex(business_hours, fill_value=0).to_numpy()

                deviation = np.mean(np.abs(actual_values - expected_values))

                # detect anomalies with Isolation Forest
                features = np.column_stack([actual_values, expected_values, np.abs(actual_values - expected_values)])
                scaler = StandardScaler()
                X = scaler.fit_transform(features)

                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                is_anomaly = iso_forest.fit_predict(X) == -1

                if is_anomaly.any():
                    anomalies.append((dist_code, week, day, deviation))

    return anomalies

def plot_anomalies_with_context(df, anomalies):
    """plot transaction patterns with highlighted anomalies and context."""
    num_anomalies = len(anomalies)
    rows = (num_anomalies // 2) + (num_anomalies % 2 > 0)  # two plots per row
    cols = min(2, num_anomalies)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
    axes = axes.flatten() if num_anomalies > 1 else [axes]

    for idx, anomaly in enumerate(anomalies):
        dist_code, week, day, _ = anomaly
        ax = axes[idx]

        # filter data for specific distributor and week
        week_data = df[(df["DistCode"] == dist_code) & (df["InsertedOnWeek"] == week)]

        # plot all days for context
        for other_day in week_data["InsertedOnDayOfWeek"].unique():
            day_data = week_data[week_data["InsertedOnDayOfWeek"] == other_day]
            actual_pattern = day_data.groupby("InsertedOnHour")["TransactionCount"].sum()
            actual_pattern = actual_pattern / actual_pattern.max() if actual_pattern.max() != 0 else 0
            color = "gray" if other_day != day else "blue"
            label = f"{other_day} (Normal)" if other_day != day else f"{day} (Anomaly)"
            ax.plot(actual_pattern.index, actual_pattern.values, label=label, linestyle="-" if other_day != day else "--")

        # highlight anomalies within the anomalous day
        anomalous_day_data = week_data[week_data["InsertedOnDayOfWeek"] == day]
        anomalous_pattern = anomalous_day_data.groupby("InsertedOnHour")["TransactionCount"].sum()
        anomalous_hours = anomalous_pattern.index[anomalous_pattern > anomalous_pattern.mean() * 1.5]  # example threshold
        ax.scatter(anomalous_hours, anomalous_pattern.loc[anomalous_hours] / anomalous_pattern.max(), color="red", zorder=5, label="Anomalous Points")

        # add expected pattern
        expected_pattern = calculate_expected_pattern(df[df["DistCode"] == dist_code], [week])
        ax.plot(expected_pattern.index, expected_pattern.values, label="Expected Pattern", linestyle="--", color="black")

        ax.set_title(f"Distributor {dist_code} - Week {week}")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Normalized Transaction Count")
        ax.legend()

    # hide unused subplots
    for ax in axes[len(anomalies):]:
        ax.set_visible(False)

    plt.suptitle("Anomalous Transaction Patterns with Context", fontsize=16)
    plt.savefig("anomalies.png")

def main():
    file_path = "./cleaned-failure.csv"
    df = preprocess_data(file_path)

    # detect individual transaction anomalies
    individual_anomalies = detect_individual_anomalies(df)
    print("Individual anomalous transactions:")
    print(individual_anomalies)

    # detect overall pattern anomalies
    anomalies = detect_anomalies(df)
    print("\nDetected pattern anomalies:")
    for anomaly in anomalies:
        print(f"Distributor {anomaly[0]} - Week {anomaly[1]} - {anomaly[2]}: Deviation {anomaly[3]:.2f}")

    # plot anomalies
    plot_anomalies_with_context(df, anomalies)

if __name__ == "__main__":
    main()