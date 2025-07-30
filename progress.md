



## Anomaly Detection in eKYC Transactions - Project Report

**1. Introduction**

This project focuses on developing an anomaly detection system for electronic Know Your Customer (eKYC) transactions. eKYC is a crucial process for verifying the identity of customers remotely, and this project aims to enhance the security and reliability of this process by identifying potentially fraudulent or problematic activity. The data used in this project is a record of transactions utilizing eKYC for verification. Each record captures details about the transaction, including the distributor involved, the time of the transaction, the status of the eKYC process, and other relevant information. 

**2. Data Preprocessing**

The raw data was provided in Fixed File Width (FWF) format. The initial step involved converting this data into a comma-separated CSV file to facilitate easier processing and analysis using standard data science tools.

**3. Data Analysis and Initial Findings**

Initial exploratory data analysis involved visualizing transaction counts by hour of day using box plots. This revealed the presence of statistical outliers, suggesting potential anomalies that warranted further investigation. These outliers were analyzed in relation to features we wanted to look for. For example, the "Transaction Count by Hour of Day" box plot, where each data point represents a day, shows outliers beyond the upper whisker for several hours (e.g., hours 8, 9, 11, 12, and 13). This suggests that on some days, significantly more transactions occurred during these hours than typically expected.

![1736135792179](image/progress/1736135792179.png)

**4. Anomaly Detection Techniques and Implementation**

The anomaly detection strategy addresses the following key areas:

1. **Distributor Behavior Over Time:** Monitoring transaction volumes per distributor to identify unusual spikes or irregularities in their activity. This aims to detect instances where a distributor processes a significantly higher or lower number of transactions than their usual pattern.
2. **Off-Hours Activity:** Identifying transactions occurring outside typical business hours (9 AM to 5 PM). This helps detect potentially unauthorized or automated activity taking place during non-operational periods.

These strategies are implemented using a Python script leveraging several key libraries: Pandas for data manipulation, scikit-learn for anomaly detection (specifically the Isolation Forest algorithm and StandardScaler for feature scaling), NumPy for numerical operations, and Matplotlib for visualization. The script's core functionality is encapsulated in the following functions:

* `preprocess_data(file_path)`: This function reads the transaction data from a CSV file specified by `file_path`. It performs several crucial preprocessing steps:
  * Strips whitespace from string columns for consistency.
  * Converts the "InsertedOn" and "LastUpdatedOn" columns to datetime objects, handling any parsing errors.
  * Removes rows with invalid timestamps.
  * Extracts relevant time-based features such as the hour of the transaction (`InsertedOnHour`), day of the week (`InsertedOnDayOfWeek`), and week of the year (`InsertedOnWeek`).
  * Calculates the transaction count for each distributor for each hour of each day of the week and merges this count back into the main DataFrame.
* `calculate_expected_pattern(df, weeks)`: This function calculates the expected transaction pattern for a given distributor. If historical data from previous weeks (specified by the `weeks` parameter) is available, it calculates the average transaction count for each hour based on this historical data. If no historical data is available (i.e., `weeks` is empty), it uses a default bell curve (Gaussian distribution) centered around 1 PM (hour 13) with a standard deviation of 4 hours. This bell curve serves as a baseline expectation in the absence of historical data. The resulting pattern is then normalized to a maximum value of 1.
* `detect_individual_anomalies(df)`: This function detects individual transactions that occur outside of business hours (8 AM to 6 PM inclusive). It first flags transactions occurring during off-hours. Then, it uses the Isolation Forest algorithm to identify anomalies within these off-hour transactions. The transaction counts are standardized using `StandardScaler` before being fed to the Isolation Forest model. The `contamination` parameter of the Isolation Forest is set to 0.05, meaning it expects approximately 5% of the data to be anomalies.
* `detect_anomalies(df, top_n=5)`: This function detects anomalies in the overall transaction patterns for the top `n` distributors (defaulting to 5). It iterates through each of these distributors and each week of data. For each week and day, it calculates the actual transaction pattern by grouping transactions by hour. It then calculates the deviation between the actual pattern and the expected pattern (calculated using `calculate_expected_pattern`). The actual and expected values, along with their absolute difference are scaled and used as features for the Isolation Forest to detect anomalous patterns.
* `plot_anomalies_with_context(df, anomalies)`: This function generates visualizations of the detected pattern anomalies. For each anomaly, it creates a plot showing the actual transaction pattern for the anomalous day (dashed colored line), the actual patterns for other days of the same week (solid gray lines), the expected transaction pattern (dashed gray line), and highlights anomalous points within the anomalous day (red dots).

**5. Key Points and Clarifications:**

* **Bell Curve Assumption:** The use of a default bell curve for expected patterns when historical data is limited is acknowledged as a strong assumption. The code prioritizes using historical data when available.
* **Isolation Forest:** Isolation Forest was chosen for its unsupervised nature, ability to handle high-dimensional data, and computational efficiency.
* **Threshold Tuning:** The need for threshold tuning (e.g., the 1.5x mean threshold for anomalous points in plots) and the potential variability of "business hours" are noted.
* **Visualizations:** The plots now provide context by showing both anomalous and normal days within a week, along with the expected pattern.

**6. Example Output and Visualization Explanation:**

The generated plots display the normalized transaction count per hour. Each detected anomaly (distributor, week, day) has its own plot.

![1736135621410](image/progress/1736135621410.png)

Each plot shows:

* A dashed colored line representing the actual transaction pattern for the anomalous day.
* Solid gray lines representing the actual transaction patterns for other (normal) days of the same week.
* A dashed gray line representing the expected transaction pattern.
* Anomalous points within the anomalous day are highlighted in red.

**7. Data Description**

The dataset contains records of eKYC transactions, with the following key fields (based on the provided sample):

* `eKYCTransactionID`: A unique identifier for each eKYC transaction attempt.
* `DistCode`: A unique 6-digit code identifying the distributor involved in the transaction.
* `CreatedBy`: The user who initiated the eKYC process. This field can identify different types of users, including distributors, sub-users (indicated by an “XX” after the distributor code), and eKYC operators (indicated by “EO”).
* `InsertedOn`: The timestamp of when the eKYC transaction was initiated.
* `LastUpdatedOn`: The timestamp of the last status update for the eKYC transaction.
* `eKYCStatus`: The status of the eKYC process (0: in progress, 1 or 2: successful, 3: failed).
* `AppID`: The source of the eKYC transaction (1: LPG Next portal, 6: consumer-facing app (currently disabled), 7: eKYC operator app).
* `ConsumerID`: The ID of the consumer for whom the eKYC is being performed.

This data allows us to analyze transaction patterns at different levels (distributor, time of day, day of week) and identify deviations that may indicate anomalous activity.
