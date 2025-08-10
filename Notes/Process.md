## Process
1) Fixed the FWF by parsing it and converting it into a comma-separated CSV file.
2) Wrote code using seaborn to analyze data and see if something can be visually discerned from the data

## Anomaly Detection
According to Shashi's notes, I figured there are multiple areas we could look for anomalies in the given dataset:

- Distributor's Behavior Over Time
  A critical area to monitor is the distributor’s behavior, particularly the number of transactions they perform over time. By tracking this, we can identify any unusual spikes or irregularities in their activity. For example, a sudden increase in transactions by a distributor in a short time period, such as during a single day, might indicate fraudulent behavior or automated transactions. To detect these anomalies, time-based analysis can be used, looking for significant changes in transaction volumes or applying outlier detection techniques like standard deviation or interquartile range.

- Unusual Attempts During Off-Hours
  Another potential anomaly we can look for is transactions occuring outside of regular business hours, which is typically, and in our case, 9 A.M to 5 P.M. Large amounts of activity taking place during off-hours, especially late at night or on weekends, could be an indication of suspicious activity. By analyzing the distribution of transactions over different hours of the day, it is possible to flag any spikes outside of normal operating hours. This can be analyzed by visualizing the distribution of transactions across different hours of the day using a bar chart, where any spikes outside business hours stand out.

  To automate the detection of such anomalies, we can use machine learning models like Isolation Forest, One-Class SVM, or KMeans clustering from the scikit-learn library. These models help identify outliers by learning the patterns of normal activity and flagging transactions that deviate significantly from those patterns. KMeans Clustering can also be used to detect anomalies. By clustering the transactions based on features such as time of day, any points that don't fit into any of the clusters might be flagged as anomalies.

- Transaction Patterns For Individual  Customers
  Monitoring individual customer behavior is also essential. By observing the success and failure rates of eKYC transactions over time for each consumer, we can detect any anomalies. For instance, if a customer experiences repeated failures or a high failure rate over a period of time, this could be a sign of fraudulent attempts or system issues. By tracking the ratio of successful to failed transactions for each customer, anomalies such as a sudden increase in failure rates or changes in behavior can be flagged. Line plots showing success and failure trends for top customers can help visualize these patterns. However, due to the sheer size of the dataset, it is, currently, challenging to plot these trends.

