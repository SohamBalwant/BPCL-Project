### Anomaly Detection in Transaction Data

#### Overview

This README outlines the method used for detecting anomalies in transactional data, with a primary focus on identifying irregularities in transaction patterns across various distributors, weeks, and days. The goal is to flag deviations, such as spikes outside of business hours or anomalies in transaction volumes that differ significantly from expected behavior.

#### Approach

The approach to anomaly detection is structured around several steps: data preprocessing, pattern identification, anomaly detection, and visualization.

First, the data undergoes preprocessing, where the transactional data is loaded from a CSV file. During this stage, basic filtering is done so that the data is easy to work with. Timestamps are then processed to extract relevant features such as the hour of the transaction, day of the week, and week of the year. These features serve as the foundation for understanding transaction patterns.

Next, we move to identifying transaction patterns. Historical data for each distributor is used to establish an expected transaction pattern. In cases where historical data is insufficient, a default pattern based on a gaussian distribution* is applied, centered around the peak transaction hour. Actual transaction patterns are then derived by grouping transactions by hour for specific days. This allows for a comparison between expected and actual transaction behavior.

**This is a heavy assumption, since general trends for transactions will rarely exactly follow the normal distribution pattern*

For anomaly detection, two types of anomalies are addressed: individual transaction anomalies and pattern anomalies. Individual anomalies are identified when transactions occur outside typical business hours (8 AM to 6 PM). To detect these anomalies, we use the Isolation Forest algorithm, a powerful unsupervised learning method. Prior to applying this model, features are standardized to ensure consistent scaling. For pattern anomalies, we calculate the deviation between actual and expected transaction patterns for each distributor and week. The average deviation during business hours is calculated, and the Isolation Forest model is applied again to detect significant deviations from the expected patterns.

The results are presented visually by plotting transaction patterns, with both expected and actual values shown for each distributor and week. Anomalous points are highlighted in red, making it easy to identify specific hours or days contributing to the anomaly. Normal days are also included in the plots for context, providing a clearer comparison between normal and anomalous behavior. To not make the plots messy, only the **Top 5** distributors are plotted. Furthermore, each anomalous day has its own graph for the week, making it significantly easier to read:

[![1735828540672](image/readme/1735828540672.png)]()

#### Why Isolation Forest?

Isolation Forest is chosen due to its advantages in anomaly detection tasks. As an unsupervised learning algorithm, it does not require labeled data, making it ideal for this task, where labeled anomaly data is often unavailable. Since our use-case was identifying intrinsic properties within the data that could be possible anomalies/statistical outliers, it makes Isolation Forest an ideal choice. Its ability to handle high-dimensional data makes it effective for transactional datasets, where multiple features are involved. Additionally, Isolation Forest is computationally efficient and scales well with large datasets, ensuring that it remains performant even with extensive transaction records.

#### Limitations

* **Threshold Tuning**: One limitation of this methodology is the need for threshold tuning. For example, the threshold for detecting anomalies (e.g., 1.5x the mean for deviations) is fixed and might need adjustments based on domain-specific insights. Additionally, the definition of "business hours" (8 AM to 6 PM) may not be universally applicable and could vary depending on the context.
* **Insufficient Historical Data**: For distributors or weeks with limited historical data, the model may rely on the default bell curve for pattern estimation, which may not be accurate. This can lead to a higher risk of false positives or missed anomalies if the default pattern is not representative of actual transaction behavior.