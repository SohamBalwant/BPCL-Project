# anomaly detection notes

quick notes on how this thing works - mainly so i don't forget later

## what it does

- looks for weird patterns in distributor transactions
- uses dbscan + isolation forest to find outliers
- checks stuff like business hours, success rates, etc.

## main features we look at

- business hours activity (9am-5pm)
- success rate
- total transactions
- weekend %
- how much the pattern varies day to day
- avg transactions per day

## how dbscan works

- groups similar distributors together based on these features
- params:
  - eps = 0.8 (how close points need to be)
  - min_samples = 3 (need at least 3 nearby points)
- marks outliers with red X in the plot

## dbscan in detail (cause i keep forgetting)

- basically works like this:

  1. picks a random point
  2. draws a circle with radius eps around it
  3. if it finds min_samples points in this circle -> it's a core point
  4. keeps expanding from core points to find more connected points
  5. points that aren't connected to any cluster = anomalies
- like throwing rocks in a pond:

  - dense areas = ripples connect = one cluster
  - isolated points = no ripples connect = anomalies
- nice thing: doesn't need us to specify number of clusters
- annoying thing: kinda sensitive to eps and min_samples values

## reading the plots

- transaction patterns:

  - yellow area = business hours
  - each line = different day
  - spikes = lots of transactions
- cluster plot:

  - colored dots = normal distributors
  - red X = anomalies
  - uses pca to squash all features into 2d

## what makes something anomaly

- super concentrated activity in weird times
- really high/low success rates
- very different from day to day
- doesn't fit normal business patterns

reminder: need at least 10 transactions to analyze a distributor

note to self: might need to tweak eps and min_samples if getting too many/few anomalies
