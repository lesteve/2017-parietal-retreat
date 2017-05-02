is_valid flag may be used as a proxy for outliers.
Another way is to train on functional data and see whether we detect
anatomical images as outliers.

ReNA to cluster in the features direction. Try with 2000 first and
then with more.

IsolationForest on ReNA clustering.

Birch (online which is good) in the samples space + LocalOutlierFactor

Not necesarily a good idea to use a masker. Maybe do a clustering of
2000 cluster for the inbrain data and 200 for the out-of-brain data.

Application goal: feedback for people put the right flag when
uploading data on NeuroVault.
