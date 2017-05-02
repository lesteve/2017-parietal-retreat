# test summaries
- Metric:
   - **is_valid** flag: _ `is_valid` flag may be used as a proxy for outliers._
   - mix different image modalities: If we learn using functional imagery and
     test using a mix of functional and anatomical volumes, we should we detect
     anatomical images as outliers.

- Feature direction:
   - **downsampling**: use an **NMI mask** to create a coarse resolution of the images
   - **ReNA** to cluster in the features direction. Try with 2000 first and
then with more.

- Samples direction:
    - IsolationForest (scales aproximatly linear with samples)
    - LOF
    - One-class SVM

- Notes:
   LOF and One-class SVM don't scale properly so we'll add Brich before running
   the anomaly detector

# Experiments
1 - ReNA + One-class SVM (using kmeans)                - Mathurin
2 - downsampling + lof(using brich)                    - Sik

# some title

IsolationForest on ReNA clustering.

Birch (online which is good) in the samples space + LocalOutlierFactor

Not necesarily a good idea to use a masker. Maybe do a clustering of
2000 cluster for the inbrain data and 200 for the out-of-brain data.

Application goal: feedback for people put the right flag when
uploading data on NeuroVault.

# Building a dataset to test stuff

This loads an valid dataset (but it only has 34 images)

```
dd = datasets.fetch_neurovault(max_images=None, mode='offline', image_terms={'collection_id':656})
```
