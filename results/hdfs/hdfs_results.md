# HDFS_v1 results (real data)

Data: Loghub HDFS_v1 (Xu et al., SOSP 2009). Blocks are ordered by their first log line; detectors are fitted on the first half without labels and evaluated on the second half against the real block labels.

## Parsing

- Log lines: 11,175,629 (0 malformed)
- Templates mined: 54
- Parsing time: 52 s (single process)
- Grouping accuracy on Loghub's hand-labelled HDFS_2k sample: 0.9975 (16 templates mined, 14 true events)

## Blocks

- Total: 575,061 (16,838 anomalous, 2.93%)
- Training half: 287,530 (10,170 anomalous; labels not used)
- Test half: 287,531 (6,668 anomalous)

## Detection on the test half

| Detector | Precision | Recall | F1 | ROC AUC | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCA (TF-IDF, 95% variance, alpha=0.001) | 0.616 | 0.084 | 0.148 | 0.689 | 562 | 351 | 6,106 |
| PCA (raw counts, 95% variance, alpha=0.001) | 0.010 | 0.148 | 0.018 | 0.655 | 990 | 102,932 | 5,678 |
| Robust z-score (max |z| > 3.5) | 0.100 | 0.006 | 0.011 | 0.712 | 39 | 351 | 6,629 |

## PCA protocol study

Precision / recall / F1 (k = number of principal components kept). The first group of rows uses the same split as above; "all blocks" fits and evaluates on every block, as earlier HDFS studies did; the semi-supervised row uses the training labels to drop anomalous blocks before fitting.

| Protocol | this parser | Loghub reference events |
|---|---:|---:|
| Unsupervised, temporal split, 90% variance | 0.997 / 0.682 / **0.810** (k=4) | 0.997 / 0.682 / **0.810** (k=4) |
| Unsupervised, temporal split, 95% variance | 0.616 / 0.084 / **0.148** (k=7) | 0.993 / 0.274 / **0.429** (k=6) |
| Unsupervised, temporal split, 99% variance | 0.095 / 0.026 / **0.041** (k=10) | 0.991 / 0.205 / **0.340** (k=9) |
| Unsupervised, fit and evaluate on all blocks, 95% variance | 0.975 / 0.637 / **0.770** (k=5) | 0.975 / 0.636 / **0.770** (k=5) |
| Semi-supervised (normal training blocks only), 95% variance | 0.707 / 0.598 / **0.648** (k=4) | 0.707 / 0.598 / **0.648** (k=3) |
