# Results

## Summary

A sleep-stage classifier was developed using one EEG channel (Fpz-Cz) and one horizontal EOG channel from the Sleep-EDF Expanded Sleep Cassette dataset.

The system was evaluated on 10 subjects using Leave-One-Subject-Out Cross-Validation (LOSO-CV), where each subject was kept completely separate from the training data.

**Final result: 74.2% ± 7.1% accuracy and 68.9% macro F1.**

This shows that reasonable sleep-stage classification is possible using only two signal channels.

## Method

The recordings were filtered from 0.3-35 Hz and divided into 30-second epochs using the provided sleep-stage annotations. Excess Wake periods before and after sleep were removed, reducing the dataset from 27,464 to 9,913 epochs.

EEG features included relative delta, theta, alpha, and beta power, alpha/delta ratio, theta/beta ratio, and spectral entropy. EOG features included variance, maximum amplitude, and zero-crossing rate.

Information from the two epochs before and after the current epoch was also included to provide temporal context.

Logistic Regression, SVM, and Random Forest were compared, with Random Forest hyperparameters tuned using grid search.

## Key Findings

### EOG made a major difference

The initial EEG-only features produced around **49-54% accuracy**. Adding EOG features increased performance to around **74-79%**.

This makes sense because EOG captures eye movements, which are especially useful for identifying REM sleep. EOG features also appeared among the most important features in the Random Forest model.

### Relative power helped with eye artifacts

Initial analysis showed unusually high delta power during some Wake epochs. Since delta activity is normally associated with deep sleep, this suggested contamination from eye movements and blinks.

Because the Fpz-Cz channel is close to the eyes, these movements can appear as slow activity in the EEG. Using relative instead of absolute band power helped reduce this effect.

### Temporal context improved performance

Using the two previous and two following epochs improved classification compared with treating every 30-second epoch independently.

This is useful because sleep stages follow a natural sequence and neighboring epochs are usually related.

### Model choice was less important

Logistic Regression, SVM, and Random Forest achieved similar performance once the improved EEG + EOG features and temporal context were added.

This suggests that **feature quality had a greater effect on performance than the choice of classifier**.

## Per-Class Performance

| Stage | Precision | Recall |   F1 |
| ----- | --------: | -----: | ---: |
| Wake  |      0.79 |   0.87 | 0.82 |
| N1    |      0.54 |   0.40 | 0.46 |
| N2    |      0.87 |   0.71 | 0.78 |
| N3    |      0.80 |   0.85 | 0.83 |
| REM   |      0.63 |   0.89 | 0.73 |

N1 was the hardest stage to classify. This is expected because N1 is a transition stage between Wake and sleep and can look similar to neighboring stages.

The largest confusion was **N2 being classified as REM**, with 110 affected epochs. There were also 44 N3 epochs classified as Wake, which could be related to movement or brief arousals.

## Subject-Level Results

Performance varied across the 10 subjects, ranging from **57.6% to 86.1% accuracy**.

This shows that generalizing to completely new subjects remains challenging. The subjects also had a wide age range, although the current dataset is too small to determine whether age caused the performance differences.

## Limitations

* Only one EEG and one EOG channel were used.
* Only 10 subjects were included.
* Some eye-related artifacts may remain.
* The system uses hand-crafted features rather than learning directly from raw signals.

## Future Work

Future improvements could include:

* Testing the system on more subjects.
* Adding chin EMG.
* Using ICA for better artifact removal.
* Testing CNN/LSTM models on raw EEG and EOG.
* Investigating the N2-REM confusion in more detail.
