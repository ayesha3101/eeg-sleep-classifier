# Results

## Summary

A sleep-stage classifier trained on single-subject-generalizable features from
EEG (Fpz-Cz) and EOG (horizontal) signals, evaluated with leave-one-subject-out
cross-validation (LOSO-CV) across 10 subjects from the Sleep-EDF Expanded
(Sleep Cassette) dataset.

**Final result: 74.2% accuracy (± 7.1%), 68.9% macro F1, LOSO-CV across 10 subjects.**

This is in line with published benchmarks for automated sleep staging using
1-2 channels of physiological signal (typical range: 70-85%, depending on
channel count and modeling approach).

## Methodology

1. **Data**: 10 subjects, Sleep-EDF Sleep Cassette study (ages 25-101, healthy,
   no sleep medication). ~20-hour ambulatory home recordings, EEG+EOG sampled
   at 100Hz.
2. **Preprocessing**: 0.3-35Hz bandpass filter (zero-phase, Butterworth order 4).
   Epochs cut to 30s using hypnogram annotations. Excess pre-/post-sleep Wake
   trimmed (kept 30 min buffer around the sleep period) to fix severe class
   imbalance -- reduced dataset from 27,464 to 9,913 epochs.
3. **Features** (10 raw features per epoch, times 5 temporal positions
   [t-2, t-1, t0, t+1, t+2] = 50 features):
   - EEG: relative band power (delta, theta, alpha, beta)
   - EEG: alpha/delta ratio, theta/beta ratio
   - EEG: spectral entropy
   - EOG: variance, max absolute amplitude, zero-crossing rate
4. **Models compared**: Logistic Regression, SVM (RBF kernel), Random Forest
   (hyperparameter-tuned via grid search).
5. **Evaluation**: subject-level train/test splitting throughout (never split
   epochs from the same subject across train and test) -- critical for an
   honest estimate, since epochs from the same subject are highly correlated.
   Final reported metric is leave-one-subject-out CV (10 folds), not a single
   split, given the small number of subjects.

## Key findings

### 1. Ocular artifact contamination in raw band power (discovered during validation)

Before trusting the pipeline, we plotted average band power per sleep stage
on real data and found Wake epochs showed anomalously high delta power --
physiologically backwards, since delta is the deep-sleep signature. Root
cause: the Fpz-Cz channel sits near the eyes, and blinks/eye movements
produce large slow deflections that leak into the delta frequency range.
Switching from absolute to **relative** band power (each band as a fraction
of total power) partially mitigated this and also improved cross-subject
comparability (absolute EEG amplitude varies by skull thickness, electrode
contact, etc., independent of sleep stage).

### 2. Adding EOG features was the single largest improvement

Initial EEG-only features (4 raw features, no temporal context) capped out
around 49-54% accuracy across all three model types -- all three converging
to a similar ceiling was itself informative: it pointed to a features
bottleneck, not a model-choice problem. Adding EOG-derived features (the
actual defining signal for REM sleep, physiologically) alongside temporal
context lifted accuracy to 74-79% across models. Feature importance analysis
on the final tuned Random Forest confirms this: EOG-derived features
(particularly `eog_max_abs` across multiple timesteps) rank among the most
important features in the entire model, alongside EEG beta power.

### 3. Temporal context matters

Sleep stages have strong sequential structure -- adjacent 30s epochs are
rarely unrelated. Concatenating each epoch's features with its ±2 neighbors'
features (rather than classifying each epoch in total isolation) measurably
improved every model tested.

### 4. Model choice mattered less than feature quality

Logistic Regression (linear), SVM (non-linear, kernel-based), and Random
Forest (tree ensemble) all converged to similar performance once given the
same improved feature set (77-79% single-split accuracy). This suggests the
classes are reasonably well-separated by the final feature representation,
and further model complexity is unlikely to yield large additional gains
without richer input features (e.g., raw-signal deep learning).

## Per-class performance (single split; consistent with LOSO-CV pattern)

| Stage | Precision | Recall | F1 | Notes |
|-------|-----------|--------|-----|-------|
| Wake  | 0.79 | 0.87 | 0.82 | Strong |
| N1    | 0.54 | 0.40 | 0.46 | Weakest class -- see below |
| N2    | 0.87 | 0.71 | 0.78 | Strong, most common stage |
| N3    | 0.80 | 0.85 | 0.83 | Strong |
| REM   | 0.63 | 0.89 | 0.73 | High recall, moderate precision |

**N1 is the hardest class to classify**, confused roughly evenly with Wake,
N2, and REM (its physiological neighbors). This matches a well-documented
property of this exact task: even trained human technicians scoring the same
recordings agree with the majority score on N1 only about 63% of the time (AASM inter-scorer reliability program, Rosenberg and Van Hout, 2013), the lowest of any stage,
since it's an inherently ambiguous, brief transitional stage. The model
struggling most on the stage humans also struggle most on is a sign the
model has learned something real about the task's actual difficulty, not
evidence of a bug.

**N2 -> REM is the largest single confusion pair** in the confusion matrix
(110 misclassified epochs). Worth further investigation in future work.

**N3 -> Wake (44 epochs)** is a smaller but physiologically surprising
confusion -- deep sleep and wakefulness are usually easy to distinguish.
Possibly reflects brief arousals/movement artifacts within N3 periods that
produce Wake-like signal characteristics. Flagged here as an open question
rather than a resolved explanation.

## Subject-level variance (LOSO-CV)

| Subject | Accuracy | Macro F1 |
|---------|----------|----------|
| SC4001 | 0.759 | 0.660 |
| SC4011 | 0.787 | 0.707 |
| SC4021 | 0.679 | 0.694 |
| SC4031 | 0.787 | 0.699 |
| SC4041 | 0.728 | 0.692 |
| SC4051 | 0.576 | 0.521 |
| SC4061 | 0.743 | 0.745 |
| SC4071 | 0.861 | 0.808 |
| SC4081 | 0.750 | 0.700 |
| SC4091 | 0.753 | 0.661 |

Performance ranges from 57.6% to 86.1% depending on the held-out subject.
This spread is plausibly related to known age-related variability in sleep
architecture within this dataset (subjects span ages 25-101; older adults
generally show less deep sleep and more fragmented sleep, which would affect
how cleanly a model trained mostly on other age groups generalizes).

## Limitations

- **Single EEG channel (Fpz-Cz) + single EOG channel.** Full clinical
  polysomnography also uses chin EMG (muscle tone, useful for REM atonia
  detection) and often a second EEG channel. This was a deliberate scope
  choice given project constraints, not an oversight -- single/dual-channel
  staging is also a legitimate, actively studied research area motivated by
  low-cost/wearable feasibility.
- **Residual ocular artifact contamination is likely not fully eliminated**
  by relative-power normalization alone; more rigorous artifact rejection
  (e.g., ICA-based EOG removal) was out of scope here.
- **Only 10 subjects.** Sufficient to demonstrate the pipeline and get a
  reasonable LOSO-CV estimate, but a larger subject pool would give a more
  stable performance estimate and reduce the influence of any single
  atypical subject (e.g., SC4051).
- **Hand-crafted features, not learned representations.** Deep learning
  approaches (CNN/LSTM/Transformer on raw or spectrogram input) are standard
  in current sleep-staging research and would likely outperform this
  approach, at the cost of significantly more data, compute, and
  implementation complexity.

## Possible future work

- Deep learning on raw or spectrogram EEG+EOG input (CNN or LSTM), to learn
  representations rather than relying on hand-crafted band power features.
- Add chin EMG as a third channel (directly informative for REM atonia).
- ICA-based artifact removal instead of relative-power normalization alone.
- Expand to more subjects and stratify evaluation by age group, given the
  observed subject-level variance.
- Investigate the N2/REM confusion specifically (e.g., via per-epoch error
  analysis on misclassified examples)