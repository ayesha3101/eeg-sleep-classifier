# Learning Roadmap: EEG Signal Processing & Biosignal ML

A guided map through everything this project touched, organized so you can
study each layer independently and apply it to future projects (other
biosignals, other time-series classification tasks).

---

## Phase 1: EEG & Biosignal Fundamentals

**What to understand:**
- What EEG actually measures (post-synaptic potentials of large neuron
  populations, picked up at the scalp -- NOT individual neuron firing)
- The 10-20 electrode placement system (why channels are named things like
  "Fpz-Cz" -- Fp=frontal pole, z=midline, Cz=central)
- The classic frequency bands and what each is associated with:
  delta (0.5-4Hz, deep sleep), theta (4-8Hz, drowsiness/light sleep),
  alpha (8-13Hz, relaxed wakefulness/eyes closed), beta (13-30Hz, active
  thinking/alertness), gamma (30Hz+, high-level cognitive binding -- and why
  we excluded it in this project)
- Why real EEG follows a "1/f" power law (lower frequencies naturally carry
  more power) -- this is why our real band-power plots looked so different
  from our idealized synthetic ones
- EOG and EMG as *complementary* signals, not just "other channels" --
  understand why REM is EOG-defined, not EEG-defined

**Where this showed up in our project:** every design decision about which
channels/bands to use, and why the ocular-artifact issue happened at all
(Fpz is close to the eyes).

**Suggested next step:** look up the AASM (American Academy of Sleep
Medicine) sleep scoring manual summary -- it's the actual clinical standard
this whole field is built on.

---

## Phase 2: Digital Signal Processing Basics

**What to understand:**
- Sampling and the Nyquist theorem (why 100Hz sampling caps you at 50Hz
  of resolvable signal -- this is why we couldn't do "real" gamma analysis)
- Filtering: what a bandpass filter does, why Butterworth specifically
  (maximally flat passband), what filter *order* controls
- Zero-phase filtering (`filtfilt`) vs single-pass (`lfilter`) -- why phase
  distortion matters for time-sensitive signals
- Fourier analysis fundamentals: how any signal can be decomposed into a
  sum of sine waves at different frequencies (this is the conceptual basis
  for everything in Phase 3)
- Power Spectral Density (PSD) and Welch's method specifically (windowing +
  averaging to reduce noise in the frequency-domain estimate)

**Where this showed up:** `preprocess.py`'s `bandpass_filter()`, and the
entire basis of `extract_features.py`'s `band_power()`.

**Suggested next step:** implement a bandpass filter and PSD calculation
completely from scratch (no scipy) on a simple synthetic sine wave, just to
build the intuition before relying on library functions again.

---

## Phase 3: Feature Engineering for Time-Series / Biosignals

**What to understand:**
- Why raw waveforms are usually a poor direct input for simple models (too
  high-dimensional, too noisy) -- and when raw input actually *is* fine
  (deep learning with enough data)
- Band power (absolute vs relative) and why relative power is usually more
  robust (corrects for amplitude differences unrelated to the signal of
  interest -- across subjects, across artifacts)
- Band ratios (alpha/delta, theta/beta) -- a general pattern: ratios often
  capture relationships that raw values miss
- Spectral entropy -- how "spread out" vs "concentrated" a signal's
  frequency content is, a single-number summary of spectral shape
  (this generalizes: entropy measures show up everywhere in time-series ML)
- Time-domain features (variance, zero-crossing rate, peak amplitude) as
  an alternative to frequency-domain features -- useful when the signal's
  *amplitude/shape* carries the information, not its frequency content
  (this is why we used time-domain features for EOG, not band power)
- Temporal context / windowing: concatenating neighboring time-steps'
  features to give a model access to sequential structure without needing
  a fully sequential model architecture (LSTM/Transformer)

**Where this showed up:** the entire evolution of `extract_features.py`,
and specifically the jump from ~50% to ~75% accuracy when context was added.

**Suggested next step:** for any new biosignal project, ask yourself:
"does the information live in the frequency content, or the shape/timing of
the raw signal?" -- that answer tells you whether to reach for spectral
features or time-domain features first.

---

## Phase 4: ML Fundamentals for This Kind of Data

**What to understand:**
- **Group/subject-level data leakage** -- the single most important lesson
  from this project. Any time your data has a natural grouping (subjects,
  sessions, patients), a random split at the individual-sample level will
  leak information and inflate your reported accuracy. Always ask: "could
  two 'independent' samples actually come from the same underlying source?"
- Class imbalance and how to handle it (`class_weight="balanced"`,
  understanding precision/recall/F1 per class rather than trusting overall
  accuracy alone)
- Cross-validation, and specifically **Leave-One-Group-Out** for small
  numbers of subjects/groups -- more trustworthy than a single split when
  your group count is small
- Hyperparameter tuning via grid search (and the tradeoff: more exhaustive
  search vs. runtime cost)
- Why comparing multiple model types on the *same* split is more
  informative than tuning one model in isolation -- it tells you whether
  your bottleneck is the model or the features

**Where this showed up:** `subject_level_split()`, `run_loso_cv()`,
`class_weight="balanced"` throughout, and the model-comparison experiment
that revealed the features (not model choice) were the real bottleneck.

**Suggested next step:** intentionally build a version of this pipeline
with a *naive* random (non-subject-level) split, and compare the reported
accuracy to the honest subject-level number -- seeing the inflated number
with your own eyes is the best way to internalize why this matters.

---

## Phase 5: Model Families -- When to Reach for What

**What to understand, at a practical level:**
- **Linear models (Logistic Regression):** fast, interpretable, a great
  baseline -- if a linear model does almost as well as fancier ones, your
  classes are probably close to linearly separable in your feature space
- **Kernel methods (SVM):** good for moderate-sized datasets with a
  moderate number of features, can capture non-linear boundaries
- **Tree ensembles (Random Forest):** robust to noisy/messy features,
  give you feature importances for free, but can overfit with very few
  features relative to their flexibility (as we saw -- RF was *worst* with
  4 features, best after we expanded to 50)
- Where all of these differ from **deep learning**: deep learning shines
  when you have lots of data and want the model to *learn* the feature
  representation itself, rather than you hand-engineering it

**Where this showed up:** `compare_models.py`, and specifically the
finding that all three models converged once features improved.

**Suggested next step:** read a bit about *why* tree ensembles can overfit
with too few features/too much depth relative to data size -- it's a good
generally-applicable intuition, not just an EEG-specific quirk.

---

## Phase 6: Evaluation Rigor

**What to understand:**
- Why accuracy alone is a poor metric under class imbalance, and what
  precision/recall/F1 actually mean per class
- Macro-averaged F1 vs weighted-averaged F1 -- macro treats every class
  equally regardless of size (good for catching a model that's ignoring
  a rare class); weighted reflects real-world class frequency
- How to *read* a confusion matrix as a diagnostic tool, not just a
  scorecard -- off-diagonal patterns tell you *which* mistakes are being
  made, which is often more informative than the accuracy number itself
- The importance of checking whether your errors make domain sense (e.g.,
  N1 confusing with its physiological neighbors) vs. looking arbitrary
  (which would suggest a real bug)

**Where this showed up:** the entire `RESULTS.md` interpretation of the
confusion matrix, and the choice to report LOSO-CV over the single split.

---

## Phase 7: Reproducible Project Practice (the "professional" half)

**What to understand:**
- Git hygiene: meaningful, incremental commits; `.gitignore` for data/
  regenerable artifacts; never committing large binary files to version
  control
- Validate-before-you-trust: structural checks (shapes, NaNs, valid labels)
  AND domain-plausibility checks (does the signal behave the way domain
  knowledge says it should?) -- structural checks alone would have missed
  the ocular-artifact issue entirely
- Build against a synthetic/mock stand-in when a real data source is
  blocked or slow, so pipeline development isn't stalled -- but always
  validate against real data before trusting results
- Explicit dependency declaration (the `joblib` lesson -- don't rely on
  transitive dependencies for things you import directly)

**Where this showed up:** basically this entire conversation's structure.

---

## Where to go next, if you want to go deeper

- **Deep learning for EEG:** look into CNN and LSTM architectures applied
  directly to raw or spectrogram EEG (search terms: "DeepSleepNet",
  "SeqSleepNet" -- well-known sleep-staging deep learning papers)
- **Other public biosignal datasets to practice on:** PhysioNet hosts many
  (ECG arrhythmia datasets, EMG datasets, gait/accelerometer datasets) --
  the *same* pipeline skeleton (download -> preprocess -> validate ->
  feature extract -> train -> evaluate with proper grouping) transfers
  directly
- **Artifact rejection techniques:** look into ICA (Independent Component
  Analysis) for EEG -- the standard, more rigorous way to remove eye-blink
  and muscle artifacts, rather than the simpler relative-power normalization
  we used here
- **Signal processing theory:** if you want the deeper "why" behind
  filtering/Fourier analysis, look for a DSP fundamentals course rather
  than just EEG-specific material -- it transfers to audio, vibration
  analysis, radar, finance time-series, anything with a time-varying signal