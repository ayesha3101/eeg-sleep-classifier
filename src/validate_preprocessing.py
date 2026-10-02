"""
validate processed real eeg data before trusting it for feature extraction
and model training. Checks structural correctness (shapes, label values,
NaNs) and visualizes real band-power patterns per stage, mirroring the
sanity check we did on synthetic data.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from extract_features import extract_full_features

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
VALID_STAGES = {"Wake", "N1", "N2", "N3", "REM"}
EXPECTED_SAMPLES_PER_EPOCH = 30 * 100  # 30s at 100Hz = 3000
EXPECTED_CHANNELS = 2  # eeg, eog


def validate():
    files = sorted(PROCESSED_DIR.glob("SC*.npz"))
    if not files:
        print(f"no processed subject files found in {PROCESSED_DIR}")
        return

    print(f"validating {len(files)} processed subject files...\n")

    all_epochs, all_labels, all_subjects = [], [], []
    problems = []

    for f in files:
        data = np.load(f)
        epochs, labels = data["epochs"], data["labels"]
        subject_id = f.stem

        if epochs.shape[0] != labels.shape[0]:
            problems.append(f"{subject_id}: epochs/labels length mismatch "
                             f"({epochs.shape[0]} vs {labels.shape[0]})")

        if epochs.shape[1] != EXPECTED_CHANNELS:
            problems.append(f"{subject_id}: unexpected channel count "
                             f"{epochs.shape[1]} (expected {EXPECTED_CHANNELS}: EEG+EOG)")

        if epochs.shape[2] != EXPECTED_SAMPLES_PER_EPOCH:
            problems.append(f"{subject_id}: unexpected epoch length "
                             f"{epochs.shape[2]} (expected {EXPECTED_SAMPLES_PER_EPOCH})")

        unknown_labels = set(np.unique(labels)) - VALID_STAGES
        if unknown_labels:
            problems.append(f"{subject_id}: unexpected label values: {unknown_labels}")

        if np.isnan(epochs).any():
            problems.append(f"{subject_id}: contains NaN values")

        if np.all(epochs == 0):
            problems.append(f"{subject_id}: all-zero data (likely a loading bug)")

        all_epochs.append(epochs)
        all_labels.append(labels)
        all_subjects.append(np.full(len(labels), subject_id))

    print("=" * 60)
    if problems:
        print(f"FOUND {len(problems)} ISSUE(S):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("All structural checks passed: consistent shapes, valid "
              "labels only, no NaNs, no all-zero epochs.")
    print("=" * 60)

    epochs = np.concatenate(all_epochs)
    labels = np.concatenate(all_labels)
    subjects = np.concatenate(all_subjects)

    print(f"\nTotal: {len(labels)} epochs across {len(files)} subjects")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u:6s}: {c:5d}  ({c/len(labels)*100:.1f}%)")

    # --- extract features and check physiological plausibility ---
    print("\nExtracting features for plausibility check...")
    features, feature_names = extract_full_features(epochs)

    stages = ["Wake", "N1", "N2", "N3", "REM"]

    band_names = feature_names[:4]
    avg_power = np.zeros((len(stages), len(band_names)))
    for i, stage in enumerate(stages):
        mask = labels == stage
        avg_power[i] = features[mask, :4].mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = np.arange(len(band_names))
    width = 0.15
    for i, stage in enumerate(stages):
        ax.bar(x + i * width, avg_power[i], width, label=stage)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(band_names)
    ax.set_ylabel("Relative power")
    ax.set_title("EEG relative band power per stage (REAL data)")
    ax.legend()

    # EOG variance plot this is the key check for whether eog actually
    # discriminates REM the way we expect physiologically
    eog_var_idx = feature_names.index("eog_variance")
    ax2 = axes[1]
    eog_means = [features[labels == s, eog_var_idx].mean() for s in stages]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    ax2.bar(stages, eog_means, color=colors)
    ax2.set_ylabel("EOG variance (movement energy)")
    ax2.set_title("EOG variance per stage (should be highest for REM)")

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent.parent / "results" / "real_band_power_by_stage.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved plausibility plot to {out_path}")

    print("\nMean EOG variance by stage (expect REM highest):")
    for s, v in zip(stages, eog_means):
        print(f"  {s:5s}: {v:.2f}")

    # save the combined arrays so feature extraction / training 
    combined_path = PROCESSED_DIR / "all_subjects_combined.npz"
    np.savez(combined_path, epochs=epochs, labels=labels, subjects=subjects)
    print(f"Saved combined dataset to {combined_path}")

    return len(problems) == 0


if __name__ == "__main__":
    ok = validate()
    if not ok:
        print("\nFix issues above before proceeding to feature extraction / training.")