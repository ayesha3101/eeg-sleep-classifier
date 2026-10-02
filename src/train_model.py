"""
train and test the sleep stage classifier using the processed eeg and eog data.
this version includes eog features, a wider ±2 epoch context window, 
leave-one-subject-out cross-validation for more reliable results, 
and gridsearchcv to tune the random forest.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score
)
import joblib

from extract_features import extract_full_features, add_temporal_context

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STAGES = ["Wake", "N1", "N2", "N3", "REM"]
CONTEXT_WINDOW = 2  


def load_combined_data():
    combined_path = PROCESSED_DIR / "all_subjects_combined.npz"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"{combined_path} not found. Run validate_preprocessing.py first."
        )
    data = np.load(combined_path)
    return data["epochs"], data["labels"], data["subjects"]


def subject_level_split(subjects, test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(np.zeros(len(subjects)), groups=subjects))
    return train_idx, test_idx


def build_features(epochs, subjects):
    features, feature_names = extract_full_features(epochs)
    features = add_temporal_context(features, subjects, window=CONTEXT_WINDOW)

    context_names = []
    offsets = list(range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1))
    for off in offsets:
        tag = f"t{off:+d}" if off != 0 else "t0"
        context_names += [f"{n}_{tag}" for n in feature_names]

    return features, context_names


def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [8, 16, None],
        "min_samples_leaf": [1, 5],
    }
    base_clf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    search = GridSearchCV(base_clf, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV macro F1: {search.best_score_:.3f}")
    return search.best_estimator_


def run_loso_cv(features, labels, subjects, clf_builder):
    """ leave one subject out cross validation: for each subject, train on
    everyone else, test on that one held out subject, repeat for all
    subjects, then average. clf_builder is a zero arg function returning
    a fresh (untrained) classifier instance for each fold.
    """
    logo = LeaveOneGroupOut()
    fold_accuracies, fold_f1s = [], []

    print(f"\nrunning leave one subject out cv ({len(np.unique(subjects))} folds)...")
    for fold_i, (train_idx, test_idx) in enumerate(logo.split(features, labels, groups=subjects)):
        clf = clf_builder()
        clf.fit(features[train_idx], labels[train_idx])
        preds = clf.predict(features[test_idx])

        acc = accuracy_score(labels[test_idx], preds)
        f1 = f1_score(labels[test_idx], preds, labels=STAGES, average="macro", zero_division=0)
        held_out_subject = subjects[test_idx][0]

        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        print(f"  Fold {fold_i+1:2d} (held out {held_out_subject}): "
              f"acc={acc:.3f}  macro_f1={f1:.3f}")

    print(f"\nLOSO-CV mean accuracy: {np.mean(fold_accuracies):.3f} "
          f"(+/- {np.std(fold_accuracies):.3f})")
    print(f"LOSO-CV mean macro F1: {np.mean(fold_f1s):.3f} "
          f"(+/- {np.std(fold_f1s):.3f})")
    return fold_accuracies, fold_f1s


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("loading combined dataset...")
    epochs, labels, subjects = load_combined_data()
    print(f"Total: {len(labels)} epochs, {len(np.unique(subjects))} subjects\n")

    print("extracting features (eeg band power + ratios + entropy, eog time-domain)...")
    print(f"adding temporal context (+/- {CONTEXT_WINDOW} epochs)...")
    features, feature_names = build_features(epochs, subjects)
    print(f"final feature matrix: {features.shape}\n")

 
    print("=" * 60)
    print("part A: single train/test split with hyperparameter tuning")
    print("=" * 60)
    train_idx, test_idx = subject_level_split(subjects)
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    print(f"train: {len(X_train)} epochs | test: {len(X_test)} epochs\n")

    print("tuning random forest hyperparameters (grid search, 3-fold CV)...")
    clf = tune_random_forest(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nheld out test accuracy (single split): {acc:.3f}\n")
    print(classification_report(y_test, y_pred, labels=STAGES, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=STAGES)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=STAGES).plot(
        ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"confusion matrix (single split, acc={acc:.2f})")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=120)
    print(f"saved confusion matrix to {RESULTS_DIR / 'confusion_matrix.png'}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(feature_names)), clf.feature_importances_, color="steelblue")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
    ax.set_ylabel("importance")
    ax.set_title("feature importance (best tuned Random Forest)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=120)
    print(f"saved feature importance to {RESULTS_DIR / 'feature_importance.png'}")

    joblib.dump(clf, RESULTS_DIR / "sleep_stage_classifier.pkl")
    print(f"saved trained model to {RESULTS_DIR / 'sleep_stage_classifier.pkl'}")

   
    print("\n" + "=" * 60)
    print("part B: Leave-one-subject-out cross-validation")
    print("(more trustworthy estimate than a single split, given only "
          f"{len(np.unique(subjects))} subjects total)")
    print("=" * 60)

    best_params = clf.get_params()
    def build_clf():
        return RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            class_weight="balanced", random_state=42, n_jobs=-1,
        )

    fold_acc, fold_f1 = run_loso_cv(features, labels, subjects, build_clf)

    fig, ax = plt.subplots(figsize=(8, 4))
    subj_ids = sorted(np.unique(subjects))
    ax.bar(range(len(fold_acc)), fold_acc, color="steelblue")
    ax.axhline(np.mean(fold_acc), color="red", linestyle="--",
               label=f"mean={np.mean(fold_acc):.2f}")
    ax.set_xticks(range(len(subj_ids)))
    ax.set_xticklabels(subj_ids, rotation=45, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_title("Leave-one-subject-out accuracy per held-out subject")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loso_cv_accuracy.png", dpi=120)
    print(f"saved LOSO-CV plot to {RESULTS_DIR / 'loso_cv_accuracy.png'}")


if __name__ == "__main__":
    main()