"""
models compared:
logistic regression: simple linear baseline
svm (rbf): non-linear model commonly used for eeg classification
random forest: our original model choice
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from extract_features import extract_full_features, add_temporal_context
from train_model import load_combined_data, subject_level_split, STAGES, CONTEXT_WINDOW

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data and extracting features...")
    epochs, labels, subjects = load_combined_data()
    features, feature_names = extract_full_features(epochs)
    features = add_temporal_context(features, subjects, window=CONTEXT_WINDOW)
    print(f"Feature matrix with temporal context: {features.shape}")

    train_idx, test_idx = subject_level_split(subjects)
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            X_train_scaled, X_test_scaled,
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf", class_weight="balanced", random_state=42),
            X_train_scaled, X_test_scaled,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, max_depth=12,
                                    class_weight="balanced", random_state=42, n_jobs=-1),
            X_train, X_test,  
        ),
    }

    results = {}
    print(f"\nTrain: {len(X_train)} epochs | Test: {len(X_test)} epochs\n")
    print("=" * 60)

    for name, (model, X_tr, X_te) in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, labels=STAGES, average="macro", zero_division=0)

        results[name] = {"accuracy": acc, "macro_f1": macro_f1}
        print(f"  Accuracy: {acc:.3f}  |  Macro F1: {macro_f1:.3f}")
        print(classification_report(y_test, y_pred, labels=STAGES, zero_division=0))

    print("=" * 60)
    print("\nSUMMARY")
    print(f"{'Model':22s} {'Accuracy':>10s} {'Macro F1':>10s}")
    for name, r in results.items():
        print(f"{name:22s} {r['accuracy']:10.3f} {r['macro_f1']:10.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, [results[n]["accuracy"] for n in names], width, label="Accuracy")
    ax.bar(x + width/2, [results[n]["macro_f1"] for n in names], width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (same subject-level test set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=120)
    print(f"\nSaved comparison chart to {RESULTS_DIR / 'model_comparison.png'}")


if __name__ == "__main__":
    main()