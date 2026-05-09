"""DecodeLabs Project 2 — Data Classification Using AI.

Loads the Iris dataset, explores it, splits into train/test sets,
trains a Decision Tree classifier, evaluates performance, and
saves a confusion matrix heatmap to disk.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuration ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"


def main():
    """Run the full classification pipeline."""
    print("=" * 50)
    print("  DecodeLabs Project 2: Data Classification")
    print("=" * 50)

    # ----------------------------------------------------------------
    # STEP 1 — Load and understand the dataset
    # ----------------------------------------------------------------
    print("\n[1] Loading the Iris dataset...\n")

    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    # Build a DataFrame for easy inspection
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = pd.Categorical.from_codes(y, class_names)

    print(f"  Shape        : {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Features     : {list(feature_names)}")
    print(f"  Classes      : {list(class_names)}")
    print(f"  Balance      : 50 samples per class (perfectly balanced)\n")

    print("  --- Sample Rows (head) ---")
    print(df.head().to_string(index=False))
    print()

    # ----------------------------------------------------------------
    # STEP 2 — Split into training and testing sets
    # ----------------------------------------------------------------
    print("[2] Splitting data into train / test sets...\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"  Training set  : {X_train.shape[0]} samples")
    print(f"  Testing set   : {X_test.shape[0]} samples")
    print(f"  Split ratio   : {1 - TEST_SIZE:.0%} / {TEST_SIZE:.0%}")
    print(f"  Random state  : {RANDOM_STATE}\n")

    # ----------------------------------------------------------------
    # STEP 3 — Train a simple classification algorithm
    # ----------------------------------------------------------------
    print("[3] Training a Decision Tree classifier...\n")

    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("  Model trained successfully.\n")

    # ----------------------------------------------------------------
    # STEP 4 — Evaluate the model
    # ----------------------------------------------------------------
    print("[4] Evaluating the model...\n")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"  Accuracy  : {acc:.2%}\n")
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix (rows=true, cols=predicted):")
    print(f"  {cm[0]}")
    print(f"  {cm[1]}")
    print(f"  {cm[2]}\n")

    # ----------------------------------------------------------------
    # STEP 5 — Save a confusion matrix heatmap
    # ----------------------------------------------------------------
    print("[5] Saving confusion matrix heatmap...\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix — Decision Tree (Iris)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved  -->  {os.path.abspath(save_path)}\n")

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    print("=" * 50)
    print("  Pipeline complete. Ready for review.")
    print("=" * 50)


if __name__ == "__main__":
    main()
