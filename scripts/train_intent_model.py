import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# ---------------------------------------------------------
# 1. Load weakly labelled training data
# ---------------------------------------------------------

DATA_PATH = "data/weak_labeled_data.csv"

print("Loading training data...")

df = pd.read_csv(DATA_PATH)

print(f"Training examples: {len(df)}")

print("\nClass distribution:")
print(df["weak_intent"].value_counts())


# ---------------------------------------------------------
# 2. Prepare input and target
# ---------------------------------------------------------

X = df["customer_message"].fillna("")
y = df["weak_intent"]


# ---------------------------------------------------------
# 3. Train / validation split
# ---------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining split: {len(X_train)}")
print(f"Validation split: {len(X_val)}")


# ---------------------------------------------------------
# 4. Build TF-IDF + Logistic Regression pipeline
# ---------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# ---------------------------------------------------------
# 5. Train
# ---------------------------------------------------------

print("\nTraining TF-IDF + Logistic Regression...")

model.fit(X_train, y_train)

print("Training complete.")


# ---------------------------------------------------------
# 6. Evaluate on validation set
# ---------------------------------------------------------

print("\nEvaluating validation set...")

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average="macro")

print(f"\nValidation Accuracy: {accuracy:.4f}")
print(f"Validation Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred, zero_division=0))


# ---------------------------------------------------------
# 7. Evaluate on locked Golden Set
# ---------------------------------------------------------

GOLDEN_PATH = "data/golden_set.csv"

print("\nLoading Golden Set...")

golden_df = pd.read_csv(GOLDEN_PATH)

print(f"Golden Set examples: {len(golden_df)}")

X_golden = golden_df["customer_message"].fillna("")
y_golden = golden_df["final_intent"]


# Predict Golden Set intents
print("\nEvaluating model on Golden Set...")

golden_pred = model.predict(X_golden)


# Calculate metrics
golden_accuracy = accuracy_score(y_golden, golden_pred)
golden_macro_f1 = f1_score(
    y_golden,
    golden_pred,
    average="macro"
)

print(f"\nGolden Set Accuracy: {golden_accuracy:.4f}")
print(f"Golden Set Macro F1: {golden_macro_f1:.4f}")

print("\nGolden Set Classification Report:")
print(
    classification_report(
        y_golden,
        golden_pred,
        zero_division=0
    )
)


# ---------------------------------------------------------
# 8. Save Golden Set predictions
# ---------------------------------------------------------

golden_results = golden_df.copy()

golden_results["predicted_intent"] = golden_pred
golden_results["correct"] = (
    golden_results["final_intent"]
    == golden_results["predicted_intent"]
)

golden_results.to_csv(
    "data/ml_predictions.csv",
    index=False
)

print("\nGolden Set predictions saved:")
print("data/ml_predictions.csv")