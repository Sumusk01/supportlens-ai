import pandas as pd

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "weak_labeled_data.csv"


# ---------------------------------------------------------
# Intent Predictor
# ---------------------------------------------------------

class IntentPredictor:

    def __init__(self):

        self.model = None

    # -----------------------------------------------------
    # Train inference model
    # -----------------------------------------------------

    def train(self):

        print("\nLoading intent training data...")

        df = pd.read_csv(DATA_PATH)

        print(
            f"Weak-labelled training examples: {len(df)}"
        )

        X = df["customer_message"].fillna("")
        y = df["weak_intent"]

        print("\nTraining intent classifier...")

        self.model = Pipeline([
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

        self.model.fit(X, y)

        print("Intent classifier ready.")

    # -----------------------------------------------------
    # Predict intent
    # -----------------------------------------------------

    def predict(self, customer_message):

        if self.model is None:

            raise RuntimeError(
                "Intent model has not been trained."
            )

        probabilities = self.model.predict_proba(
            [customer_message]
        )[0]

        classes = self.model.classes_

        best_index = probabilities.argmax()

        predicted_intent = classes[best_index]

        confidence = float(
            probabilities[best_index]
        )

        # Sort classes by probability so we can inspect
        # the next-best alternatives.
        ranked_indices = probabilities.argsort()[::-1]

        top_predictions = []

        for index in ranked_indices[:3]:

            top_predictions.append(
                {
                    "intent": classes[index],
                    "probability": float(
                        probabilities[index]
                    )
                }
            )

        return {
            "predicted_intent": predicted_intent,
            "confidence": confidence,
            "top_predictions": top_predictions,
        }


# ---------------------------------------------------------
# Manual test
# ---------------------------------------------------------

def demo():

    predictor = IntentPredictor()

    predictor.train()

    test_messages = [

        "My iPhone battery is draining really quickly",

        "The App Store won't download my apps",

        "My iPhone keeps showing no service",

        "My screen is completely unresponsive",
    ]

    print("\n" + "=" * 70)
    print("INTENT PREDICTION TEST")
    print("=" * 70)

    for message in test_messages:

        result = predictor.predict(message)

        print("\n" + "-" * 70)

        print("Customer:")
        print(message)

        print(
            f"\nPredicted intent: "
            f"{result['predicted_intent']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.4f}"
        )

        print("\nTop predictions:")

        for prediction in result["top_predictions"]:

            print(
                f"- {prediction['intent']}: "
                f"{prediction['probability']:.4f}"
            )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    demo()