import pandas as pd
import re

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

GOLDEN_FILE = "data/golden_set.csv"

# --------------------------------------------------
# 2. LOAD GOLDEN SET
# --------------------------------------------------

print("Loading Golden Set...")

df = pd.read_csv(GOLDEN_FILE)

# Remove anything accidentally left unlabeled
df = df[
    df["final_intent"].notna() &
    (df["final_intent"] != "")
].copy()

print("Golden examples:", len(df))

# --------------------------------------------------
# 3. TRUE LABELS
# --------------------------------------------------

y_true = df["final_intent"]

INTENTS = sorted(
    y_true.unique()
)

print("\nIntents:")
for intent in INTENTS:
    print("-", intent)

# --------------------------------------------------
# 4. BASELINE 1 — MAJORITY CLASS
# --------------------------------------------------

majority_class = (
    y_true.value_counts()
    .idxmax()
)

print("\n" + "=" * 70)
print("BASELINE 1 — MAJORITY CLASS")
print("=" * 70)

print(
    "Majority intent:",
    majority_class
)

majority_predictions = [
    majority_class
] * len(df)

majority_accuracy = accuracy_score(
    y_true,
    majority_predictions
)

majority_f1 = f1_score(
    y_true,
    majority_predictions,
    average="macro",
    zero_division=0
)

print(
    f"Accuracy : {majority_accuracy:.4f}"
)

print(
    f"Macro F1 : {majority_f1:.4f}"
)

# --------------------------------------------------
# 5. KEYWORD RULES
# --------------------------------------------------

KEYWORD_RULES = {

    "BATTERY_POWER": [
        "battery",
        "battery life",
        "charging",
        "charger",
        "charge",
        "drain",
        "power"
    ],

    "IOS_UPDATE": [
        "ios",
        "software update",
        "update",
        "updated",
        "updating",
        "downgrade",
        "install update"
    ],

    "APP_PROBLEM": [
        "app crashes",
        "app crash",
        "apps crash",
        "apps crashing",
        "application crash",
        "application crashing",
        "app not working",
        "app doesn't work",
        "app does not work"
    ],

    "CONNECTIVITY": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "cellular",
        "mobile data",
        "network",
        "internet",
        "connection",
        "connect"
    ],

    "ACCOUNT_APPLE_ID": [
        "apple id",
        "appleid",
        "password",
        "sign in",
        "signin",
        "login",
        "log in",
        "account",
        "icloud"
    ],

    "APP_STORE_PURCHASE": [
        "app store",
        "itunes",
        "purchase",
        "purchased",
        "refund",
        "payment",
        "subscription",
        "billing"
    ],

    "MUSIC_MEDIA": [
        "apple music",
        "music",
        "podcast",
        "podcasts"
    ],

    "SCREEN_DISPLAY": [
        "screen",
        "display",
        "touch screen",
        "touchscreen",
        "brightness",
        "lock screen",
        "home screen"
    ],

    "SETTINGS_FEATURE": [
        "settings",
        "control center",
        "notification",
        "notifications",
        "how do i",
        "how can i",
        "where do i"
    ],

    "PERFORMANCE_STABILITY": [
        "slow",
        "slower",
        "lag",
        "lagging",
        "freeze",
        "freezing",
        "frozen",
        "restart",
        "restarting",
        "crash"
    ],

    "HARDWARE_DEVICE": [
        "speaker",
        "microphone",
        "camera",
        "button",
        "headphone",
        "earphone",
        "broken",
        "physical damage"
    ]
}

# --------------------------------------------------
# 6. KEYWORD CLASSIFIER
# --------------------------------------------------

def classify_with_keywords(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    # Store all matching intents
    matches = []

    for intent, keywords in KEYWORD_RULES.items():

        for keyword in keywords:

            if keyword in text:
                matches.append(intent)
                break

    # No keyword matched
    if not matches:
        return "OTHER"

    # If exactly one intent matched
    if len(matches) == 1:
        return matches[0]

    # --------------------------------------------------
    # Resolve overlapping matches
    #
    # More specific rules get priority.
    # --------------------------------------------------

    priority = [
        "APP_STORE_PURCHASE",
        "ACCOUNT_APPLE_ID",
        "BATTERY_POWER",
        "CONNECTIVITY",
        "APP_PROBLEM",
        "MUSIC_MEDIA",
        "SCREEN_DISPLAY",
        "SETTINGS_FEATURE",
        "IOS_UPDATE",
        "PERFORMANCE_STABILITY",
        "HARDWARE_DEVICE"
    ]

    for intent in priority:

        if intent in matches:
            return intent

    return "OTHER"


# --------------------------------------------------
# 7. RUN KEYWORD BASELINE
# --------------------------------------------------

df["keyword_prediction"] = (
    df["customer_message"]
    .apply(classify_with_keywords)
)

keyword_predictions = df[
    "keyword_prediction"
]

keyword_accuracy = accuracy_score(
    y_true,
    keyword_predictions
)

keyword_f1 = f1_score(
    y_true,
    keyword_predictions,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 70)
print("BASELINE 2 — KEYWORD CLASSIFIER")
print("=" * 70)

print(
    f"Accuracy : {keyword_accuracy:.4f}"
)

print(
    f"Macro F1 : {keyword_f1:.4f}"
)

# --------------------------------------------------
# 8. DETAILED KEYWORD RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("KEYWORD CLASSIFIER — PER INTENT")
print("=" * 70)

print(
    classification_report(
        y_true,
        keyword_predictions,
        labels=INTENTS,
        zero_division=0
    )
)

# --------------------------------------------------
# 9. CONFUSION MATRIX
# --------------------------------------------------

print("\n" + "=" * 70)
print("KEYWORD CLASSIFIER — CONFUSION MATRIX")
print("=" * 70)

matrix = confusion_matrix(
    y_true,
    keyword_predictions,
    labels=INTENTS
)

confusion = pd.DataFrame(
    matrix,
    index=INTENTS,
    columns=INTENTS
)

print(confusion)

# --------------------------------------------------
# 10. SAVE PREDICTIONS
# --------------------------------------------------

output_columns = [
    "example_id",
    "customer_message",
    "final_intent",
    "keyword_prediction"
]

df[output_columns].to_csv(
    "data/baseline_predictions.csv",
    index=False
)

print(
    "\nSaved predictions to:",
    "data/baseline_predictions.csv"
)

# --------------------------------------------------
# 11. FINAL COMPARISON
# --------------------------------------------------

print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    "Model": [
        "Majority Class",
        "Keyword Rules"
    ],
    "Accuracy": [
        majority_accuracy,
        keyword_accuracy
    ],
    "Macro F1": [
        majority_f1,
        keyword_f1
    ]
})

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)