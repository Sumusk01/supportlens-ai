import re
import pandas as pd


INPUT_FILE = "data/ml_training_data.csv"
OUTPUT_FILE = "data/weak_labeled_data.csv"


# =========================================================
# 1. Intent rules
# =========================================================

RULES = {
    "IOS_UPDATE": {
        "strong": [
            "software update failed",
            "software update won't install",
            "software update wont install",
            "unable to update ios",
            "can't update ios",
            "cannot update ios",
            "ios update failed",
            "ios update stuck",
            "update is stuck",
            "update won't install",
            "update wont install",
            "update not installing",
            "won't let me update",
            "wont let me update",
            "can't update my phone",
            "cannot update my phone",
            "can't update my iphone",
            "cannot update my iphone",
            "unable to update my phone",
            "unable to update my iphone",
            "update won't work",
            "update wont work",
            "update isn't working",
            "update is not working",
            "update failed",
            "update fails",
            "update stuck",
            "update is stuck",
            "update available",
            "new ios update",
            "new update",
            "ios update available",
            "iphone update",
            "iphone update issue",
            "ios update issue",
        ],
        "keywords": [
            "ios update",
            "update",
            "updated",
            "updating",
            "upgrade",
            "installation",
            "install",
            "download",
            "software update",
            "update ios",
            "ios upgrade",
            "updating ios",
        ],
    },

    "BATTERY_POWER": {
        "strong": [
            "takes all day to charge",
            "takes almost all day to charge",
            "takes forever to charge",
            "charging very slowly",
            "charging slowly",
            "battery drains quickly",
            "battery drains fast",
            "battery dies quickly",
            "battery dies fast",
            "battery life is terrible",
            "battery life is poor",
            "won't hold a charge",
            "wont hold a charge",
            "losing battery quickly",
            "battery percentage drops",
            "battery percentage is dropping",
            "battery drains overnight",
        ],
        "keywords": [
            "battery",
            "charging",
            "charge slowly",
            "charge time",
            "battery life",
            "battery drain",
            "battery percentage",
            "power",
        ],
    },

    "APP_PROBLEM": {
        "strong": [
            "app won't open",
            "app wont open",
            "apps won't open",
            "apps wont open",
            "can't open the app",
            "can't open apps",
            "cannot open apps",
            "app keeps crashing",
            "apps keep crashing",
            "app crashes",
            "apps crash",
            "app keeps freezing",
            "apps keep freezing",
            "app is not responding",
            "apps are not responding",
            "app doesn't work",
            "app doesnt work",
            "apps don't work",
            "apps dont work",
            "app hangs",
            "apps hang",
        ],
        "keywords": [
            "app",
            "apps",
            "crash",
            "crashes",
            "crashing",
            "freeze",
            "freezes",
            "freezing",
            "hangs",
            "hanging",
            "not responding",
        ],
    },

    "CONNECTIVITY": {
        "strong": [
            "can't connect to wifi",
            "cannot connect to wifi",
            "can't connect to wi fi",
            "cannot connect to wi fi",
            "wifi won't connect",
            "wifi wont connect",
            "wifi is not connecting",
            "wifi not connecting",
            "bluetooth won't connect",
            "bluetooth wont connect",
            "bluetooth is not connecting",
            "bluetooth not connecting",
            "keeps disconnecting from wifi",
            "keeps disconnecting from bluetooth",
            "no internet connection",
            "can't get online",
            "cannot get online",
            "mobile data not working",
            "cellular data not working",
        ],
        "keywords": [
            "wifi",
            "wi fi",
            "bluetooth",
            "internet",
            "connection",
            "connectivity",
            "connecting",
            "disconnecting",
            "network",
            "signal",
            "cellular",
            "mobile data",
        ],
    },

    "ACCOUNT_APPLE_ID": {
        "strong": [
            "apple id is disabled",
            "apple id locked",
            "can't sign into apple id",
            "cannot sign into apple id",
            "can't sign in to apple id",
            "cannot sign in to apple id",
            "forgot my apple id password",
            "forgot apple id password",
            "icloud not syncing",
            "icloud isn't syncing",
            "icloud is not syncing",
            "icloud syncing issue",
            "icloud sync issue",
        ],
        "keywords": [
            "apple id",
            "icloud",
            "sign in",
            "sign into",
            "login",
            "password",
            "account",
            "syncing",
            "sync",
        ],
    },

    "APP_STORE_PURCHASE": {
        "strong": [
            "app store purchase",
            "charged for an app",
            "charged for app",
            "charged twice",
            "wrong charge",
            "unknown charge",
            "unauthorized purchase",
            "want a refund",
            "need a refund",
            "request a refund",
            "app store refund",
            "can't download from app store",
            "cannot download from app store",
        ],
        "keywords": [
            "app store",
            "purchase",
            "purchased",
            "refund",
            "billing",
            "charged",
            "charge",
            "subscription",
            "payment",
        ],
    },

    "MUSIC_MEDIA": {
        "strong": [
            "apple music isn't working",
            "apple music is not working",
            "apple music won't play",
            "apple music wont play",
            "music won't play",
            "music wont play",
            "music stopped playing",
            "downloaded music deleted",
            "songs disappeared",
            "music disappeared",
            "itunes isn't working",
            "itunes is not working",
        ],
        "keywords": [
            "apple music",
            "music",
            "song",
            "songs",
            "itunes",
            "playlist",
            "album",
            "podcast",
            "aux",
            "audio",
        ],
    },

    "SCREEN_DISPLAY": {
        "strong": [
            "screen is black",
            "screen went black",
            "black screen",
            "screen is flickering",
            "screen flickers",
            "screen keeps flickering",
            "touchscreen isn't working",
            "touchscreen is not working",
            "touch screen isn't working",
            "touch screen is not working",
            "screen won't respond",
            "screen wont respond",
            "display isn't working",
            "display is not working",
            "screen is cracked",
            "cracked screen",
        ],
        "keywords": [
            "screen",
            "display",
            "touchscreen",
            "touch screen",
            "brightness",
            "flicker",
            "flickering",
        ],
    },

    "SETTINGS_FEATURE": {
        "strong": [
            "auto correct",
            "autocorrect",
            "screen recorder",
            "control center",
            "notification settings",
            "keyboard settings",
            "keyboard isn't working",
            "keyboard is not working",
            "how do i turn",
            "how can i turn",
            "how do i enable",
            "how do i disable",
            "how can i enable",
            "how can i disable",
            "how do i change",
            "how can i change",
        ],
        "keywords": [
            "keyboard",
            "emoji",
            "autocorrect",
            "auto correct",
            "notification",
            "settings",
            "control center",
            "screen recorder",
            "wallpaper",
            "siri",
            "enable",
            "disable",
            "turn on",
            "turn off",
        ],
    },

    "PERFORMANCE_STABILITY": {
        "strong": [
            "iphone is really slow",
            "iphone is very slow",
            "iphone is so slow",
            "phone is really slow",
            "phone is very slow",
            "device is really slow",
            "device is very slow",
            "iphone is laggy",
            "phone is laggy",
            "system is laggy",
            "iphone is glitchy",
            "phone is glitchy",
            "device is glitchy",
            "system is unstable",
        ],
        "keywords": [
            "slow",
            "slower",
            "lag",
            "laggy",
            "lagging",
            "glitch",
            "glitchy",
            "stability",
            "unstable",
            "performance",
            "stuttering",
        ],
    },

    "HARDWARE_DEVICE": {
        "strong": [
            "phone is cracked",
            "iphone is cracked",
            "screen is physically damaged",
            "phone is physically damaged",
            "camera isn't working",
            "camera is not working",
            "microphone isn't working",
            "microphone is not working",
            "speaker isn't working",
            "speaker is not working",
            "can't hear people on calls",
            "cannot hear people on calls",
            "can't hear anything on my iphone",
            "cannot hear anything on my iphone",
            "volume isn't working",
            "volume is not working",
            "button isn't working",
            "button is not working",
        ],
        "keywords": [
            "camera",
            "microphone",
            "mic",
            "speaker",
            "earpiece",
            "volume",
            "button",
            "broken",
            "cracked",
            "physical damage",
            "headphone",
        ],
    },
}


# =========================================================
# 2. Text normalization
# =========================================================

def normalize_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# 3. Weak-label scoring
# =========================================================

def score_intent(text, rule):
    """
    Score an intent using:
    - strong phrases: +3
    - matching keywords: +1 each
    - relevant keyword combinations: +2
    """

    text = normalize_text(text)

    score = 0

    # Strong phrases
    for phrase in rule["strong"]:
        if phrase in text:
            score += 3

    # Individual keywords
    matched_keywords = []

    for keyword in rule["keywords"]:
        if keyword in text:
            score += 1
            matched_keywords.append(keyword)

    # Extra confidence for combinations of related words.
    # These help catch natural customer language where
    # the exact strong phrase is not present.

    keyword_set = set(matched_keywords)

    # Battery
    if "battery" in keyword_set and (
        "charge slowly" in keyword_set
        or "charge time" in keyword_set
        or "battery life" in keyword_set
        or "battery drain" in keyword_set
        or "charging" in keyword_set
    ):
        score += 2

    # Connectivity
    if (
        ("wifi" in keyword_set or "wi fi" in keyword_set
         or "bluetooth" in keyword_set)
        and (
            "connecting" in keyword_set
            or "disconnecting" in keyword_set
            or "connection" in keyword_set
        )
    ):
        score += 2

    # App problems
    if (
        ("app" in keyword_set or "apps" in keyword_set)
        and (
            "crash" in keyword_set
            or "crashes" in keyword_set
            or "crashing" in keyword_set
            or "freeze" in keyword_set
            or "freezes" in keyword_set
            or "freezing" in keyword_set
            or "hangs" in keyword_set
            or "not responding" in keyword_set
        )
    ):
        score += 2

    # Apple ID / iCloud
    if (
        ("apple id" in keyword_set or "icloud" in keyword_set)
        and (
            "password" in keyword_set
            or "sign in" in keyword_set
            or "login" in keyword_set
            or "sync" in keyword_set
            or "syncing" in keyword_set
        )
    ):
        score += 2

    # App Store / purchases
    if (
        ("app store" in keyword_set or "purchase" in keyword_set
         or "purchased" in keyword_set)
        and (
            "refund" in keyword_set
            or "charged" in keyword_set
            or "billing" in keyword_set
            or "payment" in keyword_set
        )
    ):
        score += 2

    # Screen/display
    if (
        ("screen" in keyword_set or "display" in keyword_set
         or "touchscreen" in keyword_set
         or "touch screen" in keyword_set)
        and (
            "brightness" in keyword_set
            or "flicker" in keyword_set
            or "flickering" in keyword_set
        )
    ):
        score += 2

    # Hardware
    if (
        ("camera" in keyword_set
         or "microphone" in keyword_set
         or "mic" in keyword_set
         or "speaker" in keyword_set
         or "earpiece" in keyword_set
         or "button" in keyword_set)
        and (
            "broken" in keyword_set
            or "cracked" in keyword_set
            or "volume" in keyword_set
        )
    ):
        score += 2

    return score


def assign_label(text):
    """
    Assign a weak label only when there is enough evidence.

    We deliberately abstain when:
    - there is too little evidence
    - two intents are too close
    """

    scores = {}

    for intent, rule in RULES.items():
        scores[intent] = score_intent(text, rule)

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_intent, best_score = ranked[0]
    second_score = ranked[1][1]

    # No meaningful evidence
    if best_score < 3:
        return "OTHER", best_score, "ABSTAIN"

    # Two intents have exactly the same evidence
    if best_score == second_score:
        return "OTHER", best_score, "AMBIGUOUS"

    # Require a meaningful lead over the second-best intent
    if best_score - second_score < 2:
        return "OTHER", best_score, "AMBIGUOUS"

    return best_intent, best_score, "HIGH_CONFIDENCE"


# =========================================================
# 4. Load data
# =========================================================

print("Loading historical AppleSupport data...")

df = pd.read_csv(INPUT_FILE)

print(f"Input rows: {len(df)}")


# =========================================================
# 5. Apply weak labels
# =========================================================

print("\nGenerating weak labels...")

results = df["customer_message"].apply(assign_label)

df["weak_intent"] = results.apply(lambda x: x[0])
df["weak_score"] = results.apply(lambda x: x[1])
df["label_status"] = results.apply(lambda x: x[2])


# =========================================================
# 6. Show statistics
# =========================================================

print("\nLabel distribution:")
print(df["weak_intent"].value_counts())

print("\nLabel status:")
print(df["label_status"].value_counts())


# =========================================================
# 7. Keep high-confidence training examples
# =========================================================

training_df = df[
    df["label_status"] == "HIGH_CONFIDENCE"
].copy()

print(
    f"\nHigh-confidence training examples: "
    f"{len(training_df)}"
)

print(
    f"Excluded ambiguous/weak examples: "
    f"{len(df) - len(training_df)}"
)


# =========================================================
# 8. Save FULL weak-labeled dataset
# =========================================================

FULL_OUTPUT_FILE = "data/weak_labeled_all.csv"

df.to_csv(
    FULL_OUTPUT_FILE,
    index=False
)

# Also keep the high-confidence training dataset
training_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDONE")
print(f"Full labeled dataset saved: {FULL_OUTPUT_FILE}")
print(f"High-confidence training dataset saved: {OUTPUT_FILE}")