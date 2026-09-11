import pandas as pd
import re
import random

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

FILE_PATH = "data/raw/twcs.csv"
OUTPUT_FILE = "data/golden_set.csv"

BRAND = "AppleSupport"

RANDOM_SEED = 42

# Number of examples we want
TARGET_PER_INTENT = 16

# --------------------------------------------------
# 2. CANDIDATE INTENTS
# --------------------------------------------------

candidate_intents = {

    "IOS_UPDATE": [
        "ios",
        "update",
        "updated",
        "software update",
        "install",
        "installation",
        "downgrade"
    ],

    "BATTERY_POWER": [
        "battery",
        "battery life",
        "drain",
        "charging",
        "charge",
        "charger",
        "power"
    ],

    "APP_PROBLEM": [
        "app",
        "apps",
        "crash",
        "crashing",
        "application"
    ],

    "CONNECTIVITY": [
        "wifi",
        "wi fi",
        "bluetooth",
        "cellular",
        "network",
        "internet",
        "connection",
        "connect"
    ],

    "ACCOUNT_APPLE_ID": [
        "apple id",
        "password",
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
        "credit"
    ],

    "MUSIC_MEDIA": [
        "apple music",
        "music",
        "spotify",
        "podcast",
        "podcasts"
    ],

    "SCREEN_DISPLAY": [
        "screen",
        "display",
        "touch",
        "brightness",
        "lock screen",
        "home screen"
    ],

    "SETTINGS_FEATURE": [
        "settings",
        "control center",
        "notification",
        "notifications",
        "feature",
        "how do i",
        "how can i"
    ],

    "PERFORMANCE_STABILITY": [
        "slow",
        "lag",
        "freeze",
        "freezing",
        "restart",
        "restarting",
        "performance"
    ],

    "HARDWARE_DEVICE": [
        "speaker",
        "microphone",
        "camera",
        "button",
        "broken",
        "screen damage",
        "headphone"
    ]
}

# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

print("Loading dataset...")

columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "text",
    "response_tweet_id"
]

df = pd.read_csv(
    FILE_PATH,
    usecols=columns,
    low_memory=False
)

print("Rows loaded:", len(df))

# --------------------------------------------------
# 4. PREPARE LOOKUP
# --------------------------------------------------

df["tweet_id"] = df["tweet_id"].astype(str)

tweet_lookup = df.set_index("tweet_id")[
    ["author_id", "inbound", "text"]
]

# --------------------------------------------------
# 5. FIND APPLESUPPORT CONVERSATIONS
# --------------------------------------------------

brand_tweets = df[
    (df["author_id"] == BRAND) &
    (df["inbound"] == False)
].copy()

brand_tweets["response_tweet_id"] = (
    brand_tweets["response_tweet_id"]
    .fillna("")
    .astype(str)
)

links = brand_tweets[
    ["tweet_id", "text", "response_tweet_id"]
].copy()

links["response_tweet_id"] = (
    links["response_tweet_id"].str.split(",")
)

links = links.explode("response_tweet_id")

links["response_tweet_id"] = (
    links["response_tweet_id"].str.strip()
)

links = links[
    links["response_tweet_id"].ne("")
]

customer_info = tweet_lookup.reindex(
    links["response_tweet_id"]
)

links["customer_inbound"] = (
    customer_info["inbound"].values
)

links["customer_text"] = (
    customer_info["text"].values
)

pairs = links[
    (links["customer_inbound"] == True) &
    (links["customer_text"].notna())
].copy()

print("Usable messages:", len(pairs))

# --------------------------------------------------
# 6. CLEAN TEXT
# --------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


pairs["clean_text"] = pairs[
    "customer_text"
].apply(clean_text)

# --------------------------------------------------
# 7. FIND CANDIDATES FOR EACH INTENT
# --------------------------------------------------

random.seed(RANDOM_SEED)

selected = []

used_ids = set()

for intent, keywords in candidate_intents.items():

    patterns = []

    for keyword in keywords:

        escaped = re.escape(keyword)

        escaped = escaped.replace(
            r"\ ",
            r"\s+"
        )

        patterns.append(
            rf"\b{escaped}\b"
        )

    pattern = "|".join(patterns)

    matches = pairs[
        pairs["clean_text"].str.contains(
            pattern,
            regex=True,
            na=False
        )
    ].copy()

    # Remove very short messages
    matches = matches[
        matches["clean_text"].str.len() >= 20
    ]

    # Remove duplicate tweet IDs
    matches = matches[
        ~matches["tweet_id"].isin(used_ids)
    ]

    # Random sample
    sample_size = min(
        TARGET_PER_INTENT,
        len(matches)
    )

    if sample_size > 0:

        sample = matches.sample(
            n=sample_size,
            random_state=RANDOM_SEED
        )

        for _, row in sample.iterrows():

            selected.append({
                "tweet_id": row["tweet_id"],
                "customer_message": row["customer_text"],
                "suggested_intent": intent,
                "final_intent": ""
            })

            used_ids.add(row["tweet_id"])

# --------------------------------------------------
# 8. ADD RANDOM EXAMPLES
# --------------------------------------------------

remaining = pairs[
    ~pairs["tweet_id"].isin(used_ids)
].copy()

random_count = 20

if len(remaining) >= random_count:

    random_examples = remaining.sample(
        n=random_count,
        random_state=RANDOM_SEED
    )

    for _, row in random_examples.iterrows():

        selected.append({
            "tweet_id": row["tweet_id"],
            "customer_message": row["customer_text"],
            "suggested_intent": "REVIEW_MANUALLY",
            "final_intent": ""
        })

# --------------------------------------------------
# 9. CREATE GOLDEN SET
# --------------------------------------------------

golden = pd.DataFrame(selected)

# Shuffle final dataset
golden = golden.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# Add evaluation ID
golden.insert(
    0,
    "example_id",
    range(1, len(golden) + 1)
)

# Save
golden.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nGolden-set candidate created.")

print(
    "Examples:",
    len(golden)
)

print(
    "\nSuggested intent distribution:"
)

print(
    golden["suggested_intent"].value_counts()
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)