import pandas as pd
import re

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

FILE_PATH = "data/raw/twcs.csv"
BRAND = "AppleSupport"

# --------------------------------------------------
# 2. LOAD DATA
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
# 3. PREPARE LOOKUP
# --------------------------------------------------

df["tweet_id"] = df["tweet_id"].astype(str)

tweet_lookup = df.set_index("tweet_id")[
    ["author_id", "inbound", "text"]
]

# --------------------------------------------------
# 4. FIND APPLESUPPORT -> CUSTOMER LINKS
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

print("\nUsable AppleSupport customer messages:", len(pairs))

# --------------------------------------------------
# 5. CLEAN TEXT
# --------------------------------------------------

def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove punctuation
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


pairs["clean_text"] = pairs["customer_text"].apply(clean_text)

# --------------------------------------------------
# 6. CANDIDATE INTENT KEYWORDS
# --------------------------------------------------

candidate_intents = {

    "Battery / Power": [
        "battery",
        "battery life",
        "drain",
        "charging",
        "charge",
        "charger",
        "power"
    ],

    "iOS Update / Software": [
        "ios",
        "update",
        "updated",
        "software update",
        "install",
        "installation",
        "downgrade"
    ],

    "App Problems": [
        "app",
        "apps",
        "crash",
        "crashing",
        "freeze",
        "freezing",
        "application"
    ],

    "iPhone / Device Problems": [
        "iphone",
        "ipad",
        "device",
        "phone",
        "macbook",
        "mac",
        "imac"
    ],

    "Screen / Display": [
        "screen",
        "display",
        "touch",
        "brightness",
        "lock screen",
        "home screen"
    ],

    "Connectivity": [
        "wifi",
        "wi fi",
        "bluetooth",
        "cellular",
        "network",
        "internet",
        "connection",
        "connect"
    ],

    "Apple ID / Account": [
        "apple id",
        "password",
        "login",
        "log in",
        "account",
        "icloud"
    ],

    "App Store / Purchases": [
        "app store",
        "itunes",
        "purchase",
        "purchased",
        "refund",
        "payment",
        "subscription",
        "credit"
    ],

    "Apple Music": [
        "apple music",
        "music",
        "spotify",
        "podcast",
        "podcasts"
    ],

    "Settings / Features": [
        "settings",
        "control center",
        "notification",
        "notifications",
        "feature",
        "how do i",
        "how can i"
    ],

    "Performance / Stability": [
        "slow",
        "lag",
        "freeze",
        "freezing",
        "restart",
        "restarting",
        "crash",
        "crashing",
        "performance"
    ],

    "Other / Unclear": []
}

# --------------------------------------------------
# 7. COUNT MESSAGE COVERAGE
# --------------------------------------------------

print("\n" + "=" * 70)
print("CANDIDATE INTENT COVERAGE")
print("=" * 70)

for intent, keywords in candidate_intents.items():

    if intent == "Other / Unclear":
        continue

    pattern_parts = []

    for keyword in keywords:

        # Convert spaces into flexible whitespace
        escaped = re.escape(keyword)
        escaped = escaped.replace(r"\ ", r"\s+")

        pattern_parts.append(
            rf"\b{escaped}\b"
        )

    pattern = "|".join(pattern_parts)

    matches = pairs["clean_text"].str.contains(
        pattern,
        regex=True,
        na=False
    )

    count = matches.sum()

    percentage = (
        count / len(pairs) * 100
    )

    print(
        f"{intent:30} "
        f"{count:6} messages "
        f"({percentage:5.2f}%)"
    )

# --------------------------------------------------
# 8. SHOW EXAMPLES FOR EACH CANDIDATE
# --------------------------------------------------

print("\n" + "=" * 70)
print("EXAMPLES BY CANDIDATE INTENT")
print("=" * 70)

for intent, keywords in candidate_intents.items():

    if intent == "Other / Unclear":
        continue

    pattern_parts = []

    for keyword in keywords:

        escaped = re.escape(keyword)
        escaped = escaped.replace(r"\ ", r"\s+")

        pattern_parts.append(
            rf"\b{escaped}\b"
        )

    pattern = "|".join(pattern_parts)

    matches = pairs[
        pairs["clean_text"].str.contains(
            pattern,
            regex=True,
            na=False
        )
    ]

    print(f"\n--- {intent.upper()} ---")

    examples = matches.head(5)

    for _, row in examples.iterrows():
        print("\n", row["customer_text"])