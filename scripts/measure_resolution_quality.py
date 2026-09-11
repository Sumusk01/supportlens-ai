import pandas as pd
import re

file_path = "data/raw/twcs.csv"

BRAND = "AppleSupport"

columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "text",
    "response_tweet_id"
]

print("Loading dataset...")

df = pd.read_csv(
    file_path,
    usecols=columns,
    low_memory=False
)

df["tweet_id"] = df["tweet_id"].astype(str)

# Lookup table for finding the customer tweet
tweet_lookup = df.set_index("tweet_id")[
    ["author_id", "inbound", "text"]
]

# AppleSupport replies
brand_tweets = df[
    (df["author_id"] == BRAND) &
    (df["inbound"] == False)
].copy()

brand_tweets["response_tweet_id"] = (
    brand_tweets["response_tweet_id"]
    .fillna("")
    .astype(str)
)

# One response ID per row
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

# Find the customer tweet
customer_info = tweet_lookup.reindex(
    links["response_tweet_id"]
)

links["customer_inbound"] = (
    customer_info["inbound"].values
)

links["customer_text"] = (
    customer_info["text"].values
)

# Keep valid customer -> AppleSupport pairs
pairs = links[
    (links["customer_inbound"] == True) &
    (links["customer_text"].notna())
].copy()

print("\nTotal usable AppleSupport pairs:", len(pairs))


# -------------------------------------------------
# Simple heuristic for identifying useful replies
# -------------------------------------------------

def classify_reply(text):

    text = str(text).lower()

    # Mostly administrative / escalation responses
    dm_patterns = [
        r"\bdm\b",
        r"direct message",
        r"private message",
        r"message us",
        r"contact us",
        r"reach out",
        r"call us",
        r"give us a call",
        r"we'll call",
        r"we will call"
    ]

    short_ack_patterns = [
        r"^thanks[.! ]*$",
        r"^thank you[.! ]*$",
        r"^you're welcome[.! ]*$",
        r"^glad we could help[.! ]*$"
    ]

    if any(re.search(pattern, text) for pattern in short_ack_patterns):
        return "ACKNOWLEDGEMENT"

    if any(re.search(pattern, text) for pattern in dm_patterns):
        return "ESCALATION_DM"

    # Replies containing troubleshooting/action language
    action_patterns = [
        r"\btry\b",
        r"\bcheck\b",
        r"\bgo to\b",
        r"\bopen\b",
        r"\bselect\b",
        r"\bupdate\b",
        r"\brestart\b",
        r"\breset\b",
        r"\binstall\b",
        r"\breinstall\b",
        r"\bremove\b",
        r"\bturn on\b",
        r"\bturn off\b",
        r"\bsettings\b",
        r"\bsteps\b",
        r"\bfollow\b",
        r"\btap\b",
        r"\bclick\b"
    ]

    if any(re.search(pattern, text) for pattern in action_patterns):
        return "ACTIONABLE"

    # Longer replies are more likely to contain useful information
    if len(text.split()) >= 15:
        return "INFORMATIONAL"

    return "OTHER"


pairs["reply_type"] = pairs["text"].apply(classify_reply)


print("\n--- REPLY TYPE COUNTS ---")

counts = pairs["reply_type"].value_counts()

print(counts)

print("\n--- REPLY TYPE PERCENTAGES ---")

percentages = (
    pairs["reply_type"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(percentages)


# -------------------------------------------------
# Show examples
# -------------------------------------------------

for category in [
    "ACTIONABLE",
    "INFORMATIONAL",
    "ESCALATION_DM",
    "ACKNOWLEDGEMENT",
    "OTHER"
]:

    print(f"\n===== {category} EXAMPLES =====")

    examples = pairs[
        pairs["reply_type"] == category
    ].head(3)

    for _, row in examples.iterrows():

        print("\nCUSTOMER:")
        print(row["customer_text"])

        print("\nAPPLE SUPPORT:")
        print(row["text"])

        print("-" * 60)