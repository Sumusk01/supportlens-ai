import pandas as pd

file_path = "data/raw/twcs.csv"

candidates = [
    "Uber_Support",
    "SpotifyCares",
    "AskPayPal",
    "BofA_Help",
    "AppleSupport"
]

# Only load columns needed for conversation linking
columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id"
]

print("Loading dataset...")

df = pd.read_csv(
    file_path,
    usecols=columns,
    low_memory=False
)

print("Rows loaded:", len(df))

# Convert IDs to strings so they can be matched reliably
df["tweet_id"] = df["tweet_id"].astype(str)

# Build a lookup:
# tweet_id -> author_id, inbound, text
tweet_lookup = df.set_index("tweet_id")[
    ["author_id", "inbound", "text"]
]

print("\n--- USABLE CUSTOMER -> BRAND CONVERSATIONS ---")

for brand in candidates:

    # Get tweets written by this brand
    brand_tweets = df[
        (df["author_id"] == brand) &
        (df["inbound"] == False)
    ].copy()

    # A brand tweet can have one or more response tweet IDs.
    # Split them so each response becomes a separate relationship.
    brand_tweets["response_tweet_id"] = (
        brand_tweets["response_tweet_id"]
        .fillna("")
        .astype(str)
    )

    links = brand_tweets[
        ["tweet_id", "text", "response_tweet_id"]
    ].copy()

    links["response_tweet_id"] = links[
        "response_tweet_id"
    ].str.split(",")

    links = links.explode("response_tweet_id")

    links["response_tweet_id"] = (
        links["response_tweet_id"]
        .str.strip()
    )

    # Remove empty response IDs
    links = links[
        links["response_tweet_id"].ne("")
    ]

    # Look up the tweet that the brand responded to
    customer_info = tweet_lookup.reindex(
        links["response_tweet_id"]
    )

    links["customer_author_id"] = (
        customer_info["author_id"].values
    )

    links["customer_inbound"] = (
        customer_info["inbound"].values
    )

    links["customer_text"] = (
        customer_info["text"].values
    )

    # Keep only actual customer tweets
    usable = links[
        (links["customer_inbound"] == True) &
        (links["customer_text"].notna())
    ].copy()

    print(f"\n===== {brand} =====")
    print("Brand replies:", len(brand_tweets))
    print("Usable customer -> brand pairs:", len(usable))
    print(
        "Unique customer messages:",
        usable["response_tweet_id"].nunique()
    )

    print("\nSample conversation pairs:")

    for _, row in usable.head(3).iterrows():

        print("\nCUSTOMER:")
        print(row["customer_text"])

        print("\nBRAND:")
        print(row["text"])

        print("-" * 60)