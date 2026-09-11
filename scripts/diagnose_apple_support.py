import pandas as pd

RAW_FILE = "data/raw/twcs.csv"
GOLDEN_FILE = "data/golden_set.csv"

golden = pd.read_csv(GOLDEN_FILE)

golden_ids = set(
    golden["tweet_id"].astype(str)
)

raw = pd.read_csv(
    RAW_FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
)

raw["tweet_id"] = raw["tweet_id"].astype(str)
raw["response_tweet_id"] = (
    raw["response_tweet_id"].fillna("").astype(str)
)
raw["in_response_to_tweet_id"] = (
    raw["in_response_to_tweet_id"].fillna("").astype(str)
)

golden_rows = raw[
    raw["tweet_id"].isin(golden_ids)
].copy()

print("=" * 70)
print("GOLDEN SET RAW DATA CHECK")
print("=" * 70)

print(f"Golden examples: {len(golden):,}")
print(f"Matching raw rows: {len(golden_rows):,}")

print("\nInbound distribution:")
print(golden_rows["inbound"].value_counts(dropna=False))

print("\nSample Golden Set rows:")
print("-" * 70)

for _, row in golden_rows.head(10).iterrows():

    print(f"\nTweet ID: {row['tweet_id']}")
    print(f"Inbound: {row['inbound']}")
    print(f"Author ID: {row['author_id']}")
    print(f"Response Tweet ID: {row['response_tweet_id']}")
    print(
        f"In Response To: "
        f"{row['in_response_to_tweet_id']}"
    )
    print(f"Text: {row['text']}")

print("\n" + "=" * 70)
print("GOLDEN SET LINK ANALYSIS")
print("=" * 70)

# IDs referenced by Golden Set tweets
response_ids = set()

for value in golden_rows["response_tweet_id"]:

    if value:
        for x in value.split(","):
            x = x.strip()
            if x:
                response_ids.add(x)

parent_ids = set()

for value in golden_rows["in_response_to_tweet_id"]:

    if value:
        for x in value.split(","):
            x = x.strip()
            if x:
                parent_ids.add(x)

print(f"Response IDs referenced: {len(response_ids):,}")
print(f"Parent IDs referenced: {len(parent_ids):,}")

print("\nResponse IDs that exist in raw data:")

existing_response_ids = response_ids.intersection(
    set(raw["tweet_id"])
)

print(f"{len(existing_response_ids):,}")

print("\nParent IDs that exist in raw data:")

existing_parent_ids = parent_ids.intersection(
    set(raw["tweet_id"])
)

print(f"{len(existing_parent_ids):,}")

print("\n" + "=" * 70)
print("AUTHOR ANALYSIS")
print("=" * 70)

print("\nGolden Set author IDs:")
print(
    golden_rows["author_id"]
    .value_counts()
    .head(10)
)

print("\nInbound=True examples:")
print(
    golden_rows[
        golden_rows["inbound"] == True
    ][["tweet_id", "text"]]
    .head(5)
    .to_string(index=False)
)

print("\nInbound=False examples:")
print(
    golden_rows[
        golden_rows["inbound"] == False
    ][["tweet_id", "text"]]
    .head(5)
    .to_string(index=False)
)