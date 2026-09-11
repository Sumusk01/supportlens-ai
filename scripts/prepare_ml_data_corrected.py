import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "twcs.csv"
GOLDEN_FILE = BASE_DIR / "data" / "golden_set.csv"
OUTPUT_FILE = BASE_DIR / "data" / "ml_training_data.csv"

print("Loading dataset...")

df = pd.read_csv(
    RAW_FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "in_response_to_tweet_id"
    ]
)

golden = pd.read_csv(GOLDEN_FILE)

# ---------------------------------------------------------
# 1. Normalize tweet IDs
# ---------------------------------------------------------

df["tweet_id"] = pd.to_numeric(
    df["tweet_id"], errors="coerce"
).astype("Int64")

df["in_response_to_tweet_id"] = pd.to_numeric(
    df["in_response_to_tweet_id"], errors="coerce"
).astype("Int64")

golden["tweet_id"] = pd.to_numeric(
    golden["tweet_id"], errors="coerce"
).astype("Int64")

# ---------------------------------------------------------
# 2. Get Golden Set information
# ---------------------------------------------------------

golden_support_ids = set(
    golden["tweet_id"].dropna().astype(int)
)

golden_messages = set(
    golden["customer_message"]
    .dropna()
    .astype(str)
    .str.strip()
)

print(f"Golden examples: {len(golden)}")
print(f"Golden support tweet IDs: {len(golden_support_ids)}")

# ---------------------------------------------------------
# 3. Identify AppleSupport replies to customers
# ---------------------------------------------------------

support_replies = df[
    (df["author_id"] == "AppleSupport") &
    (df["inbound"] == False) &
    (df["in_response_to_tweet_id"].notna())
].copy()

print(f"AppleSupport reply candidates: {len(support_replies)}")

# ---------------------------------------------------------
# 4. Identify inbound customer tweets
# ---------------------------------------------------------

customer_tweets = df[
    df["inbound"] == True
][
    ["tweet_id", "text"]
].copy()

customer_tweets = customer_tweets.rename(
    columns={
        "tweet_id": "customer_tweet_id",
        "text": "customer_message"
    }
)

# ---------------------------------------------------------
# 5. Connect AppleSupport reply -> customer message
# ---------------------------------------------------------

pairs = support_replies.merge(
    customer_tweets,
    left_on="in_response_to_tweet_id",
    right_on="customer_tweet_id",
    how="inner"
)

print(f"Usable customer -> AppleSupport pairs: {len(pairs)}")

# ---------------------------------------------------------
# 6. Remove Golden Set leakage
# ---------------------------------------------------------

before = len(pairs)

# Remove Golden Set support replies
pairs = pairs[
    ~pairs["tweet_id"].isin(golden_support_ids)
].copy()

# Remove exact Golden Set customer messages
pairs = pairs[
    ~pairs["customer_message"]
    .astype(str)
    .str.strip()
    .isin(golden_messages)
].copy()

after = len(pairs)

print(f"Removed from training to prevent leakage: {before - after}")
print(f"Final training pairs: {after}")

# ---------------------------------------------------------
# 7. Keep only useful columns
# ---------------------------------------------------------

training_data = pairs[
    [
        "customer_tweet_id",
        "customer_message",
        "tweet_id",
        "text"
    ]
].copy()

training_data = training_data.rename(
    columns={
        "tweet_id": "support_tweet_id",
        "text": "support_reply"
    }
)

# Remove empty messages
training_data = training_data[
    training_data["customer_message"].notna() &
    training_data["support_reply"].notna()
].copy()

training_data.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("DONE")
print(f"Saved: {OUTPUT_FILE}")
print(f"Rows: {len(training_data)}")
print()
print("Columns:")
print(list(training_data.columns))

print()
print("Sample:")
print(
    training_data[
        ["customer_message", "support_reply"]
    ].head(5).to_string(index=False)
)