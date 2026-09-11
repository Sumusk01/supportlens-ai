import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "ml_training_data.csv"
OUTPUT_FILE = BASE_DIR / "data" / "historical_replies_sample_100.csv"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SAMPLE_SIZE = 100
RANDOM_STATE = 42


# --------------------------------------------------
# Load historical AppleSupport conversations
# --------------------------------------------------

print("Loading historical AppleSupport data...")

df = pd.read_csv(INPUT_FILE)

df = df.dropna(
    subset=[
        "customer_message",
        "support_reply"
    ]
)

print(f"Total historical pairs: {len(df):,}")


# --------------------------------------------------
# Clean text
# --------------------------------------------------

df["customer_message"] = (
    df["customer_message"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df["support_reply"] = (
    df["support_reply"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# --------------------------------------------------
# Remove empty replies
# --------------------------------------------------

df = df[
    (df["customer_message"] != "")
    & (df["support_reply"] != "")
].copy()


# --------------------------------------------------
# Sample 100 historical conversations
# --------------------------------------------------

sample_df = df.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_STATE
).copy()


# --------------------------------------------------
# Create inspection-friendly dataset
# --------------------------------------------------

inspection_df = sample_df[
    [
        "customer_tweet_id",
        "customer_message",
        "support_tweet_id",
        "support_reply"
    ]
].copy()

inspection_df.insert(
    0,
    "sample_number",
    range(1, len(inspection_df) + 1)
)


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

inspection_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# --------------------------------------------------
# Print replies grouped in terminal
# --------------------------------------------------

print("\n" + "=" * 80)
print("100 HISTORICAL APPLESUPPORT REPLIES")
print("=" * 80)

for _, row in inspection_df.iterrows():

    print(
        f"\n[{row['sample_number']}]"
    )

    print(
        f"Customer:\n{row['customer_message']}"
    )

    print(
        f"\nAppleSupport:\n{row['support_reply']}"
    )

    print("-" * 80)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nInspection sample created successfully.")

print(
    f"Sample size: {len(inspection_df)}"
)

print(
    f"Saved CSV:\n{OUTPUT_FILE}"
)

print(
    "\nThe terminal output is grouped as:"
    "\nCustomer message → AppleSupport reply"
)