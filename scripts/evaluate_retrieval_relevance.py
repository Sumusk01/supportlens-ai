import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "retrieval_review_set.csv"
OUTPUT_FILE = BASE_DIR / "data" / "retrieval_relevance_summary.csv"


# --------------------------------------------------
# Load labelled retrieval review set
# --------------------------------------------------

print("Loading labelled retrieval review set...")

df = pd.read_csv(INPUT_FILE)

print(f"Total rows: {len(df):,}")


# --------------------------------------------------
# Basic validation
# --------------------------------------------------

required_columns = [
    "query_id",
    "query_message",
    "rank",
    "similarity",
    "historical_customer_message",
    "historical_support_reply",
    "relevance"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Convert relevance to numeric
df["relevance"] = pd.to_numeric(
    df["relevance"],
    errors="coerce"
)

# Remove accidental blank/invalid labels
if df["relevance"].isna().any():
    invalid_count = df["relevance"].isna().sum()

    raise ValueError(
        f"Found {invalid_count} blank or invalid relevance labels."
    )


# Check allowed labels
allowed_labels = {0, 1, 2}

actual_labels = set(df["relevance"].astype(int).unique())

unexpected_labels = actual_labels - allowed_labels

if unexpected_labels:
    raise ValueError(
        f"Unexpected relevance labels found: "
        f"{unexpected_labels}. "
        f"Only 0, 1 and 2 are allowed."
    )


# --------------------------------------------------
# Check query structure
# --------------------------------------------------

query_counts = df.groupby("query_id").size()

if not (query_counts == 5).all():
    print(
        "\nWARNING: Not every query has exactly 5 "
        "retrieved examples."
    )

number_of_queries = df["query_id"].nunique()

print(f"Unique evaluation queries: {number_of_queries}")


# --------------------------------------------------
# Label distribution
# --------------------------------------------------

label_counts = (
    df["relevance"]
    .astype(int)
    .value_counts()
    .sort_index()
)

print("\nOverall relevance distribution:")

for label in [0, 1, 2]:
    count = label_counts.get(label, 0)
    percentage = count / len(df) * 100

    if label == 0:
        meaning = "Irrelevant"
    elif label == 1:
        meaning = "Related/useful"
    else:
        meaning = "Directly relevant"

    print(
        f"  {label} ({meaning}): "
        f"{count} ({percentage:.1f}%)"
    )


# --------------------------------------------------
# Calculate query-level metrics
# --------------------------------------------------

query_results = []

for query_id, group in df.groupby("query_id"):

    # Sort by retrieval rank
    group = group.sort_values("rank")

    relevance_values = group["relevance"].astype(int).tolist()

    # Top-1 result
    top1_relevance = relevance_values[0]

    # Is at least one directly relevant result
    top5_has_direct = 2 in relevance_values

    # Is at least one useful result?
    # 1 = related/useful
    # 2 = directly relevant
    top5_has_useful = any(
        value >= 1
        for value in relevance_values
    )

    # Average relevance across all 5 retrieved examples
    average_relevance = sum(relevance_values) / len(
        relevance_values
    )

    query_results.append({
        "query_id": query_id,
        "top1_relevance": top1_relevance,
        "top5_has_direct": int(top5_has_direct),
        "top5_has_useful": int(top5_has_useful),
        "average_relevance": average_relevance
    })


query_metrics = pd.DataFrame(query_results)


# --------------------------------------------------
# Overall retrieval metrics
# --------------------------------------------------

top1_direct_rate = (
    query_metrics["top1_relevance"] == 2
).mean()

top1_useful_rate = (
    query_metrics["top1_relevance"] >= 1
).mean()

top5_direct_rate = (
    query_metrics["top5_has_direct"] == 1
).mean()

top5_useful_rate = (
    query_metrics["top5_has_useful"] == 1
).mean()

mean_relevance = (
    query_metrics["average_relevance"].mean()
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n" + "=" * 55)
print("RETRIEVAL RELEVANCE EVALUATION")
print("=" * 55)

print(
    f"\nEvaluation queries: "
    f"{number_of_queries}"
)

print(
    f"Retrieved examples per query: "
    f"{len(df) // number_of_queries}"
)

print(
    f"\nTop-1 Direct Relevance: "
    f"{top1_direct_rate:.3f} "
    f"({top1_direct_rate * 100:.1f}%)"
)

print(
    f"Top-1 Useful Relevance (1 or 2): "
    f"{top1_useful_rate:.3f} "
    f"({top1_useful_rate * 100:.1f}%)"
)

print(
    f"\nTop-5 Direct Relevance "
    f"(at least one 2): "
    f"{top5_direct_rate:.3f} "
    f"({top5_direct_rate * 100:.1f}%)"
)

print(
    f"Top-5 Useful Evidence "
    f"(at least one 1 or 2): "
    f"{top5_useful_rate:.3f} "
    f"({top5_useful_rate * 100:.1f}%)"
)

print(
    f"\nMean Relevance Across Top-5: "
    f"{mean_relevance:.3f}"
)


# --------------------------------------------------
# Relevance by retrieval rank
# --------------------------------------------------

print("\nRelevance by retrieval rank:")

rank_summary = (
    df.groupby("rank")["relevance"]
    .agg(
        mean_relevance="mean",
        direct_relevance=lambda x: (x == 2).mean(),
        useful_relevance=lambda x: (x >= 1).mean()
    )
    .reset_index()
)

for _, row in rank_summary.iterrows():

    print(
        f"  Rank {int(row['rank'])}: "
        f"mean={row['mean_relevance']:.3f}, "
        f"direct={row['direct_relevance'] * 100:.1f}%, "
        f"useful={row['useful_relevance'] * 100:.1f}%"
    )


# --------------------------------------------------
# Find complete retrieval failures
#
# Cases where all 5 retrieved examples
# were judged irrelevant.
# --------------------------------------------------

failed_queries = query_metrics[
    (query_metrics["top5_has_useful"] == 0)
]

print(
    f"\nQueries with NO useful result in top-5: "
    f"{len(failed_queries)}"
)


# --------------------------------------------------
# Save query-level summary
# --------------------------------------------------

query_metrics.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved query-level metrics to:"
    f"\n{OUTPUT_FILE}"
)

print("\nEvaluation complete.")