import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/ml_training_data.csv"

SAMPLE_SIZE = 500
TOP_K = 5
RANDOM_STATE = 42


print("Loading historical support data...")

df = pd.read_csv(DATA_PATH)

df = df[
    df["customer_message"].notna()
    & df["support_reply"].notna()
].copy()

df["customer_message"] = df["customer_message"].astype(str)
df["support_reply"] = df["support_reply"].astype(str)

print(f"Usable historical pairs: {len(df)}")


# ---------------------------------------------------------
# Create a fixed evaluation sample
# ---------------------------------------------------------

evaluation_df = df.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_STATE
).copy()

evaluation_indices = set(evaluation_df.index)


# Everything outside the evaluation sample is the retrieval index.
# This prevents the exact evaluation examples from being retrieved.
index_df = df[
    ~df.index.isin(evaluation_indices)
].copy()

print(f"Evaluation examples: {len(evaluation_df)}")
print(f"Retrieval index examples: {len(index_df)}")


# ---------------------------------------------------------
# Build TF-IDF index
# ---------------------------------------------------------

print("\nBuilding TF-IDF index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

index_vectors = vectorizer.fit_transform(
    index_df["customer_message"]
)

print(f"TF-IDF matrix shape: {index_vectors.shape}")


# ---------------------------------------------------------
# Evaluate retrieval
# ---------------------------------------------------------

top1_scores = []
top5_scores = []

results = []


print("\nEvaluating retrieval...")

for eval_index, row in evaluation_df.iterrows():

    query = row["customer_message"]

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        index_vectors
    )[0]

    top_indices = similarities.argsort()[-TOP_K:][::-1]

    retrieved_rows = index_df.iloc[top_indices]

    retrieved_messages = (
        retrieved_rows["customer_message"]
        .tolist()
    )

    retrieved_replies = (
        retrieved_rows["support_reply"]
        .tolist()
    )

    scores = similarities[top_indices]

    # A simple proxy for retrieval usefulness:
    # does the retrieved customer conversation share
    # meaningful vocabulary with the query?
    #
    # We record the strongest similarity and the top-5
    # average similarity for later analysis.

    top1_score = float(scores[0])
    top5_average = float(scores.mean())

    top1_scores.append(top1_score)
    top5_scores.append(top5_average)

    results.append({
        "evaluation_id": eval_index,
        "customer_message": query,
        "top1_similarity": top1_score,
        "top5_average_similarity": top5_average,
        "top1_historical_customer": retrieved_messages[0],
        "top1_historical_reply": retrieved_replies[0]
    })


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("SupportLens AI — Retrieval Evaluation")
print("=" * 70)

print(f"\nEvaluation examples: {len(results_df)}")

print(
    f"Average Top-1 similarity: "
    f"{results_df['top1_similarity'].mean():.4f}"
)

print(
    f"Median Top-1 similarity: "
    f"{results_df['top1_similarity'].median():.4f}"
)

print(
    f"Average Top-5 similarity: "
    f"{results_df['top5_average_similarity'].mean():.4f}"
)


# Useful similarity buckets

print("\nTop-1 similarity distribution:")

print(
    pd.cut(
        results_df["top1_similarity"],
        bins=[-0.01, 0.20, 0.30, 0.40, 0.50, 1.0],
        labels=[
            "<0.20",
            "0.20-0.30",
            "0.30-0.40",
            "0.40-0.50",
            ">=0.50"
        ]
    ).value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

OUTPUT_PATH = "data/retrieval_evaluation.csv"

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nDetailed results saved to:")
print(OUTPUT_PATH)


# ---------------------------------------------------------
# Show strongest and weakest examples
# ---------------------------------------------------------

print("\nStrongest retrieval examples:")

strongest = results_df.sort_values(
    "top1_similarity",
    ascending=False
).head(5)

for _, row in strongest.iterrows():

    print("\n" + "-" * 70)
    print(f"Similarity: {row['top1_similarity']:.4f}")
    print(f"Customer: {row['customer_message']}")
    print(
        f"Retrieved customer: "
        f"{row['top1_historical_customer']}"
    )


print("\nWeakest retrieval examples:")

weakest = results_df.sort_values(
    "top1_similarity",
    ascending=True
).head(5)

for _, row in weakest.iterrows():

    print("\n" + "-" * 70)
    print(f"Similarity: {row['top1_similarity']:.4f}")
    print(f"Customer: {row['customer_message']}")
    print(
        f"Retrieved customer: "
        f"{row['top1_historical_customer']}"
    )