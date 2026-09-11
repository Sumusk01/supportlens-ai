import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRAINING_FILE = BASE_DIR / "data" / "ml_training_data.csv"
OUTPUT_FILE = BASE_DIR / "data" / "retrieval_review_set.csv"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SAMPLE_SIZE = 100
TOP_K = 5
RANDOM_STATE = 42


# --------------------------------------------------
# Load historical customer-support pairs
# --------------------------------------------------

print("Loading training data...")

df = pd.read_csv(TRAINING_FILE)

df = df.dropna(subset=["customer_tweet_id", "customer_message", "support_reply"])

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

print(f"Total usable historical pairs: {len(df):,}")


# --------------------------------------------------
# Select held-out evaluation queries
# --------------------------------------------------

if len(df) < SAMPLE_SIZE:
    raise ValueError(
        f"Dataset contains only {len(df)} rows, "
        f"but {SAMPLE_SIZE} evaluation queries were requested."
    )

eval_df = df.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_STATE
).copy()

eval_indices = set(eval_df.index)


# --------------------------------------------------
# Build retrieval corpus
#
# IMPORTANT:
# Evaluation queries are removed from the retrieval
# corpus so the system cannot retrieve the exact
# same customer/reply pair.
# --------------------------------------------------

corpus_df = df.loc[~df.index.isin(eval_indices)].copy()

print(f"Evaluation queries: {len(eval_df)}")
print(f"Retrieval corpus: {len(corpus_df):,}")


# --------------------------------------------------
# Build TF-IDF retrieval index
# --------------------------------------------------

print("\nBuilding TF-IDF index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

corpus_matrix = vectorizer.fit_transform(
    corpus_df["customer_message"]
)

print(
    f"TF-IDF matrix shape: "
    f"{corpus_matrix.shape[0]:,} x {corpus_matrix.shape[1]:,}"
)


# --------------------------------------------------
# Retrieve top-K historical examples
# --------------------------------------------------

print("\nRetrieving historical examples...")

results = []

for count, (query_index, query_row) in enumerate(
    eval_df.iterrows(),
    start=1
):

    query_text = query_row["customer_message"]

    query_vector = vectorizer.transform([query_text])

    similarities = cosine_similarity(
        query_vector,
        corpus_matrix
    ).flatten()

    # Get highest-scoring documents
    top_indices = similarities.argsort()[-TOP_K:][::-1]

    for rank, corpus_position in enumerate(top_indices, start=1):

        retrieved_row = corpus_df.iloc[corpus_position]

        results.append({
            "query_id": str(query_row["customer_tweet_id"]),
            "query_message": query_text,

            "rank": rank,

            "similarity": round(
                float(similarities[corpus_position]),
                4
            ),

            "historical_customer_tweet_id": str(
                retrieved_row["customer_tweet_id"]
            ),

            "historical_customer_message": (
                retrieved_row["customer_message"]
            ),

            "historical_support_reply": (
                retrieved_row["support_reply"]
            ),

            # Leave this blank for manual evaluation.
            #
            # 2 = directly relevant
            # 1 = related/useful
            # 0 = irrelevant
            "relevance": ""
        })

    if count % 10 == 0:
        print(f"Processed {count}/{len(eval_df)} queries")


# --------------------------------------------------
# Save review file
# --------------------------------------------------

review_df = pd.DataFrame(results)

review_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nDone!")

print(f"Review rows: {len(review_df):,}")
print(f"Expected rows: {SAMPLE_SIZE * TOP_K:,}")

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\nRelevance labels:")
print("2 = Directly relevant")
print("1 = Related/useful")
print("0 = Irrelevant")