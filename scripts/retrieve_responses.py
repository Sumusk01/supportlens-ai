import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/ml_training_data.csv"


print("Loading historical support data...")

df = pd.read_csv(DATA_PATH)

print(f"Historical examples: {len(df)}")


# Remove empty messages
df = df[
    df["customer_message"].notna()
    & df["support_reply"].notna()
].copy()

df["customer_message"] = df["customer_message"].astype(str)
df["support_reply"] = df["support_reply"].astype(str)


print(f"Usable examples: {len(df)}")


print("\nBuilding TF-IDF index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

customer_vectors = vectorizer.fit_transform(
    df["customer_message"]
)

print(
    f"TF-IDF matrix shape: {customer_vectors.shape}"
)


def retrieve_responses(customer_message, top_k=5):

    query_vector = vectorizer.transform(
        [customer_message]
    )

    similarities = cosine_similarity(
        query_vector,
        customer_vectors
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "similarity": float(similarities[index]),
            "historical_customer": df.iloc[index]["customer_message"],
            "historical_reply": df.iloc[index]["support_reply"]
        })

    return results


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("SupportLens AI — Historical Response Retrieval")
    print("=" * 70)

    test_message = input(
        "\nEnter a customer message:\n> "
    ).strip()

    if not test_message:
        print("No message entered.")
        exit()

    results = retrieve_responses(
        test_message,
        top_k=5
    )

    print("\nTop historical matches:")

    for i, result in enumerate(results, start=1):

        print("\n" + "-" * 70)
        print(f"Match #{i}")
        print(f"Similarity: {result['similarity']:.4f}")

        print("\nHistorical customer:")
        print(result["historical_customer"])

        print("\nAppleSupport reply:")
        print(result["historical_reply"])