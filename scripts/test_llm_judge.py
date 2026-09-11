import os
import time
import pandas as pd
from llm_judge import judge_relevance


INPUT_FILE = "data/retrieval_review_set.csv"
OUTPUT_FILE = "data/llm_judge_test_10.csv"

SAMPLE_SIZE = 10
RANDOM_STATE = 42
DELAY_SECONDS = 2


def main():
    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set in the environment."
        )

    # Load manually labelled retrieval evaluation data
    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "query_id",
        "query_message",
        "historical_customer_message",
        "historical_support_reply",
        "relevance",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Select exactly 10 examples
    sample = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    results = []

    print(f"Running Gemini judge on {SAMPLE_SIZE} examples...")
    print()

    for index, row in sample.iterrows():

        print(
            f"[{index + 1}/{SAMPLE_SIZE}] "
            f"Query ID: {row['query_id']}"
        )

        try:
            result = judge_relevance(
                query_message=row["query_message"],
                historical_customer_message=(
                    row["historical_customer_message"]
                ),
                historical_support_reply=(
                    row["historical_support_reply"]
                ),
            )

            gemini_label = int(result["relevance"])
            rationale = result["rationale"]

            results.append({
                "query_id": row["query_id"],
                "query_message": row["query_message"],
                "historical_customer_message": (
                    row["historical_customer_message"]
                ),
                "historical_support_reply": (
                    row["historical_support_reply"]
                ),
                "human_relevance": int(row["relevance"]),
                "gemini_relevance": gemini_label,
                "agreement": (
                    int(row["relevance"]) == gemini_label
                ),
                "gemini_rationale": rationale,
            })

            print(
                f"  Human:  {int(row['relevance'])}"
            )
            print(
                f"  Gemini: {gemini_label}"
            )
            print(
                f"  Match:  "
                f"{int(row['relevance']) == gemini_label}"
            )
            print()

        except Exception as error:
            print(f"  ERROR: {error}")
            print()

            results.append({
                "query_id": row["query_id"],
                "query_message": row["query_message"],
                "historical_customer_message": (
                    row["historical_customer_message"]
                ),
                "historical_support_reply": (
                    row["historical_support_reply"]
                ),
                "human_relevance": int(row["relevance"]),
                "gemini_relevance": None,
                "agreement": None,
                "gemini_rationale": f"ERROR: {error}",
            })

        # Small delay to avoid hitting rate limits
        if index < len(sample) - 1:
            time.sleep(DELAY_SECONDS)

    # Save results
    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    valid = result_df[
        result_df["gemini_relevance"].notna()
    ]

    if len(valid) > 0:
        agreement = valid["agreement"].mean()

        print("=" * 50)
        print("GEMINI JUDGE TEST SUMMARY")
        print("=" * 50)
        print(f"Examples evaluated: {len(valid)}")
        print(
            f"Human-Gemini agreement: "
            f"{agreement:.2%}"
        )

    print()
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()