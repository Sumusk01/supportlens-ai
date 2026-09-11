import pandas as pd
from pathlib import Path
import sys


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------
# Import reusable agent
# ---------------------------------------------------------

from run_agent import SupportLensAgent


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

GOLDEN_PATH = BASE_DIR / "data" / "golden_set.csv"

OUTPUT_PATH = BASE_DIR / "data" / "agent_evaluation_50.csv"

SAMPLE_SIZE = 50

RANDOM_STATE = 42


# ---------------------------------------------------------
# Load evaluation data
# ---------------------------------------------------------

def load_evaluation_set():

    print("\nLoading locked Golden Set...")

    df = pd.read_csv(
        GOLDEN_PATH
    )

    print(
        f"Golden Set examples available: "
        f"{len(df)}"
    )

    # Fixed random seed makes the evaluation reproducible.
    evaluation_df = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).copy()

    print(
        f"Evaluation examples selected: "
        f"{len(evaluation_df)}"
    )

    return evaluation_df


# ---------------------------------------------------------
# Evaluate one example
# ---------------------------------------------------------

def evaluate_example(
    agent,
    row,
    example_number
):

    customer_message = str(
        row["customer_message"]
    )

    actual_intent = str(
        row["final_intent"]
    )

    print(
        f"\n[{example_number}/{SAMPLE_SIZE}] "
        f"Evaluating..."
    )

    print(
        f"Customer: "
        f"{customer_message[:120]}"
    )

    # Reuse the already initialized agent.
    result = agent.run(
        customer_message
    )

    predicted_intent = result[
        "predicted_intent"
    ]

    confidence = result[
        "intent_confidence"
    ]

    correct = (
        actual_intent
        == predicted_intent
    )

    supported_actions = result[
        "supported_actions"
    ]

    decision = result[
        "decision"
    ]

    reason = result[
        "decision_reason"
    ]

    draft = result[
        "draft_response"
    ]

    return {
        "example_id": row["example_id"],
        "customer_message": customer_message,
        "actual_intent": actual_intent,
        "predicted_intent": predicted_intent,
        "intent_confidence": confidence,
        "intent_correct": correct,
        "supported_actions": " | ".join(
            supported_actions
        ),
        "decision": decision,
        "decision_reason": reason,
        "draft_response": draft,
    }


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("SUPPORTLENS AI — FINAL AGENT EVALUATION")
    print("=" * 70)

    evaluation_df = load_evaluation_set()

    # -----------------------------------------------------
    # Initialize agent ONCE
    # -----------------------------------------------------

    print("\nInitializing evaluation agent...")

    agent = SupportLensAgent()

    print("Evaluation agent ready.")

    # -----------------------------------------------------
    # Evaluate all examples
    # -----------------------------------------------------

    results = []

    for number, (_, row) in enumerate(
        evaluation_df.iterrows(),
        start=1
    ):

        result = evaluate_example(
            agent,
            row,
            number
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    intent_accuracy = (
        results_df["intent_correct"].mean()
    )

    auto_handle_count = (
        results_df["decision"]
        .eq("AUTO-HANDLE")
        .sum()
    )

    escalate_count = (
        results_df["decision"]
        .eq("ESCALATE")
        .sum()
    )

    auto_handle_rate = (
        auto_handle_count
        / len(results_df)
    )

    escalation_rate = (
        escalate_count
        / len(results_df)
    )

    mean_confidence = (
        results_df["intent_confidence"]
        .mean()
    )

    correct_confidence = results_df.loc[
        results_df["intent_correct"],
        "intent_confidence"
    ].mean()

    incorrect_confidence = results_df.loc[
        ~results_df["intent_correct"],
        "intent_confidence"
    ].mean()

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL AGENT EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"\nEvaluation examples: "
        f"{len(results_df)}"
    )

    print(
        f"Intent accuracy: "
        f"{intent_accuracy:.4f}"
    )

    print(
        f"Mean intent confidence: "
        f"{mean_confidence:.4f}"
    )

    print(
        f"Mean confidence — correct predictions: "
        f"{correct_confidence:.4f}"
    )

    print(
        f"Mean confidence — incorrect predictions: "
        f"{incorrect_confidence:.4f}"
    )

    print(
        f"\nAUTO-HANDLE: "
        f"{auto_handle_count} "
        f"({auto_handle_rate:.2%})"
    )

    print(
        f"ESCALATE: "
        f"{escalate_count} "
        f"({escalation_rate:.2%})"
    )

    print("\nDecision distribution:")

    print(
        results_df["decision"]
        .value_counts()
    )

    print("\nIntent prediction results:")

    print(
        results_df[
            [
                "actual_intent",
                "predicted_intent",
                "intent_correct"
            ]
        ].to_string(index=False)
    )

    print("\nResults saved to:")

    print(OUTPUT_PATH)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()