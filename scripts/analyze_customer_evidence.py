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
# Import existing modules
# ---------------------------------------------------------

from retrieve_responses import retrieve_responses
from evidence_aggregator import analyze_evidence


# ---------------------------------------------------------
# Analyze a customer message
# ---------------------------------------------------------

def analyze_customer_message(customer_message, top_k=5):
    """
    Retrieve historical support cases and analyze the
    resolution actions found in those cases.
    """

    # Step 1: Retrieve historical examples
    historical_cases = retrieve_responses(
        customer_message,
        top_k=top_k
    )

    # Step 2: Analyze historical resolution evidence
    evidence_result = analyze_evidence(
        historical_cases
    )

    return evidence_result


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------

def print_analysis(customer_message, result):

    print("\n" + "=" * 70)
    print("SUPPORTLENS AI — HISTORICAL EVIDENCE")
    print("=" * 70)

    print("\nCustomer message:")
    print(customer_message)

    print("\n" + "-" * 70)
    print("TOP HISTORICAL CASES")
    print("-" * 70)

    for case in result["cases"]:

        print(
            f"\nRank {case['rank']} "
            f"(similarity={case['similarity']:.4f})"
        )

        print(
            f"Customer: {case['historical_customer']}"
        )

        print(
            f"Actions: {' | '.join(case['actions'])}"
        )

        print(
            f"Reply: {case['historical_reply']}"
        )

    print("\n" + "-" * 70)
    print("AGGREGATED ACTION EVIDENCE")
    print("-" * 70)

    for item in result["action_evidence"]:

        print(
            f"{item['action']}: "
            f"{item['support_count']}/"
            f"{item['total_cases']} "
            f"historical cases"
        )

    print("\n" + "-" * 70)
    print("SUPPORTED ACTIONS")
    print("-" * 70)

    if result["supported_actions"]:

        for action in result["supported_actions"]:
            print(f"- {action}")

    else:

        print(
            "No action has enough historical support."
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("SupportLens AI — Evidence Analysis")
    print("=" * 70)

    customer_message = input(
        "\nEnter a customer message:\n> "
    ).strip()

    if not customer_message:

        print("No message entered.")
        sys.exit()

    result = analyze_customer_message(
        customer_message,
        top_k=5
    )

    print_analysis(
        customer_message,
        result
    )