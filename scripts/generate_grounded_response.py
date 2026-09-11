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

from analyze_customer_evidence import analyze_customer_message
from draft_response import build_response


# ---------------------------------------------------------
# Generate grounded response
# ---------------------------------------------------------

def generate_grounded_response(customer_message, top_k=5):

    # Retrieve and analyze historical evidence
    evidence_result = analyze_customer_message(
        customer_message,
        top_k=top_k
    )

    # Actions supported by multiple historical cases
    supported_actions = evidence_result[
        "supported_actions"
    ]

    # Build response from those actions
    response = build_response(
        supported_actions,
        customer_message
    )

    return {
        "customer_message": customer_message,
        "historical_cases": evidence_result["cases"],
        "action_evidence": evidence_result["action_evidence"],
        "supported_actions": supported_actions,
        "response": response,
    }


# ---------------------------------------------------------
# Display final result
# ---------------------------------------------------------

def print_result(result):

    print("\n" + "=" * 70)
    print("SUPPORTLENS AI — GROUNDED RESPONSE")
    print("=" * 70)

    print("\nCustomer:")
    print(result["customer_message"])

    print("\n" + "-" * 70)
    print("SUPPORTED HISTORICAL ACTIONS")
    print("-" * 70)

    if result["supported_actions"]:

        for action in result["supported_actions"]:
            print(f"- {action}")

    else:

        print("No sufficiently supported actions.")

    print("\n" + "-" * 70)
    print("EVIDENCE")
    print("-" * 70)

    for item in result["action_evidence"]:

        if item["action"] in result["supported_actions"]:

            print(
                f"{item['action']}: "
                f"{item['support_count']}/"
                f"{item['total_cases']} cases"
            )

    print("\n" + "-" * 70)
    print("DRAFT RESPONSE")
    print("-" * 70)

    print(result["response"])


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("SupportLens AI — Grounded Response Generator")
    print("=" * 70)

    customer_message = input(
        "\nEnter a customer message:\n> "
    ).strip()

    if not customer_message:

        print("No message entered.")
        sys.exit()

    result = generate_grounded_response(
        customer_message,
        top_k=5
    )

    print_result(result)