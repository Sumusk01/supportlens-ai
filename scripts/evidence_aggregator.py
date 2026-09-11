from collections import defaultdict
from pathlib import Path
import sys


# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

SCRIPTS_DIR = BASE_DIR / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# Import our existing action extractor
from extract_resolution_actions import extract_resolution_actions


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MIN_ACTION_SUPPORT = 2


# ---------------------------------------------------------
# Extract actions from retrieved historical replies
# ---------------------------------------------------------

def extract_actions_from_cases(historical_cases):
    """
    Extract resolution actions from each retrieved historical case.

    Expected input:

    [
        {
            "historical_customer": "...",
            "historical_reply": "...",
            "similarity": 0.52
        },
        ...
    ]

    Returns a list where each case contains its extracted actions.
    """

    enriched_cases = []

    for rank, case in enumerate(historical_cases, start=1):

        reply = str(case.get("historical_reply", "")).strip()

        actions = extract_resolution_actions(reply)

        enriched_case = {
            "rank": rank,
            "similarity": float(case.get("similarity", 0.0)),
            "historical_customer": case.get(
                "historical_customer",
                ""
            ),
            "historical_reply": reply,
            "actions": actions,
        }

        enriched_cases.append(enriched_case)

    return enriched_cases


# ---------------------------------------------------------
# Aggregate action evidence
# ---------------------------------------------------------

def aggregate_action_evidence(enriched_cases):
    """
    Count how many retrieved historical cases support each action.

    An action is counted once per historical case, even if the
    extractor somehow detects the same action multiple times.
    """

    action_cases = defaultdict(list)

    total_cases = len(enriched_cases)

    for case in enriched_cases:

        for action in set(case["actions"]):

            if action == "NO_CLEAR_ACTION":
                continue

            action_cases[action].append(case["rank"])

    evidence = []

    for action, ranks in action_cases.items():

        support_count = len(ranks)

        support_ratio = (
            support_count / total_cases
            if total_cases > 0
            else 0.0
        )

        evidence.append(
            {
                "action": action,
                "support_count": support_count,
                "total_cases": total_cases,
                "support_ratio": round(support_ratio, 3),
                "supporting_ranks": ranks,
            }
        )

    # Stronger evidence first.
    # Similarity is used as a secondary signal.
    evidence.sort(
        key=lambda item: (
            item["support_count"],
            item["support_ratio"],
        ),
        reverse=True,
    )

    return evidence


# ---------------------------------------------------------
# Select supported actions
# ---------------------------------------------------------

def select_supported_actions(
    evidence,
    min_support=MIN_ACTION_SUPPORT,
):
    """
    Select actions supported by at least min_support historical cases.

    If no action reaches the threshold, return an empty list.

    This is intentionally conservative:
    weak evidence should not automatically become advice.
    """

    selected = []

    for item in evidence:

        if item["support_count"] >= min_support:

            selected.append(item["action"])

    return selected


# ---------------------------------------------------------
# Complete evidence analysis
# ---------------------------------------------------------

def analyze_evidence(
    historical_cases,
    min_support=MIN_ACTION_SUPPORT,
):
    """
    Full evidence pipeline:

        retrieved cases
            ↓
        action extraction
            ↓
        action aggregation
            ↓
        supported action selection

    Returns a dictionary containing all intermediate results.
    """

    enriched_cases = extract_actions_from_cases(
        historical_cases
    )

    evidence = aggregate_action_evidence(
        enriched_cases
    )

    supported_actions = select_supported_actions(
        evidence,
        min_support=min_support,
    )

    return {
        "cases": enriched_cases,
        "action_evidence": evidence,
        "supported_actions": supported_actions,
    }


# ---------------------------------------------------------
# Display evidence
# ---------------------------------------------------------

def print_evidence_report(result):

    print("\n" + "=" * 70)
    print("HISTORICAL EVIDENCE ANALYSIS")
    print("=" * 70)

    print("\nRetrieved cases:")

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

    print("\n" + "-" * 70)
    print("ACTION EVIDENCE")
    print("-" * 70)

    if not result["action_evidence"]:

        print("No actionable historical evidence found.")

    else:

        for item in result["action_evidence"]:

            print(
                f"{item['action']}: "
                f"{item['support_count']}/{item['total_cases']} "
                f"cases "
                f"(ranks={item['supporting_ranks']})"
            )

    print("\n" + "-" * 70)
    print("SUPPORTED ACTIONS")
    print("-" * 70)

    if result["supported_actions"]:

        print(
            " | ".join(result["supported_actions"])
        )

    else:

        print(
            "NONE — evidence is too weak for automatic action selection."
        )


# ---------------------------------------------------------
# Demonstration
# ---------------------------------------------------------

def demo():

    sample_cases = [

        {
            "similarity": 0.5235,
            "historical_customer":
                "Battery draining very quickly",
            "historical_reply":
                "Thanks for reaching out. Please check your battery "
                "usage and send us a DM so we can help.",
        },

        {
            "similarity": 0.5057,
            "historical_customer":
                "My iPhone battery is draining after the update",
            "historical_reply":
                "Please make sure your iPhone is updated and send us "
                "a DM with more details.",
        },

        {
            "similarity": 0.4636,
            "historical_customer":
                "Battery draining quickly on iOS",
            "historical_reply":
                "Please check the battery and let us know your "
                "iPhone model and iOS version.",
        },

        {
            "similarity": 0.4567,
            "historical_customer":
                "Battery draining after update",
            "historical_reply":
                "Please check the battery behavior and send us a DM "
                "if you still need help.",
        },

        {
            "similarity": 0.4487,
            "historical_customer":
                "Battery life is worse after updating",
            "historical_reply":
                "Please update your device and contact us by DM "
                "if the issue continues.",
        },
    ]

    result = analyze_evidence(sample_cases)

    print_evidence_report(result)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    demo()