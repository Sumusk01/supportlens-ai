from typing import Dict, List


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MIN_INTENT_CONFIDENCE = 0.60
MIN_SUPPORTED_ACTIONS = 1


# ---------------------------------------------------------
# Escalation decision
# ---------------------------------------------------------

def decide_escalation(
    predicted_intent: str,
    intent_confidence: float,
    supported_actions: List[str],
    action_evidence: List[Dict],
):
    """
    Decide whether a customer message should be auto-handled
    or escalated to a human.

    The policy is intentionally conservative.

    AUTO-HANDLE requires:
        1. A known intent
        2. Sufficient classifier confidence
        3. At least one historically supported action

    Otherwise, the case is escalated.
    """

    # -----------------------------------------------------
    # Normalize inputs
    # -----------------------------------------------------

    predicted_intent = str(
        predicted_intent
    ).strip().upper()

    intent_confidence = float(
        intent_confidence
    )

    supported_actions = (
        supported_actions
        if supported_actions
        else []
    )

    # -----------------------------------------------------
    # Rule 1: OTHER intent
    # -----------------------------------------------------

    if predicted_intent == "OTHER":

        return {
            "decision": "ESCALATE",
            "reason": (
                "The message was classified as OTHER, so there "
                "is insufficient intent-specific evidence for "
                "safe automatic handling."
            ),
        }

    # -----------------------------------------------------
    # Rule 2: Low classifier confidence
    # -----------------------------------------------------

    if intent_confidence < MIN_INTENT_CONFIDENCE:

        return {
            "decision": "ESCALATE",
            "reason": (
                f"Intent confidence is {intent_confidence:.2f}, "
                f"below the {MIN_INTENT_CONFIDENCE:.2f} threshold."
            ),
        }

    # -----------------------------------------------------
    # Rule 3: No historically supported actions
    # -----------------------------------------------------

    if len(supported_actions) < MIN_SUPPORTED_ACTIONS:

        return {
            "decision": "ESCALATE",
            "reason": (
                "No resolution action was supported by enough "
                "historical evidence to safely draft a response."
            ),
        }

    # -----------------------------------------------------
    # Rule 4: Strong enough evidence
    # -----------------------------------------------------

    strongest_support = 0

    for item in action_evidence:

        if item["action"] in supported_actions:

            strongest_support = max(
                strongest_support,
                item["support_count"]
            )

    if strongest_support < 2:

        return {
            "decision": "ESCALATE",
            "reason": (
                "The available resolution actions do not have "
                "sufficient repeated historical support."
            ),
        }

    # -----------------------------------------------------
    # AUTO-HANDLE
    # -----------------------------------------------------

    return {
        "decision": "AUTO-HANDLE",
        "reason": (
            f"Intent confidence is {intent_confidence:.2f} "
            f"and historical evidence supports "
            f"{len(supported_actions)} resolution action(s), "
            f"including an action observed in at least "
            f"{strongest_support} retrieved cases."
        ),
    }


# ---------------------------------------------------------
# Test cases
# ---------------------------------------------------------

def run_tests():

    tests = [

        {
            "name": "Strong evidence",
            "predicted_intent": "BATTERY_POWER",
            "intent_confidence": 0.82,
            "supported_actions": [
                "CONTACT_SUPPORT_DM",
                "ASK_FOR_DETAILS",
            ],
            "action_evidence": [
                {
                    "action": "CONTACT_SUPPORT_DM",
                    "support_count": 5,
                },
                {
                    "action": "ASK_FOR_DETAILS",
                    "support_count": 3,
                },
            ],
            "expected": "AUTO-HANDLE",
        },

        {
            "name": "OTHER intent",
            "predicted_intent": "OTHER",
            "intent_confidence": 0.90,
            "supported_actions": [
                "ASK_FOR_DETAILS",
            ],
            "action_evidence": [
                {
                    "action": "ASK_FOR_DETAILS",
                    "support_count": 3,
                },
            ],
            "expected": "ESCALATE",
        },

        {
            "name": "Low confidence",
            "predicted_intent": "BATTERY_POWER",
            "intent_confidence": 0.42,
            "supported_actions": [
                "CHECK_BATTERY",
            ],
            "action_evidence": [
                {
                    "action": "CHECK_BATTERY",
                    "support_count": 4,
                },
            ],
            "expected": "ESCALATE",
        },

        {
            "name": "No supported actions",
            "predicted_intent": "CONNECTIVITY",
            "intent_confidence": 0.80,
            "supported_actions": [],
            "action_evidence": [],
            "expected": "ESCALATE",
        },

        {
            "name": "Weak historical evidence",
            "predicted_intent": "BATTERY_POWER",
            "intent_confidence": 0.80,
            "supported_actions": [
                "CHECK_BATTERY",
            ],
            "action_evidence": [
                {
                    "action": "CHECK_BATTERY",
                    "support_count": 1,
                },
            ],
            "expected": "ESCALATE",
        },
    ]

    print("\n" + "=" * 70)
    print("SUPPORTLENS AI — ESCALATION POLICY TEST")
    print("=" * 70)

    passed = 0

    for test in tests:

        result = decide_escalation(
            predicted_intent=test["predicted_intent"],
            intent_confidence=test["intent_confidence"],
            supported_actions=test["supported_actions"],
            action_evidence=test["action_evidence"],
        )

        actual = result["decision"]

        if actual == test["expected"]:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"

        print(
            f"\n[{status}] {test['name']}"
        )

        print(
            f"Expected: {test['expected']}"
        )

        print(
            f"Actual:   {actual}"
        )

        print(
            f"Reason:   {result['reason']}"
        )

    print("\n" + "-" * 70)
    print(
        f"Tests passed: {passed}/{len(tests)}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    run_tests()