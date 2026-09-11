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
# Import pipeline components
# ---------------------------------------------------------

from intent_predictor import IntentPredictor
from retrieve_responses import retrieve_responses
from evidence_aggregator import analyze_evidence
from draft_response import build_response
from escalation import decide_escalation


# ---------------------------------------------------------
# Reusable SupportLens AI agent
# ---------------------------------------------------------

class SupportLensAgent:

    def __init__(self):

        print("\nInitializing SupportLens AI...")

        # -------------------------------------------------
        # Train intent classifier ONCE
        # -------------------------------------------------

        self.predictor = IntentPredictor()

        self.predictor.train()

    # -----------------------------------------------------
    # Run complete agent
    # -----------------------------------------------------

    def run(self, customer_message):

        # -------------------------------------------------
        # 1. Intent prediction
        # -------------------------------------------------

        intent_result = self.predictor.predict(
            customer_message
        )

        predicted_intent = intent_result[
            "predicted_intent"
        ]

        intent_confidence = intent_result[
            "confidence"
        ]

        # -------------------------------------------------
        # 2. Historical retrieval
        # -------------------------------------------------

        historical_cases = retrieve_responses(
            customer_message,
            top_k=5
        )

        # -------------------------------------------------
        # 3. Evidence analysis
        # -------------------------------------------------

        evidence_result = analyze_evidence(
            historical_cases
        )

        supported_actions = evidence_result[
            "supported_actions"
        ]

        action_evidence = evidence_result[
            "action_evidence"
        ]

        # -------------------------------------------------
        # 4. Response drafting
        # -------------------------------------------------

        draft = build_response(
            supported_actions,
            customer_message
        )

        # -------------------------------------------------
        # 5. Escalation decision
        # -------------------------------------------------

        escalation_result = decide_escalation(
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            supported_actions=supported_actions,
            action_evidence=action_evidence,
        )

        # -------------------------------------------------
        # 6. Final result
        # -------------------------------------------------

        return {
            "customer_message": customer_message,
            "predicted_intent": predicted_intent,
            "intent_confidence": intent_confidence,
            "top_predictions": intent_result[
                "top_predictions"
            ],
            "historical_cases": historical_cases,
            "action_evidence": action_evidence,
            "supported_actions": supported_actions,
            "draft_response": draft,
            "decision": escalation_result[
                "decision"
            ],
            "decision_reason": escalation_result[
                "reason"
            ],
        }


# ---------------------------------------------------------
# Backward-compatible single-message function
# ---------------------------------------------------------

def run_agent(customer_message):

    agent = SupportLensAgent()

    return agent.run(
        customer_message
    )


# ---------------------------------------------------------
# Display agent result
# ---------------------------------------------------------

def print_agent_result(result):

    print("\n" + "=" * 70)
    print("SUPPORTLENS AI — FINAL AGENT")
    print("=" * 70)

    print("\nCUSTOMER MESSAGE")
    print("-" * 70)
    print(result["customer_message"])

    print("\nINTENT")
    print("-" * 70)

    print(
        f"Predicted intent: "
        f"{result['predicted_intent']}"
    )

    print(
        f"Confidence: "
        f"{result['intent_confidence']:.4f}"
    )

    print("\nTop predictions:")

    for prediction in result["top_predictions"]:

        print(
            f"- {prediction['intent']}: "
            f"{prediction['probability']:.4f}"
        )

    print("\nHISTORICAL EVIDENCE")
    print("-" * 70)

    for item in result["action_evidence"]:

        print(
            f"- {item['action']}: "
            f"{item['support_count']}/"
            f"{item['total_cases']} cases"
        )

    print("\nSUPPORTED ACTIONS")
    print("-" * 70)

    if result["supported_actions"]:

        for action in result["supported_actions"]:
            print(f"- {action}")

    else:

        print("None")

    print("\nDRAFT RESPONSE")
    print("-" * 70)

    print(result["draft_response"])

    print("\nDECISION")
    print("-" * 70)

    print(
        result["decision"]
    )

    print("\nREASON")
    print("-" * 70)

    print(
        result["decision_reason"]
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("SupportLens AI")
    print("=" * 70)

    customer_message = input(
        "\nEnter a customer message:\n> "
    ).strip()

    if not customer_message:

        print("No message entered.")
        sys.exit()

    agent = SupportLensAgent()

    result = agent.run(
        customer_message
    )

    print_agent_result(
        result
    )