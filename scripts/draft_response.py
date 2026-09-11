from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RETRIEVAL_FILE = BASE_DIR / "data" / "retrieval_review_set.csv"


# ---------------------------------------------------------
# Response templates
# ---------------------------------------------------------

def build_response(actions, customer_message):
    """
    Build a conservative response from historically observed
    resolution actions.

    The function does not invent troubleshooting steps.
    It only converts extracted historical actions into
    response components.
    """

    parts = []

    # -----------------------------------------------------
    # Acknowledge the customer
    # -----------------------------------------------------

    parts.append(
        "Thanks for reaching out. We'd be happy to help with this."
    )

    # -----------------------------------------------------
    # Diagnostic questions
    # -----------------------------------------------------

    if "ASK_DIAGNOSTIC_QUESTION" in actions:

        parts.append(
            "Could you tell us a little more about what is happening "
            "and when the issue started?"
        )

    # -----------------------------------------------------
    # Request useful details
    # -----------------------------------------------------

    if "ASK_FOR_DETAILS" in actions:

        parts.append(
            "Please also share your device model and the iOS version "
            "currently installed."
        )

    # -----------------------------------------------------
    # Software update
    # -----------------------------------------------------

    if "UPDATE_SOFTWARE" in actions:

        parts.append(
            "Please make sure your device is updated to the latest "
            "available software."
        )

    # -----------------------------------------------------
    # Restart
    # -----------------------------------------------------

    if "RESTART_DEVICE" in actions:

        parts.append(
            "If you haven't already, please try restarting the device "
            "and check whether the issue continues."
        )

    # -----------------------------------------------------
    # Settings
    # -----------------------------------------------------

    if "CHECK_SETTINGS" in actions:

        parts.append(
            "Please check the relevant settings on your device and "
            "let us know whether the issue persists."
        )

    # -----------------------------------------------------
    # Connection
    # -----------------------------------------------------

    if "CHECK_CONNECTION" in actions:

        parts.append(
            "Please check your network connection and let us know "
            "whether the issue continues."
        )

    # -----------------------------------------------------
    # Account
    # -----------------------------------------------------

    if "CHECK_ACCOUNT" in actions:

        parts.append(
            "Please check that you're signed in with the correct "
            "Apple Account and let us know if the issue persists."
        )

    # -----------------------------------------------------
    # App Store
    # -----------------------------------------------------

    if "CHECK_APP_STORE" in actions:

        parts.append(
            "Please check the App Store and let us know what happens "
            "when you try again."
        )

    # -----------------------------------------------------
    # Battery
    # -----------------------------------------------------

    if "CHECK_BATTERY" in actions:

        parts.append(
            "Please check the battery behavior and let us know whether "
            "the issue continues."
        )

    # -----------------------------------------------------
    # Troubleshooting
    # -----------------------------------------------------

    if "PROVIDE_TROUBLESHOOTING" in actions:

        parts.append(
            "You can also try the troubleshooting steps that apply "
            "to your situation."
        )

    # -----------------------------------------------------
    # Workaround
    # -----------------------------------------------------

    if "PROVIDE_WORKAROUND" in actions:

        parts.append(
            "There may be a workaround available for this issue."
        )

    # -----------------------------------------------------
    # Information
    # -----------------------------------------------------

    if "PROVIDE_INFORMATION" in actions:

        parts.append(
            "We'll be glad to provide more information based on "
            "the details of your issue."
        )

    # -----------------------------------------------------
    # Article
    # -----------------------------------------------------

    if "PROVIDE_ARTICLE" in actions:

        parts.append(
            "We can also point you to the relevant Apple Support "
            "resource for this issue."
        )

    # -----------------------------------------------------
    # Specialist
    # -----------------------------------------------------

    if "ROUTE_TO_SPECIALIST" in actions:

        parts.append(
            "We can connect you with the appropriate specialist "
            "for further assistance."
        )

    # -----------------------------------------------------
    # Feedback
    # -----------------------------------------------------

    if "ROUTE_FEEDBACK" in actions:

        parts.append(
            "If you'd like, you can also share feedback with Apple "
            "about your experience."
        )

    # -----------------------------------------------------
    # Language support
    # -----------------------------------------------------

    if "ROUTE_LANGUAGE_SUPPORT" in actions:

        parts.append(
            "We can help route you to the appropriate language "
            "support team."
        )

    # -----------------------------------------------------
    # Direct message
    # -----------------------------------------------------

    if "CONTACT_SUPPORT_DM" in actions:

        parts.append(
            "Please send us a DM with the relevant details so we "
            "can look into this with you."
        )

    # -----------------------------------------------------
    # No clear action
    # -----------------------------------------------------

    if actions == ["NO_CLEAR_ACTION"]:

        return (
            "Thanks for reaching out. We'd be happy to help. "
            "Please send us a DM with some more details about the issue "
            "so we can take a closer look."
        )

    # -----------------------------------------------------
    # Final response
    # -----------------------------------------------------

    response = " ".join(parts)

    return response


# ---------------------------------------------------------
# Example action combinations
# ---------------------------------------------------------

def demo():

    examples = [
        {
            "customer_message": "My iPhone battery is draining really quickly.",
            "actions": [
                "CHECK_BATTERY",
                "ASK_FOR_DETAILS",
                "CONTACT_SUPPORT_DM",
            ],
        },
        {
            "customer_message": "My iPhone keeps freezing after the update.",
            "actions": [
                "UPDATE_SOFTWARE",
                "RESTART_DEVICE",
                "ASK_DIAGNOSTIC_QUESTION",
            ],
        },
        {
            "customer_message": "The App Store won't download my apps.",
            "actions": [
                "CHECK_APP_STORE",
                "ASK_DIAGNOSTIC_QUESTION",
                "CONTACT_SUPPORT_DM",
            ],
        },
    ]

    for example in examples:

        print("\n" + "=" * 70)
        print("CUSTOMER:")
        print(example["customer_message"])

        print("\nACTIONS:")
        print(" | ".join(example["actions"]))

        print("\nDRAFT RESPONSE:")
        print(
            build_response(
                example["actions"],
                example["customer_message"],
            )
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    demo()