import pandas as pd
from pathlib import Path

from extract_resolution_actions import extract_resolution_actions


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "historical_replies_sample_100.csv"
RESULT_FILE = BASE_DIR / "data" / "resolution_action_test_results.csv"


# ============================================================
# Expected labels
# ============================================================

# These are development expectations based on our manual
# inspection of the 100 historical AppleSupport replies.
#
# Some cases intentionally allow multiple valid interpretations.
#
# The purpose is to catch obvious regressions and improve
# the rule-based extractor before applying it to the full data.

EXPECTED = {

    1: ["ASK_DIAGNOSTIC_QUESTION"],

    2: ["PROVIDE_ARTICLE"],

    3: ["CONTACT_SUPPORT_DM"],

    4: ["ASK_FOR_DETAILS"],

    5: ["CONTACT_SUPPORT_DM"],

    6: [
        "ANNOUNCE_FIX_UPDATE",
        "PROVIDE_WORKAROUND",
        "PROVIDE_ARTICLE",
    ],

    7: [
        "ASK_DIAGNOSTIC_QUESTION",
        "ASK_FOR_DETAILS",
    ],

    8: [
        "ANNOUNCE_FIX_UPDATE",
        "PROVIDE_ARTICLE",
    ],

    9: ["CONTACT_SUPPORT_DM"],

    10: ["ASK_DIAGNOSTIC_QUESTION"],

    11: [
        "ASK_DIAGNOSTIC_QUESTION",
        "ASK_FOR_DETAILS",
    ],

    12: [
        "ASK_DIAGNOSTIC_QUESTION",
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    13: ["PROVIDE_TROUBLESHOOTING"],

    14: ["CONTACT_SUPPORT_DM"],

    15: [
        "RESTART_DEVICE",
        "ASK_DIAGNOSTIC_QUESTION",
        "CONTACT_SUPPORT_DM",
    ],

    16: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    17: [
        "ASK_DIAGNOSTIC_QUESTION",
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    18: [
        "ANNOUNCE_FIX_UPDATE",
        "PROVIDE_WORKAROUND",
        "PROVIDE_ARTICLE",
    ],

    19: ["ASK_FOR_DETAILS"],

    20: ["ASK_FOR_DETAILS"],

    # --------------------------------------------------------
    # Samples 21-25
    # --------------------------------------------------------

    21: ["ASK_DIAGNOSTIC_QUESTION"],

    22: [
        "ASK_DIAGNOSTIC_QUESTION",
        "RESTART_DEVICE",
        "CONTACT_SUPPORT_DM",
    ],

    23: ["CONTACT_SUPPORT_DM"],

    24: ["ASK_DIAGNOSTIC_QUESTION"],

    25: ["CONTACT_SUPPORT_DM"],

    # --------------------------------------------------------
    # Samples 26-52
    # --------------------------------------------------------

    26: ["CONTACT_SUPPORT_DM"],

    27: ["PROVIDE_ARTICLE"],

    28: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    29: [
        "PROVIDE_WORKAROUND",
        "ANNOUNCE_FIX_UPDATE",
        "PROVIDE_ARTICLE",
    ],

    30: ["CONTACT_SUPPORT_DM"],

    31: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    32: ["ASK_DIAGNOSTIC_QUESTION"],

    33: ["ASK_FOR_DETAILS"],

    34: [
        "RESTART_DEVICE",
        "ASK_DIAGNOSTIC_QUESTION",
        "CONTACT_SUPPORT_DM",
    ],

    35: [
        "ASK_DIAGNOSTIC_QUESTION",
        "RESTART_DEVICE",
    ],

    36: ["CONTACT_SUPPORT_DM"],

    37: [
        "ROUTE_TO_SPECIALIST",
        "PROVIDE_ARTICLE",
    ],

    38: ["ASK_DIAGNOSTIC_QUESTION"],

    39: [
        "ASK_DIAGNOSTIC_QUESTION",
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    40: [
        "PROVIDE_WORKAROUND",
        "ANNOUNCE_FIX_UPDATE",
        "PROVIDE_ARTICLE",
    ],

    41: [
        "ROUTE_TO_SPECIALIST",
        "PROVIDE_ARTICLE",
    ],

    42: ["ASK_DIAGNOSTIC_QUESTION"],

    43: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    44: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    45: ["CONTACT_SUPPORT_DM"],

    46: ["ASK_DIAGNOSTIC_QUESTION"],

    47: ["CONTACT_SUPPORT_DM"],

    48: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    49: ["CONTACT_SUPPORT_DM"],

    50: ["ASK_DIAGNOSTIC_QUESTION"],

    51: ["ASK_FOR_DETAILS", "CONTACT_SUPPORT_DM"],

    52: ["NO_CLEAR_ACTION"],

    # --------------------------------------------------------
    # Samples 53-80
    # --------------------------------------------------------

    53: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    54: ["CONTACT_SUPPORT_DM"],

    55: ["ASK_DIAGNOSTIC_QUESTION"],

    56: [
        "PROVIDE_TROUBLESHOOTING",
        "CONTACT_SUPPORT_DM",
    ],

    57: [
        "ANNOUNCE_FIX_UPDATE",
        "UPDATE_SOFTWARE",
        "CONTACT_SUPPORT_DM",
    ],

    58: [
        "PROVIDE_TROUBLESHOOTING",
        "PROVIDE_ARTICLE",
    ],

    59: [
        "PROVIDE_INFORMATION",
        "ROUTE_FEEDBACK",
    ],

    60: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    61: ["ASK_DIAGNOSTIC_QUESTION"],

    62: ["CONTACT_SUPPORT_DM"],

    63: [
        "PROVIDE_WORKAROUND",
        "CONTACT_SUPPORT_DM",
    ],

    64: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    65: ["ASK_FOR_DETAILS"],

    66: ["ASK_DIAGNOSTIC_QUESTION"],

    67: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    68: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    69: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    70: ["NO_CLEAR_ACTION"],

    71: [
        "PROVIDE_INFORMATION",
        "PROVIDE_ARTICLE",
    ],

    72: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    73: [
        "ASK_DIAGNOSTIC_QUESTION",
        "CONTACT_SUPPORT_DM",
    ],

    74: [
        "PROVIDE_INFORMATION",
        "PROVIDE_ARTICLE",
    ],

    75: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    76: ["ASK_DIAGNOSTIC_QUESTION"],

    77: ["NO_CLEAR_ACTION"],

    78: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    79: ["CONTACT_SUPPORT_DM"],

    80: ["NO_CLEAR_ACTION"],

    # --------------------------------------------------------
    # Samples 81-100
    # --------------------------------------------------------

    81: ["NO_CLEAR_ACTION"],

    82: [
        "ASK_FOR_DETAILS",
        "ASK_DIAGNOSTIC_QUESTION",
    ],

    83: [
        "PROVIDE_TROUBLESHOOTING",
        "ASK_DIAGNOSTIC_QUESTION",
    ],

    84: [
        "PROVIDE_ARTICLE",
        "PROVIDE_TROUBLESHOOTING",
    ],

    85: [
        "PROVIDE_ARTICLE",
        "PROVIDE_INFORMATION",
    ],

    86: ["CONTACT_SUPPORT_DM"],

    87: ["CONTACT_SUPPORT_DM"],

    88: ["PROVIDE_ARTICLE"],

    89: [
        "PROVIDE_TROUBLESHOOTING",
        "PROVIDE_ARTICLE",
    ],

    90: ["ASK_DIAGNOSTIC_QUESTION"],

    91: ["ROUTE_LANGUAGE_SUPPORT"],

    92: [
        "UPDATE_SOFTWARE",
    ],

    93: ["ASK_FOR_DETAILS"],

    94: ["CONTACT_SUPPORT_DM"],

    95: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    96: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    97: [
        "ANNOUNCE_FIX_UPDATE",
        "UPDATE_SOFTWARE",
        "PROVIDE_ARTICLE",
    ],

    98: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    99: [
        "ASK_FOR_DETAILS",
        "CONTACT_SUPPORT_DM",
    ],

    100: ["ASK_DIAGNOSTIC_QUESTION"],
}


# ============================================================
# Comparison
# ============================================================

def compare_actions(expected, predicted):
    """
    Exact set comparison.

    Order does not matter.
    """

    expected_set = set(expected)
    predicted_set = set(predicted)

    return expected_set == predicted_set


# ============================================================
# Main test
# ============================================================

def main():

    print("=" * 70)
    print("SupportLens AI - Resolution Action Test Harness")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if len(df) != 100:
        raise ValueError(
            f"Expected exactly 100 samples, found {len(df)}."
        )

    if "support_reply" not in df.columns:
        raise ValueError(
            "Column 'support_reply' not found."
        )

    results = []

    passed = 0
    failed = 0

    for index, row in df.iterrows():

        sample_number = index + 1

        reply = row["support_reply"]

        predicted = extract_resolution_actions(reply)

        expected = EXPECTED.get(sample_number)

        if expected is None:
            raise ValueError(
                f"No expected labels defined for sample {sample_number}"
            )

        is_match = compare_actions(expected, predicted)

        if is_match:
            passed += 1
        else:
            failed += 1

        results.append({
            "sample_number": sample_number,
            "support_reply": reply,
            "expected_actions": " | ".join(expected),
            "predicted_actions": " | ".join(predicted),
            "pass": is_match,
        })

        status = "PASS" if is_match else "FAIL"

        print(
            f"[{status}] Sample {sample_number:03d} | "
            f"Expected: {', '.join(expected)} | "
            f"Predicted: {', '.join(predicted)}"
        )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    accuracy = passed / len(df)

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(f"Total samples : {len(df)}")
    print(f"Passed        : {passed}")
    print(f"Failed        : {failed}")
    print(f"Exact match   : {accuracy:.2%}")

    print(f"\nDetailed results saved to:")
    print(RESULT_FILE)

    # --------------------------------------------------------
    # Failed examples
    # --------------------------------------------------------

    failed_df = results_df[
        results_df["pass"] == False
    ]

    if len(failed_df) > 0:

        print("\n" + "=" * 70)
        print("FAILED CASES")
        print("=" * 70)

        for _, row in failed_df.iterrows():

            print(f"\nSample {row['sample_number']}")

            print(
                f"Expected : {row['expected_actions']}"
            )

            print(
                f"Predicted: {row['predicted_actions']}"
            )

            print(
                f"Reply    : {row['support_reply']}"
            )

    print("\n" + "=" * 70)

    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(
            f"{failed} test(s) need rule refinement."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()