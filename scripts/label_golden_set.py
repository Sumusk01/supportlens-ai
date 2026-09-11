import pandas as pd
import os

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

FILE_PATH = "data/golden_set.csv"

INTENTS = [
    "IOS_UPDATE",
    "BATTERY_POWER",
    "APP_PROBLEM",
    "CONNECTIVITY",
    "ACCOUNT_APPLE_ID",
    "APP_STORE_PURCHASE",
    "MUSIC_MEDIA",
    "SCREEN_DISPLAY",
    "SETTINGS_FEATURE",
    "PERFORMANCE_STABILITY",
    "HARDWARE_DEVICE",
    "OTHER"
]

# --------------------------------------------------
# LOAD GOLDEN SET
# --------------------------------------------------

df = pd.read_csv(FILE_PATH)

# Make sure final_intent exists
if "final_intent" not in df.columns:
    df["final_intent"] = ""

# Treat empty values as unlabeled
df["final_intent"] = df["final_intent"].fillna("")

# --------------------------------------------------
# DISPLAY INTENTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("GOLDEN SET — HUMAN LABELING")
print("=" * 65)

for i, intent in enumerate(INTENTS, start=1):
    print(f"{i:2}. {intent}")

print("\nCommands:")
print("  q = save and quit")
print("  s = skip this example")
print("=" * 65)

# --------------------------------------------------
# LABEL LOOP
# --------------------------------------------------

for index, row in df.iterrows():

    # Skip already labelled examples
    if row["final_intent"] != "":
        continue

    print("\n" + "-" * 65)

    print(
        f"Example {row['example_id']} "
        f"({index + 1}/{len(df)})"
    )

    print("\nCUSTOMER MESSAGE:")
    print(row["customer_message"])

    print(
        "\nSuggested label:",
        row["suggested_intent"]
    )

    print("\nChoose the PRIMARY customer problem:")

    while True:

        choice = input(
            "\nEnter 1-12, s to skip, or q to quit: "
        ).strip().lower()

        # Quit
        if choice == "q":

            df.to_csv(
                FILE_PATH,
                index=False
            )

            print(
                "\nProgress saved. "
                "You can run this script again to continue."
            )

            raise SystemExit

        # Skip
        if choice == "s":

            print("Skipped.")

            break

        # Validate number
        if choice.isdigit():

            number = int(choice)

            if 1 <= number <= len(INTENTS):

                selected_intent = INTENTS[
                    number - 1
                ]

                df.at[
                    index,
                    "final_intent"
                ] = selected_intent

                # Save after every label
                df.to_csv(
                    FILE_PATH,
                    index=False
                )

                print(
                    "Saved:",
                    selected_intent
                )

                break

        print(
            "Invalid choice. "
            "Enter a number from 1-12."
        )

# --------------------------------------------------
# FINISHED
# --------------------------------------------------

df.to_csv(
    FILE_PATH,
    index=False
)

labelled = (
    df["final_intent"]
    .ne("")
    .sum()
)

print("\n" + "=" * 65)
print("LABELING COMPLETE")
print("=" * 65)

print(
    "Labelled examples:",
    labelled,
    "/",
    len(df)
)

print("\nFinal intent distribution:")

print(
    df["final_intent"]
    .value_counts()
)