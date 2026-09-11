import pandas as pd

file_path = "data/raw/twcs.csv"

df = pd.read_csv(file_path, usecols=["author_id", "inbound"])

print("\n--- AUTHOR COUNTS ---")
print(df["author_id"].value_counts().head(50))

print("\n--- INBOUND / OUTBOUND BY AUTHOR ---")
summary = (
    df.groupby("author_id")["inbound"]
    .agg(["count", "sum"])
    .sort_values("count", ascending=False)
)

print(summary.head(50))