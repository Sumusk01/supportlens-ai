import pandas as pd

file_path = "data/raw/twcs.csv"

# Read only the first 1,000 rows for initial inspection
df = pd.read_csv(file_path, nrows=1000)

print("\n--- DATASET SAMPLE INFORMATION ---")
print("Rows loaded:", len(df))

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head().to_string())

print("\n--- MISSING VALUES IN SAMPLE ---")
print(df.isnull().sum())

print("\n--- DATA TYPES ---")
print(df.dtypes)