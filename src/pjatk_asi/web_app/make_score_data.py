import pandas as pd

# Path to your CSV file
csv_file = "../../../data/01_raw/credit_data.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Remove the "is_fraud" column if it exists
if "is_fraud" in df.columns:
    df.drop(columns=["is_fraud"], inplace=True)

# Take a random sample of 200 rows
sample_df = df.sample(n=200, random_state=42)

# Print the first few rows of the sampled DataFrame
print(sample_df.head())

# Optionally, save the sampled DataFrame to a new CSV file
sample_csv_file = "../../../score_data.csv"
sample_df.to_csv(sample_csv_file, index=False)