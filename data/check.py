import pandas as pd

df = pd.read_csv("url.csv")
print("Columns:")
print(df.columns)
print("\nFirst 5 rows:")
print(df.head())
