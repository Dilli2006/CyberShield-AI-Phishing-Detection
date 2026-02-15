import pandas as pd

df = pd.read_csv("data/url.csv")

print("Original columns:")
print(df.columns)

df_small = df[["URL", "label"]]
df_small.to_csv("data/url_text.csv", index=False)

print("\nNew dataset created: data/url_text.csv")
print("Columns in new dataset:")
print(df_small.columns)
print("\nFirst 5 rows:")
print(df_small.head())
