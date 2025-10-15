# analyzer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to our data file
file_path = 'data.csv'

# --- Data Loading ---
print("Attempting to load the dataset...")
try:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(f"✅ Dataset loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: The file '{file_path}' was not found.")
    exit()

# --- Data Cleaning ---
print("\n--- Starting Data Cleaning ---")
df.dropna(subset=['CustomerID'], inplace=True)
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
print(f"✅ Data cleaning complete.")

# --- Feature Engineering ---
print("\n--- Starting Feature Engineering ---")
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
print("✅ 'TotalPrice' and 'InvoiceDate' columns prepared.")


# --- RFM Analysis ---
print("\n--- Starting RFM Analysis ---")

# To calculate recency, we need a "snapshot" date.
# We'll set it to one day after the last transaction in the dataset.
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Group by customer and calculate R, F, and M
rfm_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days, # Recency
    'InvoiceNo': 'nunique',                                      # Frequency
    'TotalPrice': 'sum'                                          # Monetary
}).reset_index()

# Rename the columns for clarity
rfm_df.rename(columns={'InvoiceDate': 'Recency',
                       'InvoiceNo': 'Frequency',
                       'TotalPrice': 'MonetaryValue'}, inplace=True)

print("\nTop 5 rows of the new RFM DataFrame:")
print(rfm_df.head())

print("\nStatistical Summary of RFM Data:")
print(rfm_df.describe())