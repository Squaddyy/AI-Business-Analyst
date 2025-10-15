# analyzer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to our data file
file_path = 'data.csv'

# --- Data Loading & Cleaning ---
print("Loading and cleaning the dataset...")
df = pd.read_csv(file_path, encoding='ISO-8859-1')
df.dropna(subset=['CustomerID'], inplace=True)
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
print("✅ Data cleaning complete.")

# --- Feature Engineering & RFM Calculation ---
print("\nCalculating RFM scores...")
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm_df.rename(columns={'InvoiceDate': 'Recency',
                       'InvoiceNo': 'Frequency',
                       'TotalPrice': 'MonetaryValue'}, inplace=True)
print("✅ RFM scores calculated.")


# --- Customer Segmentation ---
print("\n--- Starting Customer Segmentation ---")

# Create RFM quintile scores (1-5, with 5 being best)
# Note: For Recency, a lower value is better, so the labels are reversed.
rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm_df['M_Score'] = pd.qcut(rfm_df['MonetaryValue'], 5, labels=[1, 2, 3, 4, 5])

# Combine the scores to get a full RFM score
rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

# Create human-readable segments based on scores
# This uses a regex match to find customers based on their score patterns.
def create_segments(df):
    if df['R_Score'] >= 4 and df['F_Score'] >= 4:
        return 'Champions'
    elif df['R_Score'] >= 3 and df['F_Score'] >= 3:
        return 'Loyal Customers'
    elif df['R_Score'] >= 4:
        return 'Recent Customers'
    elif df['F_Score'] >= 4:
        return 'Potential Loyalists'
    elif df['R_Score'] <= 2 and df['F_Score'] <= 2:
        return 'At Risk'
    else:
        return 'Standard'

rfm_df['Segment'] = rfm_df.apply(create_segments, axis=1)

print("\nTop 5 rows of the final RFM DataFrame with Segments:")
print(rfm_df.head())

# --- Visualize the Customer Segments ---
print("\nGenerating a bar chart for customer segments...")
segment_counts = rfm_df['Segment'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=segment_counts.index, y=segment_counts.values, palette='mako')
plt.title('Customer Segmentation Distribution')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

chart_filename = 'customer_segments.png'
plt.savefig(chart_filename)
print(f"✅ Segment chart saved successfully as '{chart_filename}'")