# analyzer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- Data Loading, Cleaning, and RFM ---
print("Loading, cleaning, and calculating RFM scores...")
df = pd.read_csv('data.csv', encoding='ISO-8859-1')
df.dropna(subset=['CustomerID'], inplace=True)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm_df.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'MonetaryValue'}, inplace=True)
print("✅ RFM calculation complete.")

# --- Customer Segmentation ---
print("\nSegmenting customers...")
rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm_df['M_Score'] = pd.qcut(rfm_df['MonetaryValue'], 5, labels=[1, 2, 3, 4, 5])

def create_segments(df):
    if df['R_Score'] >= 4 and df['F_Score'] >= 4: return 'Champions'
    elif df['R_Score'] >= 3 and df['F_Score'] >= 3: return 'Loyal Customers'
    else: return 'Standard'
rfm_df['Segment'] = rfm_df.apply(create_segments, axis=1)
print("✅ Customer segmentation complete.")

# --- Predictive Modeling ---
print("\n--- Starting Predictive Modeling ---")

# 1. Prepare data for the model
model_df = rfm_df.copy()
model_df['IsHighValue'] = model_df['Segment'].apply(lambda x: 1 if x == 'Champions' else 0)

# Define features (X) and target (y)
X = model_df[['Recency', 'Frequency', 'MonetaryValue']]
y = model_df['IsHighValue']

# 2. Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("1. Data split into training and testing sets.")

# 3. Train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("2. Machine learning model trained successfully.")

# 4. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation Results ---")
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))