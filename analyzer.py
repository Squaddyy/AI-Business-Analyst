# analyzer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import google.generativeai as genai
import os

def generate_executive_summary(top_countries, segment_counts, model_accuracy):
    """Uses Gemini to write a business summary of the analysis results."""
    try:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return "Error: GEMINI_API_KEY not set."
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-pro-latest')
    except Exception as e:
        return f"Error configuring AI model: {e}"

    prompt = f"""
    You are a senior business analyst writing an executive summary.
    Based on the following data, write a concise, professional summary highlighting key insights
    for non-technical stakeholders.

    **Data Findings:**
    1. Top 10 Countries by Sales: {top_countries.to_string()}
    2. Customer Segments: {segment_counts.to_string()}
    3. Predictive Model Accuracy: {model_accuracy * 100:.2f}%.

    **Your Task:**
    Summarize the dominant market, the most significant customer segment, and the value of the predictive model.
    """
    
    response = model.generate_content(prompt)
    return response.text

# --- Main Analysis Pipeline ---
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

print("\nSegmenting customers...")
top_countries = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm_df['M_Score'] = pd.qcut(rfm_df['MonetaryValue'], 5, labels=[1, 2, 3, 4, 5])
def create_segments(df):
    if df['R_Score'] >= 4 and df['F_Score'] >= 4: return 'Champions'
    elif df['R_Score'] >= 3 and df['F_Score'] >= 3: return 'Loyal Customers'
    else: return 'Standard'
rfm_df['Segment'] = rfm_df.apply(create_segments, axis=1)
segment_counts = rfm_df['Segment'].value_counts()
print("✅ Customer segmentation complete.")

print("\nTraining predictive model...")
model_df = rfm_df.copy()
model_df['IsHighValue'] = model_df['Segment'].apply(lambda x: 1 if x == 'Champions' else 0)
X = model_df[['Recency', 'Frequency', 'MonetaryValue']]
y = model_df['IsHighValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model training complete.")

print("\n--- Generating AI-Powered Executive Summary ---")
summary = generate_executive_summary(top_countries, segment_counts, accuracy)
print(summary)
print("---------------------------------------------")