
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection System", page_icon="🔍", layout="wide")

@st.cache_resource
def load_model():
    BASE   = os.path.dirname(os.path.abspath(__file__))
    model  = joblib.load(os.path.join(BASE, '..', 'model', 'fraud_model.pkl'))
    scaler = joblib.load(os.path.join(BASE, '..', 'model', 'scaler.pkl'))
    return model, scaler

model, scaler = load_model()

st.title("🔍 Cloud-Based Fraud Detection System")
st.markdown("Real-time transaction fraud scoring using Machine Learning")

st.sidebar.header("📂 Upload Transactions")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔎 Single Transaction", "📁 Batch Analysis"])

with tab1:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['Amount'] = scaler.fit_transform(df[['Amount']])
        df['Time']   = scaler.fit_transform(df[['Time']])
        features = df.drop('Class', axis=1) if 'Class' in df.columns else df
        df['Fraud_Probability'] = model.predict_proba(features)[:, 1]
        df['Prediction']        = model.predict(features)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", len(df))
        col2.metric("Flagged as Fraud",   int(df['Prediction'].sum()))
        col3.metric("Fraud Rate",         f"{df['Prediction'].mean()*100:.2f}%")
        col4.metric("Avg Fraud Score",    f"{df['Fraud_Probability'].mean():.3f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Fraud Probability Distribution")
            fig, ax = plt.subplots()
            ax.hist(df['Fraud_Probability'], bins=50, color='steelblue', edgecolor='white')
            ax.set_xlabel("Fraud Probability")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with col_b:
            st.subheader("Fraud vs Legit Split")
            counts = df['Prediction'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(counts, labels=['Legit', 'Fraud'], autopct='%1.1f%%',
                    colors=['#2ecc71', '#e74c3c'])
            st.pyplot(fig2)

        st.subheader("🚨 High Risk Transactions (Score > 0.7)")
        high_risk = df[df['Fraud_Probability'] > 0.7].sort_values('Fraud_Probability', ascending=False)
        st.dataframe(high_risk[['Amount', 'Fraud_Probability', 'Prediction']].head(20))
    else:
        st.info("👈 Upload a CSV file from the sidebar to see the dashboard.")

with tab2:
    st.subheader("Score a Single Transaction")
    col1, col2 = st.columns(2)
    with col1:
        amount = st.slider("Transaction Amount ($)", 0.0, 5000.0, 100.0)
        time   = st.slider("Time (seconds from first txn)", 0, 172800, 50000)
    with col2:
        v1 = st.slider("V1 (PCA Feature)", -5.0, 5.0, 0.0)
        v2 = st.slider("V2 (PCA Feature)", -5.0, 5.0, 0.0)
        v3 = st.slider("V3 (PCA Feature)", -5.0, 5.0, 0.0)

    if st.button("🔍 Check for Fraud"):
        features_input = np.zeros((1, 30))
        features_input[0, 0]  = time
        features_input[0, 1]  = v1
        features_input[0, 2]  = v2
        features_input[0, 3]  = v3
        features_input[0, 29] = amount
        prob       = model.predict_proba(features_input)[0][1]
        prediction = model.predict(features_input)[0]
        if prediction == 1:
            st.error(f"🚨 FRAUD DETECTED — Probability: {prob:.2%}")
        else:
            st.success(f"✅ LEGITIMATE — Fraud Probability: {prob:.2%}")
        st.progress(prob)
        st.caption(f"Fraud Risk Score: {prob:.4f}")

with tab3:
    if uploaded_file:
        st.subheader("📥 Download Scored Transactions")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "fraud_scored_transactions.csv", "text/csv")
        st.dataframe(df.head(50))
    else:
        st.info("Upload a file first in the Dashboard tab.")
