# 🔍 Cloud-Based Fraud Detection System

A machine learning-powered web application that detects fraudulent financial transactions in real-time, built with Python and deployed on the cloud.

---

## 🚀 Live Demo
👉 **[Try the App](https://fraud-detection-system-lzen.onrender.com/)**

---

## 📌 Features

- **Real-time Fraud Scoring** — Instantly scores any transaction with a fraud probability
- **Interactive Dashboard** — Visual charts showing fraud distribution and risk breakdown
- **Batch Processing** — Upload a CSV of transactions and get all of them scored at once
- **Single Transaction Scorer** — Manually input transaction details and get an instant prediction
- **Downloadable Results** — Export scored transactions as a CSV file

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.x |
| ML Library | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Frontend/Dashboard | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Deployment | Render / Streamlit Cloud |
| Version Control | Git & GitHub |

---

## 🧠 Machine Learning

- **Algorithm:** Random Forest Classifier
- **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions
- **Class Imbalance Handling:** Undersampling majority class (fraud is only 0.17% of data)
- **Features:** 30 features — `Time`, `V1–V28` (PCA-transformed), `Amount`
- **Output:** Fraud probability score between 0 and 1

---

## 📁 Project Structure

```
fraud-detection/
├── app/
│   └── app.py              # Streamlit dashboard
├── model/
│   ├── train_model.py      # Model training script
│   ├── fraud_model.pkl     # Trained Random Forest model
│   └── scaler.pkl          # Fitted StandardScaler
├── data/
│   └── creditcard.csv      # Dataset (not committed - too large)
├── requirements.txt        # Python dependencies
├── Procfile                # Render deployment config
└── README.md
```

---

## ⚙️ Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Khmariya16/Fraud-detection-system.git
cd Fraud-detection-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model (optional — pkl files already included)
```bash
cd model
python train_model.py
cd ..
```

### 5. Run the App
```bash
streamlit run app/app.py
```

Open your browser at: `http://localhost:8501`

---

## 📊 How to Use

1. Download the [Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle
2. Open the live app or run locally
3. Upload `creditcard.csv` using the sidebar file uploader
4. View the dashboard with fraud metrics and charts
5. Use the **Single Transaction** tab to test individual transactions
6. Download scored results from the **Batch Analysis** tab

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~99% |
| Precision (Fraud) | ~95% |
| Recall (Fraud) | ~92% |
| F1-Score (Fraud) | ~93% |

---

## 🌐 Deployment

This app is deployed on **Render** using the following config:

```
Build Command:  pip install -r requirements.txt
Start Command:  streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 🤝 Acknowledgements

- Dataset: [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Built with [Streamlit](https://streamlit.io/) and [Scikit-learn](https://scikit-learn.org/)

---

## 👩‍💻 Author

**Mariya Khan**
- GitHub: [@Khmariya16](https://github.com/Khmariya16)