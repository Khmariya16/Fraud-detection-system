import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ── 1. Load Data ──────────────────────────────────────────
df = pd.read_csv('../data/creditcard.csv')
print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")

# ── 2. Preprocess ─────────────────────────────────────────
# Scale Amount and Time (other features are already PCA-scaled)
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time']   = scaler.fit_transform(df[['Time']])

X = df.drop('Class', axis=1)
y = df['Class']

# ── 3. Handle Class Imbalance (undersample majority class) ─
fraud     = df[df['Class'] == 1]
legit     = df[df['Class'] == 0].sample(n=len(fraud)*10, random_state=42)
balanced  = pd.concat([fraud, legit])

X_bal = balanced.drop('Class', axis=1)
y_bal = balanced['Class']

# ── 4. Train/Test Split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

# ── 5. Train Model ────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 7. Save Model ─────────────────────────────────────────
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ Model saved as fraud_model.pkl")