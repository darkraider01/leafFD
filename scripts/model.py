import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === Paths ===
CSV_PATH = "./processed/fd_features.csv"
MODEL_PATH = "./processed/leaf_fd_classifier.pkl"

# === Load CSV ===
df = pd.read_csv(CSV_PATH)

# === Features and Label ===
X = df[[
    "fd_geometrical",
    "fd_textural",
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM"
]]
y = df["health_status"]

# === Encode Labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# === Train Classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# === Save Model and Label Encoder ===
joblib.dump(clf, MODEL_PATH)
joblib.dump(le, MODEL_PATH.replace(".pkl", "_label_encoder.pkl"))
print(f"\n💾 Model saved to: {MODEL_PATH}")
