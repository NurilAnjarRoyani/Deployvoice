import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
data_path = "data/bukatutup.csv"
df = pd.read_csv(data_path)

# Pisahkan fitur dan label
X = df.drop(columns=["file_name", "class"])
y = df["class"]

# Encode label jadi angka
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data train-test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Buat model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f" Akurasi model: {acc:.2f}")
print("\nLaporan klasifikasi:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Simpan model dan label encoder
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/classifier.pkl")
joblib.dump(le, "models/label_encoder.pkl")
print(" Model dan label encoder berhasil disimpan di folder 'models/'.")
