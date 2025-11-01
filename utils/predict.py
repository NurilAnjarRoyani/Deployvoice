import joblib
import numpy as np

def predict_voice(features):
    try:
        # Muat model dan label encoder
        model = joblib.load("models/classifier.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        
        # Prediksi
        pred = model.predict(features)
        prob = model.predict_proba(features)
        class_name = label_encoder.inverse_transform(pred)[0]
        confidence = np.max(prob) * 100
        
        return class_name, confidence
    except Exception as e:
        print("❌ Error prediksi:", e)
        return None, 0
