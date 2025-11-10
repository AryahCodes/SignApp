import numpy as np
import json
from tensorflow.keras.models import load_model
from pathlib import Path

class LetterClassifier:
    def __init__(self):
        self.model = None
        self.labels = []
        self.is_trained = False
        self.model_path = Path("models/asl_landmark_model.h5")
        self.labels_path = Path("models/class_labels.json")

    def load_model(self):
        """Load TensorFlow model and class labels"""
        if self.model_path.exists() and self.labels_path.exists():
            print("🧠 Loading ASL landmark model...")
            self.model = load_model(self.model_path)
            with open(self.labels_path, 'r') as f:
                self.labels = json.load(f)
            self.is_trained = True
            print(f"✅ Model loaded ({len(self.labels)} classes)")
        else:
            print("⚠️ Model files not found. Please train or copy model first.")

    def preprocess_landmarks(self, landmarks):
        """Flatten normalized landmarks into a single array"""
        arr = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks]).flatten()
        return np.expand_dims(arr, axis=0)

    def predict(self, landmarks):
        """Predict ASL letter"""
        if not self.is_trained or self.model is None:
            return {"success": False, "error": "Model not loaded"}

        try:
            X = self.preprocess_landmarks(landmarks)
            probs = self.model.predict(X, verbose=0)[0]
            idx = int(np.argmax(probs))
            letter = self.labels[idx]
            confidence = float(probs[idx])
            return {
                "success": True,
                "letter": letter,
                "confidence": confidence
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_model(self):
        """No-op, TensorFlow model already saved."""
        pass