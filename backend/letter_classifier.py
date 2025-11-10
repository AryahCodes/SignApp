import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from feature_extractor import FeatureExtractor

class LetterClassifier:
    """Robust ASL letter classifier using geometric hand features."""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        self.labels = []
        print("✅ LetterClassifier initialized (RandomForest)")

    def train(self, landmarks_list, labels):
        """Train the model."""
        print(f"🎓 Training on {len(landmarks_list)} samples...")
        X = self.feature_extractor.extract_batch(landmarks_list)
        if X is None or len(X) == 0:
            print("❌ No valid landmarks extracted.")
            return False

        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, labels)
        self.labels = sorted(list(set(labels)))
        self.is_trained = True
        print(f"✅ Training complete! {len(self.labels)} classes learned.")
        return True

    def predict(self, landmarks):
        """Predict ASL letter from landmarks."""
        if not self.is_trained:
            return {'success': False, 'error': 'Model not trained'}

        features = self.feature_extractor.extract_features(landmarks)
        if features is None:
            return {'success': False, 'error': 'Invalid input'}

        X_scaled = self.scaler.transform(features.reshape(1, -1))
        probs = self.classifier.predict_proba(X_scaled)[0]
        idx = np.argmax(probs)
        return {
            'success': True,
            'letter': self.classifier.classes_[idx],
            'confidence': float(probs[idx])
        }

    def save_model(self, path="models/letter_classifier.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'labels': self.labels,
            'is_trained': self.is_trained
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Model saved → {path}")

    def load_model(self, path="models/letter_classifier.pkl"):
        if not os.path.exists(path):
            print(f"⚠️ Model not found: {path}")
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.classifier = data['classifier']
        self.scaler = data['scaler']
        self.labels = data['labels']
        self.is_trained = data['is_trained']
        print(f"✅ Model loaded with {len(self.labels)} classes")
        return True