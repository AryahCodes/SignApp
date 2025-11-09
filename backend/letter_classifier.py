import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from feature_extractor import FeatureExtractor

class LetterClassifier:
    """Classifier for recognizing ASL letters from hand landmarks"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.is_trained = False
        self.labels = []
        print("✅ LetterClassifier initialized")
    
    def train(self, landmarks_list, labels):
        """Train the classifier"""
        print(f"🎓 Training on {len(landmarks_list)} samples...")
        
        features = self.feature_extractor.extract_batch(landmarks_list)
        
        if features is None or len(features) == 0:
            print("❌ No valid features extracted")
            return False
        
        features_scaled = self.scaler.fit_transform(features)
        self.classifier.fit(features_scaled, labels)
        self.labels = list(set(labels))
        self.is_trained = True
        
        print(f"✅ Training complete! Can recognize: {sorted(self.labels)}")
        return True
    
    def predict(self, landmarks):
        """Predict letter from landmarks"""
        if not self.is_trained:
            return {
                'success': False,
                'error': 'Classifier not trained',
                'letter': None,
                'confidence': 0.0
            }
        
        features = self.feature_extractor.extract_features(landmarks)
        
        if features is None:
            return {
                'success': False,
                'error': 'Invalid landmarks',
                'letter': None,
                'confidence': 0.0
            }
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities))
        
        class_probs = {
            label: float(prob) 
            for label, prob in zip(self.classifier.classes_, probabilities)
        }
        
        return {
            'success': True,
            'letter': prediction,
            'confidence': confidence,
            'probabilities': class_probs
        }
    
    def save_model(self, filepath='models/letter_classifier.pkl'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'labels': self.labels,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath='models/letter_classifier.pkl'):
        """Load a trained model"""
        if not os.path.exists(filepath):
            print(f"⚠️  Model file not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.labels = model_data['labels']
        self.is_trained = model_data['is_trained']
        
        print(f"✅ Model loaded! Can recognize: {sorted(self.labels)}")
        return True