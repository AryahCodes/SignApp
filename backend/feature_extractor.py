import numpy as np

class FeatureExtractor:
    """
    Extract meaningful features from hand landmarks for gesture recognition
    """
    
    def __init__(self):
        print("✅ FeatureExtractor initialized")
    
    def extract_features(self, landmarks):
        """
        Extract features from 21 hand landmarks
        
        Args:
            landmarks: List of 21 landmarks with x, y, z coordinates
            
        Returns:
            numpy array of features
        """
        if not landmarks or len(landmarks) != 21:
            return None
        
        features = []
        
        # Convert landmarks to numpy array
        points = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        # Normalize relative to wrist
        wrist = points[0]
        normalized_points = points - wrist
        
        # Flatten normalized positions (63 features)
        features.extend(normalized_points.flatten())
        
        # Palm center
        palm_center = (points[0] + points[5] + points[9] + points[13] + points[17]) / 5
        
        # Finger tip distances from palm (5 features)
        finger_tips = [4, 8, 12, 16, 20]
        for tip_idx in finger_tips:
            distance = np.linalg.norm(points[tip_idx] - palm_center)
            features.append(distance)
        
        # Distances between consecutive fingertips (4 features)
        for i in range(len(finger_tips) - 1):
            dist = np.linalg.norm(points[finger_tips[i]] - points[finger_tips[i+1]])
            features.append(dist)
        
        # Finger extension ratios (5 features)
        finger_chains = [
            [1, 2, 3, 4],      # Thumb
            [5, 6, 7, 8],      # Index
            [9, 10, 11, 12],   # Middle
            [13, 14, 15, 16],  # Ring
            [17, 18, 19, 20]   # Pinky
        ]
        
        for chain in finger_chains:
            finger_length = 0
            for i in range(len(chain) - 1):
                finger_length += np.linalg.norm(points[chain[i+1]] - points[chain[i]])
            
            base_to_tip = np.linalg.norm(points[chain[-1]] - points[chain[0]])
            extension_ratio = base_to_tip / (finger_length + 1e-6)
            features.append(extension_ratio)
        
        # Hand direction (3 features)
        hand_direction = points[9] - points[0]
        hand_direction_norm = hand_direction / (np.linalg.norm(hand_direction) + 1e-6)
        features.extend(hand_direction_norm)
        
        # Palm size (1 feature)
        palm_size = np.linalg.norm(points[9] - points[0])
        features.append(palm_size)
        
        return np.array(features, dtype=np.float32)
    
    def extract_batch(self, landmarks_list):
        """Extract features from multiple landmark sets"""
        features_list = []
        for landmarks in landmarks_list:
            features = self.extract_features(landmarks)
            if features is not None:
                features_list.append(features)
        
        return np.array(features_list) if features_list else None