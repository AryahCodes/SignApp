import json
import os
from datetime import datetime

class DataCollector:
    """Collect and save hand landmark data for training"""
    
    def __init__(self, data_dir='training_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        print(f"✅ DataCollector initialized (saving to {data_dir}/)")
    
    def save_sample(self, landmarks, label):
        """Save a single training sample"""
        label_dir = os.path.join(self.data_dir, label.upper())
        os.makedirs(label_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{label.upper()}_{timestamp}.json"
        filepath = os.path.join(label_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                'label': label.upper(),
                'landmarks': landmarks,
                'timestamp': timestamp
            }, f, indent=2)
        
        return filepath
    
    def load_all_samples(self):
        """Load all training samples"""
        landmarks_list = []
        labels_list = []
        
        if not os.path.exists(self.data_dir):
            return landmarks_list, labels_list
        
        for label in os.listdir(self.data_dir):
            label_dir = os.path.join(self.data_dir, label)
            
            if not os.path.isdir(label_dir):
                continue
            
            for filename in os.listdir(label_dir):
                if not filename.endswith('.json'):
                    continue
                
                filepath = os.path.join(label_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        landmarks_list.append(data['landmarks'])
                        labels_list.append(data['label'])
                except Exception as e:
                    print(f"⚠️  Error loading {filepath}: {e}")
        
        print(f"📂 Loaded {len(landmarks_list)} samples from {self.data_dir}/")
        return landmarks_list, labels_list
    
    def get_sample_counts(self):
        """Get count of samples per label"""
        counts = {}
        
        if not os.path.exists(self.data_dir):
            return counts
        
        for label in os.listdir(self.data_dir):
            label_dir = os.path.join(self.data_dir, label)
            
            if not os.path.isdir(label_dir):
                continue
            
            json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
            counts[label] = len(json_files)
        
        return counts