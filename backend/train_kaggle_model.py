import pickle
import numpy as np
from pathlib import Path
from letter_classifier import LetterClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_data(data_dir='kaggle_processed'):
    data_dir = Path(data_dir)
    def load_split(name):
        with open(data_dir / f"{name}_data.pkl", "rb") as f:
            return pickle.load(f)
    print("📂 Loading dataset splits...")
    return load_split('train'), load_split('val'), load_split('test')

def extract_Xy(data):
    X = [d['landmarks'] for d in data]
    y = [d['label'] for d in data]
    return X, y

def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_data, val_data, test_data = load_data()

    X_train, y_train = extract_Xy(train_data)
    X_val, y_val = extract_Xy(val_data)
    X_test, y_test = extract_Xy(test_data)

    print(f"✅ Loaded: {len(X_train)} train | {len(X_val)} val | {len(X_test)} test")

    clf = LetterClassifier()
    clf.train(X_train, y_train)
    clf.save_model('models/kaggle_letter_classifier.pkl')
    clf.save_model('models/letter_classifier.pkl')

    # Validation
    y_pred, y_true = [], []
    for lms, label in zip(X_val, y_val):
        result = clf.predict(lms)
        if result['success']:
            y_pred.append(result['letter'])
            y_true.append(label)

    print("\n📊 Validation Report:")
    print(classification_report(y_true, y_pred))

    plot_confusion(y_true, y_pred, sorted(list(set(y_true))))

    # Save label list
    with open('models/class_labels.json', 'w') as f:
        json.dump(sorted(list(set(y_train))), f, indent=2)
    print("💾 Saved labels → models/class_labels.json")

    print("\n✅ Model ready! Restart backend to use new classifier.")