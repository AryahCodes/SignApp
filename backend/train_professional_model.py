import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def create_professional_model(num_classes=24):
    """
    Based on SigNN research paper architecture
    Relu(900) → Dropout → Relu(400) → Dropout → Tanh(200) → Dropout → Softmax
    
    Input: 59 features from z-score normalized FeatureExtractor
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(72,)),  # 59 features from new FeatureExtractor
        
        keras.layers.Dense(900, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.15),
        
        keras.layers.Dense(400, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),
        
        keras.layers.Dense(200, activation='tanh'),
        keras.layers.Dropout(0.4),
        
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_kaggle_data(data_dir='kaggle_processed'):
    """Load your preprocessed Kaggle data"""
    data_dir = Path(data_dir)
    
    def load_split(name):
        filepath = data_dir / f"{name}_data.pkl"
        print(f"   Loading {filepath}...")
        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    print("📂 Loading Kaggle preprocessed data...")
    train_data = load_split('train')
    val_data = load_split('val')
    test_data = load_split('test')
    
    print(f"✅ Loaded: {len(train_data)} train | {len(val_data)} val | {len(test_data)} test samples")
    return train_data, val_data, test_data

def extract_features_from_data(data, extractor):
    """
    Extract z-score normalized features from landmark data
    Using the new FeatureExtractor with z-score normalization
    """
    X = []
    y = []
    
    for item in data:
        landmarks = item['landmarks']
        label = item['label']
        
        # Extract features (will apply z-score normalization automatically)
        features = extractor.extract_features(landmarks)
        
        if features is not None:
            X.append(features)
            y.append(label)
        else:
            print(f"⚠️  Warning: Could not extract features for label {label}")
    
    return np.array(X), y

def train_professional_kaggle_model():
    """Train professional deep learning model on Kaggle preprocessed data"""
    
    print("=" * 70)
    print("🚀 TRAINING PROFESSIONAL DEEP LEARNING MODEL")
    print("📊 Dataset: Kaggle Preprocessed Data")
    print("🔬 Features: Z-Score Normalized (59 features)")
    print("🧠 Architecture: SigNN Research Paper (Relu → Dropout → Tanh → Softmax)")
    print("=" * 70)
    
    # Load preprocessed Kaggle data
    train_data, val_data, test_data = load_kaggle_data()
    
    # Initialize feature extractor
    from feature_extractor import FeatureExtractor
    extractor = FeatureExtractor()
    
    # Extract features with z-score normalization
    print("\n🔄 Extracting z-score normalized features...")
    X_train, y_train = extract_features_from_data(train_data, extractor)
    X_val, y_val = extract_features_from_data(val_data, extractor)
    X_test, y_test = extract_features_from_data(test_data, extractor)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"   Train: {X_train.shape} samples, {X_train.shape[1]} features")
    print(f"   Val:   {X_val.shape} samples, {X_val.shape[1]} features")
    print(f"   Test:  {X_test.shape} samples, {X_test.shape[1]} features")
    
    # Create label mapping
    unique_labels = sorted(set(y_train))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    print(f"\n🔤 Training on {len(unique_labels)} letters:")
    print(f"   {unique_labels}")
    
    # Convert labels to categorical (one-hot encoding)
    y_train_idx = [label_to_idx[l] for l in y_train]
    y_val_idx = [label_to_idx[l] for l in y_val]
    y_test_idx = [label_to_idx[l] for l in y_test]
    
    y_train_cat = keras.utils.to_categorical(y_train_idx, num_classes=len(unique_labels))
    y_val_cat = keras.utils.to_categorical(y_val_idx, num_classes=len(unique_labels))
    y_test_cat = keras.utils.to_categorical(y_test_idx, num_classes=len(unique_labels))
    
    # Create model
    print("\n🏗️  Building professional neural network...")
    model = create_professional_model(num_classes=len(unique_labels))
    
    print("\n📊 Model Architecture:")
    model.summary()
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=5,
            factor=0.5,
            verbose=1,
            min_lr=0.00001
        ),
        keras.callbacks.ModelCheckpoint(
            'models/professional_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n🎓 Training model (this may take 5-15 minutes)...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    
    # Get predictions for classification report
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    y_pred_labels = [idx_to_label[idx] for idx in y_pred_idx]
    
    # Print classification report
    print("\n📊 Test Set Classification Report:")
    print("=" * 70)
    print(classification_report(y_test, y_pred_labels))
    
    # Calculate per-letter accuracy
    print("\n📊 Per-Letter Accuracy:")
    for label in unique_labels:
        label_mask = np.array(y_test) == label
        if label_mask.sum() > 0:
            label_acc = (np.array(y_pred_labels)[label_mask] == label).mean()
            print(f"   {label}: {label_acc:.1%}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_labels, unique_labels)
    
    # Save model
    print("\n💾 Saving model and metadata...")
    model.save('models/professional_model.h5')
    print("   ✅ Saved: models/professional_model.h5")
    
    # Save label mappings
    with open('models/professional_label_mapping.pkl', 'wb') as f:
        pickle.dump({'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label}, f)
    print("   ✅ Saved: models/professional_label_mapping.pkl")
    
    with open('models/professional_labels.json', 'w') as f:
        json.dump(unique_labels, f, indent=2)
    print("   ✅ Saved: models/professional_labels.json")
    
    # Save training history
    history_dict = {
        'train_loss': [float(x) for x in history.history['loss']],
        'train_accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'epochs_trained': len(history.history['loss'])
    }
    
    with open('models/professional_training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    print("   ✅ Saved: models/professional_training_history.json")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"✅ Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")
    print(f"✅ Final Test Accuracy: {test_acc:.2%}")
    print(f"✅ Epochs Trained: {len(history.history['loss'])}")
    print(f"\n💾 Files saved:")
    print(f"   • models/professional_model.h5 (main model)")
    print(f"   • models/professional_model_best.h5 (best checkpoint)")
    print(f"   • models/professional_labels.json (letter list)")
    print(f"   • models/professional_label_mapping.pkl (label mappings)")
    print(f"   • models/professional_training_history.json (training stats)")
    print(f"   • models/confusion_matrix_professional.png (confusion matrix)")
    print(f"   • models/training_history_professional.png (training plots)")
    print(f"\n📝 Your OLD RandomForest model is still at:")
    print(f"   • models/kaggle_letter_classifier.pkl (UNCHANGED)")
    print(f"   • models/letter_classifier.pkl (UNCHANGED)")
    print("\n🚀 Next Steps:")
    print("   1. Create professional_letter_classifier.py (I'll give you this next)")
    print("   2. Update server.py to use new model")
    print("   3. Restart backend and test!")
    print("=" * 70)

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Letter', fontsize=14, fontweight='bold')
    plt.ylabel('True Letter', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix - Professional Deep Learning Model', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix_professional.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: models/confusion_matrix_professional.png")
    plt.close()

def plot_training_history(history):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_history_professional.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: models/training_history_professional.png")
    plt.close()

if __name__ == '__main__':
    train_professional_kaggle_model()