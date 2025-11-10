import pickle
import numpy as np
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load processed data
# -------------------------------
def load_data(data_dir):
    data_dir = Path(data_dir)
    def load_split(name):
        with open(data_dir / f"{name}.pkl", "rb") as f:
            data = pickle.load(f)
        X = np.array([d["landmarks"] for d in data])
        y = np.array([d["label"] for d in data])
        return X, y

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print(f"✅ Loaded dataset: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# -------------------------------
# Encode labels
# -------------------------------
def encode_labels(y_train, y_val, y_test):
    classes = sorted(list(set(y_train)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    print(f"🔤 Classes: {classes}")

    y_train = np.array([class_to_idx[y] for y in y_train])
    y_val = np.array([class_to_idx[y] for y in y_val])
    y_test = np.array([class_to_idx[y] for y in y_test])

    num_classes = len(classes)
    return (y_train, y_val, y_test, classes, num_classes)

# -------------------------------
# Build model (MLP)
# -------------------------------
def build_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

# -------------------------------
# Plot confusion matrix
# -------------------------------
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# -------------------------------
# Main training pipeline
# -------------------------------
if __name__ == "__main__":
    data_dir = Path("kaggle_processed")

    # Load and prepare
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(data_dir)
    y_train, y_val, y_test, classes, num_classes = encode_labels(y_train, y_val, y_test)

    input_dim = X_train.shape[1]
    model = build_model(input_dim, num_classes)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        verbose=1
    )

    # Evaluate
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

    # Predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\n🧾 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # Save model + class map
    model.save("asl_landmark_model.h5")
    with open("class_labels.json", "w") as f:
        json.dump(classes, f, indent=2)
    print("💾 Saved model → asl_landmark_model.h5")
    print("💾 Saved labels → class_labels.json")

    # Optional: visualize confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes)
