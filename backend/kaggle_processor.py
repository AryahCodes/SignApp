import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import pickle
import random

class KaggleDatasetProcessor:
    """
    Process merged ASL Alphabet datasets (real + synthetic)
    to extract and normalize MediaPipe hand landmarks.
    """

    def __init__(self, dataset_path, output_path="kaggle_processed"):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3
        )

        # Automatically detect valid folders (A–Z + Blank)
        all_folders = [f.name for f in (self.dataset_path / "asl_alphabet_train").iterdir() if f.is_dir()]
        self.valid_classes = sorted([c for c in all_folders if c.upper() in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") or c.lower() == "blank"])

        # Exclude J and Z (motion-based)
        self.valid_classes = [c for c in self.valid_classes if c.upper() not in ["J", "Z"]]

        print("✅ Processor initialized")
        print(f"📂 Dataset path: {self.dataset_path}")
        print(f"💾 Output path: {self.output_path}")
        print(f"🔤 Classes to process: {self.valid_classes}")

    # ---------- AUGMENTATION ----------
    def augment_image(self, image):
        """Simple augmentations for robustness"""
        augmentations = []

        # Random horizontal flip
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            augmentations.append("flip")

        # Random brightness
        if random.random() < 0.4:
            factor = 0.8 + 0.4 * random.random()
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
            augmentations.append("brightness")

        # Random rotation
        if random.random() < 0.4:
            angle = random.uniform(-15, 15)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            image = cv2.warpAffine(image, M, (w, h))
            augmentations.append("rotate")

        return image, augmentations

    # ---------- LANDMARK EXTRACTION ----------
    def extract_landmarks(self, image):
        """Extract normalized hand landmarks"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])

        # Normalize for size and position (centered at wrist)
        wrist = landmarks[0]
        landmarks -= wrist
        scale = np.linalg.norm(landmarks, axis=1).max()
        landmarks /= scale if scale > 0 else 1

        return landmarks.flatten()

    # ---------- FOLDER PROCESSING ----------
    def process_class_folder(self, label, max_samples=None):
        """Process all images for a given class"""
        folder = self.dataset_path / "asl_alphabet_train" / label
        if not folder.exists():
            print(f"⚠️ Missing folder: {folder}")
            return []

        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if max_samples:
            image_files = image_files[:max_samples]

        data = []
        print(f"\n📸 Processing '{label}' ({len(image_files)} images)")

        for img_path in tqdm(image_files, desc=f"Class {label}"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # Base extraction
            landmarks = self.extract_landmarks(image)
            if landmarks is not None:
                data.append({"landmarks": landmarks, "label": label})

            # Augmented version (optional)
            image_aug, _ = self.augment_image(image)
            landmarks_aug = self.extract_landmarks(image_aug)
            if landmarks_aug is not None:
                data.append({"landmarks": landmarks_aug, "label": label})

        print(f"✅ '{label}': {len(data)} samples (including augmented)")
        return data

    # ---------- MAIN PIPELINE ----------
    def process_all(self, max_samples_per_class=None):
        all_data, stats = [], {}
        print(f"\n🚀 Starting processing with {len(self.valid_classes)} classes")

        for label in self.valid_classes:
            class_data = self.process_class_folder(label, max_samples=max_samples_per_class)
            all_data.extend(class_data)
            stats[label] = len(class_data)

        print("\n✅ Done processing!")
        print(f"📊 Total samples: {len(all_data)}")
        return all_data, stats

    # ---------- SAVE HELPERS ----------
    def save_pickle(self, data, filename):
        out_file = self.output_path / filename
        with open(out_file, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 Saved {len(data)} samples → {out_file}")
        return out_file

    def split_dataset(self, data, train_ratio=0.7, val_ratio=0.15):
        np.random.shuffle(data)
        n = len(data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train, val, test = data[:n_train], data[n_train:n_train+n_val], data[n_train+n_val:]

        print("\n📊 Dataset split:")
        print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        return train, val, test

    def cleanup(self):
        self.hands.close()


if __name__ == "__main__":
    processor = KaggleDatasetProcessor(
        dataset_path="../asl_alphabet_data",
        output_path="kaggle_processed"
    )

    all_data, stats = processor.process_all(max_samples_per_class=500)
    train, val, test = processor.split_dataset(all_data)
    processor.save_pickle(train, "train.pkl")
    processor.save_pickle(val, "val.pkl")
    processor.save_pickle(test, "test.pkl")

    with open(processor.output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    processor.cleanup()
    print("\n✅ All processing complete! Data ready for training.")
