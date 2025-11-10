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
    """Process ASL Alphabet dataset (Kaggle or synthetic) to extract MediaPipe landmarks."""

    def __init__(self, dataset_path, output_path='kaggle_processed'):
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

        # Detect valid folders (A–Z + Blank)
        root = self.dataset_path / "asl_alphabet_train"
        self.valid_letters = sorted(
            [f.name for f in root.iterdir() if f.is_dir() and f.name.upper() not in ["J", "Z"]]
        )

        print("✅ Processor initialized")
        print(f"📂 Dataset path: {self.dataset_path}")
        print(f"💾 Output path: {self.output_path}")
        print(f"🔤 Letters: {self.valid_letters}")

    # ---------- AUGMENT ----------
    def augment_image(self, image):
        """Perform mild augmentations to improve generalization."""
        aug_img = image.copy()

        if random.random() < 0.5:
            aug_img = cv2.flip(aug_img, 1)

        if random.random() < 0.4:
            factor = 0.8 + 0.4 * random.random()
            aug_img = np.clip(aug_img * factor, 0, 255).astype(np.uint8)

        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            h, w = aug_img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            aug_img = cv2.warpAffine(aug_img, M, (w, h))
        return aug_img

    # ---------- LANDMARK EXTRACTION ----------
    def extract_landmarks_from_image(self, image):
        """Extract MediaPipe hand landmarks and normalize."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        landmarks = [
            {'x': lm.x, 'y': lm.y, 'z': lm.z}
            for lm in hand.landmark
        ]
        return landmarks

    # ---------- PROCESS FOLDER ----------
    def process_letter_folder(self, letter, max_samples=None):
        folder = self.dataset_path / "asl_alphabet_train" / letter
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if max_samples:
            image_files = image_files[:max_samples]

        data = []
        for img_path in tqdm(image_files, desc=f"{letter}"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            landmarks = self.extract_landmarks_from_image(image)
            if landmarks:
                data.append({'landmarks': landmarks, 'label': letter})

            # Augmentation
            aug = self.augment_image(image)
            landmarks_aug = self.extract_landmarks_from_image(aug)
            if landmarks_aug:
                data.append({'landmarks': landmarks_aug, 'label': letter})

        print(f"✅ {letter}: {len(data)} samples (augmented included)")
        return data

    # ---------- MAIN ----------
    def process_all(self, max_samples_per_letter=500):
        all_data, stats = [], {}
        print(f"🚀 Starting processing for {len(self.valid_letters)} classes")

        for letter in self.valid_letters:
            letter_data = self.process_letter_folder(letter, max_samples_per_letter)
            all_data.extend(letter_data)
            stats[letter] = len(letter_data)

        print(f"\n✅ Processing complete — total {len(all_data)} samples")
        return all_data, stats

    def split_dataset(self, data, train_ratio=0.7, val_ratio=0.15):
        np.random.shuffle(data)
        n = len(data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train, val, test = data[:n_train], data[n_train:n_train+n_val], data[n_train+n_val:]
        print(f"📊 Split: Train {len(train)} | Val {len(val)} | Test {len(test)}")
        return train, val, test

    def save_pickle(self, data, name):
        out = self.output_path / f"{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 Saved {len(data)} samples → {out}")

    def cleanup(self):
        self.hands.close()


if __name__ == "__main__":
    processor = KaggleDatasetProcessor(
        dataset_path="../asl_alphabet_data",
        output_path="kaggle_processed"
    )
    all_data, stats = processor.process_all(max_samples_per_letter=500)
    train, val, test = processor.split_dataset(all_data)
    processor.save_pickle(train, "train_data")
    processor.save_pickle(val, "val_data")
    processor.save_pickle(test, "test_data")
    with open(processor.output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    processor.cleanup()