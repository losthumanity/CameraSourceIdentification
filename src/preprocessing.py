import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

# Constants
ALPHA = 0.7
BETA = 4
GAMMA = np.log(0.01)
PATCH_SIZE = 64
TOP_K_PATCHES = 128

def patch_quality(patch):
    Q = 0
    for channel in range(3):  # R, G, B
        patch_channel = patch[:, :, channel]
        mu_c = np.mean(patch_channel)
        sigma_c = np.std(patch_channel)
        Q += ALPHA * BETA * (mu_c - mu_c**2) + (1 - ALPHA) * (1 - np.exp(GAMMA * sigma_c))
    return Q / 3

def extract_top_k_patches(image, k=TOP_K_PATCHES, patch_size=PATCH_SIZE):
    h, w, _ = image.shape
    patches = []
    qualities = []

    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            q = patch_quality(patch)
            patches.append(patch)
            qualities.append(q)

    if len(patches) == 0:
        return []

    sorted_indices = np.argsort(qualities)[::-1]
    top_indices = sorted_indices[:min(k, len(patches))]
    top_patches = [patches[i] for i in top_indices]

    if len(top_patches) < k:
        padding = [np.zeros_like(top_patches[0]) for _ in range(k - len(top_patches))]
        top_patches.extend(padding)

    return top_patches

def prepare_dataset_as_images(dataset_path, output_path):
    for mode in ['train', 'test']:
        os.makedirs(os.path.join(output_path, mode), exist_ok=True)

    split_dir = os.path.join(output_path, 'splits')
    os.makedirs(split_dir, exist_ok=True)

    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        image_files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        split_file_path = os.path.join(split_dir, f"{class_name}_split.json")

        if os.path.exists(split_file_path):
            with open(split_file_path, 'r') as f:
                split_data = json.load(f)
                train_files = split_data['train']
                test_files = split_data['test']
        else:
            train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
            split_data = {'train': train_files, 'test': test_files}
            with open(split_file_path, 'w') as f:
                json.dump(split_data, f, indent=2)

        for mode, file_list in [('train', train_files), ('test', test_files)]:
            save_dir = os.path.join(output_path, mode, class_name)
            os.makedirs(save_dir, exist_ok=True)

            for idx, file in enumerate(tqdm(file_list, desc=f"{mode.upper()} - {class_name}")):
                file_path = os.path.join(class_path, file)
                image = cv2.imread(file_path)
                if image is None:
                    continue

                patches = extract_top_k_patches(image)
                for i, patch in enumerate(patches):
                    patch_filename = f"{os.path.splitext(file)[0]}_patch_{i}.jpg"
                    patch_path = os.path.join(save_dir, patch_filename)
                    cv2.imwrite(patch_path, patch)

    print("Dataset prepared as image patches with persistent train/test splits!")

# Example usage for Kaggle
dataset_path = './camera_model_data/train/'  # or wherever your Kaggle dataset is mounted
output_path = './camera_model_data/processed_dataset'

prepare_dataset_as_images(dataset_path, output_path)