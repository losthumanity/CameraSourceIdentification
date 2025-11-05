"""
Deep Learning Pipeline for Camera Source Identification
Enhanced CNN-based classifier with PRNU feature integration
Author: Pranav Patil | Sponsored by PiLabs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import cv2
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from .prnu_extractor import PRNUExtractor


class PRNUDataset(Dataset):
    """Custom dataset for PRNU patterns"""

    def __init__(self, prnu_patterns, labels, transform=None):
        self.prnu_patterns = prnu_patterns
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.prnu_patterns)

    def __getitem__(self, idx):
        # Convert PRNU pattern to 3-channel image for CNN compatibility
        prnu = self.prnu_patterns[idx]

        # Normalize to [0, 1] range
        prnu_normalized = (prnu - prnu.min()) / (prnu.max() - prnu.min() + 1e-8)

        # Create 3-channel image
        prnu_rgb = np.stack([prnu_normalized] * 3, axis=-1)

        if self.transform:
            prnu_rgb = self.transform(prnu_rgb)
        else:
            prnu_rgb = torch.FloatTensor(prnu_rgb).permute(2, 0, 1)

        return prnu_rgb, torch.LongTensor([self.labels[idx]])[0]


class PRNUClassifier(nn.Module):
    """
    CNN classifier for PRNU-based camera identification
    Uses ResNet50 pretrained on ImageNet, fine-tuned for camera classification
    Implementation from technical specification
    """

    def __init__(self, num_classes=9):
        super(PRNUClassifier, self).__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True)

        # Replace final classification layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class CameraIdentificationPipeline:
    """Complete pipeline for camera source identification"""

    def __init__(self, num_classes=9, device=None):
        self.num_classes = num_classes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize components
        self.prnu_extractor = PRNUExtractor()
        self.model = PRNUClassifier(num_classes=num_classes).to(self.device)
        self.class_names = []

        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None

        print(f"Pipeline initialized on device: {self.device}")

    def prepare_data_from_videos(self, video_data_dir, output_dir, samples_per_class=50):
        """
        Prepare PRNU dataset from video directory structure

        Expected structure:
        video_data_dir/
        ├── camera_model_1/
        │   ├── video1.mp4
        │   ├── video2.mp4
        └── camera_model_2/
            ├── video1.mp4
            └── video2.mp4
        """
        os.makedirs(output_dir, exist_ok=True)

        all_prnus = []
        all_labels = []

        camera_folders = sorted([f for f in os.listdir(video_data_dir)
                               if os.path.isdir(os.path.join(video_data_dir, f))])

        self.class_names = camera_folders

        for class_idx, camera_name in enumerate(camera_folders):
            camera_dir = os.path.join(video_data_dir, camera_name)
            video_files = [f for f in os.listdir(camera_dir)
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

            print(f"Processing {camera_name} ({len(video_files)} videos)...")

            camera_prnus = []
            for video_file in tqdm(video_files[:samples_per_class], desc=camera_name):
                try:
                    video_path = os.path.join(camera_dir, video_file)
                    prnu = self.prnu_extractor.video_to_prnu(video_path, num_frames=15)
                    camera_prnus.append(prnu)
                except Exception as e:
                    print(f"Failed to process {video_file}: {e}")
                    continue

            # Save individual camera PRNU patterns
            if camera_prnus:
                all_prnus.extend(camera_prnus)
                all_labels.extend([class_idx] * len(camera_prnus))

                # Generate and save reference pattern
                reference = self.prnu_extractor.generate_reference_pattern(camera_dir, camera_name)

        # Save processed dataset
        dataset_path = os.path.join(output_dir, 'prnu_dataset.npz')
        np.savez(dataset_path,
                prnu_patterns=np.array(all_prnus),
                labels=np.array(all_labels),
                class_names=self.class_names)

        # Save reference patterns
        ref_path = os.path.join(output_dir, 'reference_patterns.npz')
        self.prnu_extractor.save_reference_patterns(ref_path)

        print(f"Dataset prepared: {len(all_prnus)} samples, {len(self.class_names)} classes")
        return np.array(all_prnus), np.array(all_labels)

    def load_dataset(self, dataset_path):
        """Load preprocessed PRNU dataset"""
        data = np.load(dataset_path)
        prnu_patterns = data['prnu_patterns']
        labels = data['labels']
        self.class_names = data['class_names'].tolist()

        print(f"Loaded dataset: {len(prnu_patterns)} samples, {len(self.class_names)} classes")
        return prnu_patterns, labels

    def train(self, prnu_patterns, labels, validation_split=0.2, batch_size=32,
              epochs=20, learning_rate=1e-4):
        """
        Train the PRNU classifier
        Implementation from technical specification
        """

        # Split data
        n_samples = len(prnu_patterns)
        n_val = int(n_samples * validation_split)

        # Shuffle data
        indices = np.random.permutation(n_samples)
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        # Create datasets with transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        train_dataset = PRNUDataset(prnu_patterns[train_indices],
                                   labels[train_indices],
                                   transform=transform)

        val_dataset = PRNUDataset(prnu_patterns[val_indices],
                                 labels[val_indices],
                                 transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

        # Setup optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        best_val_acc = 0.0
        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0

            for imgs, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                imgs, labels_batch = imgs.to(self.device), labels_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Validation phase
            val_acc = self.evaluate(val_loader)
            val_accuracies.append(val_acc)

            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs('./camera_model_data', exist_ok=True)
                torch.save(self.model.state_dict(), './camera_model_data/best_prnu_classifier.pth')
                print(f"✓ New best model saved! Val Acc: {val_acc:.4f}")

        return train_losses, val_accuracies

    def evaluate(self, data_loader):
        """Evaluate model on given data loader"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        return accuracy

    def predict_video_source(self, video_path):
        """Predict camera source for a given video"""
        # Extract PRNU from video
        prnu = self.prnu_extractor.video_to_prnu(video_path)

        # Prepare for model input
        prnu_normalized = (prnu - prnu.min()) / (prnu.max() - prnu.min() + 1e-8)
        prnu_rgb = np.stack([prnu_normalized] * 3, axis=-1)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        input_tensor = transform(prnu_rgb).unsqueeze(0).to(self.device)

        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            'predicted_camera': self.class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': {self.class_names[i]: prob.item()
                                for i, prob in enumerate(probabilities[0])}
        }

    def detect_deepfake(self, video_path, expected_camera=None, correlation_threshold=0.4):
        """Combined approach: CNN + PRNU correlation for deepfake detection"""

        # Method 1: CNN-based prediction
        cnn_result = self.predict_video_source(video_path)

        # Method 2: PRNU correlation (if expected camera is known)
        correlation_result = None
        if expected_camera and expected_camera in self.prnu_extractor.reference_patterns:
            correlation_result = self.prnu_extractor.detect_forgery(
                video_path, expected_camera, correlation_threshold
            )

        return {
            'cnn_prediction': cnn_result,
            'prnu_correlation': correlation_result,
            'is_likely_deepfake': (
                correlation_result['is_forged'] if correlation_result
                else cnn_result['confidence'] < 0.7
            )
        }

    def plot_training_history(self, train_losses, val_accuracies):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2.plot(val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

        plt.tight_layout()
        plt.savefig('./camera_model_data/training_history.png')
        plt.show()


def main():
    """Example usage of the complete pipeline"""
    pipeline = CameraIdentificationPipeline(num_classes=9)

    # Example workflow:
    # 1. Prepare data from videos
    # prnu_patterns, labels = pipeline.prepare_data_from_videos('./video_data/', './camera_model_data/')

    # 2. Train model
    # train_losses, val_accuracies = pipeline.train(prnu_patterns, labels, epochs=30)

    # 3. Predict video source
    # result = pipeline.predict_video_source('./test_video.mp4')
    # print("Prediction:", result)

    print("Camera Identification Pipeline initialized successfully!")


if __name__ == "__main__":
    main()