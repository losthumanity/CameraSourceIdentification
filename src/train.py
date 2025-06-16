import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np

# Constants
NUM_CLASSES = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCHS = 20
FEATURE_SAVE_PATH = '/camera_model_data/features_rn50.npy'
LABEL_SAVE_PATH = '/camera_model_data/true_labels_rn50.npy'
PREDICTION_SAVE_PATH = '/camera_model_data/predictions_rn50.npy'

# Dataset path
dataset_path = './camera_model_data/processed_dataset'

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Datasets and Dataloaders
train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'test'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Print class mapping
print("Class to index mapping:", train_dataset.class_to_idx)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ResNet50 Model for classification + feature extraction
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50FeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])  # exclude final fc
        in_features = self.resnet.fc.in_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)  # [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # flatten to [B, 2048]
        logits = self.classifier(features)
        return logits, features

model = ResNet50FeatureExtractor(num_classes=NUM_CLASSES).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
def train_model():
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss:.4f}, Accuracy: {acc:.4f}")

        # Evaluation after each epoch
        test_acc = evaluate_model()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), './camera_model_data/best_resnet50_rgb.pth')

# Evaluation and feature saving
def evaluate_model(save_results=False):
    model.eval()
    all_preds, all_labels, all_features = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, features = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {acc:.4f}")

    if save_results:
        np.save(FEATURE_SAVE_PATH, all_features)
        np.save(LABEL_SAVE_PATH, np.array(all_labels))
        np.save(PREDICTION_SAVE_PATH, np.array(all_preds))

    return acc

# Run training and save features
train_model()
evaluate_model(save_results=True)