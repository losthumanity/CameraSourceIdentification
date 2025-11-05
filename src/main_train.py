"""
Main Training Script for PRNU-based Camera Source Identification
Integrates all components for end-to-end training
Author: Pranav Patil | Sponsored by PiLabs
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.camera_pipeline import CameraIdentificationPipeline
from src.video_dataset_generator import PRNUDatasetGenerator


def load_or_generate_dataset(video_dir, dataset_dir, force_regenerate=False):
    """
    Load existing dataset or generate new one from videos
    """
    train_path = os.path.join(dataset_dir, 'train_dataset.npz')
    test_path = os.path.join(dataset_dir, 'test_dataset.npz')
    metadata_path = os.path.join(dataset_dir, 'dataset_metadata.json')

    # Check if dataset exists
    if not force_regenerate and os.path.exists(train_path) and os.path.exists(test_path):
        print("📂 Loading existing dataset...")

        # Load training data
        train_data = np.load(train_path)
        train_patterns = train_data['prnu_patterns']
        train_labels = train_data['labels']
        class_names = train_data['class_names'].tolist()

        # Load test data
        test_data = np.load(test_path)
        test_patterns = test_data['prnu_patterns']
        test_labels = test_data['labels']

        print(f"✅ Loaded dataset: {len(train_patterns)} train, {len(test_patterns)} test samples")

        return (train_patterns, train_labels), (test_patterns, test_labels), class_names

    else:
        print("🔄 Generating new dataset from videos...")

        # Generate dataset
        generator = PRNUDatasetGenerator()
        prnu_patterns, labels, class_names = generator.generate_from_videos(
            video_dir, dataset_dir, samples_per_camera=100, frames_per_video=20
        )

        # Load the split data
        return load_or_generate_dataset(video_dir, dataset_dir, force_regenerate=False)


def plot_class_distribution(labels, class_names, save_path):
    """Plot distribution of samples across classes"""
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(12, 6))
    bars = plt.bar([class_names[i] for i in unique], counts,
                   color=plt.cm.Set3(np.linspace(0, 1, len(unique))))

    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Camera Models', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_and_plot_results(pipeline, test_patterns, test_labels, class_names, output_dir):
    """Comprehensive evaluation with visualizations"""

    # Prepare test data
    from src.camera_pipeline import PRNUDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_dataset = PRNUDataset(test_patterns, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Get predictions
    pipeline.model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(pipeline.device)
            logits, _ = pipeline.model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate accuracy
    test_accuracy = np.mean(np.array(all_preds) == test_labels)
    print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Confusion Matrix
    cm = confusion_matrix(test_labels, all_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - PRNU Camera Classification', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Camera Model', fontsize=12)
    plt.ylabel('True Camera Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Classification Report
    report = classification_report(test_labels, all_preds,
                                 target_names=class_names,
                                 output_dict=True)

    print("\n📊 Classification Report:")
    print(classification_report(test_labels, all_preds, target_names=class_names))

    # Save detailed results
    results = {
        'test_accuracy': float(test_accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'class_names': class_names
    }

    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return test_accuracy, results


def main():
    parser = argparse.ArgumentParser(description='Train PRNU Camera Source Identification')
    parser.add_argument('--video_dir', type=str, default='./video_data',
                       help='Directory containing video datasets')
    parser.add_argument('--dataset_dir', type=str, default='./camera_model_data/prnu_dataset',
                       help='Directory for processed datasets')
    parser.add_argument('--output_dir', type=str, default='./camera_model_data',
                       help='Output directory for models and results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--force_regenerate', action='store_true',
                       help='Force regenerate dataset even if exists')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate existing model')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 PRNU-based Camera Source Identification System")
    print("=" * 60)

    # Load or generate dataset
    try:
        (train_patterns, train_labels), (test_patterns, test_labels), class_names = load_or_generate_dataset(
            args.video_dir, args.dataset_dir, args.force_regenerate
        )
    except Exception as e:
        print(f"❌ Error loading/generating dataset: {e}")
        print("\n💡 Make sure you have video data in the correct structure:")
        print("   Use: python src/video_dataset_generator.py --setup_structure")
        return

    # Initialize pipeline
    pipeline = CameraIdentificationPipeline(num_classes=len(class_names))
    pipeline.class_names = class_names

    # Plot class distribution
    print(f"\n📊 Dataset Overview:")
    print(f"   - Classes: {len(class_names)}")
    print(f"   - Train samples: {len(train_patterns)}")
    print(f"   - Test samples: {len(test_patterns)}")
    print(f"   - PRNU shape: {train_patterns[0].shape}")

    dist_path = os.path.join(args.output_dir, 'class_distribution.png')
    plot_class_distribution(np.concatenate([train_labels, test_labels]),
                           class_names, dist_path)

    # Check if we should only evaluate
    model_path = os.path.join(args.output_dir, 'best_prnu_classifier.pth')

    if args.eval_only and os.path.exists(model_path):
        print("📋 Evaluation mode - loading existing model...")
        pipeline.model.load_state_dict(torch.load(model_path, map_location=pipeline.device))

    else:
        print("\n🔥 Starting training...")

        # Train the model
        train_losses, val_accuracies = pipeline.train(
            train_patterns, train_labels,
            validation_split=0.2,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )

        # Plot training history
        pipeline.plot_training_history(train_losses, val_accuracies)

        print("✅ Training completed!")

    # Comprehensive evaluation
    print("\n🎯 Evaluating model...")
    test_accuracy, results = evaluate_and_plot_results(
        pipeline, test_patterns, test_labels, class_names, args.output_dir
    )

    # Load reference patterns for PRNU correlation testing
    ref_path = os.path.join(args.dataset_dir, 'reference_patterns.npz')
    if os.path.exists(ref_path):
        pipeline.prnu_extractor.load_reference_patterns(ref_path)
        print("✅ Reference patterns loaded for correlation analysis")

    print(f"\n🏆 Final Results:")
    print(f"   - Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   - Model saved: {model_path}")
    print(f"   - Results saved: {args.output_dir}")

    # Example usage demonstration
    print("\n💡 Usage Examples:")
    print("   # Predict video source:")
    print("   pipeline.predict_video_source('./test_video.mp4')")
    print("   ")
    print("   # Detect deepfake:")
    print("   pipeline.detect_deepfake('./suspicious_video.mp4', 'iPhone_13_Pro')")

    return pipeline, results


if __name__ == "__main__":
    pipeline, results = main()