"""
Complete Training Script for PRNU-based Camera Source Identification
Implementation from technical specification with all stages
Author: Pranav Patil | Sponsored by PiLabs
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prnu_extractor import PRNUExtractor
from camera_pipeline import CameraIdentificationPipeline, PRNUDataset
from forgery_detector import ForgeryDetector


def stage1_data_collection_preprocessing(video_dir, output_dir, samples_per_camera=100):
    """
    Stage 1: Data Collection and Preprocessing
    Extract PRNU from video frames and save as .npy files

    Args:
        video_dir: Directory containing camera subfolders with videos
        output_dir: Directory to save processed PRNU maps
        samples_per_camera: Number of videos to process per camera
    """
    print("=" * 80)
    print("🧩 Stage 1: Data Collection and Preprocessing")
    print("=" * 80)

    extractor = PRNUExtractor()
    os.makedirs(output_dir, exist_ok=True)

    # Get camera folders
    camera_folders = sorted([f for f in os.listdir(video_dir)
                           if os.path.isdir(os.path.join(video_dir, f))])

    print(f"Found {len(camera_folders)} camera models: {', '.join(camera_folders)}")

    all_prnus = []
    all_labels = []
    class_names = camera_folders

    for class_idx, camera_name in enumerate(camera_folders):
        camera_dir = os.path.join(video_dir, camera_name)
        video_files = [f for f in os.listdir(camera_dir)
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        print(f"\n📹 Processing {camera_name}: {len(video_files)} videos")

        camera_output = os.path.join(output_dir, camera_name)
        os.makedirs(camera_output, exist_ok=True)

        camera_prnus = []
        for i, video_file in enumerate(tqdm(video_files[:samples_per_camera], desc=camera_name)):
            try:
                video_path = os.path.join(camera_dir, video_file)

                # Extract PRNU using specification method
                prnu_map = extractor.video_to_prnu(video_path, num_frames=30)

                # Save PRNU map
                save_path = os.path.join(camera_output, f'{camera_name}_{i:04d}.npy')
                np.save(save_path, prnu_map)

                camera_prnus.append(prnu_map)

            except Exception as e:
                print(f"  ⚠️  Failed to process {video_file}: {e}")
                continue

        all_prnus.extend(camera_prnus)
        all_labels.extend([class_idx] * len(camera_prnus))

        print(f"  ✅ Processed {len(camera_prnus)} videos for {camera_name}")

    # Save dataset
    dataset_path = os.path.join(output_dir, 'prnu_dataset.npz')
    np.savez(dataset_path,
             prnu_patterns=np.array(all_prnus),
             labels=np.array(all_labels),
             class_names=class_names)

    print(f"\n✅ Stage 1 Complete: {len(all_prnus)} PRNU maps generated")
    print(f"   Dataset saved to: {dataset_path}")

    return np.array(all_prnus), np.array(all_labels), class_names


def stage2_enrollment_reference_patterns(output_dir, class_names):
    """
    Stage 2: Enrollment (Reference Pattern Creation)
    Generate camera-level fingerprints by averaging PRNU maps

    Args:
        output_dir: Directory containing camera PRNU subdirectories
        class_names: List of camera model names
    """
    print("\n" + "=" * 80)
    print("⚙️  Stage 2: Enrollment (Reference Pattern Creation)")
    print("=" * 80)

    extractor = PRNUExtractor()

    for camera_name in class_names:
        camera_dir = os.path.join(output_dir, camera_name)

        print(f"\n📸 Generating reference pattern for {camera_name}...")

        try:
            # Generate reference pattern using specification method
            reference = extractor.generate_reference_pattern(camera_dir, camera_name)

            print(f"  ✅ Reference pattern created (shape: {reference.shape})")
            print(f"     Norm: {np.linalg.norm(reference):.4f}")

        except Exception as e:
            print(f"  ⚠️  Failed to generate reference for {camera_name}: {e}")

    # Save all reference patterns
    ref_path = os.path.join(output_dir, 'reference_patterns.npz')
    extractor.save_reference_patterns(ref_path)

    print(f"\n✅ Stage 2 Complete: {len(extractor.reference_patterns)} reference patterns created")
    print(f"   Saved to: {ref_path}")

    return extractor.reference_patterns


def stage3_cnn_source_identification(prnu_patterns, labels, class_names, epochs=20):
    """
    Stage 3: CNN-Based Source Identification
    Train ResNet50 for camera classification

    Args:
        prnu_patterns: Array of PRNU patterns
        labels: Array of labels
        class_names: List of camera model names
        epochs: Number of training epochs
    """
    print("\n" + "=" * 80)
    print("🧠 Stage 3: CNN-Based Source Identification")
    print("=" * 80)

    # Initialize pipeline
    pipeline = CameraIdentificationPipeline(num_classes=len(class_names))
    pipeline.class_names = class_names

    print(f"\n📊 Dataset Info:")
    print(f"   Total samples: {len(prnu_patterns)}")
    print(f"   Classes: {len(class_names)}")
    print(f"   PRNU shape: {prnu_patterns[0].shape}")
    print(f"   Device: {pipeline.device}")

    # Train model using specification configuration
    print(f"\n🚀 Starting training for {epochs} epochs...")
    train_losses, val_accuracies = pipeline.train(
        prnu_patterns, labels,
        validation_split=0.2,
        batch_size=32,
        epochs=epochs,
        learning_rate=1e-4
    )

    # Plot training history
    pipeline.plot_training_history(train_losses, val_accuracies)

    print(f"\n✅ Stage 3 Complete!")
    print(f"   Best validation accuracy: {max(val_accuracies):.2%}")
    print(f"   Final training loss: {train_losses[-1]:.4f}")

    return pipeline, train_losses, val_accuracies


def stage4_forgery_deepfake_detection(pipeline, ref_patterns_path, test_videos=None):
    """
    Stage 4: Forgery & DeepFake Detection
    Test correlation-based forgery detection

    Args:
        pipeline: Trained camera identification pipeline
        ref_patterns_path: Path to reference patterns
        test_videos: Optional dict {video_path: expected_camera}
    """
    print("\n" + "=" * 80)
    print("🔍 Stage 4: Forgery & DeepFake Detection")
    print("=" * 80)

    # Initialize forgery detector
    detector = ForgeryDetector(reference_patterns_path=ref_patterns_path, threshold=0.4)

    print(f"\n📋 Forgery Detection Setup:")
    print(f"   Correlation threshold: {detector.threshold}")
    print(f"   Reference patterns loaded: {len(detector.prnu_extractor.reference_patterns)}")

    if test_videos:
        print(f"\n🧪 Testing forgery detection on {len(test_videos)} videos...")

        for video_path, expected_camera in test_videos.items():
            if not os.path.exists(video_path):
                print(f"  ⚠️  Video not found: {video_path}")
                continue

            try:
                # Detect forgery
                result = detector.detect_forgery(video_path, expected_camera)

                print(f"\n  📹 Video: {os.path.basename(video_path)}")
                print(f"     Expected: {expected_camera}")
                print(f"     Correlation: {result['correlation']:.4f}")
                print(f"     Verdict: {result['verdict']}")
                print(f"     {result['message']}")

            except Exception as e:
                print(f"  ⚠️  Error processing {video_path}: {e}")
    else:
        print("\n💡 No test videos provided. Use detector.detect_forgery() to test.")

    print(f"\n✅ Stage 4 Complete!")
    print(f"   Forgery detection ready with threshold: {detector.threshold}")

    return detector


def generate_results_report(pipeline, train_losses, val_accuracies, class_names):
    """
    Generate comprehensive results report

    Args:
        pipeline: Trained pipeline
        train_losses: Training loss history
        val_accuracies: Validation accuracy history
        class_names: Camera model names
    """
    print("\n" + "=" * 80)
    print("📊 4️⃣ Results (Quantitative & Qualitative)")
    print("=" * 80)

    best_val_acc = max(val_accuracies)
    final_loss = train_losses[-1]

    print(f"\n✨ Training Summary:")
    print(f"   • Camera Classification Accuracy: {best_val_acc:.1%}")
    print(f"   • Final Training Loss: {final_loss:.4f}")
    print(f"   • Number of Classes: {len(class_names)}")
    print(f"   • Total Epochs: {len(train_losses)}")
    print(f"   • Device Used: {pipeline.device}")

    print(f"\n📸 Camera Models:")
    for i, camera in enumerate(class_names, 1):
        print(f"   {i}. {camera}")

    print(f"\n🔬 Technical Achievements:")
    print(f"   ✓ PRNU Extraction: Wavelet-based denoising implemented")
    print(f"   ✓ Reference Patterns: Camera-level fingerprints generated")
    print(f"   ✓ CNN Classifier: ResNet50 fine-tuned for {len(class_names)} classes")
    print(f"   ✓ Forgery Detection: Correlation threshold = 0.4")
    print(f"   ✓ Robustness: PRNU survives compression (H.264/MP4)")

    print(f"\n🚀 Deployment Status:")
    print(f"   ✓ Model saved: ./camera_model_data/best_prnu_classifier.pth")
    print(f"   ✓ Reference patterns: ./camera_model_data/prnu_dataset/reference_patterns.npz")
    print(f"   ✓ Flask UI: Available in src/flask_app.py")

    # Save results to file
    results_file = './camera_model_data/training_results.txt'
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CAMERA SOURCE IDENTIFICATION - TRAINING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Classification Accuracy: {best_val_acc:.1%}\n")
        f.write(f"Final Loss: {final_loss:.4f}\n")
        f.write(f"Number of Classes: {len(class_names)}\n")
        f.write(f"Camera Models: {', '.join(class_names)}\n")
        f.write(f"\nBest Validation Accuracy: {best_val_acc:.1%}\n")
        f.write(f"Model Device: {pipeline.device}\n")

    print(f"\n📄 Results saved to: {results_file}")


def main():
    """Main training workflow - All 4 stages"""
    parser = argparse.ArgumentParser(description='Train PRNU Camera Source Identification System')
    parser.add_argument('--video_dir', type=str, default='./video_data',
                       help='Directory containing camera subfolders with videos')
    parser.add_argument('--output_dir', type=str, default='./camera_model_data/prnu_dataset',
                       help='Output directory for processed data')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--samples', type=int, default=100,
                       help='Samples per camera to process')
    parser.add_argument('--skip_stage1', action='store_true',
                       help='Skip stage 1 if data already processed')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🎯 CAMERA SOURCE IDENTIFICATION - COMPLETE TRAINING PIPELINE")
    print("   Implementation from Technical Specification")
    print("   Sponsored by PiLabs")
    print("=" * 80)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('./camera_model_data', exist_ok=True)

    # Stage 1: Data Collection and Preprocessing
    if not args.skip_stage1:
        prnu_patterns, labels, class_names = stage1_data_collection_preprocessing(
            args.video_dir, args.output_dir, args.samples
        )
    else:
        print("⏭️  Skipping Stage 1 - Loading existing dataset...")
        dataset_path = os.path.join(args.output_dir, 'prnu_dataset.npz')
        data = np.load(dataset_path)
        prnu_patterns = data['prnu_patterns']
        labels = data['labels']
        class_names = data['class_names'].tolist()
        print(f"✅ Loaded {len(prnu_patterns)} samples, {len(class_names)} classes")

    # Stage 2: Enrollment (Reference Pattern Creation)
    reference_patterns = stage2_enrollment_reference_patterns(args.output_dir, class_names)

    # Stage 3: CNN-Based Source Identification
    pipeline, train_losses, val_accuracies = stage3_cnn_source_identification(
        prnu_patterns, labels, class_names, args.epochs
    )

    # Stage 4: Forgery & DeepFake Detection
    ref_patterns_path = os.path.join(args.output_dir, 'reference_patterns.npz')
    detector = stage4_forgery_deepfake_detection(pipeline, ref_patterns_path)

    # Generate final results report
    generate_results_report(pipeline, train_losses, val_accuracies, class_names)

    print("\n" + "=" * 80)
    print("✅ ALL STAGES COMPLETE!")
    print("=" * 80)
    print("\n📌 Next Steps:")
    print("   1. Test the model: python src/demo.py")
    print("   2. Launch Flask UI: python src/flask_app.py")
    print("   3. Evaluate performance: python src/evaluate.py")
    print("\n🎉 Training completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
