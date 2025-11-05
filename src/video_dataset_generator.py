"""
Video Data Preparation and PRNU Dataset Generation
Converts video datasets to PRNU patterns for training
Author: Pranav Patil | Sponsored by PiLabs
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import json
import argparse
from sklearn.model_selection import train_test_split
from src.prnu_extractor import PRNUExtractor


def create_video_dataset_structure(base_dir):
    """
    Create the expected directory structure for video datasets

    Expected structure after setup:
    video_data/
    ├── Samsung_Galaxy_S21/
    │   ├── video001.mp4
    │   ├── video002.mp4
    │   └── ...
    ├── iPhone_13_Pro/
    │   ├── video001.mp4
    │   └── ...
    └── ...
    """

    camera_models = [
        'Samsung_Galaxy_S21',
        'iPhone_13_Pro',
        'OnePlus_9_Pro',
        'Xiaomi_Mi_11',
        'Google_Pixel_6',
        'Huawei_P50_Pro',
        'Sony_Xperia_1_III',
        'LG_V60_ThinQ',
        'Motorola_Edge_20'
    ]

    for camera in camera_models:
        camera_dir = os.path.join(base_dir, camera)
        os.makedirs(camera_dir, exist_ok=True)

        # Create a placeholder info file
        info_file = os.path.join(camera_dir, 'camera_info.json')
        info = {
            'camera_model': camera,
            'sensor_info': 'Unknown',
            'video_count': 0,
            'notes': 'Add video files to this directory'
        }

        if not os.path.exists(info_file):
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2)

    print(f"Created video dataset structure in {base_dir}")
    print("Please add video files to each camera model directory")


def convert_images_to_videos(image_dataset_path, video_output_path, fps=30):
    """
    Convert image dataset to video dataset for PRNU processing
    Useful if you have image-based camera datasets
    """

    os.makedirs(video_output_path, exist_ok=True)

    for camera_folder in os.listdir(image_dataset_path):
        camera_path = os.path.join(image_dataset_path, camera_folder)
        if not os.path.isdir(camera_path):
            continue

        output_camera_path = os.path.join(video_output_path, camera_folder)
        os.makedirs(output_camera_path, exist_ok=True)

        image_files = [f for f in os.listdir(camera_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Group images into videos (e.g., 30 images per video)
        images_per_video = 30
        video_count = 0

        for i in range(0, len(image_files), images_per_video):
            video_images = image_files[i:i + images_per_video]

            if len(video_images) < 10:  # Skip if too few images
                continue

            video_filename = f"{camera_folder}_video_{video_count:03d}.mp4"
            video_path = os.path.join(output_camera_path, video_filename)

            # Read first image to get dimensions
            first_img_path = os.path.join(camera_path, video_images[0])
            first_img = cv2.imread(first_img_path)
            if first_img is None:
                continue

            height, width = first_img.shape[:2]

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            # Add images to video
            for img_file in video_images:
                img_path = os.path.join(camera_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    out.write(img)

            out.release()
            video_count += 1

        print(f"Created {video_count} videos for {camera_folder}")


class PRNUDatasetGenerator:
    """Generate PRNU patterns from video datasets"""

    def __init__(self, target_size=(256, 256)):
        self.prnu_extractor = PRNUExtractor(target_size=target_size)
        self.target_size = target_size

    def generate_from_videos(self, video_data_dir, output_dir,
                           samples_per_camera=100, frames_per_video=20):
        """
        Generate PRNU dataset from video directory structure

        Args:
            video_data_dir: Directory containing camera subdirectories with videos
            output_dir: Output directory for processed dataset
            samples_per_camera: Maximum samples to generate per camera
            frames_per_video: Number of frames to extract per video
        """

        os.makedirs(output_dir, exist_ok=True)

        # Get camera directories
        camera_dirs = [d for d in os.listdir(video_data_dir)
                      if os.path.isdir(os.path.join(video_data_dir, d))]

        if len(camera_dirs) == 0:
            raise ValueError(f"No camera directories found in {video_data_dir}")

        all_prnu_patterns = []
        all_labels = []
        class_names = sorted(camera_dirs)

        print(f"Found {len(class_names)} camera models: {class_names}")

        for class_idx, camera_name in enumerate(class_names):
            camera_path = os.path.join(video_data_dir, camera_name)
            video_files = [f for f in os.listdir(camera_path)
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

            print(f"\nProcessing {camera_name}: {len(video_files)} videos")

            camera_patterns = []
            processed_count = 0

            for video_file in tqdm(video_files, desc=f"Processing {camera_name}"):
                if processed_count >= samples_per_camera:
                    break

                video_path = os.path.join(camera_path, video_file)

                try:
                    # Extract PRNU pattern from video
                    prnu_pattern = self.prnu_extractor.video_to_prnu(
                        video_path, num_frames=frames_per_video
                    )

                    camera_patterns.append(prnu_pattern)
                    processed_count += 1

                except Exception as e:
                    print(f"Error processing {video_file}: {e}")
                    continue

            if len(camera_patterns) > 0:
                all_prnu_patterns.extend(camera_patterns)
                all_labels.extend([class_idx] * len(camera_patterns))

                # Generate reference pattern for this camera
                try:
                    reference = self.prnu_extractor.generate_reference_pattern(
                        camera_path, camera_name
                    )
                    print(f"Generated reference pattern for {camera_name}")
                except Exception as e:
                    print(f"Failed to generate reference for {camera_name}: {e}")

            print(f"Generated {len(camera_patterns)} PRNU patterns for {camera_name}")

        # Convert to numpy arrays
        prnu_array = np.array(all_prnu_patterns)
        labels_array = np.array(all_labels)

        print(f"\nTotal dataset: {len(prnu_array)} samples, {len(class_names)} classes")
        print(f"PRNU pattern shape: {prnu_array[0].shape}")

        # Split into train/test
        train_patterns, test_patterns, train_labels, test_labels = train_test_split(
            prnu_array, labels_array, test_size=0.2, stratify=labels_array, random_state=42
        )

        # Save datasets
        train_path = os.path.join(output_dir, 'train_dataset.npz')
        test_path = os.path.join(output_dir, 'test_dataset.npz')

        np.savez(train_path,
                prnu_patterns=train_patterns,
                labels=train_labels,
                class_names=class_names)

        np.savez(test_path,
                prnu_patterns=test_patterns,
                labels=test_labels,
                class_names=class_names)

        # Save reference patterns
        ref_path = os.path.join(output_dir, 'reference_patterns.npz')
        self.prnu_extractor.save_reference_patterns(ref_path)

        # Save metadata
        metadata = {
            'num_classes': len(class_names),
            'class_names': class_names,
            'train_samples': len(train_patterns),
            'test_samples': len(test_patterns),
            'prnu_shape': list(prnu_array[0].shape),
            'frames_per_video': frames_per_video,
            'samples_per_camera': samples_per_camera
        }

        metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nDataset saved:")
        print(f"- Training: {train_path}")
        print(f"- Testing: {test_path}")
        print(f"- References: {ref_path}")
        print(f"- Metadata: {metadata_path}")

        return prnu_array, labels_array, class_names


def main():
    parser = argparse.ArgumentParser(description='PRNU Dataset Generation')
    parser.add_argument('--video_dir', type=str, default='./video_data',
                       help='Directory containing video datasets')
    parser.add_argument('--output_dir', type=str, default='./camera_model_data/prnu_dataset',
                       help='Output directory for processed dataset')
    parser.add_argument('--samples_per_camera', type=int, default=100,
                       help='Maximum samples per camera model')
    parser.add_argument('--frames_per_video', type=int, default=20,
                       help='Number of frames to extract per video')
    parser.add_argument('--setup_structure', action='store_true',
                       help='Create video dataset directory structure')
    parser.add_argument('--convert_images', type=str, default=None,
                       help='Convert image dataset to videos (provide image dataset path)')

    args = parser.parse_args()

    # Setup directory structure if requested
    if args.setup_structure:
        create_video_dataset_structure(args.video_dir)
        return

    # Convert images to videos if requested
    if args.convert_images:
        convert_images_to_videos(args.convert_images, args.video_dir)
        return

    # Generate PRNU dataset
    generator = PRNUDatasetGenerator()

    try:
        prnu_patterns, labels, class_names = generator.generate_from_videos(
            args.video_dir,
            args.output_dir,
            args.samples_per_camera,
            args.frames_per_video
        )

        print("\n✅ PRNU dataset generation completed successfully!")
        print(f"📊 Dataset statistics:")
        print(f"   - Total samples: {len(prnu_patterns)}")
        print(f"   - Number of classes: {len(class_names)}")
        print(f"   - PRNU pattern shape: {prnu_patterns[0].shape}")

    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        print("\n💡 Make sure your video directory has the correct structure:")
        print("   video_data/")
        print("   ├── Camera_Model_1/")
        print("   │   ├── video1.mp4")
        print("   │   └── video2.mp4")
        print("   └── Camera_Model_2/")
        print("       ├── video1.mp4")
        print("       └── video2.mp4")


if __name__ == "__main__":
    main()