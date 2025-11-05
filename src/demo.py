"""
Demo and Testing Script for Camera Source Identification
Provides easy-to-use interface for testing the trained model
Author: Pranav Patil | Sponsored by PiLabs
"""

import os
import sys
import json
import numpy as np
import torch
import cv2
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.camera_pipeline import CameraIdentificationPipeline
from src.prnu_extractor import PRNUExtractor


class CameraSourceDemo:
    """Demo interface for camera source identification"""

    def __init__(self, model_dir='./camera_model_data'):
        self.model_dir = model_dir
        self.pipeline = None
        self.load_model()

    def load_model(self):
        """Load trained model and reference patterns"""

        # Load model metadata
        metadata_path = os.path.join(self.model_dir, 'prnu_dataset', 'dataset_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Initialize pipeline
        self.pipeline = CameraIdentificationPipeline(
            num_classes=metadata['num_classes']
        )
        self.pipeline.class_names = metadata['class_names']

        # Load trained model weights
        model_path = os.path.join(self.model_dir, 'best_prnu_classifier.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")

        self.pipeline.model.load_state_dict(
            torch.load(model_path, map_location=self.pipeline.device)
        )
        self.pipeline.model.eval()

        # Load reference patterns
        ref_path = os.path.join(self.model_dir, 'prnu_dataset', 'reference_patterns.npz')
        if os.path.exists(ref_path):
            self.pipeline.prnu_extractor.load_reference_patterns(ref_path)

        print(f"✅ Model loaded successfully!")
        print(f"   - Device: {self.pipeline.device}")
        print(f"   - Classes: {len(self.pipeline.class_names)}")
        print(f"   - Camera models: {', '.join(self.pipeline.class_names)}")

    def predict_video_source(self, video_path, show_details=True):
        """
        Predict the source camera of a video

        Args:
            video_path: Path to video file
            show_details: Whether to show detailed analysis

        Returns:
            dict: Prediction results
        """

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"\n🔍 Analyzing video: {video_path}")
        print("-" * 50)

        try:
            # Get prediction
            result = self.pipeline.predict_video_source(video_path)

            if show_details:
                print(f"🎯 Predicted Camera: {result['predicted_camera']}")
                print(f"🎲 Confidence: {result['confidence']:.3f}")
                print(f"\n📊 All Probabilities:")

                # Sort probabilities for better display
                sorted_probs = sorted(result['all_probabilities'].items(),
                                    key=lambda x: x[1], reverse=True)

                for camera, prob in sorted_probs:
                    bar = "█" * int(prob * 20)  # Visual bar
                    print(f"   {camera:<20}: {prob:.3f} {bar}")

            return result

        except Exception as e:
            print(f"❌ Error analyzing video: {e}")
            return None

    def detect_deepfake(self, video_path, expected_camera=None, show_analysis=True):
        """
        Detect if a video is potentially a deepfake

        Args:
            video_path: Path to video file
            expected_camera: Expected camera model (optional)
            show_analysis: Whether to show detailed analysis

        Returns:
            dict: Detection results
        """

        print(f"\n🕵️ Deepfake Detection: {video_path}")
        print("-" * 50)

        try:
            result = self.pipeline.detect_deepfake(video_path, expected_camera)

            if show_analysis:
                # CNN Analysis
                cnn_result = result['cnn_prediction']
                print(f"🤖 CNN Analysis:")
                print(f"   Predicted: {cnn_result['predicted_camera']}")
                print(f"   Confidence: {cnn_result['confidence']:.3f}")

                # PRNU Correlation Analysis
                if result['prnu_correlation']:
                    prnu_result = result['prnu_correlation']
                    print(f"\n🔬 PRNU Correlation Analysis:")
                    print(f"   Expected Camera: {prnu_result['camera_name']}")
                    print(f"   Correlation: {prnu_result['correlation']:.3f}")
                    print(f"   Threshold: {prnu_result['threshold']}")
                    print(f"   Forgery Detected: {prnu_result['is_forged']}")

                # Final Verdict
                print(f"\n⚖️ Final Verdict:")
                verdict = "🚨 LIKELY DEEPFAKE" if result['is_likely_deepfake'] else "✅ LIKELY AUTHENTIC"
                print(f"   {verdict}")

            return result

        except Exception as e:
            print(f"❌ Error in deepfake detection: {e}")
            return None

    def batch_analyze_videos(self, video_dir, output_file=None):
        """
        Analyze all videos in a directory

        Args:
            video_dir: Directory containing videos
            output_file: Optional file to save results

        Returns:
            list: Analysis results for all videos
        """

        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        video_files = [f for f in Path(video_dir).rglob('*')
                      if f.suffix.lower() in video_extensions]

        if len(video_files) == 0:
            print(f"No video files found in {video_dir}")
            return []

        print(f"\n📁 Batch Analysis: {len(video_files)} videos")
        print("=" * 60)

        results = []

        for video_file in video_files:
            print(f"\n🎬 Processing: {video_file.name}")

            try:
                result = self.predict_video_source(str(video_file), show_details=False)
                if result:
                    result['video_path'] = str(video_file)
                    result['video_name'] = video_file.name
                    results.append(result)

                    print(f"   ✅ {result['predicted_camera']} (conf: {result['confidence']:.3f})")
                else:
                    print(f"   ❌ Failed to analyze")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue

        # Save results if requested
        if output_file and results:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")

        # Summary
        if results:
            print(f"\n📈 Summary:")
            camera_counts = {}
            for result in results:
                camera = result['predicted_camera']
                camera_counts[camera] = camera_counts.get(camera, 0) + 1

            for camera, count in sorted(camera_counts.items()):
                print(f"   {camera}: {count} videos")

        return results

    def compare_videos(self, video1_path, video2_path):
        """
        Compare two videos to see if they're from the same camera

        Args:
            video1_path: Path to first video
            video2_path: Path to second video

        Returns:
            dict: Comparison results
        """

        print(f"\n🔄 Comparing Videos:")
        print(f"   Video 1: {video1_path}")
        print(f"   Video 2: {video2_path}")
        print("-" * 50)

        # Analyze both videos
        result1 = self.predict_video_source(video1_path, show_details=False)
        result2 = self.predict_video_source(video2_path, show_details=False)

        if not result1 or not result2:
            print("❌ Failed to analyze one or both videos")
            return None

        # Extract PRNU patterns for correlation
        prnu1 = self.pipeline.prnu_extractor.video_to_prnu(video1_path)
        prnu2 = self.pipeline.prnu_extractor.video_to_prnu(video2_path)

        # Calculate PRNU correlation
        correlation = self.pipeline.prnu_extractor.correlation_coefficient(prnu1, prnu2)

        # Determine if same camera
        same_camera_cnn = result1['predicted_camera'] == result2['predicted_camera']
        same_camera_prnu = correlation > 0.6  # Threshold for same camera

        print(f"\n📊 Analysis Results:")
        print(f"   Video 1 Camera: {result1['predicted_camera']} (conf: {result1['confidence']:.3f})")
        print(f"   Video 2 Camera: {result2['predicted_camera']} (conf: {result2['confidence']:.3f})")
        print(f"   PRNU Correlation: {correlation:.3f}")
        print(f"   Same Camera (CNN): {same_camera_cnn}")
        print(f"   Same Camera (PRNU): {same_camera_prnu}")

        conclusion = same_camera_cnn and same_camera_prnu
        print(f"\n🎯 Conclusion: {'SAME CAMERA' if conclusion else 'DIFFERENT CAMERAS'}")

        return {
            'video1_result': result1,
            'video2_result': result2,
            'prnu_correlation': correlation,
            'same_camera_cnn': same_camera_cnn,
            'same_camera_prnu': same_camera_prnu,
            'conclusion': conclusion
        }


def interactive_demo():
    """Interactive demo for testing the system"""

    print("🎬 Camera Source Identification Demo")
    print("=" * 50)

    try:
        demo = CameraSourceDemo()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Make sure you've trained the model first:")
        print("   python src/main_train.py")
        return

    while True:
        print(f"\n📋 Available Commands:")
        print("1. Analyze single video")
        print("2. Deepfake detection")
        print("3. Compare two videos")
        print("4. Batch analyze directory")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == '1':
            video_path = input("Enter video path: ").strip()
            demo.predict_video_source(video_path)

        elif choice == '2':
            video_path = input("Enter video path: ").strip()
            expected_camera = input("Expected camera (optional, press Enter to skip): ").strip()
            expected_camera = expected_camera if expected_camera else None
            demo.detect_deepfake(video_path, expected_camera)

        elif choice == '3':
            video1 = input("Enter first video path: ").strip()
            video2 = input("Enter second video path: ").strip()
            demo.compare_videos(video1, video2)

        elif choice == '4':
            video_dir = input("Enter directory path: ").strip()
            output_file = input("Output file (optional, press Enter to skip): ").strip()
            output_file = output_file if output_file else None
            demo.batch_analyze_videos(video_dir, output_file)

        elif choice == '5':
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line usage
        video_path = sys.argv[1]
        demo = CameraSourceDemo()
        demo.predict_video_source(video_path)
    else:
        # Interactive mode
        interactive_demo()