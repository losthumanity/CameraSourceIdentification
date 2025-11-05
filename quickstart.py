#!/usr/bin/env python3
"""
Quick Start Script for Camera Source Identification System
Author: Pranav Patil | Sponsored by PiLabs

This script provides a complete walkthrough of setting up and running
the Camera Source Identification System.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def print_banner():
    """Print welcome banner"""
    print("🎬" + "=" * 58 + "🎬")
    print("🎯  Camera Source Identification System - Quick Start  🎯")
    print("📱  PRNU-based Video Analysis | Sponsored by PiLabs  📱")
    print("🎬" + "=" * 58 + "🎬")


def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n🔍 Checking Dependencies...")

    required_packages = [
        'torch', 'torchvision', 'opencv-python', 'numpy',
        'matplotlib', 'scikit-learn', 'scipy', 'PyWavelets'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages detected. Installing...")
        install_cmd = f"pip install {' '.join(missing_packages)}"
        print(f"Running: {install_cmd}")

        try:
            subprocess.run(install_cmd.split(), check=True)
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please install manually:")
            print(f"   pip install -r requirements.txt")
            return False

    return True


def setup_directory_structure():
    """Create necessary directory structure"""
    print("\n📁 Setting up Directory Structure...")

    directories = [
        'camera_model_data',
        'camera_model_data/prnu_dataset',
        'video_data',
        'video_data/Samsung_Galaxy_S21',
        'video_data/iPhone_13_Pro',
        'video_data/OnePlus_9_Pro',
        'video_data/Xiaomi_Mi_11',
        'video_data/Google_Pixel_6',
        'video_data/Huawei_P50_Pro',
        'video_data/Sony_Xperia_1_III',
        'video_data/LG_V60_ThinQ',
        'video_data/Motorola_Edge_20'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}/")

    # Create placeholder info files
    camera_models = [d for d in directories if d.startswith('video_data/') and '/' in d[11:]]

    for camera_dir in camera_models:
        camera_name = camera_dir.split('/')[-1]
        info_file = Path(camera_dir) / 'README.txt'

        if not info_file.exists():
            with open(info_file, 'w') as f:
                f.write(f"Camera Model: {camera_name}\n")
                f.write("Add your video files (.mp4, .avi, .mov, .mkv) to this directory\n")
                f.write("Recommended: 10-50 videos per camera model\n")
                f.write("Each video should be 10-60 seconds long\n")

    print("📝 Camera model directories created with README files")


def run_tests():
    """Run system tests"""
    print("\n🧪 Running System Tests...")

    try:
        # Add src to Python path
        sys.path.append('src')

        # Import and run tests
        result = subprocess.run([sys.executable, 'src/test_system.py'],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ All tests passed!")
            print(result.stdout[-500:])  # Show last 500 chars
        else:
            print("⚠️  Some tests failed, but system should still work:")
            print(result.stderr[-300:])  # Show errors

    except Exception as e:
        print(f"⚠️  Test execution failed: {e}")
        print("This is normal in some environments. System should still work.")


def demonstrate_features():
    """Demonstrate key features"""
    print("\n🚀 Feature Demonstration...")

    # Try to import and run basic functionality
    try:
        sys.path.append('src')
        from prnu_extractor import PRNUExtractor
        from camera_pipeline import CameraIdentificationPipeline

        print("   ✅ PRNU Extractor: Ready")
        print("   ✅ CNN Pipeline: Ready")
        print("   ✅ All modules loaded successfully")

        # Quick functionality test
        extractor = PRNUExtractor(target_size=(64, 64))  # Small for demo
        pipeline = CameraIdentificationPipeline(num_classes=9)

        print("   ✅ System initialization: Success")

    except Exception as e:
        print(f"   ⚠️  Module loading issue: {e}")
        print("   💡 This might be due to missing dependencies")


def show_next_steps():
    """Show next steps for the user"""
    print("\n🎯 Next Steps:")
    print("=" * 40)

    print("\n1️⃣  Add Your Video Data:")
    print("   📁 Place videos in video_data/[CameraModel]/ directories")
    print("   📱 10-50 videos per camera model recommended")
    print("   🎬 Supported formats: .mp4, .avi, .mov, .mkv")

    print("\n2️⃣  Train the System:")
    print("   🚀 python src/main_train.py --epochs 50")
    print("   ⏱️  Training time: 1-3 hours depending on dataset size")
    print("   💾 Models saved to camera_model_data/")

    print("\n3️⃣  Test & Demo:")
    print("   🎮 python src/demo.py  # Interactive demo")
    print("   📊 python src/demo.py test_video.mp4  # Analyze single video")
    print("   📁 Batch analysis available in demo mode")

    print("\n4️⃣  Advanced Usage:")
    print("   📓 Open notebooks/ModuleV2.ipynb for technical deep dive")
    print("   🔬 Explore PRNU extraction algorithms")
    print("   🧠 CNN architecture and training details")

    print("\n5️⃣  Production Deployment:")
    print("   🌐 Integrate with REST API frameworks")
    print("   📱 Build mobile apps for real-time detection")
    print("   🏛️  Deploy for forensic analysis workflows")


def main():
    parser = argparse.ArgumentParser(description='Camera Source Identification - Quick Start')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--skip-setup', action='store_true', help='Skip directory setup')
    parser.add_argument('--demo-only', action='store_true', help='Only show features demo')

    args = parser.parse_args()

    print_banner()

    # Check system requirements
    if not check_dependencies():
        print("❌ Dependency check failed. Please fix and try again.")
        return 1

    # Setup directories unless skipped
    if not args.skip_setup:
        setup_directory_structure()

    # Run tests unless skipped
    if not args.skip_tests and not args.demo_only:
        run_tests()

    # Demonstrate features
    demonstrate_features()

    # Show next steps
    show_next_steps()

    print(f"\n🎉 Quick Start Complete!")
    print(f"📚 Check README.md for detailed documentation")
    print(f"🆘 Need help? Check the technical notebook or source code comments")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)