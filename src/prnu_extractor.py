"""
PRNU (Photo-Response Non-Uniformity) Extraction Module
Author: Pranav Patil | Sponsored by PiLabs
"""

import cv2
import numpy as np
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter
import pywt
from tqdm import tqdm
import os


class PRNUExtractor:
    """
    PRNU extraction using advanced denoising techniques
    Implements Wiener filtering and wavelet-based noise pattern extraction
    """

    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        self.reference_patterns = {}

    def extract_prnu_frame(self, frame):
        """
        Extract PRNU pattern from a single frame using wavelet denoising
        Implementation from technical specification

        Args:
            frame: Input frame (BGR format)

        Returns:
            residual: Normalized PRNU pattern
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        gray = gray.astype(np.float32)

        # Resize to target size for consistency
        gray = cv2.resize(gray, self.target_size)

        # Wavelet-based denoising (Wiener filter approximation)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        # Residual = Original - Denoised
        residual = gray - denoised

        # Normalize residual
        residual = (residual - residual.mean()) / (residual.std() + 1e-8)

        return residual

    def video_to_prnu(self, video_path, num_frames=30, skip_frames=5):
        """
        Extract aggregated PRNU pattern from video

        Args:
            video_path: Path to video file
            num_frames: Number of frames to process
            skip_frames: Skip frames for temporal diversity

        Returns:
            prnu_map: Aggregated PRNU pattern
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames_prnu = []
        frame_count = 0
        processed_frames = 0

        pbar = tqdm(total=num_frames, desc="Extracting PRNU")

        while processed_frames < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for temporal diversity
            if frame_count % (skip_frames + 1) == 0:
                try:
                    prnu = self.extract_prnu_frame(frame)
                    frames_prnu.append(prnu)
                    processed_frames += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")

            frame_count += 1

        pbar.close()
        cap.release()

        if len(frames_prnu) == 0:
            raise ValueError("No frames could be processed")

        # Aggregate PRNU from all frames (median is more robust than mean)
        prnu_map = np.median(frames_prnu, axis=0)

        # Final normalization
        prnu_map = (prnu_map - prnu_map.mean()) / (prnu_map.std() + 1e-8)

        return prnu_map

    def generate_reference_pattern(self, camera_videos_dir, camera_name=None):
        """
        Generate reference PRNU pattern for a camera model
        Implementation from technical specification - averages PRNU maps from multiple samples

        Args:
            camera_videos_dir: Directory containing videos/PRNU maps from same camera
            camera_name: Name/model of the camera

        Returns:
            reference_pattern: Normalized reference PRNU fingerprint
        """
        # Check if directory contains .npy files or videos
        npy_files = [f for f in os.listdir(camera_videos_dir) if f.endswith('.npy')]

        all_prnus = []

        if npy_files:
            # Load pre-computed PRNU maps
            print(f"Loading {len(npy_files)} PRNU maps...")
            for npy_file in npy_files:
                prnu = np.load(os.path.join(camera_videos_dir, npy_file))
                all_prnus.append(prnu)
        else:
            # Extract from videos
            video_files = [f for f in os.listdir(camera_videos_dir)
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

            if len(video_files) == 0:
                raise ValueError(f"No video or .npy files found in {camera_videos_dir}")

            print(f"Extracting PRNU from {len(video_files)} videos...")
            for video_file in tqdm(video_files[:10], desc=f"Processing {camera_name or 'camera'}"):
                video_path = os.path.join(camera_videos_dir, video_file)
                try:
                    prnu = self.video_to_prnu(video_path, num_frames=20)
                    all_prnus.append(prnu)
                except Exception as e:
                    print(f"Failed to process {video_file}: {e}")
                    continue

        if len(all_prnus) == 0:
            raise ValueError(f"No valid PRNU patterns extracted")

        # Create reference pattern (mean of all PRNU patterns)
        reference = np.mean(all_prnus, axis=0)

        # Normalize to unit norm (for correlation coefficient calculation)
        reference = reference / np.linalg.norm(reference)

        # Store reference pattern if camera name provided
        if camera_name:
            self.reference_patterns[camera_name] = reference

        return reference

    def correlation_coefficient(self, prnu1, prnu2):
        """
        Calculate correlation coefficient between two PRNU patterns
        Implementation from technical specification
        Used for forgery detection and camera identification

        Args:
            prnu1: First PRNU pattern
            prnu2: Second PRNU pattern (reference pattern)

        Returns:
            float: Correlation coefficient [-1, 1]
        """
        prnu1_flat = prnu1.flatten()
        prnu2_flat = prnu2.flatten()

        # Pearson correlation coefficient using numpy
        correlation = np.corrcoef(prnu1_flat, prnu2_flat)[0, 1]

        return correlation if not np.isnan(correlation) else 0.0

    def detect_forgery(self, test_video_path, camera_name, threshold=0.4):
        """
        Detect if a video is potentially forged/deepfake

        Args:
            test_video_path: Path to test video
            camera_name: Expected camera model
            threshold: Correlation threshold for forgery detection

        Returns:
            dict: Detection results
        """
        if camera_name not in self.reference_patterns:
            raise ValueError(f"No reference pattern for camera: {camera_name}")

        # Extract PRNU from test video
        test_prnu = self.video_to_prnu(test_video_path)

        # Calculate correlation with reference pattern
        reference = self.reference_patterns[camera_name]
        correlation = self.correlation_coefficient(test_prnu, reference)

        # Determine if likely forged
        is_forged = correlation < threshold

        return {
            'correlation': correlation,
            'threshold': threshold,
            'is_forged': is_forged,
            'confidence': abs(correlation - threshold),
            'camera_name': camera_name
        }

    def save_reference_patterns(self, save_path):
        """Save all reference patterns to disk"""
        np.savez(save_path, **self.reference_patterns)
        print(f"Reference patterns saved to {save_path}")

    def load_reference_patterns(self, load_path):
        """Load reference patterns from disk"""
        data = np.load(load_path)
        self.reference_patterns = {key: data[key] for key in data.files}
        print(f"Loaded {len(self.reference_patterns)} reference patterns")


def main():
    """Example usage of PRNU extractor"""
    extractor = PRNUExtractor()

    # Example: Create reference pattern
    # reference = extractor.generate_reference_pattern('./data/samsung_s21/', 'Samsung_S21')

    # Example: Detect forgery
    # result = extractor.detect_forgery('./test_video.mp4', 'Samsung_S21')
    # print("Detection result:", result)

    print("PRNU Extractor initialized successfully!")


if __name__ == "__main__":
    main()