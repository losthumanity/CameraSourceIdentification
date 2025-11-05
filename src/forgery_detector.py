"""
Forgery and Deepfake Detection Module
Uses PRNU correlation for detecting forged or deepfake videos
Implementation from technical specification
Author: Pranav Patil | Sponsored by PiLabs
"""

import numpy as np
import cv2
import os
from typing import Dict, Optional
from .prnu_extractor import PRNUExtractor


class ForgeryDetector:
    """
    Detects forgeries and deepfakes using PRNU correlation
    Implementation from technical specification
    """
    
    def __init__(self, reference_patterns_path: Optional[str] = None, threshold: float = 0.4):
        """
        Initialize forgery detector
        
        Args:
            reference_patterns_path: Path to reference patterns .npz file
            threshold: Correlation threshold for forgery detection (default: 0.4)
        """
        self.prnu_extractor = PRNUExtractor()
        self.threshold = threshold
        
        if reference_patterns_path and os.path.exists(reference_patterns_path):
            self.prnu_extractor.load_reference_patterns(reference_patterns_path)
            print(f"✅ Loaded reference patterns for {len(self.prnu_extractor.reference_patterns)} cameras")
    
    def correlation_coefficient(self, prnu1: np.ndarray, prnu2: np.ndarray) -> float:
        """
        Calculate correlation coefficient between two PRNU patterns
        Implementation from technical specification
        
        Args:
            prnu1: First PRNU pattern
            prnu2: Second PRNU pattern (reference)
        
        Returns:
            float: Correlation coefficient
        """
        prnu1_flat, prnu2_flat = prnu1.flatten(), prnu2.flatten()
        return np.corrcoef(prnu1_flat, prnu2_flat)[0, 1]
    
    def detect_forgery(self, test_video_path: str, expected_camera: str) -> Dict:
        """
        Detect if a video is potentially forged or deepfake
        Implementation from technical specification
        
        Args:
            test_video_path: Path to video to test
            expected_camera: Expected camera model name
        
        Returns:
            dict: Detection results with correlation, threshold, and verdict
        """
        if expected_camera not in self.prnu_extractor.reference_patterns:
            raise ValueError(f"No reference pattern for camera: {expected_camera}")
        
        # Extract PRNU from test video
        test_prnu = self.prnu_extractor.video_to_prnu(test_video_path)
        
        # Get reference pattern
        reference_pattern = self.prnu_extractor.reference_patterns[expected_camera]
        
        # Calculate correlation
        corr = self.correlation_coefficient(test_prnu, reference_pattern)
        
        # Determine if forged
        is_forged = corr < self.threshold
        
        # Generate warning message
        if is_forged:
            message = "⚠️ Possible Forgery Detected!"
            verdict = "FORGED/DEEPFAKE"
        else:
            message = "✅ Video appears authentic"
            verdict = "AUTHENTIC"
        
        return {
            'correlation': float(corr),
            'threshold': self.threshold,
            'is_forged': is_forged,
            'verdict': verdict,
            'message': message,
            'expected_camera': expected_camera,
            'confidence': abs(corr - self.threshold)
        }
    
    def batch_detect(self, video_paths: list, expected_cameras: list) -> list:
        """
        Detect forgeries in multiple videos
        
        Args:
            video_paths: List of video paths
            expected_cameras: List of expected camera models (same length as video_paths)
        
        Returns:
            list: Detection results for each video
        """
        if len(video_paths) != len(expected_cameras):
            raise ValueError("Number of videos and expected cameras must match")
        
        results = []
        for video_path, camera in zip(video_paths, expected_cameras):
            try:
                result = self.detect_forgery(video_path, camera)
                result['video_path'] = video_path
                results.append(result)
            except Exception as e:
                results.append({
                    'video_path': video_path,
                    'error': str(e),
                    'verdict': 'ERROR'
                })
        
        return results
    
    def compare_prnu_patterns(self, video1_path: str, video2_path: str) -> Dict:
        """
        Compare PRNU patterns between two videos
        Useful for determining if two videos are from same camera
        
        Args:
            video1_path: Path to first video
            video2_path: Path to second video
        
        Returns:
            dict: Comparison results
        """
        prnu1 = self.prnu_extractor.video_to_prnu(video1_path)
        prnu2 = self.prnu_extractor.video_to_prnu(video2_path)
        
        corr = self.correlation_coefficient(prnu1, prnu2)
        
        same_camera = corr > self.threshold
        
        return {
            'video1': video1_path,
            'video2': video2_path,
            'correlation': float(corr),
            'threshold': self.threshold,
            'same_camera': same_camera,
            'confidence': abs(corr - self.threshold)
        }
    
    def analyze_video_authenticity(self, video_path: str, camera_name: str = None) -> Dict:
        """
        Comprehensive authenticity analysis of a video
        
        Args:
            video_path: Path to video
            camera_name: Optional expected camera name
        
        Returns:
            dict: Comprehensive analysis results
        """
        # Extract PRNU
        test_prnu = self.prnu_extractor.video_to_prnu(video_path)
        
        # Calculate stats
        prnu_mean = float(np.mean(test_prnu))
        prnu_std = float(np.std(test_prnu))
        prnu_energy = float(np.sum(test_prnu ** 2))
        
        result = {
            'video_path': video_path,
            'prnu_statistics': {
                'mean': prnu_mean,
                'std': prnu_std,
                'energy': prnu_energy
            },
            'prnu_shape': test_prnu.shape
        }
        
        # If camera name provided, check correlation
        if camera_name and camera_name in self.prnu_extractor.reference_patterns:
            forgery_result = self.detect_forgery(video_path, camera_name)
            result.update(forgery_result)
        else:
            # Check against all available cameras
            correlations = {}
            for cam_name, ref_pattern in self.prnu_extractor.reference_patterns.items():
                corr = self.correlation_coefficient(test_prnu, ref_pattern)
                correlations[cam_name] = float(corr)
            
            if correlations:
                best_match = max(correlations.items(), key=lambda x: x[1])
                result['best_match_camera'] = best_match[0]
                result['best_match_correlation'] = best_match[1]
                result['all_correlations'] = correlations
                result['is_forged'] = best_match[1] < self.threshold
                result['verdict'] = 'FORGED/DEEPFAKE' if result['is_forged'] else 'AUTHENTIC'
        
        return result


def main():
    """Example usage of forgery detector"""
    print("Forgery Detector Module")
    print("=" * 50)
    
    # Example usage:
    # detector = ForgeryDetector(reference_patterns_path='./camera_model_data/prnu_dataset/reference_patterns.npz')
    # result = detector.detect_forgery('./test_video.mp4', 'Samsung_S21')
    # print(f"Result: {result['verdict']}")
    # print(f"Correlation: {result['correlation']:.3f}")
    # print(result['message'])
    
    print("Initialize detector with reference patterns to use detection features.")


if __name__ == "__main__":
    main()
