"""
Comprehensive test suite for the Camera Source Identification System
Author: Pranav Patil | Sponsored by PiLabs
"""

import unittest
import numpy as np
import tempfile
import os
import cv2
from unittest.mock import patch, MagicMock
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.prnu_extractor import PRNUExtractor
from src.camera_pipeline import CameraIdentificationPipeline, PRNUDataset


class TestPRNUExtractor(unittest.TestCase):
    """Test PRNU extraction functionality"""

    def setUp(self):
        self.extractor = PRNUExtractor(target_size=(128, 128))  # Smaller for testing

        # Create test frame
        self.test_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    def test_extract_prnu_frame(self):
        """Test PRNU extraction from single frame"""
        prnu = self.extractor.extract_prnu_frame(self.test_frame)

        # Check output properties
        self.assertEqual(prnu.shape, (128, 128))
        self.assertAlmostEqual(np.mean(prnu), 0, places=3)  # Zero mean
        self.assertAlmostEqual(np.std(prnu), 1, places=1)   # Unit variance

    def test_correlation_coefficient(self):
        """Test correlation coefficient calculation"""
        prnu1 = np.random.randn(64, 64)
        prnu2 = prnu1 + 0.1 * np.random.randn(64, 64)  # Similar pattern
        prnu3 = np.random.randn(64, 64)  # Different pattern

        # High correlation for similar patterns
        corr_high = self.extractor.correlation_coefficient(prnu1, prnu2)
        self.assertGreater(corr_high, 0.5)

        # Low correlation for different patterns
        corr_low = self.extractor.correlation_coefficient(prnu1, prnu3)
        self.assertLess(abs(corr_low), 0.5)

    def test_reference_pattern_operations(self):
        """Test saving/loading reference patterns"""
        # Create mock reference patterns
        self.extractor.reference_patterns = {
            'camera1': np.random.randn(128, 128),
            'camera2': np.random.randn(128, 128)
        }

        # Test save/load
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            self.extractor.save_reference_patterns(tmp.name)

            # Create new extractor and load
            new_extractor = PRNUExtractor()
            new_extractor.load_reference_patterns(tmp.name)

            # Check patterns match
            self.assertEqual(len(new_extractor.reference_patterns), 2)
            self.assertIn('camera1', new_extractor.reference_patterns)

            # Cleanup
            os.unlink(tmp.name)


class TestPRNUDataset(unittest.TestCase):
    """Test PRNU dataset functionality"""

    def setUp(self):
        # Create mock PRNU patterns
        self.prnu_patterns = np.random.randn(10, 64, 64)
        self.labels = np.random.randint(0, 3, 10)

    def test_dataset_creation(self):
        """Test dataset initialization"""
        dataset = PRNUDataset(self.prnu_patterns, self.labels)

        self.assertEqual(len(dataset), 10)

        # Test item retrieval
        prnu_tensor, label = dataset[0]
        self.assertEqual(prnu_tensor.shape, (3, 64, 64))  # 3-channel RGB
        self.assertIsInstance(label.item(), int)


class TestCameraIdentificationPipeline(unittest.TestCase):
    """Test the complete pipeline"""

    def setUp(self):
        self.pipeline = CameraIdentificationPipeline(num_classes=3)
        self.pipeline.class_names = ['Camera1', 'Camera2', 'Camera3']

    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertEqual(self.pipeline.num_classes, 3)
        self.assertEqual(len(self.pipeline.class_names), 3)
        self.assertIsNotNone(self.pipeline.model)
        self.assertIsNotNone(self.pipeline.prnu_extractor)

    @patch('cv2.VideoCapture')
    def test_predict_video_source_mock(self, mock_capture):
        """Test video prediction with mocked video"""
        # Mock video capture
        mock_cap = MagicMock()
        mock_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
            for _ in range(15)
        ] + [(False, None)]

        # Mock model prediction
        with patch.object(self.pipeline.model, 'eval'), \
             patch('torch.no_grad'), \
             patch.object(self.pipeline.model, '__call__') as mock_forward:

            # Mock model output
            import torch
            mock_logits = torch.tensor([[2.0, 1.0, 0.5]])
            mock_features = torch.randn(1, 2048)
            mock_forward.return_value = (mock_logits, mock_features)

            # Test prediction
            result = self.pipeline.predict_video_source('dummy_path.mp4')

            self.assertIsInstance(result, dict)
            self.assertIn('predicted_camera', result)
            self.assertIn('confidence', result)
            self.assertIn('all_probabilities', result)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def create_test_video(self, filename, frames=30):
        """Create a test video file"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (320, 240))

        for _ in range(frames):
            # Create frame with some pattern
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            out.write(frame)

        out.release()

    def test_video_to_prnu_extraction(self):
        """Test complete video to PRNU extraction"""
        extractor = PRNUExtractor(target_size=(64, 64))

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            self.create_test_video(tmp.name, frames=10)

            try:
                prnu = extractor.video_to_prnu(tmp.name, num_frames=5)

                # Check output
                self.assertEqual(prnu.shape, (64, 64))
                self.assertAlmostEqual(np.mean(prnu), 0, places=3)

            finally:
                os.unlink(tmp.name)

    def test_forgery_detection_workflow(self):
        """Test the forgery detection workflow"""
        extractor = PRNUExtractor(target_size=(64, 64))

        # Create reference pattern
        reference = np.random.randn(64, 64)
        reference = reference / np.linalg.norm(reference)
        extractor.reference_patterns['test_camera'] = reference

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            self.create_test_video(tmp.name, frames=10)

            try:
                result = extractor.detect_forgery(tmp.name, 'test_camera', threshold=0.3)

                # Check result structure
                self.assertIn('correlation', result)
                self.assertIn('is_forged', result)
                self.assertIn('confidence', result)
                self.assertIsInstance(result['is_forged'], bool)

            finally:
                os.unlink(tmp.name)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def test_invalid_video_path(self):
        """Test handling of invalid video paths"""
        extractor = PRNUExtractor()

        with self.assertRaises(ValueError):
            extractor.video_to_prnu('nonexistent_video.mp4')

    def test_empty_directory(self):
        """Test handling of empty directories"""
        extractor = PRNUExtractor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                extractor.generate_reference_pattern(tmp_dir, 'empty_camera')

    def test_corrupted_frame_handling(self):
        """Test handling of corrupted/invalid frames"""
        extractor = PRNUExtractor()

        # Test with invalid frame
        invalid_frame = np.array([])

        with self.assertRaises((ValueError, AttributeError)):
            extractor.extract_prnu_frame(invalid_frame)


def run_performance_benchmark():
    """Benchmark PRNU extraction performance"""
    print("\n🚀 Performance Benchmark")
    print("=" * 40)

    extractor = PRNUExtractor(target_size=(256, 256))

    # Test different frame sizes
    frame_sizes = [(240, 320), (480, 640), (720, 1280), (1080, 1920)]

    import time

    for height, width in frame_sizes:
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        start_time = time.time()
        prnu = extractor.extract_prnu_frame(frame)
        end_time = time.time()

        processing_time = end_time - start_time
        print(f"📊 {width}x{height}: {processing_time:.3f}s")


if __name__ == '__main__':
    # Run unit tests
    print("🧪 Running Camera Source Identification Tests")
    print("=" * 50)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestPRNUExtractor,
        TestPRNUDataset,
        TestCameraIdentificationPipeline,
        TestIntegration,
        TestErrorHandling
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n📈 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("✅ All tests passed!")

        # Run performance benchmark if all tests pass
        run_performance_benchmark()
    else:
        print("❌ Some tests failed!")

        # Print failure details
        if result.failures:
            print("\n💥 Failures:")
            for test, traceback in result.failures:
                print(f"   {test}: {traceback}")

        if result.errors:
            print("\n⚠️ Errors:")
            for test, traceback in result.errors:
                print(f"   {test}: {traceback}")

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)