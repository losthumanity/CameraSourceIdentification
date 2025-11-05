# Example Data Directory Structures

## Raw Video Dataset Structure
```
video_data/
├── Samsung_Galaxy_S21/
│   ├── video001.mp4
│   ├── video002.mp4
│   ├── video003.mp4
│   └── ... (10-50 videos per camera)
├── iPhone_13_Pro/
│   ├── video001.mp4
│   ├── video002.mp4
│   └── ...
├── OnePlus_9_Pro/
├── Xiaomi_Mi_11/
├── Google_Pixel_6/
├── Huawei_P50_Pro/
├── Sony_Xperia_1_III/
├── LG_V60_ThinQ/
└── Motorola_Edge_20/
```

## Generated PRNU Dataset Structure
```
camera_model_data/
├── prnu_dataset/
│   ├── train_dataset.npz         # Training PRNU patterns
│   ├── test_dataset.npz          # Test PRNU patterns
│   ├── reference_patterns.npz    # Camera reference fingerprints
│   └── dataset_metadata.json     # Dataset info
├── best_prnu_classifier.pth      # Trained CNN model
├── training_history.png          # Training curves
├── confusion_matrix.png          # Evaluation results
├── class_distribution.png        # Dataset statistics
└── evaluation_results.json       # Detailed metrics
```

## Model Files & Outputs
- `best_prnu_classifier.pth`: Trained ResNet50 model weights
- `reference_patterns.npz`: PRNU fingerprints for each camera
- `evaluation_results.json`: Accuracy, precision, recall metrics
- `training_history.png`: Loss/accuracy curves during training