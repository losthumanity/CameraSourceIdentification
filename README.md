# Camera Source Identification 📸🔍

A deep learning-based system to identify the source camera model of an image using PRNU (Photo Response Non-Uniformity) noise patterns.

This project is based on the IEEE Signal Processing Society Camera Model Identification Challenge (SP Cup 2018) and supports training and evaluating a CNN pipeline on image residuals.

---

## 🗂 Dataset

This project uses the official dataset from [Kaggle - IEEE SP Cup 2018: Camera Model Identification Challenge](https://www.kaggle.com/c/sp-society-camera-model-identification/data). The dataset contains images from 10 different camera models for training and evaluation.

Due to licensing restrictions, please download the dataset manually from the above link and place it in a suitable folder (e.g., `data/`).

---

## 🧠 Features

- 📥 Preprocessing pipeline to extract PRNU noise (residuals)
- 🧼 Noise-aware training data generation
- 🧠 CNN-based architecture for classification
- 📊 Evaluation metrics and logging
- 🧪 Jupyter notebooks for model exploration

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/losthumanity/CameraSourceIdentification.git
cd CameraSourceIdentification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset
Download from: https://www.kaggle.com/c/sp-society-camera-model-identification/data
Place the unzipped contents inside a data/ directory (not tracked by Git).

---

## 🧱 Project Structure
```bash
CameraSourceIdentification/
├── camera_model_data/
├── notebooks/
│   └── ModuleV2.ipynb
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── preprocessing.py
├── requirements.txt
├── .gitignore
└── README.md
```

### 🧪 Example Commands
Train the model
```bash
python src/train.py
```
Evaluate the model
```bash
python src/evaluate.py
```

### 📌 Notes
The extracted PRNU features and processed noise data are stored under camera_model_data/
To avoid Git bloat, large files and dataset contents are not tracked (see .gitignore)

### 📚 References
SP Cup 2018 Challenge on Kaggle

Lukas, J., Fridrich, J., & Goljan, M. (2006). Digital camera identification from sensor pattern noise. IEEE Transactions on Information Forensics and Security.

### 🙋‍♂️ Author
Pranav Patil
If you find this work useful, consider giving this repo a ⭐!
