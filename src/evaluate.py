import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Load ResNet50 features and labels
features = np.load('./camera_model_data/processed_dataset/features_rn50.npy')
true_labels = np.load('./camera_model_data/processed_dataset/true_labels_rn50.npy')

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Initialize classifiers
svm = SVC(kernel='rbf', probability=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train classifiers on ResNet50 features
svm_preds = svm.fit(features_scaled, true_labels).predict(features_scaled)
knn_preds = knn.fit(features_scaled, true_labels).predict(features_scaled)
rf_preds = rf.fit(features_scaled, true_labels).predict(features_scaled)

# Ensemble majority voting
votes = np.vstack([svm_preds, knn_preds, rf_preds]).T
ensemble_preds, _ = mode(votes, axis=1, keepdims=True)
ensemble_preds = ensemble_preds.flatten()

# Save accuracies for plotting later
svm_acc = accuracy_score(true_labels, svm_preds) * 100
knn_acc = accuracy_score(true_labels, knn_preds) * 100
rf_acc = accuracy_score(true_labels, rf_preds) * 100
ensemble_acc = accuracy_score(true_labels, ensemble_preds) * 100

print(f"SVM Accuracy: {svm_acc:.2f}%")
print(f"KNN Accuracy: {knn_acc:.2f}%")
print(f"Random Forest Accuracy: {rf_acc:.2f}%")
print(f"Ensemble Accuracy: {ensemble_acc:.2f}%")

# Save results to CSV
df = pd.DataFrame({
    'SVM': svm_preds,
    'KNN': knn_preds,
    'RF': rf_preds,
    'Ensemble': ensemble_preds,
    'True_Label': true_labels
})
df.to_csv('./camera_model_data/processed_dataset/resnet50_classifier_outputs.csv', index=False)
print("Saved ResNet50 classifier predictions and ensemble.")