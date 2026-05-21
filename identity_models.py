import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def load_dataset_and_preprocess(dataset_path, target_size=(64, 64)):
    X, y = [], []
    print("[INFO] Extracting flattened facial feature maps from dataset... Please wait.")
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset directory '{dataset_path}' not found!")
        return np.array(X), np.array(y)

    for student_name in os.listdir(dataset_path):
        student_folder = os.path.join(dataset_path, student_name)
        if not os.path.isdir(student_folder):
            continue
            
        print(f"[PROCESSING] Loading images for student: {student_name}")
        for image_name in os.listdir(student_folder):
            image_path = os.path.join(student_folder, image_name)
            
            # Load in grayscale for uniform structural analysis
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Resize to ensure all feature vector lengths match perfectly
            img_resized = cv2.resize(img, target_size)
            X.append(img_resized.flatten())
            y.append(student_name)
                
    return np.array(X), np.array(y)

if __name__ == "__main__":
    DATASET_PATH = "dataset" 
    
    # 1. Load and Flatten Image Data
    X_raw, y = load_dataset_and_preprocess(DATASET_PATH)
    
    if len(X_raw) == 0:
        print("[ERROR] No valid pictures found in the dataset directory!")
        exit()

    print(f"[SUCCESS] Loaded a total of {len(X_raw)} images into raw vector arrays.")

    # 2. Encode Class Labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    # 3. Train/Test Split (80% Train, 20% Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_encoded, test_size=0.2, random_state=42)

    # 4. Feature Extraction via PCA (Dimensionality Reduction to capture variance)
    print("[INFO] Projecting images into lower-dimensional structural feature space (PCA)...")
    n_components = min(50, len(X_train_raw))
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    
    X_train = pca.fit_transform(X_train_raw)
    X_test = pca.transform(X_test_raw)

    # Save PCA transformer so Ramlah can reduce live frames the exact same way
    with open("pca_transformer.pkl", "wb") as f:
        pickle.dump(pca, f)

    # 5. Initialize Classifiers
    models = {
        "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True),
        "KNN (K=3)": KNeighborsClassifier(n_neighbors=3),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    model_names = []
    accuracies = []
    latencies = []

    print("\n" + "="*65)
    print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Latency per Face (ms)':<20}")
    print("="*65)

    best_accuracy = 0
    best_model = None
    best_model_name = ""

    # 6. Training and Evaluation
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        start_time = time.time()
        predictions = model.predict(X_test)
        end_time = time.time()
        
        acc = accuracy_score(y_test, predictions) * 100
        total_time_ms = (end_time - start_time) * 1000
        latency_per_face = total_time_ms / len(X_test) if len(X_test) > 0 else 0
        
        model_names.append(name)
        accuracies.append(acc)
        latencies.append(latency_per_face)
        
        print(f"{name:<25} | {acc:<9.2f}% | {latency_per_face:<20.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    print("="*65)
    print(f"[RESULT] Optimal Model: {best_model_name} with {best_accuracy:.2f}% Accuracy")

    # 7. Save Trained Model Weights
    with open("best_identity_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print("[INFO] 'best_identity_model.pkl' has been successfully exported!")

    # 8. Performance Visualization Export
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Model Accuracy Comparison (%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 110)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    ax2.bar(model_names, latencies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Inference Latency Comparison (ms)')
    ax2.set_ylabel('Time per Face (ms)')
    for i, v in enumerate(latencies):
        ax2.text(i, v + 0.2, f"{v:.2f}ms", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("model_comparison_metrics.png")
    print("[SUCCESS] Analytical chart 'model_comparison_metrics.png' has been saved successfully!")