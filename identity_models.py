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
import urllib.request

xml_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
    xml_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(xml_path):
        print("[INFO] Downloading face detector XML... Please wait.")
        urllib.request.urlretrieve(xml_url, xml_path)
    face_cascade.load(xml_path)

def load_dataset_and_preprocess(dataset_path, target_size=(64, 64)):
    X_data, labels_list = [], []
    print("[INFO] Loading dataset, cropping tightly, and applying Data Augmentation...")
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset directory '{dataset_path}' not found!")
        return np.array(X_data), np.array(labels_list)

    total_images_processed = 0
    faces_detected_count = 0

    for student_name in os.listdir(dataset_path):
        student_folder = os.path.join(dataset_path, student_name)
        if not os.path.isdir(student_folder):
            continue
            
        print(f"[PROCESSING] Augmenting data for: {student_name}")
        for image_name in os.listdir(student_folder):
            image_path = os.path.join(student_folder, image_name)
            
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            total_images_processed += 1
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            
            if len(faces) > 0:
                (x, coord_y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                face_crop = img[coord_y:coord_y+h, x:x+w]
                faces_detected_count += 1
            else:
                h_img, w_img = img.shape
                start_x, start_y = int(w_img * 0.15), int(h_img * 0.15)
                end_x, end_y = int(w_img * 0.85), int(h_img * 0.85)
                face_crop = img[start_y:end_y, start_x:end_x]
            
            img_resized = cv2.resize(face_crop, target_size)
            
            # --- THE MAGIC TRICK: DATA AUGMENTATION ---
            # 1. Original Face
            X_data.append(img_resized.flatten())
            labels_list.append(student_name)
            
            # 2. Horizontally Flipped Face (Mirror Image)
            img_flipped = cv2.flip(img_resized, 1)
            X_data.append(img_flipped.flatten())
            labels_list.append(student_name)
            
            # 3. Slightly Darker Face (Lighting Variation)
            img_darker = cv2.convertScaleAbs(img_resized, alpha=0.8, beta=-15)
            X_data.append(img_darker.flatten())
            labels_list.append(student_name)
            
            # 4. Slightly Brighter Face
            img_brighter = cv2.convertScaleAbs(img_resized, alpha=1.2, beta=15)
            X_data.append(img_brighter.flatten())
            labels_list.append(student_name)
                
    # 1 image * 4 variations = Huge Dataset automatically!
    print(f"[METRICS] Successfully extracted and augmented {faces_detected_count * 4} unique facial samples!")
    return np.array(X_data), np.array(labels_list)

if __name__ == "__main__":
    DATASET_PATH = "dataset" 
    
    X_raw, y_labels = load_dataset_and_preprocess(DATASET_PATH)
    
    if len(X_raw) == 0:
        print("[ERROR] No valid pictures found!")
        exit()

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_labels)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_encoded, test_size=0.2, random_state=42)

    print("[INFO] Projecting features into PCA space...")
    n_components = min(50, len(X_train_raw))
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    
    X_train = pca.fit_transform(X_train_raw)
    X_test = pca.transform(X_test_raw)

    with open("pca_transformer.pkl", "wb") as f:
        pickle.dump(pca, f)

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

    with open("best_identity_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print("[INFO] 'best_identity_model.pkl' saved successfully!")

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
    print("[SUCCESS] New analytical chart saved!")