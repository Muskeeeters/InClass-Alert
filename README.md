# 🛡️ InClass Alert: Dual-Engine Smart Attendance System

An advanced, real-time facial recognition and anti-spoofing attendance system built for modern educational environments. **InClass Alert** eliminates manual roll calls and proxy attendance by deploying a dual-model architecture that not only verifies identity but mathematically blocks digital presentation attacks (photos/screens).

## ✨ Key Features
* **Dual-Engine Authentication:** Simultaneously processes Identity and Liveness in real-time.
* **Temporal Smoothing:** Utilizes a 50-frame majority-voting algorithm to eliminate flickering and prevent false positives.
* **Premium UI/UX:** Built with a custom Streamlit Dark Mode featuring glassmorphism elements, dynamic sidebars, and real-time bounding boxes.
* **Lightweight & Fast:** Engineered to run on standard laptop hardware without requiring a heavy GPU, processing frames in milliseconds.

## 🧠 The Architecture
Instead of relying on a single, easily fooled deep learning model, InClass Alert splits the workload:

1. **The Identity Engine (FaceNet + PCA + SVM)**
   * Extracts 128-D geometric facial embeddings using FaceNet.
   * Compresses 4,096 raw pixels into 50 core mathematical components via PCA.
   * Classifies the identity using a Support Vector Machine (SVM) trained for 100% accuracy on low-volume datasets (50 images/student).
2. **The Anti-Spoofing Engine (Texture Analysis + Logistic Regression)**
   * Bypasses geometric spoofing by analyzing flat screen glare, moiré patterns, and bezels.
   * Extracts $64 \times 64$ grayscale pixel textures from precise bounding boxes.
   * Uses Logistic Regression to draw a strict boundary between "Human Skin Texture" and "Digital Screen Texture."

## 🛠️ Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV
* **Machine Learning:** Scikit-Learn (SVM, PCA, Logistic Regression)
* **Deep Learning Feature Extraction:** Keras FaceNet
* **Frontend/Dashboard:** Streamlit (Custom CSS injected)

## 🚀 How to Run Locally
1. Clone the repository: `git clone https://github.com/your-username/InClass-Alert.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the Master Dashboard: `streamlit run app.py`

## 👥 The Team
* **Ramlah Munir** (Live Authentication & Liveness)
* **Haleema Zafar** (Identity Engine & Comparative Analysis)
* **Muhammad Talal** (Enrollment UI & Database)
