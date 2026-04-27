# InClass Alert: Smart Attendance with Liveness Detection 🛡️📸

### Machine Learning Fundamentals (BCS-6A) - Project Proposal

**InClass Alert** is an automated, vision-based attendance system designed to eliminate manual roll calls and proxy attendance. By combining **FaceNet** embeddings with **ML Classifiers**, the system identifies students while simultaneously performing **Liveness Detection** to distinguish real faces from spoof attempts (photos or screens).

---

## 2. Team Members (BCS-6A)
* **Muhammad Talal** (FA23-BCS-028)
* **Haleema Zafar** (FA23-BCS-008)
* **Ramlah Munir** (FA23-BCS-037) 

**Submitted To:** Dr. Sheraz Anjum

---

## 🚀 Key Features
* **Facial Recognition:** Multi-class classification identifying students from 128-D embeddings.
* **Anti-Spoofing (Liveness):** Binary classification (Real vs. Fake) to prevent digital/print attacks.
* **Comparative Analysis:** Performance evaluation across **SVM, KNN, and Random Forest**.
* **Real-time Performance:** Optimized for low inference latency (ms/frame).

## 🛠️ Tech Stack
- **Language:** Python
- **Feature Extraction:** FaceNet (Pre-trained Deep Learning Model)
- **ML Classifiers:** Scikit-Learn (SVM, Logistic Regression, Random Forest, KNN)
- **Computer Vision:** OpenCV
- **Data Source:** Kaggle (Liveness) & Self-Collected (Student IDs)

## 📁 Project Structure
```text
├── data/               # Raw and processed datasets (Gitignored)
├── models/             # Saved .pkl files for SVM/RF/LR
├── notebooks/          # Jupyter notebooks for EDA and Training
├── src/
│   ├── preprocess.py   # Image processing and embedding extraction
│   ├── train.py        # Model training and comparison scripts
│   └── inference.py    # Real-time camera feed attendance script
├── requirements.txt    # Project dependencies
└── README.md
