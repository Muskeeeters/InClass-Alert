import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os
from keras_facenet import FaceNet

# Create folder for model if it doesn't exist
os.makedirs('models', exist_ok=True)

print("Booting up FaceNet for precise cropping...")
embedder = FaceNet()

def capture_dataset(label_name, num_frames=100):
    cap = cv2.VideoCapture(0)
    frames = []
    print(f"\n[GET READY] Collecting {num_frames} frames for: {label_name.upper()}")
    print("Press 'c' to start capturing...")
    
    while True:
        ret, frame = cap.read()
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
            
    print(f"Capturing {label_name}...")
    collected = 0
    
    while collected < num_frames:
        ret, frame = cap.read()
        if not ret: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = embedder.extract(rgb_frame, threshold=0.80)
        
        if detections:
            det = detections[0]
            x, y, w, h = det['box']
            
            # --- EXACT SAME CROP AS APP.PY ---
            pad_x = int(w * 0.20)
            pad_y = int(h * 0.20)
            y1 = max(0, y - pad_y)
            y2 = min(frame.shape[0], y + h + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(frame.shape[1], x + w + pad_x)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                frames.append(resized.flatten()) # Flatten to 4096 pixels
                collected += 1
                
                # Draw box so you know it's working
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Captured: {collected}/{num_frames}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(10) # Small delay
        
    cap.release()
    cv2.destroyAllWindows()
    return frames

# --- 1. COLLECT DATA ---
print("--- LIVENESS MODEL TRAINING ---")
print("First, look at the camera normally. Move your head slightly so it learns depth.")
real_data = capture_dataset("REAL (Normal Face)", 100)

print("\nNow, hold up a PHONE SCREEN with a face on it to the camera.")
print("PRO TIP: Wiggle the phone slightly so the camera catches the screen glare!")
fake_data = capture_dataset("FAKE (Phone Screen)", 100)

# --- 2. PREPARE DATA ---
X = np.array(real_data + fake_data)
y = np.array([1]*100 + [0]*100) # 1 = Real, 0 = Fake

# --- 3. TRAIN LOGISTIC REGRESSION ---
print("\nTraining Logistic Regression on precise facial textures...")
model = LogisticRegression(max_iter=2000, C=0.1) 
model.fit(X, y)

accuracy = model.score(X, y)
print(f"Training Complete! Model Accuracy: {accuracy * 100:.2f}%")

# --- 4. SAVE MODEL ---
joblib.dump(model, 'models/new_liveness_model.pkl')
print("\nSaved as 'models/new_liveness_model.pkl'. You are ready to integrate!")