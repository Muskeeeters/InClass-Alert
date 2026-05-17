import cv2
import numpy as np
from keras_facenet import FaceNet
import joblib
import os

# 1. Mute TensorFlow system warnings locally
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 2. Bulletproof Path - Always finds the model in the exact folder this script lives in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'liveness_model.pkl')

print("Loading your Liveness Engine...")
try:
    liveness_model = joblib.load(model_path)
except FileNotFoundError:
    print(f"\n❌ ERROR: Could not find 'liveness_model.pkl' in {BASE_DIR}")
    print("Please ensure the downloaded Kaggle model is inside your /models folder!")
    exit()

print("Loading FaceNet...")
# Initialize the FaceNet embedder
embedder = FaceNet()

print("Starting Webcam... Press 'q' to quit.")
# 3. Open the default webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Is your webcam being used by another app?")
        break

    # OpenCV uses BGR format, but FaceNet requires RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 4. Detect faces and extract embeddings
    # Using threshold 0.80 for variable webcam lighting
    detections = embedder.extract(rgb_frame, threshold=0.80)

    for detection in detections:
        # Extract the bounding box coordinates
        x, y, w, h = detection['box']
        
        # Extract the 128-D vector
        embedding = detection['embedding']

        # 5. Make a Prediction (The Stricter Way)
        # Get exact probabilities instead of a simple 1 or 0
        probabilities = liveness_model.predict_proba([embedding])[0]
        fake_prob = probabilities[0]
        real_prob = probabilities[1]

        # 6. Draw the UI on the frame (Stricter Threshold)
        # We DEMAND the model be at least 75% confident it is a live person
        if real_prob > 0.75:
            label = f"REAL ({real_prob*100:.1f}%)"
            color = (0, 255, 0) # Green box
        else:
            # If it's less than 75% sure it's real, default to Fake/Spoof
            label = f"FAKE/SPOOF ({fake_prob*100:.1f}%)"
            color = (0, 0, 255) # Red box

        # Draw the rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Put the text label slightly above the rectangle
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show the final image
    cv2.imshow("InClass Alert - Module 2 Liveness Test", frame)

    # Listen for the 'q' key to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()