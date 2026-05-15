import os
import shutil
import time
import cv2
import numpy as np
from PIL import Image

# Base setup for local storage
DATASET_DIR = "dataset"
IMAGES_TO_CAPTURE = 50

def init_system():
    # Ensures our root directory exists before doing any file operations
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

def get_students():
    init_system()
    return [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

def get_image_count(student_name):
    # Quick sweep of a specific student's directory
    path = os.path.join(DATASET_DIR, student_name)
    if os.path.exists(path):
        return len(os.listdir(path))
    return 0

def get_stats():
    # Calculate dataset volume for the dashboard header
    students = get_students()
    total_images = sum(get_image_count(s) for s in students)
    return len(students), total_images

def check_student_exists(student_name):
    return os.path.exists(os.path.join(DATASET_DIR, student_name))

def delete_student_record(student_name):
    # Completely wipes the directory and all associated image data
    path = os.path.join(DATASET_DIR, student_name)
    if os.path.exists(path):
        shutil.rmtree(path)

def process_batch_upload(student_name, uploaded_files):
    # Creates the directory if it doesn't exist
    path = os.path.join(DATASET_DIR, student_name)
    os.makedirs(path, exist_ok=True)
    
    # We yield progress back to the UI so it can update its progress bar seamlessly
    total_files = len(uploaded_files)
    for i, file in enumerate(uploaded_files):
        # PIL handles various weird image formats and color profiles well
        image = Image.open(file)
        img_array = np.array(image)
        
        # If image is RGB (standard for web), convert to BGR for OpenCV standard
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
        file_path = os.path.join(path, f"{student_name}_batch_{i}.jpg")
        cv2.imwrite(file_path, img_array)
        
        yield i + 1, total_files

def capture_webcam_stream(student_name):
    # Acts as a generator so the UI can render frames in real-time without 
    # needing to import or handle OpenCV logic directly.
    path = os.path.join(DATASET_DIR, student_name)
    os.makedirs(path, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    count = 0
    
    while count < IMAGES_TO_CAPTURE:
        ret, frame = cap.read()
        if not ret:
            break
            
        # UI expects RGB, so we convert it for the frontend
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save the raw BGR frame to disk so CV2 training scripts won't break later
        file_path = os.path.join(path, f"{student_name}_{count}.jpg")
        cv2.imwrite(file_path, frame)
        
        count += 1
        yield frame_rgb, count, IMAGES_TO_CAPTURE
        
        # Small sleep ensures we capture slight changes in facial position
        time.sleep(0.05)
        
    cap.release()