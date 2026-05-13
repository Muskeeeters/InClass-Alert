import streamlit as st
import cv2
import os
import time
from PIL import Image
import numpy as np

# --- Configuration ---
DATASET_DIR = "dataset"
IMAGES_TO_CAPTURE = 50

# Ensure the dataset directory exists
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

st.set_page_config(page_title="InClass Alert | Module 1", page_icon="🛡️", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("📸 InClass Alert")
st.sidebar.subheader("Module 1: Data Engine")
app_mode = st.sidebar.radio("Select Mode", ["View Database", "Webcam Enrollment", "Dataset Upload"])

if app_mode == "View Database":
    st.title("🗄️ Enrolled Students Database")
    st.markdown("Check which students are currently registered in the system.")
    
    students = os.listdir(DATASET_DIR)
    
    if not students:
        st.info("No students enrolled yet. Go to 'Webcam Enrollment' or 'Dataset Upload' to add data.")
    else:
        # Display a clean grid of registered students
        cols = st.columns(3)
        for index, student in enumerate(students):
            student_path = os.path.join(DATASET_DIR, student)
            if os.path.isdir(student_path):
                img_count = len([name for name in os.listdir(student_path) if os.path.isfile(os.path.join(student_path, name))])
                
                with cols[index % 3]:
                    st.success(f"**{student}**")
                    st.write(f"Images: {img_count}")

elif app_mode == "Webcam Enrollment":
    st.title("🎥 Live Webcam Enrollment")
    student_name = st.text_input("Enter Student Name:")
    
    if st.button("Start Camera & Register"):
        if student_name.strip() == "":
            st.warning("⚠️ Enter a name first.")
        elif os.path.exists(os.path.join(DATASET_DIR, student_name)):
            st.warning(f"⚠️ {student_name} already exists!")
        else:
            student_path = os.path.join(DATASET_DIR, student_name)
            os.makedirs(student_path)
            
            cap = cv2.VideoCapture(0)
            st.info("Capturing... Please move your head slightly.")
            progress_bar = st.progress(0)
            frame_window = st.image([])
            
            count = 0
            while count < IMAGES_TO_CAPTURE:
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(frame_rgb)
                    
                    img_name = os.path.join(student_path, f"{student_name}_{count}.jpg")
                    cv2.imwrite(img_name, frame)
                    count += 1
                    progress_bar.progress(count / IMAGES_TO_CAPTURE)
                    time.sleep(0.1)
                    
            cap.release()
            cv2.destroyAllWindows()
            st.success(f"✅ Successfully registered {student_name}!")


elif app_mode == "Dataset Upload":
    st.title("📁 Bulk Dataset Upload")
    st.markdown("Upload existing photos of a student from your computer.")
    
    student_name_upload = st.text_input("Enter Student Name for these photos:")
    uploaded_files = st.file_uploader("Select multiple images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if st.button("Save Uploaded Dataset"):
        if student_name_upload.strip() == "":
            st.warning("⚠️ Enter a name first.")
        elif not uploaded_files:
            st.warning("⚠️ Please select images to upload.")
        else:
            student_path = os.path.join(DATASET_DIR, student_name_upload)
            if not os.path.exists(student_path):
                os.makedirs(student_path)
                
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Convert uploaded file to OpenCV format and save
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                # Convert RGB to BGR for OpenCV saving
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                img_name = os.path.join(student_path, f"{student_name_upload}_upload_{i}.jpg")
                cv2.imwrite(img_name, img_array)
                
                progress_bar.progress((i + 1) / total_files)
                
            st.success(f"✅ Successfully saved {total_files} images for {student_name_upload}!")