import streamlit as st
import cv2
import numpy as np
import joblib
import pickle
from keras_facenet import FaceNet
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="InClass Alert | Smart Attendance", page_icon="🛡️", layout="wide")

# --- 2. LOAD ALL ENGINES (Cached for high-speed performance) ---
# --- 2. LOAD ALL ENGINES (Cached for high-speed performance) ---
@st.cache_resource
def load_ml_engines():
    # Pure function: NO Streamlit UI commands inside here!
    embedder = FaceNet()
    
    liveness_path = 'models/new_liveness_model.pkl'
    if not os.path.exists(liveness_path):
        raise FileNotFoundError(f"Missing {liveness_path}! Please run the liveness training script.")
    
    liveness = joblib.load(liveness_path)
    
    with open('best_identity_model.pkl', 'rb') as f:
        identity = pickle.load(f)
    with open('pca_transformer.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
        
    return embedder, liveness, identity, pca, le

# We handle the UI and errors OUTSIDE the cached function
try:
    embedder, liveness_model, identity_model, pca, le = load_ml_engines()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()
# --- 3. ATTENDANCE DATABASE ---
if "attendance_sheet" not in st.session_state:
    st.session_state.attendance_sheet = {name: "Pending" for name in le.classes_}

# --- 4. THE 50-FRAME DUAL SCANNER ---
def mark_attendance_scanner(target_student):
    cap = cv2.VideoCapture(0)
    frames_collected = 0
    real_votes = 0
    id_match_votes = 0
    
    st.info(f"🎥 Scanning started for {target_student.title()}. Please look directly at the camera...")
    progress_bar = st.progress(0)
    
    while frames_collected < 50:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam failed to open. Check your permissions.")
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using FaceNet's highly accurate bounding boxes
        detections = embedder.extract(rgb_frame, threshold=0.80)
        
        if detections:
            det = detections[0] # Focus on the primary face
            x, y, w, h = det['box']
            
            # --- THE UNIFIED CROP ---
            # Expand the box by 20% to capture jawline and hair, matching enrollment data
            pad_x = int(w * 0.20)
            pad_y = int(h * 0.20)
            
            y1 = max(0, y - pad_y)
            y2 = min(frame.shape[0], y + h + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(frame.shape[1], x + w + pad_x)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                # Convert to grayscale, resize to 64x64, and flatten ONCE
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (64, 64))
                flattened_face = resized_face.flatten() # 4096 features
                
                # --- STEP A: LIVENESS CHECK ---
                # Predict (Real vs Fake) using texture analysis
                prob_liveness = liveness_model.predict_proba([flattened_face])[0]
                
                # Dynamically find the index for "1" (Real) so it never inverts
                real_idx = list(liveness_model.classes_).index(1)
                is_real = prob_liveness[real_idx] > 0.60
                
                if is_real:
                    real_votes += 1
                    
                # --- STEP B: IDENTITY CHECK ---
                # Compress the 4096 features using Haleema's PCA, then predict
                emb_pca = pca.transform([flattened_face])
                pred_id = identity_model.predict(emb_pca)[0]
                pred_name = le.inverse_transform([pred_id])[0]
                
                if pred_name.lower() == target_student.lower():
                    id_match_votes += 1
            
            # --- UI UPDATES ---
            frames_collected += 1
            progress_bar.progress(frames_collected / 50.0)
            
            # Draw visual feedback on the camera window
            color = (0, 255, 0) if is_real else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            status_text = "REAL" if is_real else "SPOOF"
            display_name = pred_name.title() if face_crop.size > 0 else "Scanning..."
            
            cv2.putText(frame, f"{status_text} | ID: {display_name}", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
        cv2.imshow("InClass Alert - Security Scanner", frame)
        
        # Emergency quit button 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    progress_bar.empty()
    
    # --- FINAL VERDICT (Majority Voting) ---
    # We require 25+ frames (50%) to be both REAL and MATCH the requested student
    if real_votes >= 25 and id_match_votes >= 25:
        return "Present"
    elif real_votes < 25:
        return f"Spoof Detected ({real_votes}/50 Real)"
    else:
        return f"Identity Mismatch ({id_match_votes}/50 Match)"


# --- 5. THE MAIN DASHBOARD UI ---
st.sidebar.title("🛡️ InClass Alert")
st.sidebar.markdown("### BCS-6A Final Project")
st.sidebar.caption("By Ramlah, Haleema, and Talal")

app_mode = st.sidebar.radio("System Navigation", ["Live Attendance Dashboard", "Enroll New Student"])

if app_mode == "Enroll New Student":
    # Loads Talal's UI perfectly
    try:
        import enrollment_gui 
    except ImportError:
        st.error("Could not find 'enrollment_gui.py'. Make sure it's in the same folder.")
else:
    st.title("📊 Today's Attendance Roster")
    st.markdown("Select a student below to initiate the dual-layer authentication scan.")
    st.divider()
    
    # Table Headers
    col1, col2, col3 = st.columns([2, 1, 1])
    col1.markdown("**Student Name**")
    col2.markdown("**Verification Status**")
    col3.markdown("**Action**")
    
    # Render a row for every student dynamically
    for student in le.classes_:
        with st.container():
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.write(f"🎓 **{student.title()}**")
            
            # Status Badge Logic
            current_status = st.session_state.attendance_sheet[student]
            if current_status == "Present":
                c2.success("✅ Present")
            elif current_status == "Pending":
                c2.info("⏳ Pending")
            else:
                c2.error(f"🚨 {current_status}") 
                
            # Scan Button Logic
            if c3.button(f"Scan Profile", key=f"btn_{student}"):
                final_result = mark_attendance_scanner(student)
                st.session_state.attendance_sheet[student] = final_result
                st.rerun() # Instantly refreshes the UI to show the new result