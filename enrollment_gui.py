import streamlit as st
# Importing your logic file to handle data operations
import enrollment as logic

# st.set_page_config(page_title="InClass Alert | Admin", layout="wide")

# We use session state to queue toast messages so they render upon script rerun
if "toast_msg" not in st.session_state:
    st.session_state.toast_msg = None

# Injecting custom CSS to override Streamlit's default themes. 
# This handles the dark mode colors, glassmorphism cards, and the toast animations.
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0b1120;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, sans-serif;
}


.glass-card {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    transition: transform 0.2s ease, border-color 0.2s ease;
    margin-bottom: 1rem;
}
.glass-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.15);
}

.stats-container {
    background: linear-gradient(145deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.8) 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.stats-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1.2;
}
.stats-label {
    font-size: 0.875rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

.stButton>button {
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    background-color: #1e293b;
    color: #f8fafc;
    border: 1px solid #334155;
    transition: all 0.2s;
}
.stButton>button:hover {
    background-color: #0ea5e9;
    border-color: #0ea5e9;
    box-shadow: 0 0 15px rgba(14, 165, 233, 0.3);
}

.toast-container {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: rgba(15, 23, 42, 0.95);
    backdrop-filter: blur(10px);
    color: #f1f5f9;
    padding: 16px 24px;
    border-radius: 8px;
    border-left: 4px solid #0ea5e9;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    z-index: 9999;
    font-weight: 500;
    animation: slideInRight 0.4s cubic-bezier(0.16, 1, 0.3, 1), 
               fadeOutRight 0.4s cubic-bezier(0.16, 1, 0.3, 1) 4.5s forwards;
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(100%); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes fadeOutRight {
    to { opacity: 0; transform: translateX(100%); }
}

.stTextInput input {
    background-color: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid #334155 !important;
    color: white !important;
}
.stTextInput input:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 1px #0ea5e9 !important;
}
</style>
""", unsafe_allow_html=True)


def render_toast():
    # Renders frontend JS notification and cleans itself up after 4.9s
    if st.session_state.toast_msg:
        toast_html = f"""
        <div class="toast-container" id="custom-toast">
            {st.session_state.toast_msg}
        </div>
        <script>
            setTimeout(function(){{
                var el = document.getElementById('custom-toast');
                if(el) el.remove();
            }}, 4900);
        </script>
        """
        st.markdown(toast_html, unsafe_allow_html=True)
        st.session_state.toast_msg = None


@st.dialog("Confirm Deletion")
def delete_modal(student_name):
    # Modal blocks the main thread to prevent accidental data loss
    st.write(f"Are you sure you want to permanently delete **{student_name}**?")
    st.caption("This action removes all images associated with this student and cannot be undone.")
    
    col1, col2 = st.columns(2)
    if col1.button("Cancel", use_container_width=True):
        st.rerun()
    if col2.button("Confirm Delete", type="primary", use_container_width=True):
        logic.delete_student_record(student_name)
        st.session_state.toast_msg = f"Dataset for {student_name} deleted successfully."
        st.rerun()


# Main app routing
with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0px; color:#f8fafc;'>InClass Alert</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:14px;'>Admin Console • Data Engine</p>", unsafe_allow_html=True)
    st.divider()
    app_mode = st.radio("Navigation", ["Database Dashboard", "Webcam Enrollment", "Dataset Upload"])


render_toast()


if app_mode == "Database Dashboard":
    st.title("Database Dashboard")
    st.caption("Manage registered identities and monitor dataset volume.")
    
    total_students, total_images = logic.get_stats()
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="stats-container">
            <div class="stats-value">{total_students}</div>
            <div class="stats-label">Total Students</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="stats-container">
            <div class="stats-value">{total_images}</div>
            <div class="stats-label">Total Images</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        # Prevent division by zero if the system is fresh
        avg_images = round(total_images / total_students) if total_students > 0 else 0
        st.markdown(f"""
        <div class="stats-container">
            <div class="stats-value">{avg_images}</div>
            <div class="stats-label">Avg Images / Student</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    
    search_query = st.text_input("Search Students", placeholder="Type a name to filter...")
    students = logic.get_students()
    
    if search_query:
        students = [s for s in students if search_query.lower() in s.lower()]
    
    if not students:
        st.info("No records found.")
    else:
        cols = st.columns(4)
        for index, student in enumerate(students):
            img_count = logic.get_image_count(student)
            
            with cols[index % 4]:
                st.markdown(f"""
                <div class="glass-card">
                    <h3 style="margin:0 0 10px 0; font-size:18px;">{student}</h3>
                    <p style="margin:0 0 15px 0; color:#94a3b8; font-size:14px;">Total Files: {img_count}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Delete Record", key=f"del_{student}", use_container_width=True):
                    delete_modal(student)


elif app_mode == "Webcam Enrollment":
    st.title("Webcam Enrollment")
    st.caption("Real-time identity capture system.")
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        student_name = st.text_input("Identity Name", placeholder="Enter full name")
    
    if st.button("Initialize Capture Sequence", type="primary"):
        if student_name.strip() == "":
            st.warning("Identity Name is required.")
        elif logic.check_student_exists(student_name):
            st.warning("Identity already exists in the system.")
        else:
            st.info("System Ready. Please face the camera.")
            progress_bar = st.progress(0, text="Starting capture...")
            frame_window = st.image([])
            
            # The UI blindly consumes the generator from the logic file,
            # decoupling the heavy CV2 processing from the frontend.
            for frame_rgb, current, total in logic.capture_webcam_stream(student_name):
                frame_window.image(frame_rgb)
                progress_bar.progress(current / total, text=f"Processing {current}/{total} frames...")
            
            # Clean up placeholders
            frame_window.empty()
            progress_bar.empty()
            
            st.session_state.toast_msg = f"Identity {student_name} successfully enrolled."
            st.rerun()


elif app_mode == "Dataset Upload":
    st.title("Batch Dataset Upload")
    st.caption("Import existing images into the data engine.")
    st.divider()
    
    student_name_upload = st.text_input("Target Identity Name", placeholder="Enter full name")
    
    uploaded_files = st.file_uploader("Select Media Files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if st.button("Process & Save Batch", type="primary"):
        if student_name_upload.strip() == "":
            st.warning("Target Identity Name is required.")
        elif not uploaded_files:
            st.warning("Please select at least one valid image file.")
        else:
            progress_bar = st.progress(0, text="Initializing batch process...")
            
            # Iterating through the logic generator allows us to update the UI progressively
            # without writing OS/File logic in this file.
            batch_process = logic.process_batch_upload(student_name_upload, uploaded_files)
            
            for current, total in batch_process:
                progress_bar.progress(current / total, text=f"Writing file {current} of {total}...")
                
            progress_bar.empty()
            st.session_state.toast_msg = f"Batch upload complete: {total} images saved for {student_name_upload}."
            st.rerun()