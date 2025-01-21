import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

def load_model():
    """Load YOLOv8 model from best.pt in same directory"""
    try:
        return YOLO('best.pt')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model):
    """Run inference on captured image and return results"""
    results = model.predict(image)
    return results[0]

def draw_boxes(img_array, results):
    """Draw bounding boxes and labels on the image"""
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = results.names[class_id]
        
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2%}"
        cv2.putText(img_array, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return img_array

def show_about_section():
    """Display an about section with app details"""
    st.sidebar.title("About SpeakSigns")
    st.sidebar.info("""
    **SpeakSigns** is a cutting-edge app designed to translate hand signs into text, promoting digital inclusivity for the Deaf and Hard of Hearing community.
    
    **Features**:
    - Real-time hand sign detection using YOLOv8
    - User-friendly mobile interface
    - Supports multiple hand sign classifications
    
    Developed as part of the project on **Advanced Sign Language-to-Text Translation for Digital Inclusivity**.
    """)

def show_permission_info():
    """Display permission information and instructions"""
    st.info("""
    ### 📸 Camera Permission Required
    
    This app requires camera access to detect hand signs. When you click 'Take Photo':
    
    1. Your browser will ask for camera permission.
    2. Click 'Allow' or 'Accept' when prompted.
    3. If denied, manage permissions via the browser settings or refresh the page.
    """)

def main():
    st.set_page_config(
        page_title="SpeakSigns",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    
    st.title("🔤 SpeakSigns: Hand Sign Detection")
    
    # About section in the sidebar
    show_about_section()
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Ensure 'best.pt' is in the same directory as this script.")
        return
    
    # Show camera permission info
    show_permission_info()
    
    # Add a divider
    st.markdown("---")
    
    # Camera input
    st.markdown("### 📷 Take a Photo")
    st.markdown("Position your hand sign in the camera view and take a photo.")
    img_file = st.camera_input("Enable camera", key="camera")
    
    if img_file is not None:
        image = Image.open(img_file)
        
        # Analyze button
        detect_button = st.button("🔍 Analyze Hand Sign")
        
        if detect_button:
            try:
                # Show processing message
                with st.spinner("Analyzing your hand sign..."):
                    results = process_image(image, model)
                    output_image = draw_boxes(image, results)
                
                # Display results
                st.image(output_image, caption="Detected Hand Signs", use_column_width=True)
                
                # Display detection details
                st.subheader("📊 Detection Results:")
                
                if len(results.boxes) == 0:
                    st.warning("No hand signs detected. Please try again with a clearer view.")
                else:
                    detected_signs = []
                    for box in results.boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = results.names[class_id]
                        detected_signs.append((class_name, confidence))
                        
                        # Display each detection in detail
                        st.markdown(f"""
                        <div style='padding: 15px; border-radius: 10px; background-color: #f0f2f6; margin: 10px 0;'>
                            <strong>Sign:</strong> {class_name}<br>
                            <strong>Confidence:</strong> {confidence:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Highlight most confident detection
                    most_confident_sign = max(detected_signs, key=lambda x: x[1])
                    st.success(f"✅ **Detected Sign:** {most_confident_sign[0]} (Confidence: {most_confident_sign[1]:.1%})")
            except Exception as e:
                st.error("❌ Error processing the image.")
                st.error(f"Details: {str(e)}")

if __name__ == "__main__":
    main()
