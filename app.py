import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

def load_model():
    try:
        return YOLO('best.pt')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model):
    results = model.predict(image)
    return results[0]

def draw_boxes(img_array, results):
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

def main():
    st.set_page_config(
        page_title="Hand Sign Detector",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    st.title("Hand Sign Detection")
    
    # Permission guidance at the top
    with st.expander("📸 Camera Access Help", expanded=True):
        st.markdown("""
        ### Enable Your Camera
        
        To use this app, you'll need to:
        
        1. Click 'Enable camera' below
        2. Look for the camera permission popup in your browser
        3. Click 'Allow' to enable camera access
        
        **If you don't see the camera:**
        - Check the camera icon in your browser's address bar
        - Make sure you're using a secure (HTTPS) connection
        - Try refreshing the page
        
        **Troubleshooting:**
        - If you accidentally denied access, click the camera icon in your address bar to reset permissions
        - On mobile, you may need to enable camera access in your phone's settings
        """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Model loading failed. Please try again later.")
        return
    
    # Camera section with clear instructions
    st.markdown("### Take a Photo")
    st.info("👉 Click 'Enable camera' below to start. Your browser will ask for permission.")
    
    # Camera input
    img_file = st.camera_input("Enable camera")
    
    if img_file is None:
        st.markdown("##### Waiting for camera access...")
        st.markdown("If nothing happens, check that you've allowed camera access in your browser.")
    else:
        try:
            image = Image.open(img_file)
            
            # Process image button
            if st.button("🔍 Detect Hand Sign", type="primary"):
                with st.spinner("Processing your image..."):
                    results = process_image(image, model)
                    output_image = draw_boxes(image, results)
                
                st.image(output_image, caption="Detected Hand Signs", use_column_width=True)
                
                if len(results.boxes) == 0:
                    st.warning("No hand signs detected. Try taking another photo with better lighting.")
                else:
                    st.success("Detection complete! See results below:")
                    for box in results.boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = results.names[class_id]
                        
                        st.markdown(f"""
                        <div style='padding: 15px; border-radius: 10px; background-color: #f0f2f6; margin: 10px 0;'>
                            <strong>Sign:</strong> {class_name}<br>
                            <strong>Confidence:</strong> {confidence:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("Want to try again? Just take another photo!")
                
        except Exception as e:
            st.error("Error processing image. Please try again.")
            st.error(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()
