import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Path to the pre-trained YOLO model
model_path = 'best.pt'  # Update with your model's path

# Load the YOLO model
model = YOLO(model_path)
st.success("Model loaded successfully!")

# Use the camera input on mobile (or desktop with webcam)
camera_input = st.camera_input("Capture a frame")

if camera_input is not None:
    # If camera is used, process the image
    image = Image.open(camera_input)
    st.image(image, caption="Captured Image", use_column_width=True)
    
    # Run inference on the captured image
    results = model(image)
    st.write("Predictions:", results.pandas().xywh)

else:
    st.warning("Please capture an image using your camera.")
