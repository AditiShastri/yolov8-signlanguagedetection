import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO('best.pt')

# Check if the webcam is accessible
def check_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("ERROR: Could not access the camera.")
        return False
    cap.release()
    return True

# Display live webcam stream and process frames
def webcam_stream():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("ERROR: Failed to open the webcam. Please check your camera settings.")
        return

    st.write("Webcam successfully opened. Starting video stream...")

    # Create an empty frame placeholder to continuously update
    frame_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Warning: Failed to grab frame from the webcam.")
            break

        # Predict with the model
        results = model(frame)

        # Log the results to see its content
        # st.write("YOLO Results:", results)

        # Check if the model made predictions
        if len(results[0].boxes) > 0:  # Checking if there are any predicted boxes in the first result
            # Render results on the frame
            annotated_frame = results[0].plot()

            # Convert BGR (OpenCV format) to RGB (Streamlit format)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display the image in Streamlit
            frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)
        else:
            # In case no predictions are made, display the current frame without annotations
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Check for user exit (pressing "q" will close the stream)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            st.write("Exit key pressed. Closing webcam.")
            break

    cap.release()

# Streamlit app
def main():
    st.title("Sign Language Recognition with YOLO")

    # Troubleshooting webcam
    st.write("Checking webcam status...")
    if check_webcam():
        st.write("Webcam is available and working.")
        webcam_stream()
    else:
        st.error("Unable to access webcam. Please make sure the camera is connected and not being used by another application.")

if __name__ == "__main__":
    main()
