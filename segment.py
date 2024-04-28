import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

# Load YOLOv8 model
model = YOLO('best.pt')

st.title("Image Segmentation with YOLOv8")
st.subheader("Object Detection")

st.write(
    "This web app demonstrates image segmentation using the YOLOv8 object detection algorithm."
)

def preprocess_image(image_data):
    # Convert BytesIO to PIL Image
    image = Image.open(BytesIO(image_data))
    return image

def perform_segmentation(image_data):
    image = preprocess_image(image_data)

    # Perform segmentation
    results = model(image)

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #     im.show()  # show image

    # # Plot detection results on the original image
    # annotated_image = results.plot()

    return im

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image data
    image_data = uploaded_file.read()

    # Perform segmentation and get the annotated image
    annotated_image = perform_segmentation(image_data)

    # Display the segmented image
    st.image(annotated_image, caption="Segmented Image", use_column_width=True)

# st.sidebar.info("EMBEDDED PROJECT")
