import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2
import fitz  # PyMuPDF

# Download function for the YOLO model
def download_file(file_name, url):
    if not os.path.exists(file_name):
        st.write(f"Downloading {file_name}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, file_name)
            st.success(f"{file_name} downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download {file_name}: {e}")
            return None
    return YOLO(file_name)

# Load the YOLO model
model_path = "yolov10x_best.pt"
model_url = "https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov10x_best.pt"
model = download_file(model_path, model_url)

if model is None:
    st.error("Failed to load the model. Please check your internet connection and try again.")
    st.stop()

# Streamlit app UI
st.markdown("""
    <div style="text-align: center;">
        <h2 style="color: pink; font-size: 18px;">Roushni Sareen</h2>
        <h1 style="color: blue; font-size: 60px;">Document Segmentation using YOLOv10x , and performing OCR and image analysis</h1>
    </div>
""", unsafe_allow_html=True)
st.markdown("<p style='font-size: 30px;'>Upload the image/document here.</p>", unsafe_allow_html=True)

# File uploader (supports image and PDF formats)
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()

    # --- Handling Image Files (jpg, jpeg, png) ---
    if file_type in ["jpg", "jpeg", "png"]:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error opening image file: {e}")
            st.stop()

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        try:
            # Run inference on the image
            results = model(image)
            annotated_image = results[0].plot()
        except Exception as e:
            st.error(f"Error during inference: {e}")
            st.stop()

        # Display the annotated image
        st.image(annotated_image, caption="Detected Objects", use_container_width=True)

        # Convert annotated image for download
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        success, img_encoded = cv2.imencode('.jpg', annotated_image_bgr)
        if success:
            st.download_button(
                label="Download Annotated Image",
                data=img_encoded.tobytes(),
                file_name="annotated_image.jpg",
                mime="image/jpeg"
            )

    # --- Handling PDF Files ---
    elif file_type == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            temp_pdf_path = temp_pdf_file.name
            temp_pdf_file.write(uploaded_file.getvalue())

        try:
            doc = fitz.open(temp_pdf_path)
        except Exception as e:
            st.error(f"Error opening PDF: {e}")
            os.remove(temp_pdf_path)
            st.stop()

        for i in range(doc.page_count):
            try:
                page = doc.load_page(i)
                zoom = 2  # increase resolution if needed
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            except Exception as e:
                st.error(f"Error processing page {i+1}: {e}")
                continue  # Skip to next page

            # Display the original PDF page
            st.image(img, caption=f"Page {i+1}", use_container_width=True)

            try:
                # Run YOLO inference on the page image
                results = model(img)
                annotated_image = results[0].plot()
            except Exception as e:
                st.error(f"Error running inference on page {i+1}: {e}")
                continue

            # Display the annotated page image
            st.image(annotated_image, caption=f"Detected Objects - Page {i+1}", use_container_width=True)

            # Convert image for download
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            success, img_encoded = cv2.imencode('.jpg', annotated_image_bgr)
            if success:
                st.download_button(
                    label=f"Download Annotated Image - Page {i+1}",
                    data=img_encoded.tobytes(),
                    file_name=f"annotated_image_page_{i+1}.jpg",
                    mime="image/jpeg"
                )

        os.remove(temp_pdf_path)
