import streamlit as st
import cv2
from PIL import Image
import base64
from ultralytics import YOLO
import tempfile
import os
import fitz
from pdf2image import convert_from_path
import numpy as np
import gdown
import pytesseract
from pytesseract import Output
import supervision as sv
import groq

# Model Download and Loading
MODEL_DIR = 'models'
MODEL_FILENAME = 'yolov10x_best.pt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
FILE_ID = "15YJAUuHYJQlMm0_rjlC-e_VJPmAvjeiE"
FILE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def download_model():
    """Download YOLO model if not present."""
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading YOLOv10x model from Google Drive...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        gdown.download(FILE_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully.")
    return MODEL_PATH

@st.cache_resource
def load_model():
    """Load the YOLO model."""
    model_path = download_model()
    return YOLO(model_path)

@st.cache_resource
def initialize_groq_client():
    """Initialize Groq Client securely."""
    api_key = "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"
    if not api_key:
        st.error("Groq API Key not set!")
        return None
    return groq.Groq(api_key=api_key)

def get_image_description(client, image_path):
    """Use Groq to describe an image."""
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]}
        ],
        model="llama-3.2-11b-vision-preview",
        stream=False,
    )
    return chat_completion.choices[0].message.content

def perform_ocr(image, detections, client):
    """Extract text and images from document sections."""
    section_annotations = {}
    for idx, (box, label) in enumerate(zip(detections.xyxy, detections.class_id)):
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_image = image[y_min:y_max, x_min:x_max]

        section_labels = {
            0: 'Caption', 1: 'Footnote', 2: 'Formula', 3: 'List-item',
            4: 'Page-footer', 5: 'Page-header', 7: 'Section-header',
            8: 'Table', 9: 'Text', 10: 'Title'
        }

        if label in section_labels:
            section_name = section_labels[label]
            ocr_result = pytesseract.image_to_string(cropped_image, config='--psm 6').strip()

            if section_name not in section_annotations:
                section_annotations[section_name] = []
            section_annotations[section_name].append(ocr_result)
        else:
            temp_image_path = f"temp_image_{idx}.png"
            cv2.imwrite(temp_image_path, cropped_image)
            description = get_image_description(client, temp_image_path)
            os.remove(temp_image_path)

            section_annotations.setdefault('Picture', []).append(description)

    return section_annotations

def annotate_image(image, detections):
    """Draw bounding boxes on the image."""
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    return label_annotator.annotate(scene=annotated_image, detections=detections)

def process_image(model, image, client):
    """Run YOLO model and process the image."""
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)
    annotated_image = annotate_image(image, detections)
    return annotated_image, perform_ocr(image, detections, client)

def main():
    st.set_page_config(page_title="YOLOv10x Document Segmentation", layout="wide")
    st.title("📄 Document Segmentation with YOLOv10x")

    uploaded_file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file:
        model = load_model()
        if model is None:
            st.error("Error loading YOLO model!")
            st.stop()

        client = initialize_groq_client()
        if client is None:
            st.error("Groq client not initialized!")
            st.stop()

        file_type = uploaded_file.name.split('.')[-1].lower()

        if file_type in ["jpg", "jpeg", "png"]:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.spinner("Processing..."):
                annotated_image, annotations = process_image(model, image_np, client)

            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_column_width=True)

            for section, texts in annotations.items():
                st.subheader(f"📌 {section}")
                for text in texts:
                    st.markdown(f"- {text}")

            _, img_encoded = cv2.imencode('.jpg', annotated_image)
            st.download_button("Download Annotated Image", img_encoded.tobytes(), "annotated.jpg", "image/jpeg")

        elif file_type == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
                temp_pdf_path = temp_pdf_file.name
                temp_pdf_file.write(uploaded_file.getvalue())

            try:
                pages = convert_from_path(temp_pdf_path, dpi=300)
                for i, page in enumerate(pages, start=1):
                    st.image(page, caption=f"Page {i}", use_container_width=True)
                    page_np = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

                    with st.spinner(f"Processing Page {i}..."):
                        annotated_image, annotations = process_image(model, page_np, client)

                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption=f"Annotated Page {i}", use_column_width=True)

                    for section, texts in annotations.items():
                        st.subheader(f"📌 {section}")
                        for text in texts:
                            st.markdown(f"- {text}")

            except Exception as e:
                st.error(f"Error processing PDF: {e}")
            finally:
                os.remove(temp_pdf_path)

if __name__ == "__main__":
    main()
