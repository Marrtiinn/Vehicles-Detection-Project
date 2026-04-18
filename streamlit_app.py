import streamlit as st
import os
from PIL import Image
from utils.detector import YOLOModel
from utils.visualization import draw_boxes

# Setup page configuration
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="centered"
)

# Initialize model only once using Streamlit caching
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLOModel()

# Page UI
st.title("🔍 YOLO Object Detection")
st.markdown("Upload an image to detect objects, or use the demo image below.")

# Load the model with a spinner
with st.spinner("Loading model..."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Sidebar for settings
st.sidebar.title("⚙️ Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.01, 
    max_value=1.0, 
    value=0.25, 
    step=0.01
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Note:** Streamlit Cloud deployments run on CPUs by default, "
    "so inference may take a few moments."
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Define path for demo image
# Get the absolute path to the directory containing this script
base_dir = os.path.dirname(os.path.abspath(__file__))
demo_image_path = os.path.join(base_dir, "assets", "demo.png")
image_to_process = None

if uploaded_file is not None:
    try:
        image_to_process = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Invalid image file uploaded. Error: {e}")
else:
    # Use demo image if exists
    if os.path.exists(demo_image_path):
        st.info("Using demo image. Upload an image above to test your own.")
        try:
            image_to_process = Image.open(demo_image_path).convert("RGB")
        except Exception as e:
            st.warning("Failed to load demo image.")
    else:
        st.info("Please upload an image to begin.")

# Detection logic
if image_to_process is not None:
    # Only run inference if user clicks button
    if st.button("🚀 Run Detection", type="primary"):
        with st.spinner("Running inference..."):
            try:
                # 1. Run prediction
                detections = model.predict(image_to_process, conf_threshold=conf_threshold)
                
                # 2. Draw boxes
                annotated_img_np = draw_boxes(image_to_process, detections)
                annotated_img = Image.fromarray(annotated_img_np)
                
                # 3. Display results
                st.success(f"Detected {len(detections)} object(s).")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_to_process, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(annotated_img, caption="Annotated Image", use_container_width=True)
                    
                # 4. Display JSON data in expander
                with st.expander("Show detailed detection data"):
                    st.json(detections)
                    
            except Exception as e:
                st.error(f"An error occurred during inference: {e}")
