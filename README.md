# YOLO Object Detection - Streamlit Deployment

A clean, modular, and production-ready project scaffold for running YOLO object detection locally and deploying it on Streamlit Cloud.

## 📁 Folder Structure

```
yolo-terminal-deployment/
│
├── app.py                 # CLI script for running detection locally
├── streamlit_app.py       # Main Streamlit web application
├── requirements.txt       # Python dependencies
├── packages.txt           # OS-level dependencies for Streamlit Cloud (libgl1)
├── README.md              # Project documentation
│
├── model/                 # Contains model weights and labels
│   ├── best.pt            # Your trained YOLOv11 model
│   └── labels.txt         # Class names, one per line
│
├── utils/                 # Core functionality decoupled from UI
│   ├── detector.py        # YOLOModel class for running inference
│   └── visualization.py   # OpenCV functions for drawing bounding boxes
│
└── assets/                # Static assets
    └── demo.png           # Demo image for the Streamlit app
```

## 🚀 How to Run Locally

### 1. Install Dependencies
Make sure you have Python installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Local Testing via CLI
You can test the model without any UI using the provided local testing script:
```bash
python app.py --image assets/demo.png --output output.jpg --conf 0.25
```

### 3. Run the Streamlit UI Locally
Start the web application:
```bash
streamlit run streamlit_app.py
```

## ☁️ Streamlit Cloud Deployment Steps

1. **Push to GitHub**: Commit this entire folder structure to a new public or private GitHub repository.
2. **Login to Streamlit Cloud**: Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. **Deploy App**: Click "New app" and select your repository.
4. **Set Main File path**: Ensure the "Main file path" is set to `streamlit_app.py`.
5. **Deploy**: Click "Deploy!" Streamlit will automatically install `packages.txt` (for OpenCV's libgl1 requirement) and `requirements.txt`. 

## 🧠 Features Included
- **Modular Design**: UI logic is cleanly separated from ML inference and visualization.
- **Auto Device Detection**: Automatically uses GPU if available, or seamlessly falls back to CPU for Streamlit Cloud.
- **Robust Visualization**: Clean colors, adaptive bounding box thickness, and dynamic text scaling.
- **Streamlit Extras**: Loading spinners, confidence threshold slider, and error handling for robust user experience.
