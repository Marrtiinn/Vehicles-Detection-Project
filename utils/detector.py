import os
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

class YOLOModel:
    def __init__(self, model_path="model/best.pt", labels_path="model/labels.txt"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Resolve absolute paths based on the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, model_path)
        self.labels_path = os.path.join(base_dir, labels_path)
        
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}. Error: {e}")
            
        self.class_names = self._load_labels()
        
    def _load_labels(self):
        try:
            with open(self.labels_path, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            # Fallback to model's built-in names if labels.txt is missing
            if hasattr(self.model, 'names'):
                return self.model.names
            return []

    def predict(self, image, conf_threshold=0.25):
        """
        Run inference on an image.
        Accepts PIL Image or numpy array.
        Returns a list of dictionaries with detection info.
        """
        # Run YOLO inference
        results = self.model.predict(image, conf=conf_threshold, device=self.device, verbose=False)
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                # Extract coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class name safely
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                if isinstance(class_name, dict): # Handle Ultralytics .names dict format
                    class_name = self.class_names.get(class_id, f"Class {class_id}")
                
                detections.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                })
                
        return detections
