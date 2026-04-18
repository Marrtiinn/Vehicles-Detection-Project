import cv2
import numpy as np
from PIL import Image

def draw_boxes(image, detections):
    """
    Draw bounding boxes, class names, and confidences on an image.
    Accepts PIL Image or numpy array. Returns numpy array (RGB).
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()
        
    # Make sure we have a writable copy
    img_np = np.ascontiguousarray(img_np, dtype=np.uint8)
    
    # Adaptive thickness based on image size
    height, width = img_np.shape[:2]
    thickness = max(1, int(min(width, height) / 400))
    font_scale = max(0.5, min(width, height) / 1000)
    
    # Define a clean color palette (RGB)
    colors = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
        (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199)
    ]

    for det in detections:
        box = det["box"]
        conf = det["confidence"]
        cls_id = det["class_id"]
        cls_name = det["class_name"]
        
        x1, y1, x2, y2 = map(int, box)
        color = colors[cls_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, thickness)
        
        # Label text
        label = f"{cls_name} {conf:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw filled rectangle for text background
        # Adjust y1 so text box is above the bounding box if there's room
        text_y_start = max(0, y1 - text_height - baseline - 5)
        text_y_end = text_y_start + text_height + baseline + 5
        
        cv2.rectangle(
            img_np, 
            (x1, text_y_start), 
            (x1 + text_width + 10, text_y_end), 
            color, 
            -1
        )
        
        # Draw text (white)
        text_color = (255, 255, 255)
        
        cv2.putText(
            img_np, 
            label, 
            (x1 + 5, text_y_end - baseline - 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            text_color, 
            thickness, 
            cv2.LINE_AA
        )
        
    return img_np
