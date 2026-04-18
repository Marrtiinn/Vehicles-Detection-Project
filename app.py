import os
import argparse
from PIL import Image
from utils.detector import YOLOModel
from utils.visualization import draw_boxes

def main():
    parser = argparse.ArgumentParser(description="Run YOLO object detection locally.")
    parser.add_argument("--image", type=str, default="assets/demo.png", help="Path to input image")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save output image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    # Resolve paths relative to the script directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, args.image) if not os.path.isabs(args.image) else args.image
    output_path = os.path.join(base_dir, args.output) if not os.path.isabs(args.output) else args.output

    print(f"[*] Initializing YOLOModel...")
    try:
        model = YOLOModel()
        print(f"[+] Model loaded successfully. Using device: {model.device}")
    except Exception as e:
        print(f"[-] Failed to load model: {e}")
        return

    print(f"[*] Loading image from {image_path}...")
    if not os.path.exists(image_path):
        print(f"[-] Image not found at {image_path}.")
        return
        
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[-] Failed to open image: {e}")
        return

    print(f"[*] Running inference (threshold: {args.conf})...")
    detections = model.predict(image, conf_threshold=args.conf)
    
    print(f"[+] Inference complete! Found {len(detections)} objects.")
    for i, det in enumerate(detections):
        print(f"    - Object {i+1}: {det['class_name']} (Conf: {det['confidence']:.2f}) at {det['box']}")

    print(f"[*] Drawing bounding boxes...")
    annotated_np = draw_boxes(image, detections)
    annotated_img = Image.fromarray(annotated_np)
    
    print(f"[*] Saving output to {output_path}...")
    annotated_img.save(output_path)
    print(f"[+] Done! Output saved successfully.")

if __name__ == "__main__":
    main()
