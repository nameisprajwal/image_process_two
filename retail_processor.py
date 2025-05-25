import argparse
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import easyocr
from collections import Counter
from tqdm import tqdm
from db_handler import DatabaseHandler

class RetailShelfProcessor:
    def __init__(self, use_db=True):
        # Initialize CUDA device if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize YOLOv8 model
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'], gpu=True if self.device == 'cuda' else False)
        
        # Initialize database handler
        self.db_handler = DatabaseHandler() if use_db else None

    def load_image(self, image_path):
        """Load and preprocess image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        return img

    def detect_objects(self, image):
        """Detect objects using YOLOv8."""
        results = self.yolo_model(image, device=self.device)
        return results[0]

    def perform_ocr(self, image, bbox):
        """Perform OCR on a specific region of the image."""
        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2]
        
        # Ensure ROI is not empty
        if roi.size == 0:
            return ""
        
        try:
            text_results = self.reader.readtext(roi)
            return " ".join([text[1] for text in text_results])
        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    def process_image(self, image_path):
        """Main processing pipeline."""
        # Load image
        image = self.load_image(image_path)
        original_image = image.copy()
        
        # Detect objects
        results = self.detect_objects(image)
        
        # Process detections
        products = []
        product_counts = Counter()
        
        # Get all detections
        boxes = results.boxes
        
        for box in tqdm(boxes, desc="Processing detections"):
            # Get coordinates and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            # Perform OCR on the region
            ocr_text = self.perform_ocr(original_image, (x1, y1, x2, y2))
            
            # Create product entry
            product_name = ocr_text if ocr_text else class_name
            product_counts[product_name] += 1
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{product_name}: {conf:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)

        # Prepare output data
        output_data = {
            "products": [
                {"name": name, "count": count}
                for name, count in product_counts.items()
            ]
        }
        
        # Save results
        cv2.imwrite("output.jpg", image)
        with open("results.json", "w") as f:
            json.dump(output_data, f, indent=2)
        
        # Save to database if enabled
        if self.db_handler:
            self.db_handler.save_results(str(image_path), output_data)
            
        return output_data

def main():
    parser = argparse.ArgumentParser(description="Retail Shelf Image Processor")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--no-db", action="store_true", help="Disable database storage")
    args = parser.parse_args()
    
    processor = RetailShelfProcessor(use_db=not args.no_db)
    results = processor.process_image(args.image)
    
    print("\nProcessing complete!")
    print("\nDetected products:")
    for product in results["products"]:
        print(f"- {product['name']}: {product['count']} items")
    print("\nResults saved to results.json")
    print("Annotated image saved to output.jpg")

if __name__ == "__main__":
    main() 