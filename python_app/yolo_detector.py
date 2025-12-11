import cv2
import numpy as np
from ultralytics import YOLO
import math
import torch

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model and move to appropriate device
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Use half-precision for faster inference
        if self.device == 'cuda':
            self.model.model.half()
            print("✅ Using CUDA with half-precision (FP16)")
        else:
            print("⚠️ CUDA not available, using CPU")
        
        print("YOLO model loaded successfully!")
        

    def bbox_overlap(self, bbox1, bbox2, overlap_threshold=0.15):
        """Check if two bounding boxes overlap significantly"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return False
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou >= overlap_threshold
    
    def detect_objects(self, frame, confidence_threshold=0.20):
        """Detect all objects in frame with class conflict resolution"""
        try:
            results = self.model(frame, conf=confidence_threshold, verbose=False)
            all_detections = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.cpu().numpy()
                    
                    for i in range(len(boxes)):
                        class_id = int(boxes.cls[i])
                        confidence = boxes.conf[i]
                        bbox = boxes.xyxy[i].astype(int)
                        class_name = self.model.names[class_id]
                        
                        # Only detect specific classes
                        if class_name not in ['Monster', 'Farm', 'Human', 'Cursor', 'Portal']:
                            continue
                        
                        x1, y1, x2, y2 = bbox
                        obj_center_x = (x1 + x2) // 2
                        obj_center_y = (y1 + y2) // 2
                        
                        area = (x2 - x1) * (y2 - y1)
                        is_monster = (class_name == 'Monster')
                        is_farm = (class_name == 'Farm')
                        is_cursor = (class_name == 'Cursor')
                        
                        distance_from_center = math.sqrt((obj_center_x - frame.shape[1]//2) ** 2 + 
                                                        (obj_center_y - frame.shape[0]//2) ** 2)
                        
                        all_detections.append({
                            'class_name': class_name,
                            'confidence': confidence,
                            'center': (obj_center_x, obj_center_y),
                            'distance_from_center': distance_from_center,
                            'is_monster': is_monster,
                            'is_farm': is_farm,
                            'is_cursor': is_cursor,
                            'area': area,
                            'bbox': (x1, y1, x2, y2)
                        })
            
            
            return all_detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []