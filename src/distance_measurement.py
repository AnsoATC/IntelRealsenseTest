# distance_measurement.py
# Script to test Intel RealSense D455 for object detection and distance measurement using YOLO models.

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

# Configuration
MODEL_PATH = "../models/yolo11n.pt"  # Change to "../models/yolov8n.pt" for YOLOv8
CONF_THRESHOLD = 0.5  # Confidence threshold for YOLO detections
WINDOW_NAME = "RealSense D455 - Distance Measurement"

def initialize_realsense():
    """
    Initialize the Intel RealSense D455 camera pipeline for color and depth streams.
    Returns the pipeline and configuration objects.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color and depth streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start pipeline
    profile = pipeline.start(config)
    
    # Get depth scale for distance conversion
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # Create alignment object to align depth to color
    align = rs.align(rs.stream.color)
    
    return pipeline, align, depth_scale

def load_yolo_model(model_path):
    """
    Load the YOLO model from the specified path.
    Returns the YOLO model object.
    """
    return YOLO(model_path)

def calculate_distance(depth_frame, bbox, depth_scale):
    """
    Calculate the median distance to an object within the bounding box.
    Args:
        depth_frame: RealSense depth frame
        bbox: Bounding box coordinates [x_min, y_min, x_max, y_max]
        depth_scale: Depth scale factor for conversion to meters
    Returns:
        Distance in centimeters
    """
    x_min, y_min, x_max, y_max = map(int, bbox)
    
    # Extract depth values within the bounding box
    depth_roi = depth_frame[y_min:y_max, x_min:x_max].astype(float)
    
    # Filter out zero values (invalid depth)
    depth_roi = depth_roi[depth_roi > 0]
    
    if depth_roi.size == 0:
        return None
    
    # Calculate median depth in millimeters and convert to centimeters
    median_depth_mm = np.median(depth_roi)
    distance_cm = median_depth_mm * depth_scale * 100  # mm to cm
    
    return distance_cm

def main():
    """
    Main function to run the RealSense D455 test for object detection and distance measurement.
    """
    # Initialize RealSense pipeline
    pipeline, align, depth_scale = initialize_realsense()
    
    # Load YOLO model
    model = load_yolo_model(MODEL_PATH)
    
    # Initialize FPS calculation
    prev_time = time.time()
    
    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            
            # Align depth frame to color frame
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Run YOLO inference
            results = model(color_image, conf=CONF_THRESHOLD)
            
            # Process detections
            for result in results:
                for box in result.boxes:
                    # Get bounding box coordinates
                    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    label = model.names[cls]
                    
                    # Calculate distance
                    distance_cm = calculate_distance(depth_image, [x_min, y_min, x_max, y_max], depth_scale)
                    
                    # Draw bounding box and label
                    cv2.rectangle(color_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    text = f"{label}: {conf:.2f}"
                    if distance_cm is not None:
                        text += f", {distance_cm:.1f} cm"
                    
                    cv2.putText(color_image, text, (int(x_min), int(y_min) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(color_image, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow(WINDOW_NAME, color_image)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Cleanup
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
