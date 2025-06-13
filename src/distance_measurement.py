# distance_measurement.py
# Script to test Intel RealSense D455 for object detection and distance measurement using YOLO models.
# Displays class name and distance with colored backgrounds, bounding box center point, and depth map.

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

# Configuration
MODEL_PATH = "../models/yolo11n.pt"  # Change to "../models/yolov8n.pt" for YOLOv8
CONF_THRESHOLD = 0.5  # Confidence threshold for YOLO detections
WINDOW_NAME_RGB = "RealSense D455 - RGB and Detection"
WINDOW_NAME_DEPTH = "RealSense D455 - Depth Map"

def initialize_realsense():
    """
    Initialize the Intel RealSense D455 camera pipeline for color and depth streams.
    Returns the pipeline, configuration objects, and depth scale.
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

def get_text_background(color_image, text, position, font_scale, font_thickness, bg_color):
    """
    Draw text with a colored background rectangle.
    Args:
        color_image: Image to draw on
        text: Text to display
        position: (x, y) coordinates for text
        font_scale: Font scale for text
        font_thickness: Thickness of text
        bg_color: Background color (BGR)
    """
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    x, y = position
    bg_top_left = (x, y - text_height - 5)
    bg_bottom_right = (x + text_width, y + 5)
    cv2.rectangle(color_image, bg_top_left, bg_bottom_right, bg_color, -1)
    cv2.putText(color_image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

def main():
    """
    Main function to run the RealSense D455 test for object detection and distance measurement.
    Displays bounding box center, large text with class-colored backgrounds, and depth map.
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
            
            # Create colorized depth map
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Run YOLO inference
            results = model(color_image, conf=CONF_THRESHOLD)
            
            # Process detections
            for result in results:
                for box in result.boxes:
                    # Get bounding box coordinates
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].cpu().numpy())
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    label = model.names[cls]
                    
                    # Calculate distance
                    distance_cm = calculate_distance(depth_image, [x_min, y_min, x_max, y_max], depth_scale)
                    
                    # Draw bounding box (green)
                    bbox_color = (0, 255, 0)
                    cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), bbox_color, 2)
                    
                    # Draw center point of bounding box
                    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
                    cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # Prepare text for class and distance
                    class_text = f"{label} ({conf:.2f})"
                    distance_text = f"Distance: {distance_cm:.1f} cm" if distance_cm is not None else "Distance: N/A"
                    
                    # Draw text with background
                    text_y = y_min - 10 if y_min - 40 > 0 else y_min + 40
                    get_text_background(color_image, class_text, (x_min, text_y), 0.8, 2, bbox_color)
                    get_text_background(color_image, distance_text, (x_min, text_y + 25), 0.8, 2, bbox_color)
            
            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            get_text_background(color_image, f"FPS: {fps:.1f}", (10, 30), 1.0, 2, (0, 0, 255))
            
            # Display RGB and depth frames
            cv2.imshow(WINDOW_NAME_RGB, color_image)
            cv2.imshow(WINDOW_NAME_DEPTH, depth_colormap)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Cleanup
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
