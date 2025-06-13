IntelRealsenseTest
A Python project to test functionalities of the Intel RealSense D455 camera on Ubuntu 22.04. This project is designed for the Teknofest 2025 competition, focusing on autonomous navigation tasks.
Project Structure
IntelRealsenseTest/
├── models/                 # Pretrained YOLO models
├── src/                    # Source code for tests
│   ├── distance_measurement.py
│   └── __init__.py
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
└── .gitignore              # Git ignore file

Setup Instructions

Clone the repository:
git clone <repository-url>
cd IntelRealsenseTest


Install dependencies:
pip install -r requirements.txt


Place YOLO models:

Download yolo11n.pt and yolov8n.pt from the Ultralytics YOLO repository.
Place them in the models/ directory.


Run the distance measurement test:
python src/distance_measurement.py



Usage

The distance_measurement.py script:
Streams video from the RealSense D455.
Detects objects using YOLO (configurable between yolo11n.pt and yolov8n.pt).
Calculates the median distance to detected objects in centimeters.
Displays bounding boxes, distances, and FPS.


To switch YOLO models, modify MODEL_PATH in distance_measurement.py.

Requirements

Intel RealSense D455 camera
Ubuntu 22.04
Python 3.8+
RealSense SDK (pyrealsense2)

Future Tests

Depth map visualization
Additional RealSense features (e.g., point cloud, IMU)

License
MIT License

