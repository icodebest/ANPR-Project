# Import necessary libraries
from ultralytics import YOLO  # Ultralytics YOLO for object detection
import cv2  # OpenCV for image and video processing
import util  # Custom utility functions (assumed to be in the same directory)
from sort.sort import *  # SORT (Simple Online and Realtime Tracking) algorithm
from util import get_car, read_license_plate, write_csv  # Additional utility functions

# Dictionary to store the results for each frame
results = {}

# Initialize SORT tracker
mot_tracker = Sort()

# Load YOLO models for dete cting vehicles and license plates
coco_model = YOLO('yolov8n.pt')  # YOLO model for detecting vehicles
license_plate_detector = YOLO('./models/license_plate_detector.pt')  # YOLO model for detecting license plates

# Load video from file
cap = cv2.VideoCapture('./sample.mp4')

# List of class IDs representing vehicles in COCO dataset
vehicles = [2, 3, 5, 7]

# Read frames from the video
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}  # Initialize results dictionary for the current frame

        # Detect vehicles using the COCO YOLO model
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Update SORT tracker with the detected vehicle bounding boxes
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates using the license plate YOLO model
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to a tracked vehicle using SORT tracker
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop the license plate region from the frame
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Preprocess the license plate image
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read the license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                # If a valid license plate number is detected, store the results
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# Write the results to a CSV file
write_csv(results, './test.csv')
