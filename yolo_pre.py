import torch 
from ultralytics import YOLO 
import cv2
import threading
import logging
import argparse
from collections import deque
import sys

buffer_size = 20

def getModel(modelPath: str):   
    model = YOLO(model=modelPath)
    return model

def startLiveFeedDetection(__model:YOLO):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open camera")
        sys.exit(1)
    window_name = "live capture with detection mode"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 800)
    # start the live capture in a while loop
    try:
        while True:
            # capture frame by frame from the camera
            ret, frame = cap.read() 
            if not ret:
                logging.warning("Failed to grab frame")
                break
            # perform inference on the frame 
            results = __model.predict(frame)
            for result in results:
                boxes = result.boxes.xyxy # get the boxes coordinates 
                confidences = result.boxes.conf  # get the confidence levels 
                class_ids = result.boxes.cls # get the class ids 
                class_names = result.names 

                for box, confidence, class_id  in zip(boxes, confidences, class_ids):
                    # capture the corners 
                    x1, y1, x2, y2 = map(int, box)
                    # rectangle for the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # label 
                    label = f"Class: {class_names[int(class_id)]}, Confidence: {confidence:.2f}"
                    #place the label onto the image at a spcific point 
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow(window_name, frame)
                    # break the loop if q is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    except Exception as e:
        logging.warning(f"Experienced exception {e}")
        cap.release() 
        cv2.destroyAllWindows()

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelPath", default=None, type=str, help="The path to the model")
    parser.add_argument("--startCap", action="store_true", default=True, help="Start the live capture detection using the yolo model")
    return parser

def startCapture(modelPath):
    model = getModel(modelPath)
    startLiveFeedDetection(model)

if __name__ == "__main__":
    parser = parseArgs()
    args = parser.parse_args()
    if not args.modelPath:
        parser.print_help()
        sys.exit(1)
    else:
        if args.startCap:
            startCapture(args.modelPath)