## YoloLiveFeed
In this simple python program, i use the yolo pretrained model and i design for it an cv2 interface to get camera input and then the frames are each passed through the model for prediction after which the bounding boxes are placed onto the frame using the cv2 api.
The output is the then show usig cv2.imshow(window_name, frame) and since we are in a while loop, a live video feed it obtained with the model predictions emebedded into the frames.

## Results
Example of an image passed through the yolov8 model
![Alt text](https://github.com/timothy-voiuhy/Yolo-LiveFeed/blob/main/tmppulkcjx0.PNG)