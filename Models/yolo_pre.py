from ultralytics import YOLO
import cv2
import logging
import torch 
import random
import argparse
import sys

buffer_size = 20

def trainModel(model_path:str = None,savePath:str = None, epochs = 3,
               imagesize = 640, datasetPath:str = None, dataset:str = None,
               project_dir = None, fraction = 1, freeze = None, verbose= True,
               plots = False):
    """Note that the coco8 is a smaller model compared the 27GB large coco dataset
    Args:
        verbose: Whether to be verbose while trainig
        imagesizes: Image size to use while training
        dataset: dataset name to train on. The following are supported:
            coco
            VOC
            ImageNet
        plots: whether to plot training metrics
        project_dir: Incase of multiple experimentations, this helps organize the directories into projects
        fraction: fraction of the dataset to use, a lower fraction means lower training data incase of resource limited enviroments
    """
    RANDOM_SEED = 42
    # dataset
    __dataset = "coco8.yaml"
    if dataset == "coco":
        __dataset = "coco8.yaml"
    elif dataset == "voc":
        __dataset = "VOC"
    elif dataset == "ImageNet":
        __dataset = "ImageNet"
    # where the model is to saved
    if savePath is None:
        savePath = "yoloCustom"+str(random.randint(0, 100))+".pt"
    # model
    if model_path is None:
        model_path = "yolo11s"
    model = YOLO(model_path)
    device = None
    # device option
    if torch.cuda.is_available():
        device = torch.device("gpu")
    if torch.cuda.device_count() > 1:
        nG = torch.cuda.device_count()
        device = [0, nG]
    # project
    if project_dir is None:
        project_dir = "project"+str(random.randint(1, 1000))
    # train the model
    model.train(data=__dataset,
                epochs=epochs,
                imgsz=imagesize,
                device = device,
                save=True,
                project=project_dir,
                exists_ok=True,
                verbose=verbose,
                seed=RANDOM_SEED,
                fraction = fraction,
                freeze=freeze,
                plots=plots
                ) 
    # save the model
    if not savePath.endswith(".pt"):
        if savePath.endswith("/"):
            savePath = savePath.rstrip("/")
        savePath = savePath+".pt"
    model.save(savePath)

def trackVideo(modelPath:str, filepath:str):
    """
    Args:
        modelPath: path to the yolo model
        filepath: path to the video file
    """
    model = YOLO(modelPath)
    model.track(source=filepath, show=True)

def getModel(modelPath: str):   
    model = YOLO(modelPath)
    return model

def startLiveFeedDetection(__model:YOLO):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open camera")
        sys.exit(1)
    window_name = "live capture with detection mode"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(window_name, 800, 800)
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
    g1= parser.add_argument_group("Training Arguments", "These arguments should be specified if the model is to go into training mode ie if the --train is specified")
    g1.add_argument("--train", action="store_true", help="Train the model :: Note that the model shall also automatically validated")
    g1.add_argument("--validate", action="store_true", default=True, help="Whether or not to validate the model")
    g1.add_argument("--imageSize", type=int, help="Image size to use when training the model")
    g1.add_argument("--epochs", default=3, type=int, help="The number of epochs for which to train the model")
    g1.add_argument("--dataset", type=str, default="coco8.yaml", help="The dataset on which to train the model.\nBy default this is set to coco8.yaml")
    g1.add_argument("--datasetPath", type=str, help="The path to the dataset if any. Other wise coco8 shall be used.")
    g1.add_argument("--dataset", type=str, help="Specify the dataset to use. The following are supported: coco, VOC, ImageNet")
    g1.add_argument("--resume", action="store_true", help="yolo enables interrupted trainings to continue. Specify this arg if you want to resume training your model")

    parser.add_argument("--trackVid", action="store_true", help="do live image segmentation on a video file")
    parser.add_argument("--vidPath", default=None, help="the path to the video file to track incase --trackVid is choosen")
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
        if args.trackVid and args.vidPath:
            trackVideo(args.modelPath, args.vidPath)
