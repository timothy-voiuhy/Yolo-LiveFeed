import cv2.mat_wrapper
from ultralytics import YOLO
import cv2
import logging
import torch 
import random
from helperfunctions import getURL, createLogger, checkConnection
import argparse
import queue
import sys
import requests
import time
import multiprocessing
import threading
from PIL import Image

console_logger = createLogger(is_consoleLogger=True)
buffer_size = 50

processManager= multiprocessing.Manager()
frameQueue = processManager.Queue(maxsize= buffer_size)

def calculateDistance():
    """Here a model that has been trained on perspective data can be used
    Or normal mathematical algorithms that can calculate perspective can be used"""

def calculateDirection(frame:torch.Tensor, boundingbox:tuple = None):
    """The frame(image) is to be split into 9 portions ie
    Center(leftC, RightC, UpC, DownC) LeftCorner(LCUp, LCDown) 
    RightCorner(RCUp, RCDown), then depending on where the greatest percentange of the
    bounding box falls, then there that direction we shall take"""
    # get the width and length of the image
    if boundingbox:
        bbx1, bby1, bbx3, bby3 = boundingbox

    # for this task we do not need the 3 color channels, i just need a single color_channel
    frame_RC = frame[0] # the R out of the RGB
    frame_RC_shape = frame_RC.shape
    frame_len, frame_width = int(frame_RC_shape[0]),int(frame_RC_shape[1])

    y_divs = []
    y_divisor = frame_len//3
    y_max = y_divisor*3
    x_divs = []
    x_direct_positions = ["L", "C", "R"]
    y_direct_positions = ["U", "C", "D"]
    x_divisor = frame_width//3
    x_max = x_divisor*3
    y_count = x_count = 0
    for __ in range(4):
        y_divs.append(y_count)
        x_divs.append(x_count)
        y_count += y_divisor
        x_count += x_divisor
    direction_box_coordinates = []
    for y_div in y_divs:
        for x_div in x_divs:
            direction_box_coordinates.append((x_div, y_div))
    divs= []
    X1 =  X2  = X3 = X4 = Y1 = Y2 = Y3 = Y4 = 0
    # todo: notice here that this data can be collected each and a new ai model trained to automatically the direction
    for coordinate in direction_box_coordinates:
        X1, Y1 = coordinate
        div = []
        div_dict = {"div": "",  "Pos":""}
        if X1 < x_max and Y1 < y_max:
            P1 = coordinate
            Y4 =  Y1 + y_divisor
            P4 = (X1, Y4)
            X2 = X1 + x_divisor
            P2 = (X2, Y1)
            P3 = (X2, Y4)
            div.append(P1)
            div.append(P2)
            div.append(P3)
            div.append(P4)
            div_dict["div"] = div
            Dpos = []
            for x_div, x_dPos in zip(x_divs, x_direct_positions):
                if x_div == X1:
                    Dpos.append(x_dPos)
            for y_div, y_dPos in zip(y_divs, y_direct_positions):
                if y_div == Y1:
                    Dpos.append(y_dPos)
            div_dict["Pos"] = str(Dpos[0])+str(Dpos[1])
            divs.append(div_dict)
    # in this method we find the divs of interest and find where the greatest area falls and that is the div which we want
    # DSOI = [] #divs of interest
    # for __div_dict in divs:
    #     div = __div_dict["div"]
    #     if div[0][0] < bbx1 and div[1][0] > bbx1: # the bounding box falls in the range of the div's x values

    # an easier method is to just find the center of the bounding box and look for the div where it falls because the div that has the greatest areas still has the center of the bounding box
    # bbCx, bbCy = bbx3//2, bby3//2    
    bbCx = bbx1+((bbx3-bbx1)//2)
    bbCy = bby1+((bby3-bby1)//1)
    for __div_dict in divs:
        div = __div_dict["div"]
        if div[0][0] < bbCx and div[1][0] > bbCx:
            if div[0][1] < bbCy and div[3][1] > bbCy:
                DOI = __div_dict 
                return DOI["Pos"]

def getPerspectiveRatio():
    """"""

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
    metrics =  model.val() # validate the model
    print(metrics.box.map)
    # print(metrics.box.map50)
    # print(metrics.box.map75)
    # print(metrics.box.maps)

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
    sys.exit(1)

def getModel(modelPath: str):   
    model = YOLO(modelPath)
    return model

def displayImsInThread(sharedQueue:queue.Queue):
    while True:
        # if sharedQueue.qsize() > buffer_size -10:
        frame = frameQueue.get()
        cv2.imshow("Live queue Cap", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

def startLiveFeedDetection(__model:YOLO, url = None, queueMaxSize = 20):
    # displayThread = threading.Thread(target=displayImsInThread)
    # displayProcess = multiprocessing.Process(target=displayImsInThread, args=(frameQueue, ))
    # displayProcess.start()
    if url is not None:
        if checkConnection(url):
            cap = cv2.VideoCapture(url)
        else:
            logging.error("Failed to reach out to the remote server. Cannot continue")
            sys.exit()
    else:
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
            # if frameQueue.qsize() > buffer_size-10:
            #     displayProcess.start()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2,0,1)
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
                    # direction 
                    # box_tensor = torch.from_numpy(box)
                    direction = calculateDirection(frame=frame_tensor, boundingbox=(x1, y1, x2, y2))
                    # label = f"Direction: {direction}"
                    
                    label = f"Conf: {confidence:.2f} Class: {class_names[class_id.item()]}"
                    #place the label onto the image at a spcific point 
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow(window_name, frame)
                    # break the loop if q is pressed
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                    # frameQueue.put(frame)
    except Exception as e:
        logging.warning(f"Experienced exception {e}")
        cap.release() 
        cv2.destroyAllWindows()

def parseArgs():

    parser = argparse.ArgumentParser()
    g1= parser.add_argument_group("Training Arguments", "These arguments should be specified if the model is to go into training mode ie if the --train is specified")
    g1.add_argument("--train", action="store_true", help="Train the model :: Note that the model shall also automatically validated")
    g1.add_argument("--validate", action="store_true", default=True, help="Whether or not to validate the model")
    g1.add_argument("--imageSize", type=int, default=640, help="Image size to use when training the model")
    g1.add_argument("--epochs", default=3, type=int, help="The number of epochs for which to train the model")
    g1.add_argument("--dataset", type=str, default="coco8.yaml", help="The dataset on which to train the model.\nBy default this is set to coco8.yaml The following are supported: coco, VOC, ImageNet")
    g1.add_argument("--datasetPath", type=str, help="The path to the dataset if any. Other wise coco8 shall be used.")
    g1.add_argument("--resume", action="store_true", help="yolo enables interrupted trainings to continue. Specify this arg if you want to resume training your model")

    g2 = parser.add_argument_group("Phone camera live feed", "These arguments are provided when you are going to use a live feed from a url on a remote device forexample a phone")
    g2.add_argument("--https", action="store_true", default=False, help="Whether to use https for a more secure connection")
    g2.add_argument("-u", "--url", type=str, default=None, help="the url from which to get the live feed")
    g2.add_argument("-i", "--ip", type=str, default=None, help="Ip from which to get the live camera feed")
    g2.add_argument("-p", "--port", type=int, default=8080, help="the port that the live feed is being served on the remote server")

    parser.add_argument("--trackVid", action="store_true", help="do live image segmentation on a video file")
    parser.add_argument("--vidPath", default=None, help="the path to the video file to track incase --trackVid is choosen")
    parser.add_argument("--modelPath", default=None, type=str, help="The path to the model")
    parser.add_argument("--startCap", action="store_true", default=False, help="Start the live capture detection using the yolo model")
    return parser

def startCapture(modelPath, url = None, ip = None, port = None, https = None):
    if ip is not None:
        url = getURL(ip=ip, port=port)
    elif url is not None:
        url = getURL(url)
    model = getModel(modelPath)
    startLiveFeedDetection(model, url = url)

if __name__ == "__main__":
    parser = parseArgs()
    args = parser.parse_args()
    if not args.modelPath:
        parser.print_help()
        sys.exit(1)
    else:
        if args.startCap:
            startCapture(args.modelPath, ip = args.ip, port = args.port, https = args.https)
        if args.trackVid and args.vidPath:
            trackVideo(args.modelPath, args.vidPath)
    if args.train:
        trainModel(imagesize=args.imagesize, epochs=args.epochs)