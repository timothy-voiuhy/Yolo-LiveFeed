import asyncio 
import random
import os
import sys
import hashlib
import torch
import logging
from ultralytics import YOLO
import argparse
import cv2

LOG_DIR = "/home/ec2-user/AI_ML/LOGS/"
MODEL_PATH = "/home/ec2-user/AI_ML/Yolo-LiveFeed/Models/CrackSegBest.pt"

class awsModelServer:
    def __init__(self) -> None:
        pass

def createLogger(log_level = None, name = None, mode = 'a', log_format = None, is_consoleLogger = False, filename = None):
    if log_level is None:
        log_level = logging.INFO
    if log_format is None:
        log_format = "%(asctime)s -%(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    if is_consoleLogger is False:
        if name is None:
            name = "logger"+str(random.randint(1,100))
    else:
        if name is None:
            name = __name__
    logger = logging.getLogger(name)
    if is_consoleLogger is True:
        file_handler = logging.StreamHandler()
    else:
        if filename is None:
            filename = sys.executable.rstrip(".py")+".log"
        file_handler = logging.FileHandler(filename=filename, mode=mode)
    file_handler.setLevel(log_level)
    formater = logging.Formatter(log_format)
    file_handler.setFormatter(formater)
    logger.addHandler(file_handler)
    logger.setLevel(log_level)
    return logger

console_logger = createLogger(is_consoleLogger=True)
file_logger = createLogger(log_level=logging.INFO, filename=LOG_DIR+"model.log")

def predict_single_image(image_path, model=None, device=None):
    """Predicts the class of a single crack image.

    Args:
        image_path: Path to the image file.
        model: The trained PyTorch model.
        device: The device to run the prediction on (CPU or GPU).

    Returns:
        The predicted class (0 for no crack, 1 for crack).
    """
    image = cv2.imread(image_path)
    try:
        results = model.predict(image)
        for result in results:
            boxes = result.boxes.xyxy # get the boxes coordinates 
            confidences = result.boxes.conf  # get the confidence levels 
            class_ids = result.boxes.cls # get the class ids 
            class_names = result.names 
            
            for box, confidence, class_id  in zip(boxes, confidences, class_ids):
                # capture the corners 
                x1, y1, x2, y2 = map(int, box)
                # rectangle for the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # direction 
                # box_tensor = torch.from_numpy(box)
                # direction = calculateDirection(image=image_tensor, boundingbox=(x1, y1, x2, y2))
                # label = f"Direction: {direction}"

                label = f"Conf: {confidence:.2f} Class: {class_names[class_id.item()]}"
                #place the label onto the image at a spcific point 
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                return image
    except Exception as e:
        file_logger.error(f"Encountered error: {e}")
        return f"error: {e}".encode()

async def handleClient(reader:asyncio.StreamReader, writer:asyncio.StreamWriter):
    addr = writer.get_extra_info('peername')
    file_logger.info(f"Connected to : {addr}")
    # read the data sent by the client
    image_name = "received_image"+str(random.randint(0, 10000))
    hashed_image_name = hashlib.md5(image_name.encode()).hexdigest()+".jpg"; # to make sure the names are different
    with open(hashed_image_name, "wb") as image_file:
        while True:
            data = await reader.read(1024)
            if not data or b"END_OF_IMAGE" in data:
                break # end data stream 
            image_file.write(data[:-len(b"END_OF_IMAGE")])
        logging.info("Image recieved from client")
    image = predict_single_image(hashed_image_name)
    if isinstance(image, bytes):
        writer.write(image)
    else:
        writer.write(image.tobytes( order="little"))
    await writer.drain()
    os.remove(hashed_image_name)
    writer.close()
    await writer.wait_closed()

async def runServer(address = "127.0.0.1", server_port = None):
    try:
        server = await asyncio.start_server(handleClient, address, server_port)
        addr = server.sockets[0].getsockname()
        logging.info(f"Serving Model Server on : {addr}")
        async with server:
            await server.serve_forever()
    except KeyboardInterrupt as e:
        logging.info("Closing server")
        server.close()
        await server.wait_closed()

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p" "--port", type=int, help="the port onto which to run the server. This must be exposed on the ec2 instace")
    return parser
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = YOLO(MODEL_PATH)

    parser = parseArgs()
    args = parser.parse_args()
    if args.port:
        runServer(server_port=args.port)