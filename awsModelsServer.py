import asyncio 
import random
import os
import sys
import hashlib
import logging
from ultralytics import YOLO

MODEL_PATH = "/home/"

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

def predict_single_image(image_path, model=None, device=None, model_path=None):
    """Predicts the class of a single crack image.

    Args:
        image_path: Path to the image file.
        model: The trained PyTorch model.
        device: The device to run the prediction on (CPU or GPU).

    Returns:
        The predicted class (0 for no crack, 1 for crack).
    """


async def handleClient(reader:asyncio.StreamReader, writer:asyncio.StreamWriter):
    addr = writer.get_extra_info('peername')
    logging.info(f"Connected to : {addr}")
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
    predicted_class = int(predict_single_image(hashed_image_name))
    writer.write(predicted_class.to_bytes(4, byteorder ="little"))
    await writer.drain()
    os.remove(hashed_image_name)
    writer.close()
    await writer.wait_closed()

async def runServer(address = "127.0.0.1", server_port = 8082):
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