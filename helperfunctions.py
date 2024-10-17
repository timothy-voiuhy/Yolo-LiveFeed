import requests 
import sys
import random
import logging

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

def getURL(url:str=None, ip:str=None, port:int = None, https = False):
    if url is not None:
        return url
    if https is False:
        console_logger.info("Using http for cam connection: Note that this is not safe")
        return f"http://{ip}:{port}/video"
    else:
        return f"https://{ip}:{port}/video"

def checkConnection(url:str = None):
    if url is not None:
        resp = requests.head(url)
        if resp.ok:
            return True
    else: 
        return False