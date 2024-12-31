import os
import logging
from datetime import datetime

def get_logger(args):
    log_path = args.save_dir
    
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S_")

    log_name = os.path.join(log_path, formatted_time+'log.txt')

    logger = logging.getLogger('RawVideo')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_name)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)


    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
