import logging
from pathlib import Path


class App_Logger:

    def __init__(self):
        # creates a custom logger
        logger = logging.getLogger()

        # set logging level to our custom logger
        logger.setLevel(logging.DEBUG)

        # create logs folder if it doesn't exist
        Path("../logs").mkdir(parents=True, exist_ok=True)

        # define file handler and style formatter
        file_handler = logging.FileHandler(f"../logs/app.log")
        formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s',
                datefmt="%d-%m-%Y %H:%M:%S")

        # add the file handler and style formatter
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # add the fielhandler to the logger
        logger.addHandler(file_handler)

        self.logger = logger

    def get_app_logger(self) -> logging.Logger:
        return self.logger