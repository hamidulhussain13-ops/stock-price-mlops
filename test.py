from src.logger import logging
from src.exception import CustomException
import sys

logging.info("Project Started")

try:
    a = 10 / 0

except Exception as e:
    logging.info("Exception occurred")
    raise CustomException(e, sys)