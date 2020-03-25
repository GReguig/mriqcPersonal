import logging
import sys

logger = logging.getLogger("mri_classif")
LOG_FORMAT = '%(asctime)s %(name)s:%(levelname)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    filemode='a',
                    format=LOG_FORMAT,
                    datefmt='%H:%M:%S',
                    level=logging.WARNING)

