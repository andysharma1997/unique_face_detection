import logging


def get_logger(name):
    log_format = '%(levelname)s : %(asctime)-15s %(filename)s:%(lineno)d %(funcName)-8s --> %(message)s'
    logging.basicConfig(format=log_format)
    logger = logging.getLogger(name)
    logger.setLevel('INFO')
    return logger
