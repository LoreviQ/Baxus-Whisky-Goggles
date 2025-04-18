import logging

def setup_logger(verbose: bool):
    logger = logging.getLogger("BaxusLogger")
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.WARNING)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger