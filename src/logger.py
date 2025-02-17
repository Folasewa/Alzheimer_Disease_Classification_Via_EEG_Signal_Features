import logging

def setup_logger(name,log_file, level=logging.INFO):
    """Function to setup a logger"""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

if __name__ == "__main__":
    # Example usage:
    logger = setup_logger("logger", "logs.log")
    logger.info("This is an info message")
