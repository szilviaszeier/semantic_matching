import logging


def setup_loging_config(filename=None, level=logging.INFO,
                        format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S'):
    logging.basicConfig(
        filename=filename,
        filemode='w',
        level=level,
        format=format,
        datefmt=datefmt)
