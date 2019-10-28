import logging
import shutil
import time
import numpy as np
import sys

from config_loader import Config


def get_logger(exp):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    format_string = "%(asctime)s — %(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(
        "{}/{}_{}.log".format(Config.get("logDir"), time.strftime("%Y%m%d-%H%M%S"), exp), mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


def build_metrics_filename(exp, g, m, task):
    t = "{}vs{}".format(task[0]["name"], task[1]["name"])
    return "{}_{}_{}_{}_{}_metrics.npy".format(time.strftime("%Y%m%d-%H%M%S"), exp, g, m, t)


def copy_experiment_configuration(gene, mode):
    dest = "experiment_configurations/{}_{}_{}_experiment_configuration.json".format(
        time.strftime("%Y%m%d-%H%M%S"), gene, mode
    )
    shutil.copy("experiment_configurations/experiment.json", dest)


def save_metrics(exp, g, m, task, metrics):
    np.save("logs/{}".format(build_metrics_filename(exp, g, m, task)), metrics)





