import logging
import shutil
import time
import numpy as np
import sys

from config_loader import Config


def get_logger(exp, g, m, task):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
                     "%(lineno)d — %(message)s")
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(
        "{0}/{1}".format(Config.get("logDir"), build_log_filename(exp, g, m, task, metrics=False)), mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


def build_log_filename(exp, g, m, task, metrics=True):
    t = "{}vs{}".format(task[0]["name"], task[1]["name"])
    suffix = "metrics.npy" if metrics else "log.log"
    return "{}_{}_{}_{}_{}_{}".format(time.strftime("%Y%m%d-%H%M%S"), exp, g, m, t, suffix)


def copy_experiment_configuration(gene, mode):
    dest = "experiment_configurations/{}_{}_{}_experiment_configuration.json".format(
        time.strftime("%Y%m%d-%H%M%S"), gene, mode
    )
    shutil.copy("experiment_configurations/experiment.json", dest)


def save_metrics(exp, g, m, task, metrics):
    np.save("logs/{}".format(build_log_filename(exp, g, m, task, metrics=True)), metrics)





