import shutil
import time
import numpy as np
from config_loader import Config
from src.bayesian_cnn_exp import bayesian_cnn_exp
from src.bayesian_mlp_exp import bayesian_mlp_exp
from src.fixed_cnn_exp import fixed_cnn_exp


def build_log_filename():
    exp = Config.get("experiment")
    g = Config.get("gene")
    m = Config.get("mode")
    t = "{}vs{}".format(Config.get("task")[0]["name"], Config.get("task")[1]["name"])
    return "logs/{}_{}_{}_{}_{}_final_metric_log.npy".format(time.strftime("%Y%m%d-%H%M%S"), exp, g, m, t)


exp = {"bMLP": bayesian_mlp_exp, "fixedCNN": fixed_cnn_exp, "bayesianCNN": bayesian_cnn_exp}

gene = Config.get("gene")
mode = Config.get("mode")
experiment = Config.get("experiment")

f = exp[experiment]
metrics = f(gene, mode)


#Saving results metrics
np.save(build_log_filename(), metrics)
# copying the configuration json file with experiments details
dest = "experiment_configurations/{}_{}_{}_experiment_configuration.json".format(
    time.strftime("%Y%m%d-%H%M%S"), gene, mode
)
shutil.copy("experiment_configurations/experiment.json", dest)