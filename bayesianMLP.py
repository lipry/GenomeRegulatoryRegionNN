import shutil
import time
import numpy as np
from config_loader import Config
from src.bayesian_cnn_exp import bayesian_cnn_exp
from src.bayesian_mlp_exp import bayesian_mlp_exp
from src.fixed_cnn_exp import fixed_cnn_exp
from src.utilities import build_log_filename

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