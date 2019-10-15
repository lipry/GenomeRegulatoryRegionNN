import time
from skopt.space import Categorical, Real, Integer
from config_loader import Config


def conf_to_params(c):
    def build_real(x):
        return Real(x["low"], x["high"], name=x["name"])

    def build_integer(x):
        return Integer(x["low"], x["high"], name=x["name"])

    def build_categorical(x):
        return Categorical(x["categories"], name=x["name"])

    m = {'Real': build_real, 'Integer': build_integer, 'Categorical': build_categorical}

    return m[c["type"]](c)


def build_log_filename():
    exp = Config.get("experiment")
    g = Config.get("gene")
    m = Config.get("mode")
    t = "{}vs{}".format(Config.get("task")[0]["name"], Config.get("task")[1]["name"])
    return "logs/{}_{}_{}_{}_{}_final_metric_log.npy".format(time.strftime("%Y%m%d-%H%M%S"), exp, g, m, t)


# Alessandro Code
def get_hidden_layers_combinations(hidden_layers, max_level=3, allow_empty_first_level=True):
    hiddenLayersList = []
    # First round manually...
    h1 = [[i] for i in hidden_layers[0]]
    hiddenLayersList.append(h1)
    # ...and from the second iteratively
    for i in range(1, max_level):
        tempList = []
        for k in hiddenLayersList[-1]:
            for j in hidden_layers[i]:
                if k[-1] > j:
                    tempitem = list(k)
                    tempitem.append(j)
                    tempList.append(tempitem)
        hiddenLayersList.append(tempList)
    # Add level [] if requested
    if allow_empty_first_level:
        hiddenLayersList.insert(0, [])
    # Sort the list according to the total number of neurons in the entire net
    tempLL = []
    for ll in hiddenLayersList:
        tempLL.append(sorted(ll, key=lambda x: sum(x)))

    return tempLL


def get_parameters_space(experiment, BOconfig):
    d = {'bayesianMLP': bayesianMLP_param_space, 'bayesianCNN': bayesianCNN_param_space}

    return d[experiment](BOconfig)


def bayesianMLP_param_space(BOconfig):
    hidden_layers_comb = get_hidden_layers_combinations(
        BOconfig["hiddenLayers"], 3, BOconfig["allowFirstLevelZero"])
    params = [conf_to_params(conf) for conf in BOconfig["hyperparameters"]]
    m = len(hidden_layers_comb[-1]) - 1
    params.append(Integer(0, m, name="hidden_layer_choice"))

    return params


def bayesianCNN_param_space(BOconfig):
    return [conf_to_params(conf) for conf in BOconfig["hyperparameters"]]
