from skopt.space import Categorical, Real, Integer
from skopt.utils import use_named_args

from config_loader import Config

from skopt import gp_minimize
from skopt.callbacks import DeltaYStopper

from src.models import train_bayesian_mlp
from src.utilities import filter_by_tasks, import_epigenetic_dataset, split, conf_to_params


def bayesian_mlp_exp(gene, mode):
    BOconfig = Config.get("bayesianOpt")

    hidden_layers_comb = get_hidden_layers_combinations(
        BOconfig["hiddenLayers"], 3, BOconfig["allowFirstLevelZero"])
    mlp_parameters_space = get_parameters_space(hidden_layers_comb)

    @use_named_args(mlp_parameters_space)
    def fitness_mlp(learning_rate, num_hidden_layer, hidden_layer_choice):

        model, hist = train_bayesian_mlp(X_train_int, y_train_int, (X_val, y_val), features_size,
                                         learning_rate, num_hidden_layer, hidden_layer_choice, hidden_layers_comb)

        val_auprc = hist.history['val_auprc'][-1]
        print()
        print("Validation Loss: {}".format(val_auprc))
        print()
        return -val_auprc

    print()
    print("Importing Epigenetic data...")
    print()
    X, y, features_size = filter_by_tasks(*import_epigenetic_dataset("data", gene), Config.get("task"),
                                          perc=Config.get("samplePerc"))

    print("Datasets length: {}, {}".format(len(X), len(y)))
    print("Features sizes: {}".format(features_size))

    metrics = {'losses': [], 'auprc': [], 'auroc': []}
    delta_stopper = DeltaYStopper(n_best=BOconfig["n_best"], delta=BOconfig["delta"])

    for ext_holdout in range(Config.get("nExternalHoldout")):
        print()
        print("{}/{} EXTERNAL HOLDOUTS".format(ext_holdout, Config.get("nExternalHoldout")))
        print()
        X_train, X_test, y_train, y_test = split(X, y, random_state=42, proportions=None, mode=mode)

        # Internal holdouts
        X_train_int, X_val, y_train_int, y_val = split(X_train, y_train, random_state=42, proportions=None, mode=mode)

        print("Searching Parameters...")
        print()

        min_res = gp_minimize(func=fitness_mlp,
                              dimensions=mlp_parameters_space,
                              acq_func=BOconfig["acq_function"],
                              callback=[delta_stopper],
                              n_calls=BOconfig["nBayesianOptCall"])

        print()
        print("Training with best parameters found: {}".format(min_res.x))
        print()
        print(X_train)

        model, _ = train_bayesian_mlp(X_train, y_train, None, features_size,
                                      min_res.x[0], min_res.x[1], min_res.x[2], hidden_layers_comb)

        eval_score = model.evaluate(X_test, y_test)
        #K.clear_session()

        print("Metrics names: ", model.metrics_names)
        print("Final Scores: ", eval_score)
        metrics['losses'].append(eval_score[0])
        metrics['auprc'].append(eval_score[1])
        metrics['auroc'].append(eval_score[2])

    return metrics


# Alessandro Code
def get_hidden_layers_combinations(hidden_layers, max_level = 3, allow_empty_first_level=True):
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


def get_parameters_space(hidden_layers_comb):
    params = [conf_to_params(conf) for conf in Config.get("bayesianOpt")["hyperparameters"]]
    max = len(hidden_layers_comb[-1])-1
    params.append(Integer(0, max, name="hidden_layer_choice"))

    return params


