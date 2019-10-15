import shutil
import time
import numpy as np
from keras.callbacks import EarlyStopping
from skopt import gp_minimize
from skopt.callbacks import DeltaYStopper
from skopt.utils import use_named_args

from config_loader import Config
from src.dataset_utils import get_data, filter_by_tasks, split
from src.models import get_training_function
from src.utilities import get_parameters_space, get_hidden_layers_combinations, build_log_filename

if __name__ == "__main__":
    # TODO: manage multiple gene and mode
    # TODO: manage logging
    experiment = Config.get("experiment")

    gene = Config.get("gene")
    mode = Config.get("mode")

    training_func = get_training_function(experiment)

    if experiment in ['bayesianCNN', 'bayesianMLP']:
        BOconfig = Config.get("bayesianOpt")
        mlp_parameters_space = get_parameters_space(experiment, BOconfig)

    def fitness(*params):
        # Preprocessing parameters for different experiments
        if experiment == "bayesianCNN":
            es = EarlyStopping(monitor='val_loss', patience=Config.get("ESValPatience"),
                               min_delta=Config.get("ESValMinDelta"), baseline=Config.get("ESValThresholdBaseline"))

            model, hist = training_func(X_train_int, y_train_int, (X_val, y_val), es, *params[0])

        else:
            print(*params[0])
            hidden_layers_comb = get_hidden_layers_combinations(
                BOconfig["hiddenLayers"], 3, BOconfig["allowFirstLevelZero"])

            model, hist = training_func(X_train_int, y_train_int, (X_val, y_val), features_size, hidden_layers_comb, *params[0])

        val_auprc = hist.history['val_auprc'][-1]
        print()
        print("Validation Loss: {}".format(val_auprc))
        print()
        return -val_auprc

    print()
    print("Importing Epigenetic data...")
    print()
    X, y, features_size = filter_by_tasks(*get_data(experiment, "data", gene), Config.get("task"),
                                          perc=Config.get("samplePerc"))

    print("Datasets length: {}, {}".format(len(X), len(y)))
    print("Features sizes: {}".format(features_size))

    metrics = {'losses': [], 'auprc': [], 'auroc': []}

    for ext_holdout in range(Config.get("nExternalHoldout")):
        print()
        print("{}/{} EXTERNAL HOLDOUTS".format(ext_holdout, Config.get("nExternalHoldout")))
        print()
        X_train, X_test, y_train, y_test = split(X, y, random_state=42, proportions=None, mode=mode)

        if experiment in ['bayesianCNN', 'bayesianMLP']:
            # Internal holdouts
            X_train_int, X_val, y_train_int, y_val = split(X_train, y_train, random_state=42, proportions=None, mode=mode)

            print("Searching Parameters...")
            print()

            delta_stopper = DeltaYStopper(n_best=BOconfig["n_best"], delta=BOconfig["delta"])
            print(BOconfig["nBayesianOptCall"])
            print(type(BOconfig["nBayesianOptCall"]))
            min_res = gp_minimize(func=fitness,
                                  dimensions=mlp_parameters_space,
                                  acq_func=BOconfig["acq_function"],
                                  callback=[delta_stopper],
                                  n_calls=BOconfig["nBayesianOptCall"])

            print()
            print("Training with best parameters found: {}".format(min_res.x))
            print()

            print("EXPERIMENT: ", experiment)
            if experiment == "bayesianMLP":
                hidden_layers_comb = get_hidden_layers_combinations(
                    BOconfig["hiddenLayers"], 3, BOconfig["allowFirstLevelZero"])
                model, _ = training_func(X_train, y_train, None, features_size, hidden_layers_comb, *min_res.x)
                print(model)

            if experiment == "bayesianCNN":
                es = EarlyStopping(monitor='va_auprc', patience=Config.get("ESTestPatience"),
                                   min_delta=Config.get("ESTestMinDelta"))
                model, _ = training_func(X_train, y_train, None, es, *min_res.x)

        else:
            # fixedCNN need only training
            model, _ = training_func(X_train, y_train, None, Config.get("type"))

        print(model)
        eval_score = model.evaluate(X_test, y_test)
        # K.clear_session()

        print("Metrics names: ", model.metrics_names)
        print("Final Scores: ", eval_score)
        metrics['losses'].append(eval_score[0])
        metrics['auprc'].append(eval_score[1])
        metrics['auroc'].append(eval_score[2])

    # Saving results metrics
    np.save(build_log_filename(), metrics)
    # copying the configuration json file with experiments details
    dest = "experiment_configurations/{}_{}_{}_experiment_configuration.json".format(
        time.strftime("%Y%m%d-%H%M%S"), gene, mode
    )
    shutil.copy("experiment_configurations/experiment.json", dest)