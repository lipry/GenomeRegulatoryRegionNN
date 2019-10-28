import numpy as np
from keras.callbacks import EarlyStopping
from skopt import gp_minimize
from skopt.callbacks import DeltaYStopper

from config_loader import Config
from src.dataset_utils import get_data, split, encoding_labels
from src.logging_utils import save_metrics, copy_experiment_configuration, get_logger
from src.models import get_training_function
from src.utilities import get_parameters_space, get_hidden_layers_combinations

if __name__ == "__main__":
    # TODO: TESTING LOGGING
    # TODO: fixare full balanced
    experiment = Config.get("experiment")

    root_logger = get_logger(experiment)

    genes = Config.get("gene")
    modes = Config.get("mode")
    tasks = Config.get("task")

    for gene in genes:
        for task in tasks:
            for mode in modes:

                root_logger.debug("EXPERIMENT: {}, GENE: {}, MODE: {}\nTASK: {}".format(experiment, gene, mode, task))

                training_func = get_training_function(experiment)

                if experiment in ['bayesianCNN', 'bayesianMLP']:
                    BOconfig = Config.get("bayesianOpt")
                    mlp_parameters_space = get_parameters_space(experiment, BOconfig)

                def fitness(*params):
                    # Preprocessing parameters for different experiments
                    if experiment == "bayesianCNN":
                        es = EarlyStopping(monitor='val_loss',
                                           patience=Config.get("ESValPatience"),
                                           min_delta=Config.get("ESValMinDelta"),
                                           baseline=Config.get("ESValThresholdBaseline"))

                        model, hist = training_func(root_logger, X_train_int, y_train_int, X_val, y_val, es, *params[0])

                    else:
                        hidden_layers_comb = get_hidden_layers_combinations(
                            BOconfig["hiddenLayers"], 3, BOconfig["allowFirstLevelZero"])

                        model, hist = training_func(root_logger, X_train_int, y_train_int, X_val, y_val, features_size, hidden_layers_comb, *params[0])

                    val_auprc = hist.history['val_auprc'][-1]
                    root_logger.debug("BAYESIAN OPTIMIZER - Validation auprc: {}".format(val_auprc))
                    return -val_auprc

                X, y = get_data(experiment, "data", gene, Config.get("samplePerc"))
                features_size = len(X[0])

                root_logger.debug("Datasets length: {}".format(len(X)))
                root_logger.debug("Features sizes: {}".format(features_size))

                metrics = {'losses': [], 'auprc': [], 'auroc': []}

                for ext_holdout in range(Config.get("nExternalHoldout")):
                    root_logger.debug("{}/{} EXTERNAL HOLDOUTS".format(ext_holdout+1, Config.get("nExternalHoldout")))

                    X_train, X_test, y_train, y_test = split(X, y, task, proportions=np.array([1, 1, 1, 2, 2, 1, 10]),
                                                             mode=mode)
                    root_logger.debug("Train size: {}, Test size: {}".format(len(X_train), len(X_test)))

                    if experiment in ['bayesianCNN', 'bayesianMLP']:
                        # Internal holdouts, internal holdouts is always unbalaced.
                        X_train_int, X_val, y_train_int, y_val = split(X_train, y_train, task, random_state=42,
                                                                       proportions=None, mode='u')
                        root_logger.debug("Internal Train size: {}, Validation size: {}".format(len(X_train_int), len(X_val)))

                        root_logger.debug("BAYESIAN OPTIMIZER - Started to search params")

                        delta_stopper = DeltaYStopper(n_best=BOconfig["n_best"], delta=BOconfig["delta"])
                        min_res = gp_minimize(func=fitness,
                                              dimensions=mlp_parameters_space,
                                              acq_func=BOconfig["acq_function"],
                                              callback=[delta_stopper],
                                              n_calls=BOconfig["nBayesianOptCall"])

                        root_logger.debug("BAYESIAN OPTIMIZER - Best parameters found: {}".format(min_res.x))

                        if experiment == "bayesianMLP":
                            hidden_layers_comb = get_hidden_layers_combinations(
                                BOconfig["hiddenLayers"], 3, BOconfig["allowFirstLevelZero"])
                            model, _ = training_func(root_logger, X_train, y_train, None, None, features_size, hidden_layers_comb, *min_res.x)

                        if experiment == "bayesianCNN":
                            es = EarlyStopping(monitor='va_auprc', patience=Config.get("ESTestPatience"),
                                               min_delta=Config.get("ESTestMinDelta"))
                            model, _ = training_func(root_logger, X_train, y_train, None, None, es, *min_res.x)

                    else:
                        # fixedCNN need only training
                        model, _ = training_func(root_logger, X_train, y_train, None, None, Config.get("type"))

                    y_test = encoding_labels(y_test)
                    eval_score = model.evaluate(X_test, y_test)
                    # K.clear_session()

                    root_logger.debug("Metrics names: {}".format(model.metrics_names))
                    root_logger.debug("Final Scores: {}".format(eval_score))
                    metrics['losses'].append(eval_score[0])
                    metrics['auprc'].append(eval_score[1])
                    metrics['auroc'].append(eval_score[2])

                save_metrics(experiment, gene, mode, task, metrics)
                copy_experiment_configuration(Config.get("gene"), Config.get("mode"))
