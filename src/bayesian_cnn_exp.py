from keras.callbacks import EarlyStopping
from skopt import gp_minimize
from skopt.callbacks import DeltaYStopper
from skopt.utils import use_named_args

from config_loader import Config
from src.models import train_bayesian_cnn
from src.utilities import split, filter_by_tasks, import_sequence_dataset, conf_to_params


def bayesian_cnn_exp(gene, mode):
    BOconfig = Config.get("bayesianOpt")
    mlp_parameters_space = [conf_to_params(conf) for conf in Config.get("bayesianOpt")["hyperparameters"]]
    #mlp_parameters_space = [Categorical([5,10], name="kernel_space_1"),
    #                        Categorical([32, 64], name="units_2"),
    #                        Categorical([5, 10], name="kernel_space_2"),
    #                        Categorical([32, 64], name="dense_1"),
    #                        Categorical([32, 64], name="dense_2")]

    @use_named_args(mlp_parameters_space)
    def fitness_mlp(kernel_space_1, units_2, kernel_space_2, dense_1, dense_2):
        es = EarlyStopping(monitor='val_loss', patience=Config.get("ESValPatience"),
                           min_delta=Config.get("ESValMinDelta"), baseline=0.2)

        model, hist = train_bayesian_cnn(X_train_int, y_train_int, (X_val, y_val), es,
                                         kernel_space_1, units_2, kernel_space_2, dense_1, dense_2)

        val_auprc = hist.history['val_auprc'][-1]
        print()
        print("Validation Loss: {}".format(val_auprc))
        print()
        return -val_auprc

    print()
    print("Importing Epigenetic data...")
    print()
    X, y, features_size = filter_by_tasks(*import_sequence_dataset("data", gene), Config.get("task"),
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

        es = EarlyStopping(monitor='val_loss', patience=Config.get("ESPatience"), min_delta=Config.get("ESMinDelta"))
        model, _ = train_bayesian_cnn(X_train, y_train, None, es,
                                      min_res.x[0], min_res.x[1], min_res.x[2], min_res.x[3])

        eval_score = model.evaluate(X_test, y_test)
        # K.clear_session()

        print("Metrics names: ", model.metrics_names)
        print("Final Scores: ", eval_score)
        metrics['losses'].append(eval_score[0])
        metrics['auprc'].append(eval_score[1])
        metrics['auroc'].append(eval_score[2])

    return metrics