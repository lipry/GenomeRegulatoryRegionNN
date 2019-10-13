import time

from config_loader import Config
import numpy as np

from src.models import train_fixed_cnn
from src.utilities import import_sequence_dataset, filter_by_tasks, split


def fixed_cnn_exp(gene, mode):
    X, y, features_size = filter_by_tasks(*import_sequence_dataset("data", gene), Config.get("task"),
                                          perc=Config.get("samplePerc"))

    metrics = {'losses': [], 'auprc': [], 'auroc': []}
    for ext_holdout in range(Config.get("nExternalHoldout")):
        print()
        print("{}/{} EXTERNAL HOLDOUTS".format(ext_holdout, Config.get("nExternalHoldout")))
        print()
        X_train, X_test, y_train, y_test = split(X, y, random_state=42, proportions=None, mode=mode)

        # Internal holdouts
        X_train_int, X_val, y_train_int, y_val = split(X_train, y_train, random_state=42, proportions=None, mode=mode)

        model, _ = train_fixed_cnn(X_train_int, y_train_int, (X_val, y_val), Config.get("type"))

        #external holdout
        model, history = train_fixed_cnn(X_train_int, y_train_int, None, Config.get("type"))

        eval_score = model.evaluate(X_test, y_test)

        print("Metrics names: ", model.metrics_names)
        print("Final Scores: ", eval_score)
        metrics['losses'].append(eval_score[0])
        metrics['auprc'].append(eval_score[1])
        metrics['auroc'].append(eval_score[2])

    return metrics
