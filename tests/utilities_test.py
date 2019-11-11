import pytest
from skopt.space import Real, Integer, Categorical

from src.utilities import conf_to_params, bayesianMLP_param_space


@pytest.mark.parametrize("c, expected", [({"name": "learning_rate", "type": "Real", "low": 0.1, "high": 0.2},
                                Real(low=0.1, high=0.2, prior='uniform', transform='identity')),
                               ({"name": "num_hidden_layer", "type": "Integer", "low": 0, "high": 3},
                                Integer(low=0, high=3)),
                                ({"name": "categorical", "type": "Categorical", "categories": [0, 1, 2]},
                                 Categorical([0, 1, 2], name="categorical"))])
def test_conf_to_params(c, expected):
    assert conf_to_params(c) == expected


# def test_bayesianMLP_param_space():
#     BOconfig = {
#         "nBayesianOptCall": 10,
#         "n_best": 5,
#         "delta": 0.015,
#         "acq_function": "LCB",
#         "hyperparameters": [
#             {"name": "learning_rate", "type": "Real", "low": 0.1, "high": 0.2},
#             {"name": "num_hidden_layer", "type": "Integer", "low": 0, "high": 3}
#         ],
#         "hiddenLayers": [
#             [2, 4, 8, 16, 32, 64, 128, 256],
#             [2, 4, 8, 16, 32, 64, 128],
#             [2, 4, 8, 16, 32, 64]
#         ],
#         "allowFirstLevelZero": true
#     }
#     parameters = bayesianMLP_param_space(BOconfig)
