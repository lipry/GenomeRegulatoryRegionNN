from keras.utils import multi_gpu_model
from sklearn.preprocessing import LabelEncoder

from config_loader import Config
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Flatten
from keras.models import Model
from keras.optimizers import SGD, Nadam

from src.dataset_utils import encoding_labels
from src.metrics import auprc, auroc


def bayesian_mlp(features_size, hidden_layer_configuration):
    inputs = Input(shape=(features_size,))
    x = inputs
    for neurons in hidden_layer_configuration:
        x = Dense(neurons, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def train_bayesian_mlp(root_logger, X_train_int, y_train_int, X_val, y_val, features_size,
                       hidden_layers_comb, learning_rate, num_hidden_layer, hidden_layer_choice):
    if num_hidden_layer > 0:
        hidden_layer_choice = int(hidden_layer_choice * len(hidden_layers_comb[num_hidden_layer])
                                  / len(hidden_layers_comb[-1]))
        hidden_layer_configuration = hidden_layers_comb[num_hidden_layer][hidden_layer_choice]
    else:
        hidden_layer_configuration = []

    root_logger.debug("Training with parameters: ")
    root_logger.debug('Learning rate: {}'.format(learning_rate))
    root_logger.debug('number of hidden layers: {}'.format(num_hidden_layer))
    root_logger.debug('hidden layers configuration: {}'.format(hidden_layer_configuration))

    model = bayesian_mlp(features_size, hidden_layer_configuration)
    parallel_model = multi_gpu_model(model, gpus=4)

    sgd_opt = SGD(lr=learning_rate,
                  decay=Config.get('decay'),
                  momentum=Config.get('momentum'),
                  nesterov=Config.get('nesterov'))

    parallel_model.compile(loss='binary_crossentropy',
                           optimizer=sgd_opt,
                           metrics=[auprc, auroc])

    es = EarlyStopping(monitor='val_loss', patience=Config.get("ESTestPatience"), min_delta=Config.get("ESTestMinDelta"))

    validation_set = None
    if X_val is not None and y_val is not None:
        y_val = encoding_labels(y_val)
        validation_set = (X_val, y_val)

    y_train_int = encoding_labels(y_train_int)
    history = parallel_model.fit(x=X_train_int,
                                 y=y_train_int,
                                 validation_data=validation_set,
                                 epochs=Config.get('epochs'),
                                 batch_size=Config.get("batchSize"),
                                 callbacks=[es],
                                 verbose=Config.get("kerasVerbosity"), workers=Config.get("fitWorkers"))

    return parallel_model, history


def fixed_cnn(type='default'):
    if type not in ['default', 'simple']:
        raise ValueError("Type must be either default or simple, you have: {}".format(type))

    inputs = Input(shape=(200, 5))
    x = convolutional1Dstack(64, 4, 'relu', inputs, layers=3)
    x = MaxPooling1D(pool_size=2)(x)
    units = 128
    kernel = 3
    if type is 'simple':
        units = 32
        kernel = 5
    x = convolutional1Dstack(units, kernel, 'relu', x, layers=3)
    x = MaxPooling1D(pool_size=2)(x)
    if type is 'simple':
        units = 16
        kernel = 5
    x = convolutional1Dstack(units, kernel, 'relu', x, layers=3)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)

    return model


def convolutional1Dstack(units, kernel_size, activation, x, layers=3):
    for _ in range(layers):
        x = Conv1D(units, kernel_size=kernel_size, activation=activation)(x)

    return x


def train_fixed_cnn(root_logger, X_train_int, y_train_int, X_val, y_val, type):
    model = fixed_cnn(type)

    #nadam_opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)
    nadam_opt = Nadam(lr="learningRate", beta_1="nadamBeta1", beta_2="nadamBeta2")

    model.compile(loss='binary_crossentropy',
                  optimizer=nadam_opt,
                  metrics=[auprc, auroc])

    es = EarlyStopping(monitor='val_loss', patience=Config.get("ESPatience"), min_delta=Config.get("ESMinDelta"))

    validation_set = None
    if X_val is not None and y_val is not None:
        y_val = encoding_labels(y_val)
        validation_set = (X_val, y_val)

    y_train_int = encoding_labels(y_train_int)
    history = model.fit(x=X_train_int,
                        y=y_train_int,
                        validation_data=validation_set,
                        epochs=Config.get('epochs'),
                        batch_size=Config.get("batchSize"),
                        callbacks=[es],
                        verbose=Config.get("kerasVerbosity"))

    return model, history


def bayesian_cnn(kernel_size_1, units_2, kernel_size_2, dense_units_1, dense_units_2):
    inputs = Input(shape=(200, 5))
    x = Conv1D(64, kernel_size=kernel_size_1, activation='relu')(inputs)
    x = Conv1D(64, kernel_size=kernel_size_1, activation='relu')(x)
    x = Conv1D(64, kernel_size=kernel_size_1, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(units_2, kernel_size=kernel_size_2, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(dense_units_1, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(dense_units_2, activation='relu')(x)
    x = Dropout(0.1)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=predictions)

    return model


def train_bayesian_cnn(root_logger, X_train_int, y_train_int, X_val, y_val, es,
                       ks1, u2, ks2, d1, d2):
    root_logger.debug("Kernel space 1: {}".format(ks1))
    root_logger.debug("Units 2: {}".format(u2))
    root_logger.debug("Kernel space 2: {}".format(ks2))
    root_logger.debug("Dense 1: {}".format(d1))
    root_logger.debug("Dense 2: {}".format(d2))

    model = bayesian_cnn(kernel_size_1=ks1,
                         units_2=u2,
                         kernel_size_2=ks2,
                         dense_units_1=d1,
                         dense_units_2=d2)

    parallel_model = multi_gpu_model(model, gpus=4)

    nadam_opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)

    parallel_model.compile(loss='binary_crossentropy',
                           optimizer=nadam_opt,
                           metrics=[auprc, auroc])

    # Building dataset
    validation_set = None
    if X_val is not None and y_val is not None:
        y_val = encoding_labels(y_val)
        validation_set = (X_val, y_val)

    y_train_int = encoding_labels(y_train_int)
    history = parallel_model.fit(x=X_train_int,
                                 y=y_train_int,
                                 validation_data=validation_set,
                                 epochs=Config.get('epochs'),
                                 batch_size=Config.get("batchSize"),
                                 callbacks=[es],
                                 verbose=Config.get("kerasVerbosity"))

    return parallel_model, history


def get_training_function(experiment):
    training_dict = {'bayesianCNN': train_bayesian_cnn,
                     'bayesianMLP': train_bayesian_mlp,
                     'fixedCNN': train_fixed_cnn}

    return training_dict[experiment]
