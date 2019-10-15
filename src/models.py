from config_loader import Config
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Flatten
from keras.models import Model
from keras.optimizers import SGD, Nadam
from src.metrics import auprc, auroc


def bayesian_mlp(features_size, hidden_layer_configuration):
    inputs = Input(shape=(features_size,))
    x = inputs
    for neurons in hidden_layer_configuration:
        x = Dense(neurons, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def train_bayesian_mlp(X_train_int, y_train_int, validation_set, features_size,
                       hidden_layers_comb, learning_rate, num_hidden_layer, hidden_layer_choice):
    if num_hidden_layer > 0:
        print("num_hidden_layer: ", num_hidden_layer)
        print("hidden_layer_choice: ", hidden_layer_choice)
        hidden_layer_choice = int(hidden_layer_choice * len(hidden_layers_comb[num_hidden_layer])
                                  / len(hidden_layers_comb[-1]))
        hidden_layer_configuration = hidden_layers_comb[num_hidden_layer][hidden_layer_choice]
    else:
        hidden_layer_configuration = []

    print('Learning rate: ', learning_rate)
    print('number of hidden layers: ', num_hidden_layer)
    # TODO: fix prrint hiddne config
    print('hidden layesr configuration:', hidden_layer_configuration)
    print()

    model = bayesian_mlp(features_size, hidden_layer_configuration)

    sgd_opt = SGD(lr=learning_rate,
                  decay=Config.get('decay'),
                  momentum=Config.get('momentum'),
                  nesterov=Config.get('nesterov'))

    model.compile(loss='binary_crossentropy',
                  optimizer=sgd_opt,
                  metrics=[auprc, auroc])

    es = EarlyStopping(monitor='val_loss', patience=Config.get("ESTestPatience"), min_delta=Config.get("ESTestMinDelta"))

    history = model.fit(x=X_train_int,
                        y=y_train_int,
                        validation_data=validation_set,
                        epochs=Config.get('epochs'),
                        batch_size=128,
                        callbacks=[es],
                        verbose=Config.get("kerasVerbosity"))

    return model, history


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


def train_fixed_cnn(X_train_int, y_train_int, validation_set, type):
    model = fixed_cnn(type)

    nadam_opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)

    model.compile(loss='binary_crossentropy',
                  optimizer=nadam_opt,
                  metrics=[auprc, auroc])

    es = EarlyStopping(monitor='val_loss', patience=Config.get("ESPatience"), min_delta=Config.get("ESMinDelta"))

    history = model.fit(x=X_train_int,
                        y=y_train_int,
                        validation_data=validation_set,
                        epochs=Config.get('epochs'),
                        batch_size=100,
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


def train_bayesian_cnn(X_train_int, y_train_int, validation_set, es,
                       ks1, u2, ks2, d1, d2):

    model = bayesian_cnn(kernel_size_1=ks1,
                         units_2=u2,
                         kernel_size_2=ks2,
                         dense_units_1=d1,
                         dense_units_2=d2)

    nadam_opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)

    model.compile(loss='binary_crossentropy',
                  optimizer=nadam_opt,
                  metrics=[auprc, auroc])

    history = model.fit(x=X_train_int,
                        y=y_train_int,
                        validation_data=validation_set,
                        epochs=Config.get('epochs'),
                        batch_size=1000,
                        callbacks=[es],
                        verbose=Config.get("kerasVerbosity"))

    return model, history


def get_training_function(experiment):
    training_dict = {'bayesianCNN': train_bayesian_cnn,
                     'bayesianMLP': train_bayesian_mlp,
                     'fixedCNN': train_fixed_cnn}

    return training_dict[experiment]
