import tensorflow as tf
from keras import backend as K


def auprc(y_true, y_pred):
    score, up_opt = tf.metrics.auc(y_true, y_pred, curve="PR", summation_method="careful_interpolation")
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def auroc(y_true, y_pred):
    score, up_opt = tf.metrics.auc(y_true, y_pred, curve='ROC', summation_method="careful_interpolation")
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score
