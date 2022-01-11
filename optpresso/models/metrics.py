import tensorflow as tf
from tensorflow.keras import backend as K


def pull_error(y_true):
    """pull_error should be changed as more data is collected on
    the variance of shot pull times. Expected to return the empirical error.

    Approximate function for std dev of pull time based on actual time
    Shorter shots have less error inherently, longer shots have greater error
    Err = .1x, where x is the pull time, refer to notebook for std dev graph
    """
    return 0.1019 * y_true - 0.1649


def adjusted_mse(y_true, y_pred):
    diff = tf.square(y_true - y_pred)

    y_sq_err = tf.square(pull_error(y_true))
    diff.shape.assert_is_compatible_with(y_sq_err.shape)
    # Clear everything below the square err of the corresponding pull time
    output = tf.cast(diff > y_sq_err, diff.dtype) * diff
    return tf.reduce_mean(output, axis=-1)


def psuedo_huber_loss(y_true, y_pred):
    """Modified Huber Loss, thanks to @maxentile for this!"""
    fall_off = 3
    diff = K.abs(y_true - y_pred)
    return fall_off * (K.sqrt(1 + (diff / fall_off) ** 2) - 1)


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)
