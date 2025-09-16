import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import config


def combined_loss(y_true, y_pred):
    alpha, mu, sigma = y_pred[..., :config.K_MIXTURE], y_pred[..., config.K_MIXTURE:2 * config.K_MIXTURE], y_pred[...,
                                                                                                           2 * config.K_MIXTURE:]
    y_true_reshaped = tf.expand_dims(y_true, -1)

    prob = tf.exp(-0.5 * tf.square((y_true_reshaped - mu) / sigma)) / (sigma * tf.sqrt(2 * np.pi))
    mdn_loss_val = -tf.reduce_mean(tf.math.log(tf.reduce_sum(alpha * prob, axis=-1) + K.epsilon()))

    pred_mean = tf.reduce_sum(alpha * mu, axis=-1)
    mse_loss_val = tf.reduce_mean(tf.square(pred_mean - y_true))

    return config.MSE_WEIGHT * mse_loss_val + config.MDN_WEIGHT * mdn_loss_val


def mu_rmse(y_true, y_pred):
    alpha = y_pred[..., :config.K_MIXTURE]
    mu = y_pred[..., config.K_MIXTURE:2 * config.K_MIXTURE]
    pred_mu = K.sum(alpha * mu, axis=-1)
    return K.sqrt(K.mean(K.square(pred_mu - y_true)))


def mae(y_true, y_pred):
    alpha = y_pred[..., :config.K_MIXTURE]
    mu = y_pred[..., config.K_MIXTURE:2 * config.K_MIXTURE]
    pred_mean = tf.reduce_sum(alpha * mu, axis=-1)
    return tf.reduce_mean(tf.abs(pred_mean - y_true))


def r_squared(y_true, y_pred):
    alpha = y_pred[..., :config.K_MIXTURE]
    mu = y_pred[..., config.K_MIXTURE:2 * config.K_MIXTURE]
    pred_mean = tf.reduce_sum(alpha * mu, axis=-1)
    SS_res = tf.reduce_sum(tf.square(y_true - pred_mean))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def smape(y_true, y_pred):
    alpha = y_pred[..., :config.K_MIXTURE]
    mu = y_pred[..., config.K_MIXTURE:2 * config.K_MIXTURE]
    pred_mean = tf.reduce_sum(alpha * mu, axis=-1)
    numerator = tf.abs(pred_mean - y_true)
    denominator = tf.abs(y_true) + tf.abs(pred_mean)
    return tf.reduce_mean(2.0 * numerator / tf.maximum(denominator, K.epsilon())) * 100.0