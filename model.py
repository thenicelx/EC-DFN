import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, LSTM, Dense, TimeDistributed, Concatenate,
                                     Reshape, Lambda, Layer, Multiply, Add, Softmax)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import config


class MultiScaleGCN(Layer):
    def __init__(self, adj_micro, adj_meso, adj_macro, **kwargs):
        super().__init__(**kwargs)
        self.L_micro_tf = self.calculate_normalized_laplacian(adj_micro)
        self.L_meso_tf = self.calculate_normalized_laplacian(adj_meso)
        self.L_macro_tf = self.calculate_normalized_laplacian(adj_macro)

    def calculate_normalized_laplacian(self, adj_matrix):
        adj_with_self_loops = adj_matrix + np.eye(adj_matrix.shape[0])
        degree_matrix = np.sum(adj_with_self_loops, axis=1)
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(degree_matrix, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        normalized_laplacian = d_mat_inv_sqrt.dot(adj_with_self_loops).dot(d_mat_inv_sqrt)
        return tf.constant(normalized_laplacian, dtype=tf.float32)

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W_micro = self.add_weight(shape=(feature_dim, feature_dim), initializer='glorot_uniform', trainable=True,
                                       name='W_micro')
        self.W_meso = self.add_weight(shape=(feature_dim, feature_dim), initializer='glorot_uniform', trainable=True,
                                      name='W_meso')
        self.W_macro = self.add_weight(shape=(feature_dim, feature_dim), initializer='glorot_uniform', trainable=True,
                                       name='W_macro')
        super().build(input_shape)

    def call(self, inputs):
        support_micro = tf.matmul(inputs, self.W_micro)
        support_meso = tf.matmul(inputs, self.W_meso)
        support_macro = tf.matmul(inputs, self.W_macro)
        output_micro = tf.matmul(self.L_micro_tf, support_micro)
        output_meso = tf.matmul(self.L_meso_tf, support_meso)
        output_macro = tf.matmul(self.L_macro_tf, support_macro)
        return [output_micro, output_meso, output_macro]


class CrossScaleGate(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.W_g = self.add_weight(shape=(int(feature_dim) * 3, 3), initializer='glorot_uniform', name='gate_W_g')
        super().build(input_shape)

    def call(self, inputs):
        micro, meso, macro = inputs
        combined = K.concatenate(inputs, axis=-1)
        gates = K.softmax(K.dot(combined, self.W_g), axis=-1)
        return Add()([
            Multiply()([Lambda(lambda x: x[..., 0:1])(gates), micro]),
            Multiply()([Lambda(lambda x: x[..., 1:2])(gates), meso]),
            Multiply()([Lambda(lambda x: x[..., 2:3])(gates), macro])
        ])


def build_full_model(time_steps, n_grids, n_features, adj_matrices):
    inputs = Input(shape=(time_steps, n_grids, n_features))
    gcn_out_list = MultiScaleGCN(*adj_matrices)(inputs)
    temporal_feats = []

    for feat_tensor in gcn_out_list:
        x_permuted = tf.transpose(feat_tensor, perm=[0, 2, 1, 3])
        x = TimeDistributed(LSTM(128))(x_permuted)
        temporal_feats.append(x)

    fused_feat = CrossScaleGate()(temporal_feats)

    mdn_params = Dense(3 * config.K_MIXTURE)(fused_feat)
    mdn_reshaped = Reshape((n_grids, 3 * config.K_MIXTURE))(mdn_params)

    alpha, mu, sigma = tf.split(mdn_reshaped, 3, axis=-1)
    alpha = Softmax(axis=-1)(alpha)
    sigma = Lambda(lambda x: tf.exp(x) + K.epsilon())(sigma)

    final_output = Concatenate(axis=-1)([alpha, mu, sigma])
    return Model(inputs, final_output)