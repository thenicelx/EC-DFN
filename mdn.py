import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, LSTM, Dense, TimeDistributed, Concatenate,
                                     Reshape, Lambda, Layer, Multiply, Add, Softmax)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class MultiScaleGCN(Layer):
    def __init__(self, adj_micro, adj_meso, adj_macro, **kwargs):
        super().__init__(**kwargs)
        self.adj_micro_np = np.array(adj_micro)
        self.adj_meso_np = np.array(adj_meso)
        self.adj_macro_np = np.array(adj_macro)
        self.adj_micro_tf_c = tf.constant(self.adj_micro_np, dtype=tf.float32)
        self.adj_meso_tf_c = tf.constant(self.adj_meso_np, dtype=tf.float32)
        self.adj_macro_tf_c = tf.constant(self.adj_macro_np, dtype=tf.float32)

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
        timesteps, n_grids, features = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        reshaped_inputs = tf.reshape(inputs, (-1, n_grids, features))

        micro = tf.matmul(self.adj_micro_tf_c, reshaped_inputs) @ self.W_micro
        meso = tf.matmul(self.adj_meso_tf_c, reshaped_inputs) @ self.W_meso
        macro = tf.matmul(self.adj_macro_tf_c, reshaped_inputs) @ self.W_macro

        output_shape = (-1, timesteps, n_grids, features)
        micro = tf.reshape(micro, output_shape)
        meso = tf.reshape(meso, output_shape)
        macro = tf.reshape(macro, output_shape)
        return [micro, meso, macro]

    def get_config(self):
        config = super().get_config()
        config.update({
            'adj_micro': self.adj_micro_np.tolist(),
            'adj_meso': self.adj_meso_np.tolist(),
            'adj_macro': self.adj_macro_np.tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['adj_micro'] = np.array(config.pop('adj_micro'))
        config['adj_meso'] = np.array(config.pop('adj_meso'))
        config['adj_macro'] = np.array(config.pop('adj_macro'))
        return cls(**config)


class CrossScaleGate(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.W_g = self.add_weight(shape=(int(feature_dim) * 3, 3), initializer='glorot_uniform', name='gate_W_g')
        super().build(input_shape)

    def call(self, inputs):
        micro, meso, macro = inputs
        combined = K.concatenate([micro, meso, macro], axis=-1)
        gates = K.softmax(K.dot(combined, self.W_g), axis=-1)
        gated_micro = Multiply()([Lambda(lambda x: x[..., 0:1])(gates), micro])
        gated_meso = Multiply()([Lambda(lambda x: x[..., 1:2])(gates), meso])
        gated_macro = Multiply()([Lambda(lambda x: x[..., 2:3])(gates), macro])
        return Add()([gated_micro, gated_meso, gated_macro])


def build_full_model(time_steps, n_grids, n_features, adj_matrices, lstm_units, k_mixture):
    """Builds the full model architecture."""
    inputs = Input(shape=(time_steps, n_grids, n_features), name='model_input')
    adj_micro, adj_meso, adj_macro = adj_matrices

    gcn_out_list = MultiScaleGCN(adj_micro, adj_meso, adj_macro, name='MultiScaleGCN')(inputs)

    temporal_feats = []
    for i, feat_tensor in enumerate(gcn_out_list):
        scale_name = ['micro', 'meso', 'macro'][i]
        x = Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1, 3]), name=f'transpose_{scale_name}')(feat_tensor)
        x = TimeDistributed(LSTM(lstm_units, name=f'lstm_{scale_name}'), name=f'timedist_lstm_{scale_name}')(x)
        temporal_feats.append(x)

    fused_feat = CrossScaleGate(name='CrossScaleGate')(temporal_feats)

    mdn_params = Dense(3 * k_mixture, name='mdn_params_dense')(fused_feat)
    mdn_params = Reshape((n_grids, 3 * k_mixture), name='mdn_params_reshape')(mdn_params)

    alpha = Softmax(axis=-1, name='alpha_softmax')(mdn_params[..., :k_mixture])
    mu = mdn_params[..., k_mixture:2 * k_mixture]
    sigma = Lambda(lambda x: tf.exp(x) + K.epsilon(), name='sigma_exp')(mdn_params[..., 2 * k_mixture:])

    final_output = Concatenate(axis=-1, name='final_output')([alpha, mu, sigma])

    return Model(inputs, final_output, name='TrafficPredictorModel')