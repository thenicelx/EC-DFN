def combined_loss(y_true, y_pred, K_mixture=3, mse_weight=λ, mdn_weight=1-λ):

    #Combined loss function: It combines MSE loss and MDN loss to balance prediction accuracy and
    #uncertainty quantification capability
    # - mse_weight: score of MSE loss (λ)
    # - mdn_weight: score of MDN loss (1- λ)

    batch_size = tf.shape(y_pred)[0]
    n_grids = tf.shape(y_pred)[1]
    alpha = y_pred[..., :K_mixture]
    mu = y_pred[..., K_mixture:2 * K_mixture]
    sigma = y_pred[..., 2 * K_mixture:]
    y_true = K.reshape(y_true, (batch_size, n_grids, 1))
    sigma = K.clip(sigma, K.epsilon(), 1e6)
    pi = tf.constant(np.pi, dtype=tf.float32)
    prob = K.exp(-0.5 * K.square((y_true - mu) / sigma)) / (sigma * K.sqrt(2 * pi))

    #Calculate MDN loss (Negative Log-Likelihood Loss)
    mdn_loss = -K.mean(K.log(K.sum(alpha * prob, axis=-1) + K.epsilon()))

    #Calculate MSE loss
    pred_mean = K.sum(alpha * mu, axis=-1)
    mse_loss = K.mean(K.square(pred_mean - K.squeeze(y_true, axis=-1)))

    return mse_weight * mse_loss + mdn_weight * mdn_loss