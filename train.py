import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import config
from data_loader import load_grid_coordinates, load_feature_data
from graph_builder import MultiScaleGraphBuilder
from model import build_full_model
from metrics import combined_loss, mu_rmse, mae, r_squared, smape

class InverseTransformMAPE(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, scalers_list):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.scalers = scalers_list
        self.n_grids = y_val.shape[1]

    def on_epoch_end(self, epoch, logs=None):
        y_pred_raw = self.model.predict(self.x_val, verbose=0)
        alpha = y_pred_raw[..., :config.K_MIXTURE]
        mu = y_pred_raw[..., config.K_MIXTURE:2*config.K_MIXTURE]
        y_pred_scaled = np.sum(alpha * mu, axis=-1)

        y_true_orig = np.zeros_like(self.y_val)
        y_pred_orig = np.zeros_like(y_pred_scaled)

        for i in range(self.n_grids):
            y_true_orig[:, i] = self.scalers[i].inverse_transform(self.y_val[:, i].reshape(-1, 1)).flatten()
            y_pred_orig[:, i] = self.scalers[i].inverse_transform(y_pred_scaled[:, i].reshape(-1, 1)).flatten()

        non_zero_mask = y_true_orig != 0
        mape_orig = np.mean(
            np.abs((y_true_orig[non_zero_mask] - y_pred_orig[non_zero_mask]) / y_true_orig[non_zero_mask])) * 100
        print(f" — val_mape_orig: {mape_orig:.6f}%", flush=True)

def create_dataset(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back, :, 0])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    grid_df = load_grid_coordinates(config.GRID_COORDS_FILE)
    raw_data = {name: load_feature_data(f, config.MAX_TIMESTEPS, config.N_GRIDS) for name, f in config.FEATURE_FILES.items()}

    scalers, scaled_data = {}, {}
    for name, data in raw_data.items():
        scaled_values = np.zeros_like(data, dtype=np.float32)
        grid_scalers = [StandardScaler().fit(data[:, i].reshape(-1, 1)) for i in range(config.N_GRIDS)]
        for i in range(config.N_GRIDS):
            scaled_values[:, i] = grid_scalers[i].transform(data[:, i].reshape(-1, 1)).flatten()
        scaled_data[name] = scaled_values
        if name == 'speed':
            scalers['speed_scalers_list'] = grid_scalers

    X_stacked_features = np.stack(list(scaled_data.values()), axis=2)
    N_FEATURES = X_stacked_features.shape[2]
    print(f"Number of features N_FEATURES: {N_FEATURES}")

    graph_builder = MultiScaleGraphBuilder(grid_df)
    print("Building graphs...")
    adj_micro = graph_builder.build_micro_graph(scaled_data['speed'])
    adj_meso = graph_builder.build_meso_graph(adj_micro)
    adj_macro = graph_builder.build_macro_graph(adj_meso)
    print("Graphs built.")

    X_data, y_data = create_dataset(X_stacked_features, config.LOOK_BACK)
    split_idx = int(0.8 * len(X_data))
    X_train, y_train = X_data[:split_idx], y_data[:split_idx]
    X_test, y_test = X_data[split_idx:], y_data[split_idx:]

    model = build_full_model(config.LOOK_BACK, config.N_GRIDS, N_FEATURES, (adj_micro, adj_meso, adj_macro))
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
                  loss=combined_loss,
                  metrics=[mu_rmse, mae, r_squared, smape])
    model.summary()

    for path in [config.CHECKPOINT_FILEPATH, config.FINAL_MODEL_FILEPATH, config.BEST_MAE_MODEL_PATH]:
        if os.path.exists(path):
            os.remove(path)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=config.CHECKPOINT_FILEPATH, save_best_only=True,
                                           monitor='val_loss', mode='min', save_weights_only=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=config.BEST_MAE_MODEL_PATH, save_best_only=True,
                                           monitor='val_mae', mode='min', save_weights_only=True),
        InverseTransformMAPE(x_val=X_test, y_val=y_test, scalers_list=scalers['speed_scalers_list'])
    ]

    print("Starting model training...")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                        callbacks=callbacks, verbose=1)
    print("Training finished.")

    if history.history:
        val_metric_keys = [key for key in history.history if key.startswith('val_')]
        if val_metric_keys:
            df_history = pd.DataFrame({'epoch': list(range(1, len(history.history[val_metric_keys[0]]) + 1))})
            for key in val_metric_keys: df_history[key] = history.history[key]
            df_history.to_csv(config.HISTORY_CSV_PATH, index=False)
            print(f"Validation metrics for all epochs saved to {config.HISTORY_CSV_PATH}")

            if 'val_mae' in df_history.columns:
                best_mae_epoch_idx = df_history['val_mae'].idxmin()
                print(f"\nBest performing epoch (lowest val_mae) based on history: {best_mae_epoch_idx + 1}")
                print(df_history.iloc[best_mae_epoch_idx])

                if os.path.exists(config.BEST_MAE_MODEL_PATH):
                    print(f"Loading weights from best val_mae model at {config.BEST_MAE_MODEL_PATH}...")
                    model.load_weights(config.BEST_MAE_MODEL_PATH)
                    print("Predicting on the test set using the best val_mae model...")
                    y_pred_raw_best_mae = model.predict(X_test)
                    alpha = y_pred_raw_best_mae[..., :3]
                    mu = y_pred_raw_best_mae[..., 3:6]
                    pred_mean_scaled = np.sum(alpha * mu, axis=-1)

                    y_test_orig = np.zeros_like(y_test)
                    pred_mean_orig = np.zeros_like(pred_mean_scaled)

                    for i in range(config.N_GRIDS):
                        y_test_orig[:, i] = scalers['speed_scalers_list'][i].inverse_transform(y_test[:, i].reshape(-1, 1)).flatten()
                        pred_mean_orig[:, i] = scalers['speed_scalers_list'][i].inverse_transform(pred_mean_scaled[:, i].reshape(-1, 1)).flatten()

                    df_predictions = pd.DataFrame({
                        'true_value_original': y_test_orig.flatten(),
                        'predicted_value_original': pred_mean_orig.flatten()
                    })
                    df_predictions.to_csv(config.PREDICTIONS_CSV_PATH, index=False)
                    print(f"True and predicted values (original scale) for the best val_mae epoch have been saved to {config.PREDICTIONS_CSV_PATH}.")

    if os.path.exists(config.CHECKPOINT_FILEPATH):
        print(f"Loading weights from best val_loss model at {config.CHECKPOINT_FILEPATH} for final model saving.")
        model.load_weights(config.CHECKPOINT_FILEPATH)
    model.save_weights(config.FINAL_MODEL_FILEPATH)
    print(f"Final model saved to {config.FINAL_MODEL_FILEPATH}.")