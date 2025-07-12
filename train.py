import os
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial

# Import from other modules
import config as cfg
from data_loader import (load_grid_coordinates, load_and_scale_features, create_dataset)
from graph_builder import MultiScaleGraphBuilder
from model import build_full_model
from metrics import combined_loss, mu_rmse, mae, r_squared, smape


def cleanup_files():
    """Removes old model and result files."""
    for f in [cfg.BEST_VAL_LOSS_MODEL_PATH, cfg.BEST_VAL_MAE_MODEL_PATH, cfg.FINAL_MODEL_PATH]:
        if os.path.exists(f):
            print(f"Deleting existing file: {f}")
            os.remove(f)


def main():
    """Main function to run the training pipeline."""
    cleanup_files()

    # 1. Load Data
    grid_df = load_grid_coordinates()
    n_grids = len(grid_df)

    scaled_data, scalers = load_and_scale_features(cfg.FEATURE_FILES, cfg.MAX_TIMESTEPS, n_grids)

    # 2. Build Graphs
    graph_builder = MultiScaleGraphBuilder(grid_df)
    print("Building graphs...")
    adj_micro = graph_builder.build_micro_graph(scaled_data['speed'], cfg.DTW_WINDOW, cfg.THETA_MICRO)
    adj_meso = graph_builder.build_meso_graph(adj_micro)
    adj_macro = graph_builder.build_macro_graph(adj_meso, cfg.ALPHA_MACRO)
    print("Graphs built successfully.")

    # 3. Create Datasets
    feature_list = [scaled_data[key] for key in ['speed', 'start', 'end', 'pm25', 'aqi']]
    x_stacked = np.stack(feature_list, axis=2)
    n_features = x_stacked.shape[2]

    X_data, y_data = create_dataset(x_stacked, cfg.LOOK_BACK)
    split_idx = int(cfg.TRAIN_SPLIT * len(X_data))
    X_train, y_train = X_data[:split_idx], y_data[:split_idx]
    X_test, y_test = X_data[split_idx:], y_data[split_idx:]

    # 4. Build and Compile Model
    model = build_full_model(
        time_steps=cfg.LOOK_BACK,
        n_grids=n_grids,
        n_features=n_features,
        adj_matrices=(adj_micro, adj_meso, adj_macro),
        lstm_units=cfg.LSTM_UNITS,
        k_mixture=cfg.K_MIXTURE
    )

    # Use partial to pass extra arguments to loss and metrics
    loss_fn = partial(combined_loss, k_mixture=cfg.K_MIXTURE, mse_weight=cfg.MSE_WEIGHT, mdn_weight=cfg.MDN_WEIGHT)
    metrics_list = [
        partial(mu_rmse, k_mixture=cfg.K_MIXTURE),
        partial(mae, k_mixture=cfg.K_MIXTURE),
        partial(r_squared, k_mixture=cfg.K_MIXTURE),
        partial(smape, k_mixture=cfg.K_MIXTURE)
    ]
    # Keras requires the name attribute for partial functions
    for m in metrics_list:
        m.__name__ = m.func.__name__

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.LEARNING_RATE),
        loss=loss_fn,
        metrics=metrics_list
    )
    model.summary()

    # 5. Train Model
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=cfg.BEST_VAL_LOSS_MODEL_PATH, save_best_only=True,
                                           monitor='val_loss', mode='min'),
        tf.keras.callbacks.ModelCheckpoint(filepath=cfg.BEST_VAL_MAE_MODEL_PATH, save_best_only=True, monitor='val_mae',
                                           mode='min')
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    print("Training complete.")

    # 6. Evaluate and Save Results
    if history.history:
        df_history = pd.DataFrame(history.history)
        df_history['epoch'] = df_history.index + 1
        df_history.to_csv(cfg.METRICS_CSV_PATH, index=False)
        print(f"Validation metrics saved to {cfg.METRICS_CSV_PATH}")

        if 'val_mae' in df_history.columns and os.path.exists(cfg.BEST_VAL_MAE_MODEL_PATH):
            best_epoch_idx = df_history['val_mae'].idxmin()
            print(f"\nBest epoch (lowest val_mae): {best_epoch_idx + 1}")
            print(df_history.iloc[best_epoch_idx])

            model.load_weights(cfg.BEST_VAL_MAE_MODEL_PATH)
            y_pred_raw = model.predict(X_test)
            pred_mean_scaled = np.sum(
                y_pred_raw[..., :cfg.K_MIXTURE] * y_pred_raw[..., cfg.K_MIXTURE:2 * cfg.K_MIXTURE], axis=-1)

            # Inverse transform
            speed_scalers = scalers.get('speed_scalers_list', [])
            y_test_orig = np.zeros_like(y_test)
            pred_mean_orig = np.zeros_like(pred_mean_scaled)

            for i in range(n_grids):
                y_test_orig[:, i] = speed_scalers[i].inverse_transform(y_test[:, i].reshape(-1, 1)).flatten()
                pred_mean_orig[:, i] = speed_scalers[i].inverse_transform(
                    pred_mean_scaled[:, i].reshape(-1, 1)).flatten()

            df_preds = pd.DataFrame({
                'true_value': y_test_orig.flatten(),
                'predicted_value': pred_mean_orig.flatten()
            })
            df_preds.to_csv(cfg.PREDICTIONS_CSV_PATH, index=False)
            print(f"Predictions from best epoch saved to {cfg.PREDICTIONS_CSV_PATH}")

    model.save(cfg.FINAL_MODEL_PATH)
    print(f"Final model saved to {cfg.FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()