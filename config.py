
# Data parameters
MAX_TIMESTEPS = 5856
N_GRIDS_DEFAULT = 564

# --- 更新为新的 geo/ 和 features/ 文件夹路径 ---
GRID_COORD_PATH = 'data/geo/grid_coordinates.csv'
FEATURE_FILES = {
    'speed': 'data/features/grid_speed_matrix.csv',
    'start': 'data/features/start_feature_matrix_filtered.csv',
    'end': 'data/features/end_feature_matrix_filtered.csv',
    'pm25': 'data.features/pm25_feature_matrix.csv',
    'aqi': 'data/features/aqi_feature_matrix.csv'
}
# ----------------------------------------------

# Model parameters
LOOK_BACK = 6
LSTM_UNITS = 128
K_MIXTURE = 3

# Graph Builder parameters
DTW_WINDOW = 4 * 24
THETA_MICRO = 0.8
ALPHA_MACRO = 0.85

# Training parameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0013
MSE_WEIGHT = 0.2
MDN_WEIGHT = 0.8
TRAIN_SPLIT = 0.8

# File paths
BEST_VAL_LOSS_MODEL_PATH = 'best_val_loss_model.h5'
BEST_VAL_MAE_MODEL_PATH = 'best_val_mae_model.h5'
FINAL_MODEL_PATH = 'traffic_forecast_model.h5'
METRICS_CSV_PATH = 'all_epochs_validation_metrics.csv'
PREDICTIONS_CSV_PATH = 'best_epoch_predictions.csv'