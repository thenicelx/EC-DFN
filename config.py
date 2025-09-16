import os

DATA_FEATURES_DIR = os.path.join("data", "Q-traffic Events Dataset")
DATA_GEO_DIR = os.path.join("data", "Q-traffic Dataset")

FEATURE_FILES = {
    'speed': os.path.join(DATA_FEATURES_DIR, 'grid_speed_matrix.csv'),
    'start': os.path.join(DATA_FEATURES_DIR, 'start_feature_matrix_filtered.csv'),
    'end': os.path.join(DATA_FEATURES_DIR, 'end_feature_matrix_filtered.csv'),
    'pm25': os.path.join(DATA_FEATURES_DIR, 'pm25_feature_matrix.csv'),
    'aqi': os.path.join(DATA_FEATURES_DIR, 'aqi_feature_matrix.csv')
}
GRID_COORDS_FILE = os.path.join(DATA_GEO_DIR, 'grid_coordinates.csv')

MAX_TIMESTEPS = 5856
N_GRIDS = 564

LOOK_BACK = 6
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 0.001
K_MIXTURE = 3

MSE_WEIGHT = 0.2
MDN_WEIGHT = 0.8

CHECKPOINT_FILEPATH = 'best_model.h5'
FINAL_MODEL_FILEPATH = 'traffic_forecast_model.h5'
BEST_MAE_MODEL_PATH = 'best_val_mae_model.h5'
HISTORY_CSV_PATH = 'all_epochs_validation_metrics.csv'
PREDICTIONS_CSV_PATH = 'best_val_mae_epoch_predictions_original_scale.csv'