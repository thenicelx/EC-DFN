import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_grid_coordinates(file_path='data/Q-traffic Dataset/grid_coordinates.csv'):
    grid_df = pd.read_csv(file_path, sep='\t')
    grid_df['center_lon'] = (grid_df['min_lon'] + grid_df['max_lon']) / 2
    grid_df['center_lat'] = (grid_df['min_lat'] + grid_df['max_lat']) / 2
    return grid_df.sort_values('grid_id').reset_index(drop=True)

def load_feature_data(file_path, max_timesteps, n_grids):
    df = pd.read_csv(file_path, skiprows=1, header=None)
    data = df.values[:max_timesteps, :n_grids]
    return pd.DataFrame(data).ffill().bfill().values

def load_and_scale_features(feature_files, max_timesteps, n_grids):
    raw_data = {name: load_feature_data(path, max_timesteps=max_timesteps, n_grids=n_grids)
                for name, path in feature_files.items()}

    scalers = {}
    scaled_data = {}
    for name in raw_data.keys():
        scaled_values = np.zeros_like(raw_data[name], dtype=np.float32)
        grid_scalers_for_feature = []
        for i in range(n_grids):
            grid_scaler = StandardScaler()
            scaled_values[:, i] = grid_scaler.fit_transform(raw_data[name][:, i].reshape(-1, 1)).flatten()
            grid_scalers_for_feature.append(grid_scaler)
        scaled_data[name] = scaled_values
        if name == 'speed':
            scalers['speed_scalers_list'] = grid_scalers_for_feature

    return scaled_data, scalers

def create_dataset(data_cube, look_back_steps):
    X_s, y_s = [], []
    if len(data_cube) >= look_back_steps + 1:
        for i in range(len(data_cube) - look_back_steps):
            X_s.append(data_cube[i:i + look_back_steps])
            y_s.append(data_cube[i + look_back_steps, :, 0]) # Target is the speed feature
    return np.array(X_s), np.array(y_s)