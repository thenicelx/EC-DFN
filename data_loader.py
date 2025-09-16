import pandas as pd

def load_grid_coordinates(file_path):
    grid_df = pd.read_csv(file_path, sep='\t')
    grid_df['center_lon'] = (grid_df['min_lon'] + grid_df['max_lon']) / 2
    grid_df['center_lat'] = (grid_df['min_lat'] + grid_df['max_lat']) / 2
    return grid_df.sort_values('grid_id').reset_index(drop=True)

def load_feature_data(file_path, max_timesteps, n_grids):
    df = pd.read_csv(file_path, skiprows=1, header=None)
    data = df.values[:max_timesteps, :n_grids]
    return pd.DataFrame(data).ffill().bfill().values