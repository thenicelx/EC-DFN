### (1) Experimental Environment:

```
tensorflow>=2.6.0
numpy>=1.19.2
pandas>=1.3.5
scikit-learn>=1.2.1
scipy>=1.10.0
networkx>=3.2.1
python-louvain>=0.16
geopy>=2.4.1
fastdtw>=0.3.4

```

### (2) DataSets

> **"Q-traffic Dataset":**

> This dataset originates from (https://github.com/JingqingZ/BaiduTraffic). In this paper, this dataset integrates multi-modal urban dynamics including: road network topology within geographical bounds [116.10, 39.69,116.71, 40.18], traffic status records from 45,148 road segments (Apr-May 2017), 564 spatial rasters identified through incident frequency analysis, and offline/online auxiliary features sampled at 15-minute intervals.
>
> **"Q-traffic Events Dataset":**

> Derived from navigation search queries, this dataset captures: crowd-sourced event indicators (564 × 5856 dimensional tensor) and real-time adverse weather annotations.

### (3) Module Descriptions

> **"config.py"**:  Centralizes all hyperparameters, file paths, and model configurations for the EC-DFN model, making it convenient for unified modification and parameter tuning.
> **"data_loader.py"**:  Responsible for all functions related to data loading and preprocessing, including reading data from CSV files, performing data standardization, and creating time-series datasets suitable for model training.
> **"graph_builder.py"**:  Used for constructing multi-scale graphs.
> **"model.py"**:  Defines the project's core deep learning model architecture. It not only contains custom Keras layers for spatial feature extraction (such as MultiScaleGCN and CrossScaleGate), but its core lies in the final output layer which integrates a Mixture Density Network (MDN) module. This module can predict the complete probability distribution of traffic flow, thus better capturing data uncertainty.
> **"metrics.py"**:  Contains the loss function and all metrics used for model evaluation (such as MAE, RMSE, R-squared, etc.).
> **"train.py"**:  The main entry point script of the project.
