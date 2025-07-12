(1) Experimental Environment:
```
tensorflow>=2.6.0
tensorflow-estimator>=2.15.0
h5py>=3.1.0
numpy>=1.19.2
absl-py>=0.15.0
astunparse>=1.6.3
flatbuffers>=1.12
gast>=0.4.0
google-pasta>=0.2.0
grpcio>=1.71.0
opt-einsum>=3.3.0
protobuf>=3.20.1
six>=1.15.0
termcolor>=1.1.0
wrapt>=1.12.1
tensorboard>=2.19.0
Markdown>=3.8
pandas>=1.3.5
scikit-learn>=1.2.1
scipy>=1.10.0
joblib>=1.4.2
threadpoolctl>=3.5.0
python-dateutil>=2.9.0.post0
pytz>=2024.1
networkx>=3.2.1
python-louvain>=0.16
geopy>=2.4.1
fastdtw>=0.3.4
geographiclib>=2.0
```

(2) DataSets

    “geo” Sourced from Liao, Zhang, Wu, McIlwraith, Chen, Yang, Guo and Wu (2018), this dataset integrates multi-modal
    urban dynamics including: road network topology within geographical bounds [116.10, 39.69,116.71, 40.18], traffic 
    status records from 45,148 road segments (Apr-May 2017), 564 spatial rasters identified through incident frequency
    analysis, and offline/online auxiliary features sampled at 15-minute intervals.
  
    “features” Derived from navigation search queries, this dataset captures: crowd-sourced event indicators
    (564 × 5856 dimensional tensor) and real-time adverse weather annotations.。
