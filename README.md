(1) Experimental Environment:

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

(2) DataSets

    "Q-traffic Dataset" ：
    Sourced from Liao, Zhang, Wu, McIlwraith, Chen, Yang, Guo and Wu (2018), this dataset integrates multi-modal
    urban dynamics including: road network topology within geographical bounds [116.10, 39.69,116.71, 40.18], traffic 
    status records from 45,148 road segments (Apr-May 2017), 564 spatial rasters identified through incident frequency
    analysis, and offline/online auxiliary features sampled at 15-minute intervals.
  
    "Q-traffic Events Dataset"：
     Derived from navigation search queries, this dataset captures: crowd-sourced event indicators
    (564 × 5856 dimensional tensor) and real-time adverse weather annotations.。
