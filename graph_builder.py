import numpy as np
import networkx as nx
from geopy.distance import geodesic
from fastdtw import fastdtw
from community import community_louvain
from scipy.sparse import csr_matrix


class MultiScaleGraphBuilder:
    def __init__(self, grid_df):
        self.grid_df = grid_df
        self.n_grids = len(grid_df)
        self.coords = list(zip(grid_df.center_lat, grid_df.center_lon))

    def _spatial_mask(self, radius_km):
        mask = np.zeros((self.n_grids, self.n_grids))
        for i in range(self.n_grids):
            for j in range(i + 1, self.n_grids):
                if geodesic(self.coords[i], self.coords[j]).km <= radius_km:
                    mask[i, j] = mask[j, i] = 1
        return mask

    def build_micro_graph(self, speed_features, dtw_window, theta):
        spatial_mask = self._spatial_mask(0.5)
        dtw_sim = np.zeros((self.n_grids, self.n_grids))
        current_dtw_window = min(dtw_window, speed_features.shape[0])

        for i in range(self.n_grids):
            for j in range(i + 1, self.n_grids):
                if spatial_mask[i, j] == 1 and current_dtw_window > 0:
                    hist_i = speed_features[-current_dtw_window:, i]
                    hist_j = speed_features[-current_dtw_window:, j]
                    if len(hist_i) > 0 and len(hist_j) > 0:
                        distance, _ = fastdtw(hist_i, hist_j)
                        if distance is not None:
                            similarity = 1 / (1 + distance)
                            dtw_sim[i, j] = dtw_sim[j, i] = similarity

        return np.where((dtw_sim > theta) & (spatial_mask == 1), 1.0, 0.0)

    def build_meso_graph(self, adj_micro):
        G = nx.Graph(csr_matrix(adj_micro).toarray())
        partition = community_louvain.best_partition(G)
        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)

        adj_meso = np.zeros_like(adj_micro)
        for comm in communities.values():
            for i in comm:
                for j in comm:
                    if geodesic(self.coords[i], self.coords[j]).km <= 2:
                        adj_meso[i, j] = 1
        return adj_meso

    def build_macro_graph(self, adj_meso, alpha):
        out_degree = adj_meso.sum(axis=1)
        transition = adj_meso / np.maximum(out_degree[:, None], 1e-6)
        pr = np.ones(self.n_grids) / self.n_grids
        for _ in range(50):
            pr = alpha * transition.T.dot(pr) + (1 - alpha) / self.n_grids
        return np.outer(pr, pr)