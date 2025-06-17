# cluster_generator.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  
import torch  
import pickle
from feature_cluster import FeatureClusterer
from config import PATCH_DIR, N_CLUSTERS, PCA_DIM

def generate_cluster_map(output_path="cluster_map.pkl"):
    clusterer = FeatureClusterer(n_clusters=N_CLUSTERS, pca_dim=PCA_DIM)
    clusterer.fit(PATCH_DIR)
    cluster_map = clusterer.get_cluster_map()

    with open(output_path, "wb") as f:
        pickle.dump(cluster_map, f)

    print(f"[INFO] 聚类完成，结果保存在: {output_path}")

if __name__ == "__main__":
    generate_cluster_map()
