import os
import numpy as np
import torch
from skimage import io, transform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm

class FeatureClusterer:
    def __init__(self, n_clusters=10, pca_dim=50):
        self.pca = PCA(n_components=pca_dim)
        self.kmeans = KMeans(n_clusters=n_clusters)

    def fit(self, patches_dir, max_wsi=None):
        all_features, self.filenames = [], []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")

        wsi_list = [wsi_id for wsi_id in os.listdir(patches_dir) if os.path.isdir(os.path.join(patches_dir, wsi_id))]
        if max_wsi:
            wsi_list = wsi_list[:max_wsi]

        tqdm_bar = tqdm(wsi_list, desc="Clustering - Loading patches")

        for idx, wsi_id in enumerate(tqdm_bar):
            wsi_path = os.path.join(patches_dir, wsi_id)
            patch_files = os.listdir(wsi_path)
            tqdm_bar.set_postfix_str(f"WSI {idx+1}/{len(wsi_list)}: {wsi_id} ({len(patch_files)} patches)")

            for patch_file in patch_files:
                patch_path = os.path.join(wsi_path, patch_file)
                try:
                    patch = io.imread(patch_path)
                    thumbnail = transform.resize(patch, (50, 50), anti_aliasing=True)
                    all_features.append(thumbnail.flatten())
                    self.filenames.append((wsi_id, patch_file))
                except Exception as e:
                    print(f"[ERROR] Failed to load patch: {patch_path} | {e}")
                    continue

        if not all_features:
            raise ValueError("No valid patches found for clustering.")

        print(f"[INFO] Feature matrix shape: {len(all_features)} x {len(all_features[0])}")
        print("[INFO] Starting PCA...")
        self.pca.fit(all_features)
        reduced = self.pca.transform(all_features)

        print("[INFO] Starting KMeans clustering...")
        self.kmeans.fit(reduced)

        unique, counts = np.unique(self.kmeans.labels_, return_counts=True)
        print(f"[INFO] KMeans Clusters: {unique.tolist()} | Counts: {counts.tolist()}")

    def get_cluster_map(self):
        return {
            filename: cluster
            for filename, cluster in zip(self.filenames, self.kmeans.labels_)
        }


def load_case_patches_with_clusters(case_id, patches_dir, cluster_map, max_patches=200):
    all_patches = []
    cluster_labels = []
    for (wsi_id, patch_name), cluster in cluster_map.items():
        if wsi_id.startswith(case_id):
            img_path = os.path.join(patches_dir, wsi_id, patch_name)
            try:
                img = io.imread(img_path)
                img_tensor = torch.FloatTensor(img).permute(2, 0, 1) / 255.0
                all_patches.append(img_tensor.unsqueeze(0))
                cluster_labels.append(cluster)
            except Exception as e:
                print(f"[ERROR] Skipping corrupted patch {img_path}: {e}")
                continue

    if len(all_patches) > max_patches:
        all_patches = all_patches[:max_patches]
        cluster_labels = cluster_labels[:max_patches]

    if len(all_patches) == 0:
        return None, None
    return torch.cat(all_patches, dim=0), np.array(cluster_labels)
