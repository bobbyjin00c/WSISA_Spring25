import os
import numpy as np
import torch
import config
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from losses import cox_loss
from model import DeepConvSurv
from feature_cluster import load_case_patches_with_clusters
import matplotlib.pyplot as plt


def plot_training_curves(train_loss, val_loss, cluster_id):
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Train Loss', marker='o')
    plt.plot(val_loss, label='Val Loss', marker='o')
    plt.title(f'Loss Curve for Cluster {cluster_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'training_curve_cluster_{cluster_id}.png')
    plt.close()


class SurvivalModel:
    def __init__(self):
        self.cluster_models = {}
        self.aggregator = None
        self.feature_columns = None

    def train_cluster_models(self, dataset, n_epochs=config.EPOCHS, min_cluster_size=10):
        from torch.nn.utils import clip_grad_norm_

        clusters, counts = np.unique([s['cluster'] for s in dataset.samples], return_counts=True)
        valid_clusters = clusters[counts >= min_cluster_size]

        print(f"[INFO] Total samples: {len(dataset)}")
        print(f"[INFO] Valid clusters: {valid_clusters.tolist()}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.enabled = False

        for cluster in valid_clusters:
            print("\n" + "-" * 60)
            print(f"[INFO] 开始训练聚类 {cluster} 模型")
            print("-" * 60)

            cluster_indices = [i for i, s in enumerate(dataset.samples) if s['cluster'] == cluster]
            cluster_subset = torch.utils.data.Subset(dataset, cluster_indices)

            train_idx, val_idx = train_test_split(
                np.arange(len(cluster_subset)),
                test_size=0.2,
                stratify=[cluster_subset[i]['status'].item() for i in range(len(cluster_subset))]
            )

            train_loader = DataLoader(torch.utils.data.Subset(cluster_subset, train_idx),
                                      batch_size=config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(torch.utils.data.Subset(cluster_subset, val_idx),
                                    batch_size=config.BATCH_SIZE, shuffle=False)

            model = DeepConvSurv(dropout_rate=config.DROPOUT_RATE).to(device)
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config.LEARNING_RATE,
                                         weight_decay=config.WEIGHT_DECAY)

            patience = config.EARLY_STOPPING_PATIENCE
            min_delta = config.EARLY_STOPPING_MIN_DELTA
            best_val_loss = float('inf')
            epochs_no_improve = 0

            train_loss_list = []
            val_loss_list = []

            for epoch in range(n_epochs):
                model.train()
                train_loss_total, train_batches = 0.0, 0

                for batch in tqdm(train_loader, desc=f"Train Cluster {cluster} Epoch {epoch+1}"):
                    images = batch['image'].to(device)
                    if images.size(0) == 1:
                        continue
                    times = batch['time'].to(device)
                    statuses = batch['status'].to(device)

                    optimizer.zero_grad()
                    risks = model(images)
                    loss = cox_loss(risks, times, statuses)

                    if not torch.isfinite(loss):
                        continue

                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                    train_loss_total += loss.item()
                    train_batches += 1

                avg_train_loss = train_loss_total / max(train_batches, 1)

                model.eval()
                val_loss_total, val_batches = 0.0, 0
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Val Cluster {cluster} Epoch {epoch+1}"):
                        images = batch['image'].to(device)
                        if images.size(0) == 1:
                            continue
                        times = batch['time'].to(device)
                        statuses = batch['status'].to(device)
                        if torch.sum(statuses) == 0:
                            continue
                        risks = model(images)
                        loss = cox_loss(risks, times, statuses)
                        if not torch.isfinite(loss):
                            continue
                        val_loss_total += loss.item()
                        val_batches += 1

                avg_val_loss = val_loss_total / max(val_batches, 1)

                train_loss_list.append(avg_train_loss)
                val_loss_list.append(avg_val_loss)

                print(f"[Epoch {epoch+1}/{n_epochs}] Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} | EarlyStop: {epochs_no_improve}/{patience}")

                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    self.cluster_models[cluster] = model.state_dict().copy()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"[Early Stop] Cluster {cluster} early stopped at epoch {epoch+1}")
                    break

            plot_training_curves(train_loss_list, val_loss_list, cluster)

        # 恢复为完整模型
        for cluster in self.cluster_models:
            model = DeepConvSurv(dropout_rate=config.DROPOUT_RATE).to(device)
            model.load_state_dict(self.cluster_models[cluster])
            self.cluster_models[cluster] = model

    def aggregate_features(self, clinical_df):
        valid_case_ids = []
        case_features = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for case_id in clinical_df['cases.submitter_id']:
            all_patches, cluster_labels = load_case_patches_with_clusters(
                case_id, config.PATCH_DIR, self.cluster_map, max_patches=config.MAX_PATCHES_PER_CASE
            )
            if all_patches is None:
                continue

            unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
            total_patches = len(cluster_labels)
            cluster_weights = {c: count / total_patches for c, count in zip(unique_clusters, counts)}

            weighted_features = []
            for cluster in self.cluster_models.keys():
                cluster_indices = np.where(cluster_labels == cluster)[0]
                if len(cluster_indices) == 0:
                    weighted_features.append(0)
                    continue
                cluster_patches = all_patches[cluster_indices].to(device)
                model = self.cluster_models[cluster]
                model.eval()
                with torch.no_grad():
                    risks = model(cluster_patches)
                weight = cluster_weights.get(cluster, 0)
                weighted_risk = weight * risks.mean().item()
                weighted_features.append(weighted_risk)

            valid_case_ids.append(case_id)
            case_features.append(weighted_features)

        feature_df = pd.DataFrame(case_features, index=valid_case_ids,
                                  columns=[f'Cluster_{c}' for c in self.cluster_models]).fillna(0)

        final_df = clinical_df.merge(
            feature_df.reset_index().drop_duplicates(subset='index').set_index('index'),
            left_on='cases.submitter_id',
            right_index=True,
            how='inner'
        ).reset_index(drop=True)

        return final_df.dropna()

    def train_cox(self, train_feature_df):
        self.validate_train_data(train_feature_df, min_samples=20, min_features=3)
        assert 'time' in train_feature_df.columns
        assert 'status' in train_feature_df.columns

        self.feature_columns = [c for c in train_feature_df.columns if c.startswith('Cluster_')]

        clean_df = train_feature_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=self.feature_columns + ['time', 'status'])

        # 方差筛选
        selector = VarianceThreshold(threshold=1e-4)
        selector.fit(clean_df[self.feature_columns])
        valid_cols = [col for col, keep in zip(self.feature_columns, selector.get_support()) if keep]

        if not valid_cols:
            raise ValueError("所有聚类特征都被认为方差过低")

        self.feature_columns = valid_cols

        # 标准化
        scaler = StandardScaler()
        clean_df[self.feature_columns] = scaler.fit_transform(clean_df[self.feature_columns])
        self.scaler = scaler

        # 第一次拟合：获取 p 值
        temp_cox = CoxPHFitter(penalizer=0.3, l1_ratio=0.2)
        temp_cox.fit(clean_df[['time', 'status'] + self.feature_columns], duration_col='time', event_col='status')

        # 筛选显著簇
        summary = temp_cox.summary
      # top_n_features = df.columns[:TOP_N_FEATURES]
        significant_features = summary[summary['p'] <= 0.2].index.tolist()
        if not significant_features:
            raise ValueError("没有显著聚类特征（p ≤ 0.1）")

        self.feature_columns = significant_features
        print(f"[INFO] 保留显著簇特征: {self.feature_columns}")

        # 重新训练最终模型
        self.aggregator = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
        self.aggregator.fit(clean_df[['time', 'status'] + self.feature_columns],
                            duration_col='time', event_col='status')

        # 打印完整 summary
        print("\n[INFO] Cox 模型 summary：\n")
        print(self.aggregator.summary.to_string())

        # 自动保存 summary 到 CSV
        os.makedirs("logs", exist_ok=True)
        summary_path = os.path.join("logs", "cox_summary.csv")
        self.aggregator.summary.to_csv(summary_path)
        print(f"[INFO] Cox summary 已保存到: {summary_path}")

    def evaluate(self, df):
        if self.aggregator is None:
            raise RuntimeError("Cox 模型尚未训练。请先调用 train_cox() 或 load_checkpoint()。")

        if not self.feature_columns:
            raise RuntimeError("模型未定义 feature_columns，可能尚未训练。")

        required_cols = {"time", "status"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"输入数据缺少必要列: {required_cols - set(df.columns)}")

        df = df.copy()

        all_required = list(self.scaler.feature_names_in_)
        for col in all_required:
            if col not in df.columns:
                df[col] = 0.0
        df = df[all_required + ["time", "status"]]

        df[all_required] = self.scaler.transform(df[all_required])
        pred = -self.aggregator.predict_partial_hazard(df[self.feature_columns])
        return concordance_index(df["time"], pred, df["status"])
    
    def save_checkpoint(self, path="saved/survival_checkpoint.pkl"):
        import pickle, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "cox_model": self.aggregator,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns
            }, f)
        print(f"[INFO] Cox 模型和 scaler 已保存到: {path}")

    def load_checkpoint(self, path="saved/survival_checkpoint.pkl"):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.aggregator = data["cox_model"]
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]
        print(f"[INFO] 成功加载 checkpoint: {path}")

    def validate_train_data(self, df, min_samples=20, min_features=3):
        if 'time' not in df.columns or 'status' not in df.columns:
            raise ValueError("[ERROR] 缺失 'time' 或 'status' 列")

        cluster_cols = [col for col in df.columns if col.startswith("Cluster_")]
        if len(cluster_cols) < min_features:
            raise ValueError(f"[ERROR] 聚类特征数不足，仅有 {len(cluster_cols)} 个，至少应为 {min_features}")

        if df.shape[0] < min_samples:
            raise ValueError(f"[ERROR] 样本数量过少，仅有 {df.shape[0]} 条，至少应为 {min_samples}")

        constant_cols = [col for col in cluster_cols if df[col].nunique() <= 1]
        if len(constant_cols) == len(cluster_cols):
            raise ValueError(f"[ERROR] 所有聚类特征都是常量或全缺失，无法训练")

        print(f"[CHECK] ✅ 数据检查通过：{df.shape[0]} 样本，{len(cluster_cols)} 聚类特征")


"""
    def train_cox(self, train_feature_df):
        assert 'time' in train_feature_df.columns
        assert 'status' in train_feature_df.columns

        self.feature_columns = [c for c in train_feature_df.columns if c.startswith('Cluster_')]

        clean_df = train_feature_df.replace([np.inf, -np.inf], np.nan).dropna(subset=self.feature_columns + ['time', 'status'])

        selector = VarianceThreshold(threshold=1e-4)
        selected = selector.fit_transform(clean_df[self.feature_columns])
        valid_cols = [col for col, keep in zip(self.feature_columns, selector.get_support()) if keep]

        if not valid_cols:
            raise ValueError("所有聚类特征都被认为方差过低")

        self.feature_columns = valid_cols
        scaler = StandardScaler()
        clean_df[self.feature_columns] = scaler.fit_transform(clean_df[self.feature_columns])
        self.scaler = scaler

        self.aggregator = CoxPHFitter(penalizer=0.1)
        self.aggregator.fit(clean_df[['time', 'status'] + self.feature_columns],
                            duration_col='time', event_col='status')
        print(self.aggregator.summary)

    def evaluate(self, test_feature_df):
        if self.aggregator is None:
            raise ValueError("Cox 模型未训练")

        test_feature_df[self.feature_columns] = self.scaler.transform(test_feature_df[self.feature_columns])
        test_risks = self.aggregator.predict_expectation(test_feature_df[self.feature_columns])

        c_index = concordance_index(
            event_times=test_feature_df['time'],
            predicted_scores=-test_risks,
            event_observed=test_feature_df['status']
        )
        return c_index

"""