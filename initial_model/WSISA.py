import os
import re
import pandas as pd
import numpy as np
from openslide import OpenSlide
from skimage import io, transform, exposure
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# ----------------------
# 第一部分：数据对齐与临床数据处理
# ----------------------
def process_clinical_data(tsv_dir):
    clinical = pd.read_csv(f"{tsv_dir}/clinical.tsv", sep='\t')
    follow_up = pd.read_csv(f"{tsv_dir}/follow_up.tsv", sep='\t')
    
    # 列名映射（基于实际字段）
    follow_up = follow_up.rename(columns={
        "follow_ups.days_to_follow_up": "days_to_last_follow_up",
        "follow_ups.progression_or_recurrence": "status"
    })
    
    # 生成生存时间（假设无死亡时间）
    follow_up["days_to_death"] = np.nan  # 标记为缺失
    
    # 合并生存数据
    survival_df = follow_up.groupby("cases.submitter_id").agg({
        "days_to_last_follow_up": "max",
        "status": "last"
    }).reset_index()
    
    # 清洗逻辑
    merged = pd.merge(
        clinical[["cases.submitter_id", "demographic.gender", "demographic.age_at_index"]],
        survival_df,
        on="cases.submitter_id"
    )
    
    # merged["time"] = merged["days_to_last_follow_up"]
    merged["time"] = pd.to_numeric(merged["days_to_last_follow_up"], errors='coerce')
    merged["status"] = (merged["status"] == "Yes").astype(int)  # 转换二分类
    
    return merged.dropna(subset=["time", "status"])
def align_svs_clinical(svs_dir, clinical_df):
    """对齐SVS文件与临床数据"""
    # 从文件名提取case_id
    def extract_case_id(f):
        match = re.match(r'^([A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)',  f)
        return match.group(1) if match else None
    
    svs_files = [f for f in os.listdir(svs_dir) if f.endswith('.svs')]
    svs_cases = [extract_case_id(f) for f in svs_files]
    
    # 筛选匹配的临床数据
    filtered_clinical = clinical_df[
        clinical_df['cases.submitter_id'].isin(svs_cases)
    ].copy()
    
    print(f"成功匹配 {len(filtered_clinical)}/{len(svs_cases)} 病例")
    return filtered_clinical, svs_files
def split_patients(clinical_df, test_size=0.2, val_size=0.1, random_state=42):
    """分层划分患者到训练、验证、测试集"""
    # 获取唯一患者ID及对应标签
    patients = clinical_df[['cases.submitter_id', 'status']].drop_duplicates()
    patient_ids = patients['cases.submitter_id'].values
    status = patients['status'].values
    
    # 先划分测试集
    train_val_ids, test_ids = train_test_split(
        patient_ids, 
        test_size=test_size, 
        stratify=status,
        random_state=random_state
    )
    
    # 获取剩余患者的标签
    remaining_status = patients[patients['cases.submitter_id'].isin(train_val_ids)]['status'].values
    
    # 划分验证集（占剩余样本的比例）
    val_ratio = val_size / (1 - test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids, 
        test_size=val_ratio, 
        stratify=remaining_status,
        random_state=random_state
    )
    
    return train_ids, val_ids, test_ids
# ----------------------
# 第二部分：WSI预处理与特征提取
# ----------------------
class WSIPreprocessor:
    """WSI处理与Patch采样"""
    def __init__(self, patch_size=512, thumbnail_size=50, sample_ratio=0.1):
        self.patch_size = patch_size
        self.thumbnail_size = thumbnail_size
        self.sample_ratio = sample_ratio
    

    def process_slide(self, svs_path, output_dir):
        slide = OpenSlide(svs_path)
        wsi_id = os.path.basename(svs_path).split('.')[0]
        os.makedirs(f"{output_dir}/{wsi_id}", exist_ok=True)
            
        level = 0
        dims = slide.level_dimensions[level]
        # 检查WSI尺寸是否足够大
        if dims[0] < self.patch_size or dims[1] < self.patch_size:
            print(f"WSI {wsi_id} is too small ({dims[0]}x{dims[1]}) for patch size {self.patch_size}. Skipping.")
            return None
                    
        num_patches = int((dims[0]*dims[1]*self.sample_ratio) / (self.patch_size**2))
        coords = np.random.randint(0, [dims[0]-self.patch_size, dims[1]-self.patch_size], 
                                  size=(num_patches, 2))
            
        valid_patches = []
        for i, (x, y) in enumerate(coords):
            try:
                patch = slide.read_region((x, y), level, (self.patch_size, self.patch_size))
                patch = np.array(patch.convert('RGB'))
                    
                if exposure.is_low_contrast(patch, fraction_threshold=0.3):
                    continue
                    
                    # 保存时处理可能的错误
                try:
                    io.imsave(f"{output_dir}/{wsi_id}/patch_{i}.png", patch)
                except Exception as e:
                    print(f"Error saving patch {i} for {wsi_id}: {e}")
                    continue
                    
                thumbnail = transform.resize(patch, (self.thumbnail_size,)*2)
                valid_patches.append(thumbnail)
            except Exception as e:
                print(f"Error processing patch {i} for {wsi_id}: {e}")
                continue
            
        return np.stack(valid_patches) if valid_patches else None
# ----------------------
# 第三部分：特征聚类与分析
# ----------------------
class FeatureClusterer:
    """特征聚类与降维"""
    def __init__(self, n_clusters=10, pca_dim=50):
        self.pca = PCA(n_components=pca_dim)
        self.kmeans = KMeans(n_clusters=n_clusters)
        
    # def fit(self, patches_dir):
    #     """训练聚类模型"""
    #     all_features, self.filenames = [], []
        
    #     # 收集所有缩略图特征
    #     for wsi_id in os.listdir(patches_dir):
    #         for patch_file in os.listdir(f"{patches_dir}/{wsi_id}"):
    #             patch = io.imread(f"{patches_dir}/{wsi_id}/{patch_file}")
    #             thumbnail = transform.resize(patch, (50, 50))
    #             all_features.append(thumbnail.flatten())
    #             self.filenames.append((wsi_id, patch_file))
                
    #     # 降维与聚类
    #     self.pca.fit(all_features)
    #     reduced = self.pca.transform(all_features)
    #     self.kmeans.fit(reduced)

    def fit(self, patches_dir):
        all_features, self.filenames = [], []
        # 设备配置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")
        for wsi_id in os.listdir(patches_dir):
            wsi_path = os.path.join(patches_dir, wsi_id)
            if not os.path.isdir(wsi_path):
                continue  # 跳过非目录文件
            for patch_file in os.listdir(wsi_path):
                patch_path = os.path.join(wsi_path, patch_file)
                try:
                    patch = io.imread(patch_path)
                    thumbnail = transform.resize(patch, (50, 50))
                    all_features.append(thumbnail.flatten())
                    self.filenames.append((wsi_id, patch_file))
                except Exception as e:
                    print(f"Error loading {patch_path}: {e}")
                    continue  # 跳过损坏的文件
        
        # 继续执行PCA和KMeans
        if not all_features:
            raise ValueError("No valid patches found for clustering.")
        
        print(f"Feature matrix shape: {len(all_features)}x{len(all_features[0])}")
        self.pca.fit(all_features)
        reduced = self.pca.transform(all_features)
        self.kmeans.fit(reduced)
        unique, counts = np.unique(self.kmeans.labels_, return_counts=True)
        print(f"KMeans results - Clusters: {unique} Counts: {counts}")
        
    def get_cluster_map(self):
        """生成聚类分配字典"""
        return {
            filename: cluster 
            for filename, cluster in zip(self.filenames, self.kmeans.labels_)
        }

    def get_cluster_labels(self, case_id, patches_dir):
        """获取指定病例所有patch的聚类标签"""
        cluster_labels = []
        for (wsi_id, patch_name), cluster in self.cluster_map.items():
            if wsi_id.startswith(case_id):
                cluster_labels.append(cluster)
        return np.array(cluster_labels)
def load_case_patches_with_clusters(case_id, patches_dir, cluster_map):
    """加载patch及其聚类标签"""
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
                continue
    if len(all_patches) == 0:
        return None, None
    return torch.cat(all_patches, dim=0), np.array(cluster_labels)
# ----------------------
# 第四部分：深度生存模型
# ----------------------
class WSIDataset(Dataset):
    def __init__(self, cluster_map, clinical_df, patches_dir):
        self.samples = []

        print("\n[DEBUG] Cluster map samples:")
        for (wsi_id, _), cluster in list(cluster_map.items())[:5]:
            print(f"WSI ID: {wsi_id} | Cluster: {cluster}")

        for (wsi_id, patch_name), cluster in cluster_map.items():
            case_id_match = re.match(r'^([A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)', wsi_id)
            if not case_id_match:
                continue
            case_id = case_id_match.group(1)
            clinical_info = clinical_df[clinical_df['cases.submitter_id'] == case_id]
            if not clinical_info.empty:
                img_path = os.path.join(patches_dir, wsi_id, patch_name)
                if os.path.exists(img_path):
                    self.samples.append({
                        'image': img_path,
                        'cluster': cluster,
                        'time': clinical_info['time'].values[0],
                        'status': clinical_info['status'].values[0]
                    })

        print(f"[DEBUG] 实际加载样本数: {len(self.samples)}")
        if not self.samples:
            print("[WARNING] 没有找到匹配样本，此数据集为空。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['image'])
        img_tensor = torch.FloatTensor(img).permute(2, 0, 1) / 255.0
        return {
            'image': img_tensor,
            'cluster': torch.tensor(sample['cluster'], dtype=torch.long),
            'time': torch.tensor(sample['time'], dtype=torch.float),
            'status': torch.tensor(sample['status'], dtype=torch.float)
        }
# ----------------------
# 深度卷积生存网络定义
# ----------------------
class DeepConvSurv(nn.Module):
    def __init__(self, in_channels=3, dropout_rate=0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 10 * 10, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.risk_predictor = nn.Linear(32, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        risk = self.risk_predictor(features)
        return risk
# ----------------------
# Cox损失函数实现
# ----------------------
def cox_loss(pred_risk, surv_time, surv_status, eps=1e-6):
    pred_risk = pred_risk.view(-1)
    surv_time = surv_time.view(-1)
    surv_status = surv_status.view(-1)

    if torch.sum(surv_status) == 0:
        return torch.tensor(0.0, device=pred_risk.device, requires_grad=True)

    sort_idx = torch.argsort(-surv_time)
    pred_risk = pred_risk[sort_idx]
    surv_status = surv_status[sort_idx]

    hazard_ratio = torch.exp(pred_risk - pred_risk.max())
    cumsum_hr = torch.cumsum(hazard_ratio, dim=0)
    log_risk = torch.log(cumsum_hr + eps)
    loss = -torch.sum((pred_risk - log_risk) * surv_status)

    return loss / (torch.sum(surv_status) + eps)
class SurvivalModel:
    """生存分析流程"""
    def __init__(self):
        self.cluster_models = {}
        self.aggregator = None
        self.feature_columns = None  # 保存特征列名
        

    def train_cluster_models(self, dataset, n_epochs=10, min_cluster_size=10):
        import math
        from sklearn.model_selection import train_test_split
        from torch.nn.utils import clip_grad_norm_

        clusters, counts = np.unique([s['cluster'] for s in dataset.samples], return_counts=True)
        valid_clusters = clusters[counts >= min_cluster_size]

        print("Cluster validation:")
        print(f"Total samples: {len(dataset)}")
        print(f"Valid clusters: {valid_clusters}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.enabled = False  # 为防止 cuDNN plan 报错
        print(f"Training on {device}")

        for cluster in valid_clusters:
            print(f"\n[INFO] 开始训练聚类 {cluster} 模型")

            cluster_indices = [i for i, s in enumerate(dataset.samples) if s['cluster'] == cluster]
            cluster_subset = torch.utils.data.Subset(dataset, cluster_indices)

            train_idx, val_idx = train_test_split(
                np.arange(len(cluster_subset)),
                test_size=0.2,
                stratify=[cluster_subset[i]['status'].item() for i in range(len(cluster_subset))]
            )
            train_loader = DataLoader(torch.utils.data.Subset(cluster_subset, train_idx), batch_size=8, shuffle=True)
            val_loader = DataLoader(torch.utils.data.Subset(cluster_subset, val_idx), batch_size=8, shuffle=False)

            model = DeepConvSurv().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
            best_val_loss = math.inf

            for epoch in range(n_epochs):
                model.train()
                train_loss_total, train_batches = 0.0, 0
                for batch in train_loader:
                    images = batch['image'].to(device)
                    if images.size(0) == 1:
                        continue  # 跳过 batch size 为 1 的情况，防止 BatchNorm 报错

                    times = batch['time'].to(device)
                    statuses = batch['status'].to(device)

                    if torch.any(torch.isnan(images)):
                        continue

                    optimizer.zero_grad()
                    risks = model(images)
                    loss = cox_loss(risks, times, statuses)

                    if not torch.isfinite(loss):
                        print(f"[WARNING] 非有限训练损失，跳过")
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
                    for batch in val_loader:
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

                print(f"Cluster {cluster} | Epoch {epoch+1}/{n_epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.cluster_models[cluster] = model.state_dict().copy()

            print(f"[INFO] 聚类 {cluster} 训练完成，最佳 Val Loss: {best_val_loss:.4f}")

        for cluster in self.cluster_models:
            model = DeepConvSurv().to(device)
            model.load_state_dict(self.cluster_models[cluster])
            self.cluster_models[cluster] = model



    def aggregate_features(self, clinical_df):
        """处理缺失case的情况，并添加加权逻辑"""
        valid_case_ids = []
        case_features = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for case_id in clinical_df['cases.submitter_id']:
            # 加载所有patch并获取聚类标签
            all_patches, cluster_labels = load_case_patches_with_clusters(case_id, PATCH_DIR, cluster_map)  # 需补充此函数
            if all_patches is None:
                continue
            
            # 统计每个聚类的patch数量
            unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
            total_patches = len(cluster_labels)
            cluster_weights = {c: count/total_patches for c, count in zip(unique_clusters, counts)}
            
            # 加权特征生成
            weighted_features = []
            for cluster in self.cluster_models.keys():
                # 获取该簇的patch索引
                cluster_indices = np.where(cluster_labels == cluster)[0]
                if len(cluster_indices) == 0:
                    weighted_features.append(0)  # 无patch时填充0
                    continue
                    
                # 提取对应patch并计算特征
                cluster_patches = all_patches[cluster_indices].to(device)
                model = self.cluster_models[cluster]
                model.eval()
                with torch.no_grad():
                    risks = model(cluster_patches)
                
                # 应用权重（该簇比例 * 平均风险）
                weight = cluster_weights.get(cluster, 0)
                weighted_risk = weight * risks.mean().item()
                weighted_features.append(weighted_risk)
            
            valid_case_ids.append(case_id)
            case_features.append(weighted_features)
        
        # 创建DataFrame（后续逻辑保持不变）
        feature_df = pd.DataFrame(
            case_features,
            index=valid_case_ids,
            columns=[f'Cluster_{c}' for c in self.cluster_models]
        ).fillna(0)  # 填充可能的缺失值
        
        final_df = clinical_df.merge(
            feature_df, 
            left_on='cases.submitter_id',
            right_index=True,
            how='inner'
        )
        return final_df.dropna()
    

    def train_cox(self, train_feature_df):
        """训练Cox比例风险模型，修复低方差/NaN/inf等问题"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import VarianceThreshold

        assert 'time' in train_feature_df.columns, "Missing survival time"
        assert 'status' in train_feature_df.columns, "Missing event status"

        # 提取聚类特征列
        self.feature_columns = [c for c in train_feature_df.columns if c.startswith('Cluster_')]

        # 清除包含NaN/inf的行
        clean_df = train_feature_df.replace([np.inf, -np.inf], np.nan).dropna(subset=self.feature_columns + ['time', 'status'])

        # 过滤方差过低的列（阈值可调）
        selector = VarianceThreshold(threshold=1e-4)
        selected = selector.fit_transform(clean_df[self.feature_columns])
        valid_cols = [col for col, keep in zip(self.feature_columns, selector.get_support()) if keep]

        if not valid_cols:
            raise ValueError("所有聚类特征都被认为方差过低，无法训练Cox模型。")

        print(f"[INFO] 保留的特征列: {valid_cols}")
        self.feature_columns = valid_cols

        # 标准化
        scaler = StandardScaler()
        clean_df[self.feature_columns] = scaler.fit_transform(clean_df[self.feature_columns])
        self.scaler = scaler

        # 拟合 Cox 模型（加正则）
        self.aggregator = CoxPHFitter(penalizer=0.1)
        self.aggregator.fit(
            clean_df[['time', 'status'] + self.feature_columns],
            duration_col='time',
            event_col='status'
        )

        print("\nCox模型训练结果：")
        print(self.aggregator.summary)


    def evaluate(self, test_feature_df):
        """评估测试集并返回C-index"""
        if self.aggregator is None:
            raise ValueError("必须先训练Cox模型")
            
        # 应用相同的特征标准化
        test_feature_df[self.feature_columns] = self.scaler.transform(
            test_feature_df[self.feature_columns]
        )
        
        # 预测风险评分
        test_risks = self.aggregator.predict_expectation(
            test_feature_df[self.feature_columns]
        )
        
        # 计算C-index（注意符号方向）
        c_index = concordance_index(
            event_times=test_feature_df['time'],
            predicted_scores=-test_risks,  # 高风险对应短生存时间
            event_observed=test_feature_df['status']
        )
        return c_index
# ----------------------
# 主执行流程
# ----------------------
def filter_clinical_by_existing_patches(clinical_df, cluster_map):
    cluster_case_ids = set([
        re.match(r'^([A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)', wsi_id).group(1)
        for (wsi_id, _) in cluster_map.keys()
    ])
    return clinical_df[clinical_df['cases.submitter_id'].isin(cluster_case_ids)]
if __name__ == "__main__":
    # 添加测试代码验证网络结构
    test_input = torch.randn(32, 3, 512, 512)  # 模拟batch_size=32的输入
    model = DeepConvSurv()
    print("Feature extractor output shape:", model.feature_extractor(test_input).shape)


    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TSV_DIR = os.path.join(BASE_DIR, "WSISA/clinical_data")
    SVS_DIR = os.path.join(BASE_DIR, "WSISA/svs_images")
    PATCH_DIR = os.path.join(BASE_DIR, "WSISA/processed_patches")

    clinical_df = process_clinical_data(TSV_DIR)
    filtered_clinical, svs_files = align_svs_clinical(SVS_DIR, clinical_df)
    train_ids, val_ids, test_ids = split_patients(filtered_clinical)

    train_clinical = filtered_clinical[filtered_clinical['cases.submitter_id'].isin(train_ids)]
    val_clinical = filtered_clinical[filtered_clinical['cases.submitter_id'].isin(val_ids)]
    test_clinical = filtered_clinical[filtered_clinical['cases.submitter_id'].isin(test_ids)]

    processor = WSIPreprocessor(sample_ratio=0.1)
    if not os.path.exists(PATCH_DIR) or len(os.listdir(PATCH_DIR)) == 0:
        print("开始WSI切片处理...")
        for svs_file in svs_files:
            processor.process_slide(f"{SVS_DIR}/{svs_file}", PATCH_DIR)
    else:
        print("检测到已存在切片数据，跳过WSI切片步骤")

    print("特征聚类")
    clusterer = FeatureClusterer(n_clusters=10)
    clusterer.fit(PATCH_DIR)
    cluster_map = clusterer.get_cluster_map()

    train_clinical = filter_clinical_by_existing_patches(train_clinical, cluster_map)
    val_clinical = filter_clinical_by_existing_patches(val_clinical, cluster_map)
    test_clinical = filter_clinical_by_existing_patches(test_clinical, cluster_map)

    print(f"[INFO] 训练集匹配病例数: {len(train_clinical)}")
    print(f"[INFO] 验证集匹配病例数: {len(val_clinical)}")
    print(f"[INFO] 测试集匹配病例数: {len(test_clinical)}")

    print("准备训练数据")
    train_dataset = WSIDataset(cluster_map, train_clinical, PATCH_DIR)
    val_dataset = WSIDataset(cluster_map, val_clinical, PATCH_DIR)
    test_dataset = WSIDataset(cluster_map, test_clinical, PATCH_DIR)

    print("survival")
    survival_model = SurvivalModel()
    survival_model.cluster_map = cluster_map
    print("DeepConvSurv")

    # 修复 clusters 未定义的错误
    from collections import Counter
    cluster_labels = [s['cluster'] for s in train_dataset.samples]
    cluster_counts = Counter(cluster_labels)
    clusters = list(cluster_counts.keys())
    counts = list(cluster_counts.values())
    print("Cluster validation:")
    print(f"Total cases in dataset: {len(train_dataset.samples)}")
    print(f"Unique clusters detected: {len(clusters)}")
    print(f"Cluster size distribution:\n{pd.Series(counts).describe()}")

    survival_model.train_cluster_models(train_dataset, n_epochs=5)

    print("特征聚合")
    train_feature_df = survival_model.aggregate_features(train_clinical)

    print("\n训练Cox聚合模型...")
    survival_model.train_cox(train_feature_df)

    print("测试集评估")
    test_feature_df = survival_model.aggregate_features(test_clinical)

    missing_cols = set(survival_model.feature_columns) - set(test_feature_df.columns)
    for c in missing_cols:
        test_feature_df[c] = 0.0

    c_index = survival_model.evaluate(test_feature_df)
    print(f"\n测试集C-index: {c_index:.3f}")