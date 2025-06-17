# main.py

import os
import config
import joblib
from clinical_utils import process_clinical_data, align_svs_clinical, split_patients, filter_clinical_by_existing_patches
from preprocess import WSIPreprocessor 
from feature_cluster import FeatureClusterer
from dataset import WSIDataset
from survival_model import SurvivalModel

REQUIRED_CONFIG_KEYS = [
    "BATCH_SIZE", "EPOCHS", "DROPOUT_RATE", "LEARNING_RATE", "WEIGHT_DECAY",
    "N_CLUSTERS", "PCA_DIM", "PATCH_SIZE", "MAX_PATCHES_PER_CASE",
    "TSV_DIR", "SVS_DIR", "PATCH_DIR"
]

# 参数完整性检查
for key in REQUIRED_CONFIG_KEYS:
    if not hasattr(config, key):
        raise AttributeError(f"[配置错误] 缺少 config.{key}，请在 config.py 中补充。")

# 路径存在性检查
DIRS_TO_CHECK = [config.TSV_DIR, config.SVS_DIR, config.PATCH_DIR]
for path in DIRS_TO_CHECK:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[路径错误] 文件夹不存在：{path}")
if __name__ == "__main__":
    print("[INFO] 加载临床数据...")
    clinical_df = process_clinical_data(config.TSV_DIR)
    filtered_clinical, svs_files = align_svs_clinical(config.SVS_DIR, clinical_df)

    train_ids, val_ids, test_ids = split_patients(filtered_clinical,
                                                  test_size=config.TEST_SIZE,
                                                  val_size=config.VAL_SIZE,
                                                  random_state=config.RANDOM_STATE)

    train_clinical = filtered_clinical[filtered_clinical['cases.submitter_id'].isin(train_ids)]
    val_clinical = filtered_clinical[filtered_clinical['cases.submitter_id'].isin(val_ids)]
    test_clinical = filtered_clinical[filtered_clinical['cases.submitter_id'].isin(test_ids)]

    processor = WSIPreprocessor(patch_size=config.PATCH_SIZE,
                                thumbnail_size=config.THUMBNAIL_SIZE,
                                sample_ratio=config.SAMPLE_RATIO)

    if not os.path.exists(config.PATCH_DIR) or len(os.listdir(config.PATCH_DIR)) == 0:
        print("[INFO] 开始WSI切片处理...")
        for svs_file in svs_files:
            processor.process_slide(os.path.join(config.SVS_DIR, svs_file), config.PATCH_DIR)
    else:
        print("[INFO] 检测到已存在切片数据，跳过WSI切片步骤")

    # ---------- 聚类缓存逻辑 ----------
    if os.path.exists("cluster_map.pkl"):
        print("[INFO] 载入缓存的聚类结果 cluster_map.pkl")
        cluster_map = joblib.load("cluster_map.pkl")
    else:
        print("[INFO] 开始特征聚类...")
        clusterer = FeatureClusterer(n_clusters=config.N_CLUSTERS, pca_dim=config.PCA_DIM)
        clusterer.fit(config.PATCH_DIR)
        cluster_map = clusterer.get_cluster_map()
        joblib.dump(cluster_map, "cluster_map.pkl")
        print("[INFO] 聚类结果已保存为 cluster_map.pkl")

    # 过滤未提取特征的病例
    train_clinical = filter_clinical_by_existing_patches(train_clinical, cluster_map)
    val_clinical = filter_clinical_by_existing_patches(val_clinical, cluster_map)
    test_clinical = filter_clinical_by_existing_patches(test_clinical, cluster_map)

    print("[INFO] 构建训练/验证/测试数据集")
    train_dataset = WSIDataset(cluster_map, train_clinical, config.PATCH_DIR)
    val_dataset = WSIDataset(cluster_map, val_clinical, config.PATCH_DIR)
    test_dataset = WSIDataset(cluster_map, test_clinical, config.PATCH_DIR)

    survival_model = SurvivalModel()
    survival_model.cluster_map = cluster_map

    print("[INFO] 开始训练 DeepConvSurv 聚类模型")
    survival_model.train_cluster_models(train_dataset, n_epochs=config.EPOCHS)

    print("[INFO] 提取训练特征")
    train_feature_df = survival_model.aggregate_features(train_clinical)
    os.makedirs("saved", exist_ok=True)
    train_feature_df.to_pickle("saved/train_features.pkl")

    print("[INFO] 提取测试特征")
    test_feature_df = survival_model.aggregate_features(test_clinical)
    test_feature_df.to_pickle("saved/test_features.pkl")

    print("[INFO] 训练 Cox 聚合模型")
    survival_model.train_cox(train_feature_df)
    survival_model.save_checkpoint("saved/survival_checkpoint.pkl")
    
    print("[INFO] 测算测试集 C-index")
    test_cindex = survival_model.evaluate(test_feature_df)
    print(f"[结果] 测试集 C-index: {test_cindex:.4f}")

