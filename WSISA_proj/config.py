# config.py

import os
import torch

BASE_DIR = "F:/WSISA"

# -------------------------
# 路径设置
# -------------------------
TSV_DIR = os.path.join(BASE_DIR, "clinical_data")
SVS_DIR = os.path.join(BASE_DIR, "svs_images")
PATCH_DIR = os.path.join(BASE_DIR, "processed_patches")

# -------------------------
# Patch 提取配置
# -------------------------
PATCH_SIZE = 512
THUMBNAIL_SIZE = 50
SAMPLE_RATIO = 0.2
MAX_PATCHES_PER_CASE = 150   

# -------------------------
# 聚类与PCA配置
# -------------------------
N_CLUSTERS = 8
PCA_DIM = 100

# -------------------------
# 模型训练配置
# -------------------------
EPOCHS = 30
BATCH_SIZE = 8     
DROPOUT_RATE = 0.15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# -------------------------
# 数据划分与随机性
# -------------------------
RANDOM_STATE = 42
VAL_SIZE = 0.1
TEST_SIZE = 0.2

# -------------------------
# Early Stopping
# -------------------------
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# 特征簇筛选
# -------------------------
