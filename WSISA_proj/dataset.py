# dataset.py

import os
import re
import torch
from torch.utils.data import Dataset
from skimage import io

class WSIDataset(Dataset):
    def __init__(self, cluster_map, clinical_df, patches_dir):
        self.samples = []

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