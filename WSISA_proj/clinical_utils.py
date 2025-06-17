# clinical_utils.py

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def process_clinical_data(tsv_dir):
    clinical = pd.read_csv(f"{tsv_dir}/clinical.tsv", sep='\t')
    follow_up = pd.read_csv(f"{tsv_dir}/follow_up.tsv", sep='\t')

    follow_up = follow_up.rename(columns={
        "follow_ups.days_to_follow_up": "days_to_last_follow_up",
        "follow_ups.progression_or_recurrence": "status"
    })
    follow_up["days_to_death"] = np.nan

    survival_df = follow_up.groupby("cases.submitter_id").agg({
        "days_to_last_follow_up": "max",
        "status": "last"
    }).reset_index()

    merged = pd.merge(
        clinical[["cases.submitter_id", "demographic.gender", "demographic.age_at_index"]],
        survival_df,
        on="cases.submitter_id"
    )

    merged["time"] = pd.to_numeric(merged["days_to_last_follow_up"], errors='coerce')
    merged["status"] = (merged["status"] == "Yes").astype(int)
    return merged.dropna(subset=["time", "status"])

def align_svs_clinical(svs_dir, clinical_df):
    def extract_case_id(f):
        match = re.match(r'^([A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)',  f)
        return match.group(1) if match else None

    svs_files = [f for f in os.listdir(svs_dir) if f.endswith('.svs')]
    svs_cases = [extract_case_id(f) for f in svs_files]

    filtered_clinical = clinical_df[
        clinical_df['cases.submitter_id'].isin(svs_cases)
    ].copy()

    print(f"\u6210\u529f\u5339\u914d {len(filtered_clinical)}/{len(svs_cases)} \u75c5\u4f8b")
    return filtered_clinical, svs_files

def split_patients(clinical_df, test_size=0.2, val_size=0.1, random_state=42):
    patients = clinical_df[['cases.submitter_id', 'status']].drop_duplicates()
    patient_ids = patients['cases.submitter_id'].values
    status = patients['status'].values

    train_val_ids, test_ids = train_test_split(
        patient_ids,
        test_size=test_size,
        stratify=status,
        random_state=random_state
    )

    remaining_status = patients[patients['cases.submitter_id'].isin(train_val_ids)]['status'].values
    val_ratio = val_size / (1 - test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_ratio,
        stratify=remaining_status,
        random_state=random_state
    )

    return train_ids, val_ids, test_ids

def filter_clinical_by_existing_patches(clinical_df, cluster_map):
    cluster_case_ids = set([
        re.match(r'^([A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)', wsi_id).group(1)
        for (wsi_id, _) in cluster_map.keys()
    ])
    return clinical_df[clinical_df['cases.submitter_id'].isin(cluster_case_ids)]
