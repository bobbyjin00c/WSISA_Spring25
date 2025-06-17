# load_and_evaluate.py
import pandas as pd
from survival_model import SurvivalModel

test_df = pd.read_pickle("saved/test_features.pkl")

model = SurvivalModel()
model.load_checkpoint("saved/survival_checkpoint.pkl")

print("[INFO] 使用加载的模型测算 C-index")
cindex = model.evaluate(test_df)
print(f"[结果] 测试集 C-index: {cindex:.4f}")
