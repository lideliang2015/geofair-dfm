import os

# 1. 基础目录
BASE_DIR = r"E:\geofair-dfm"
DATA_DIR = r"E:\GeoFair-Building-v1.0"
RESULT_DIR = os.path.join(BASE_DIR, "results")

# 2. 权重目录
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# 3. 旧代码库路径（用于导入 dataset 和 models）
OLD_CODE_DIR = r"E:\geofair_experiments"

# 4. 确保目录存在
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
