import os, torch, pandas as pd, numpy as np
from dataset import YourDataset  # 从你原仓库复用
from models import SegFormer      # 从你原仓库复用
from config import DATA_DIR, WEIGHTS_DIR, RESULT_DIR

def run():
    out_csv = os.path.join(RESULT_DIR, "baseline_metrics.csv")
    assert not os.path.exists(out_csv), f"⚠️ {out_csv} 已存在，拒绝覆盖！"

    # 从你原来的 evaluate_segformer.py 中复制数据加载和模型加载的代码
    # ...
    
    results = []
    # 加载模型并推理
    # for images, labels, meta in test_loader:
    #     preds = model(images)
    #     for i in range(len(images)):
    #         tile_id = meta['tile_id'][i]
    #         iou = calculate_iou(preds[i], labels[i])
    #         results.append({"tile_id": tile_id, "continent": meta['continent'][i], "IoU": iou})
    
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("✅ 推理完成，结果已存至", out_csv)

if __name__ == "__main__":
    run()