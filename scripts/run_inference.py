import os, sys, torch, pandas as pd, numpy as np
from tqdm import tqdm
import torch.nn as nn

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

RESULT_DIR = r"E:\geofair-dfm\results"
DATASET_ROOT = r"E:\GeoFair-Building-v1.0"
WEIGHT_PATH = r"E:\experiments\segformer_fair_best.pth"
METADATA_PATH = r"E:\GeoFair-Building-v1.0\metadata.csv" # 🚨 新增：读取 metadata

os.makedirs(RESULT_DIR, exist_ok=True)

sys.path.insert(0, r'E:/geofair_experiments')
from dataset import get_dataloaders
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class LocalSegFormer(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        return nn.functional.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    out_csv = os.path.join(RESULT_DIR, "baseline_metrics.csv")
    out_feat = os.path.join(RESULT_DIR, "deep_features.npy")
    mask_dir = os.path.join(RESULT_DIR, "masks")
    
    if os.path.exists(out_csv):
        raise FileExistsError(f"⚠️ {out_csv} 已存在，拒绝覆盖！请先清理 results 目录。")
    os.makedirs(mask_dir, exist_ok=True)

    # 1. 读取 CSV 建立 tile_id 到经纬度的映射字典
    print("正在读取 metadata.csv...")
    meta_df = pd.read_csv(METADATA_PATH)
    meta_dict = meta_df.set_index('tile_id').to_dict('index')
    print(f"✅ 成功读取 {len(meta_dict)} 条元数据")

    # 2. 加载数据
    print("正在加载数据...")
    _, _, test_loader = get_dataloaders(DATASET_ROOT, batch_size=8, num_workers=4)
    print("✅ 数据加载成功")

    # 3. 构建模型架构
    print("正在构建模型结构...")
    try:
        config_ = SegformerConfig.from_pretrained("nvidia/mit-b2", local_files_only=True)
        config_.num_labels = 1 
        hf_model = SegformerForSemanticSegmentation(config_)
        model = LocalSegFormer(hf_model).to(device)
        
        print(f"正在加载本地权重: {WEIGHT_PATH}")
        checkpoint = torch.load(WEIGHT_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print("✅ 权重加载成功")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 挂载特征 Hook
    features_list = []
    def hook_fn(module, input, output):
        if hasattr(output, 'last_hidden_state'):
            features_list.append(output.last_hidden_state.detach().cpu())
        elif isinstance(output, tuple):
            features_list.append(output[0].detach().cpu())
        else:
            features_list.append(output.detach().cpu())

    model.model.segformer.register_forward_hook(hook_fn)

    results = []
    
    # 5. 推理
    print("开始跑推理...")
    with torch.no_grad():
        for images, masks, metadata in tqdm(test_loader, desc='Running Inference'):
            images = images.to(device)
            logits = model(images)
            preds = torch.sigmoid(logits).cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            # 获取当前 batch 的 tile_ids 和 大洲
            tile_ids = metadata['tile_id'] if 'tile_id' in metadata else [f"tile_{len(results)+i}" for i in range(len(images))]
            continents = metadata['continent']
            
            for i in range(len(images)):
                tile_id = tile_ids[i]
                continent = continents[i]
                
                # 从我们刚才建立的字典中补全完整的经纬度和相关属性
                m_info = meta_dict.get(tile_id, {})
                lon = m_info.get('longitude', 0.0)
                lat = m_info.get('latitude', 0.0)
                building_density = m_info.get('building_density', 0.0)
                hdi_level = m_info.get('hdi_level', 'unknown')

                # 关键：修复 Asia 全为 0 的问题。如果你的标签是 0/255，需要改成 > 0
                pred = (preds[i] > 0.5).astype(np.uint8)
                target = (masks_np[i] > 0.5).astype(np.uint8)
                
                if pred.ndim == 4: pred = pred.squeeze(0)
                if target.ndim == 4: target = target.squeeze(0)
                
                intersection = np.sum((pred == 1) & (target == 1))
                union = np.sum((pred == 1) | (target == 1))
                iou = intersection / (union + 1e-6)

                np.save(os.path.join(mask_dir, f"{tile_id}.npy"), pred)

                # 保存到最终的结果中
                results.append({
                    "tile_id": tile_id,
                    "continent": continent,
                    "longitude": lon,
                    "latitude": lat,
                    "hdi_level": hdi_level,
                    "building_density": building_density,
                    "IoU": iou
                })

    # 6. 保存结果
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"✅ 推理完成！结果保存在 {out_csv}")

    if len(features_list) > 0:
        features_array = torch.cat(features_list, dim=0).numpy()
        if len(features_array.shape) == 4:
            if features_array.shape[-1] == 512:
                features_array = features_array.mean(axis=(1, 2))
            else:
                features_array = features_array.mean(axis=(2, 3))
        features_array = features_array[:len(results)]
        np.save(out_feat, features_array)
        print(f"✅ 特征已保存！形状: {features_array.shape}")

if __name__ == "__main__":
    run()
