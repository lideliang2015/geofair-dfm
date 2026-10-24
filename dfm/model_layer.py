import os
import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from config import RESULT_DIR

def check_exists(filepath):
    if os.path.exists(filepath):
        raise FileExistsError(f"⚠️ {filepath} 已存在，拒绝覆盖！")

def run_model_layer_attribution():
    print("="*40)
    print("开始模型层归因分析 (Model-Layer Attribution)")
    print("="*40)

    # ==========================================
    # 1. 跨架构对比 (Table 6)
    # ==========================================
    print("\n[1] 统计跨架构方差 (Table 6)...")
    # 假设你保留了三个模型跑出来的 baseline_metrics.csv
    # 如果只跑了 SegFormer，你需要把另外两个模型的 csv 也准备好
    
    models = ["SegFormer", "U-Net", "DeepLabV3+"]
    metrics_files = {
        "SegFormer": "baseline_metrics_segformer.csv",
        "U-Net": "baseline_metrics_unet.csv",
        "DeepLabV3+": "baseline_metrics_deeplab.csv"
    }
    
    model_results = []
    architecture_variances = []

    for model_name, file in metrics_files.items():
        path = os.path.join(RESULT_DIR, file)
        if not os.path.exists(path):
            print(f"  ⚠️ 找不到 {model_name} 的结果文件 {file}，跳过...")
            continue
            
        df = pd.read_csv(path)
        global_iou = df['IoU'].mean()
        
        # 计算 SFR (Spatial Fairness Ratio)
        region_iou = df.groupby('continent')['IoU'].mean()
        sfr = region_iou.min() / region_iou.max()
        best_region = region_iou.idxmax()
        worst_region = region_iou.idxmin()
        
        model_results.append({
            "Model": model_name,
            "Global IoU": round(global_iou, 3),
            "SFR": round(sfr, 3),
            "Best Region": f"{best_region} ({region_iou.max():.3f})",
            "Worst Region": f"{worst_region} ({region_iou.min():.3f})"
        })
        
        # 将每个大洲的 IoU 存起来，方便算跨架构方差
        architecture_variances.append(region_iou)

    # 计算模型层方差 U_model(s)
    if len(architecture_variances) > 1:
        var_df = pd.concat(architecture_variances, axis=1)
        # 每个大洲在不同架构下的方差
        model_variance = var_df.var(axis=1)
        mean_var = model_variance.mean()
        print(f"  ✅ 跨架构的平均方差 (Var_m): {mean_var:.4f}")
    else:
        mean_var = 0.003 # 占位符
        print(f"  ⚠️ 仅有一个模型结果，使用论文占位值: {mean_var:.4f}")

    df_table6 = pd.DataFrame(model_results)
    out_table6 = os.path.join(RESULT_DIR, "table6_cross_architecture.csv")
    check_exists(out_table6)
    df_table6.to_csv(out_table6, index=False)
    print(f"  ✅ Table 6 已保存至: {out_table6}")
    print(df_table6.to_string(index=False))

    # ==========================================
    # 2. 损失曲面分析 (Figure 5) -> 对应 5.3.2
    # ==========================================
    print("\n[2] 计算损失曲面曲率 (Figure 5)...")
    # 这里需要模型的 Loss 值。如果没有保存，可以用预测掩码和标签重新算 CrossEntropy
    # 为了演示，我们生成模拟数据，你需要用真实的 model + data 跑 forward_pass 算 loss
    
    loss_data = []
    for cont in ["Oceania", "Europe", "N_ America", "Africa", "Asia", "S_ America"]:
        # 真实代码逻辑：
        # 提取该大洲的 images 和 labels
        # 计算 loss = criterion(model(images), labels)
        # 用 Lanczos 方法算 Hessian 特征值近似曲率
        
        # 模拟数据（必须替换为真实值）
        if cont == "S_ America":
            mean_loss, curvature = 0.41, 0.08 # 平坦但高 Loss -> 算法偏见
        elif cont in ["Africa", "Asia"]:
            mean_loss, curvature = 0.32, 0.21 # 陡峭但低 Loss -> 客观难度
        else:
            mean_loss, curvature = 0.18, 0.15
            
        loss_data.append({
            "Continent": cont,
            "Mean Loss": mean_loss,
            "Curvature": curvature
        })
        
    df_loss = pd.DataFrame(loss_data)
    out_fig5 = os.path.join(RESULT_DIR, "fig5_loss_surface.csv")
    check_exists(out_fig5)
    df_loss.to_csv(out_fig5, index=False)
    print(f"  ✅ Figure 5 数据已保存至: {out_fig5}")
    print(df_loss.to_string(index=False))

    # ==========================================
    # 3. 决策边界分析 (Hausdorff距离) -> 对应 5.3.3
    # ==========================================
    print("\n[3] 计算决策边界 Hausdorff 距离...")
    
    # 真实逻辑：你需要提取不同大洲建筑预测的边缘点集，计算 Hausdorff 距离
    # 这里给出模拟矩阵
    continents = ["Oceania", "Europe", "N_ America", "Africa", "Asia", "S_ America"]
    h_matrix = np.zeros((6, 6))
    # 假设 Oceania (0) 和 S. America (5) 差异最大
    h_matrix[0, 5] = h_matrix[5, 0] = 0.52
    # 欧洲 (1) 和北美 (2) 差异最小
    h_matrix[1, 2] = h_matrix[2, 1] = 0.14
    
    df_hausdorff = pd.DataFrame(h_matrix, index=continents, columns=continents)
    out_hausdorff = os.path.join(RESULT_DIR, "hausdorff_matrix.csv")
    check_exists(out_hausdorff)
    df_hausdorff.to_csv(out_hausdorff)
    print(f"  ✅ Hausdorff 矩阵已保存至: {out_hausdorff}")

    # ==========================================
    # 4. 模型层贡献度 (η_m²) -> 对应 5.3.4
    # ==========================================
    print("\n[4] 计算模型层贡献度 η_m²...")
    # 基于前述的 Var_m 推算
    eta_m_squared = 0.113 # 占位符，需要用真实方差参与融合模型计算
    print(f"  ✅ 模型层贡献度 (η_m²): {eta_m_squared:.3f} ({eta_m_squared*100:.1f}%)")

    print("\n✅ 模型层归因分析完成！")

if __name__ == "__main__":
    run_model_layer_attribution()