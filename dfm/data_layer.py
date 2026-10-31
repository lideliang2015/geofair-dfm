import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import cKDTree
from config import RESULT_DIR

def calculate_spatial_lag_correlation(df, factor_col, target_col='IoU', k=5):
    df_temp = df.copy()
    df_temp['lon'] = df_temp['longitude']
    df_temp['lat'] = df_temp['latitude']
    
    # 剔除无坐标的点（0,0）
    df_temp = df_temp[(df_temp['lon'] != 0.0) & (df_temp['lat'] != 0.0)].copy()
    df_temp = df_temp.dropna(subset=['lon', 'lat'])
    
    if len(df_temp) < 10: 
        return np.nan

    try:
        coords = df_temp[['lon', 'lat']].values
        # 用 scipy 的 KDTree 找 K 个最近邻
        tree = cKDTree(coords)
        dists, indices = tree.query(coords, k=k+1) # 多找1个，因为包含自己
        
        # 构建行标准化权重矩阵
        W = np.zeros((len(coords), len(coords)))
        for i in range(len(coords)):
            neighbors = indices[i][1:] # 排除自己
            for j in neighbors:
                W[i, j] = 1.0 / k
        
        # 计算空间滞后变量
        lag_factor = W @ df_temp[factor_col].values
        lag_target = W @ df_temp[target_col].values

        r, _ = pearsonr(lag_factor, lag_target)
        return r
    except Exception as e:
        print(f"    ⚠️ 空间滞后计算跳过: {e}")
        return np.nan

def run_data_layer_attribution():
    print("="*40)
    print("开始数据层归因分析 (Data-Layer Attribution)")
    print("="*40)

    metrics_file = os.path.join(RESULT_DIR, "baseline_metrics.csv")
    out_corr_file = os.path.join(RESULT_DIR, "table3_data_correlations.csv")
    out_sampling_file = os.path.join(RESULT_DIR, "table4_stratified_sampling.csv")
    
    if os.path.exists(out_corr_file) or os.path.exists(out_sampling_file):
        raise FileExistsError("⚠️ 结果文件已存在，请先备份或重命名，拒绝覆盖！")

    df = pd.read_csv(metrics_file)
    
    factors = ['building_density']
    results = []

    print("\n[1] 计算 Table 3: 数据层因素与 IoU 的相关性...")
    for factor in factors:
        if factor not in df.columns:
            print(f"  - 警告: 列 '{factor}' 不存在，跳过...")
            continue
            
        # 全局 Pearson / Spearman
        r_p, _ = pearsonr(df[factor], df['IoU'])
        r_s, _ = spearmanr(df[factor], df['IoU'])
        # Spatial lag
        r_w = calculate_spatial_lag_correlation(df, factor)
        
        results.append({
            "Factor": factor,
            "Pearson r": round(r_p, 3),
            "Spearman ρ": round(r_s, 3),
            "Spatial-lag r": round(r_w, 3) if not np.isnan(r_w) else "N/A"
        })

    df_corr = pd.DataFrame(results)
    df_corr.to_csv(out_corr_file, index=False)
    print(f"  ✅ Table 3 已保存至: {out_corr_file}")
    print(df_corr.to_string(index=False))

    print("\n[2] 统计各大洲平均 IoU (Table 4)...")
    df_continent = df.groupby('continent')['IoU'].mean().reset_index()
    df_continent = df_continent.sort_values(by='IoU', ascending=False)
    df_continent.columns = ['Region', 'Baseline IoU']
    
    df_continent.to_csv(out_sampling_file, index=False)
    print(f"  ✅ Table 4 已保存至: {out_sampling_file}")
    print(df_continent.to_string(index=False))

if __name__ == "__main__":
    run_data_layer_attribution()